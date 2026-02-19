# Copyright (C) 2020-2026 Mitsubishi Electric Research Laboratories (MERL)
# Copyright (c) 2019 Tor Aksel N. Heirung
#
# SPDX-License-Identifier: AGPL-3.0-or-later
# SPDX-License-Identifier: MIT

# Code purpose:  Define the methods involving just the Polytope class

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import numpy as np
from scipy.spatial import ConvexHull

from pycvxset.common import compute_irredundant_affine_set_using_cdd, convex_set_minimum_volume_circumscribing_rectangle
from pycvxset.common.constants import PYCVXSET_ZERO

if TYPE_CHECKING:
    from pycvxset.ConstrainedZonotope import ConstrainedZonotope
    from pycvxset.Ellipsoid import Ellipsoid
    from pycvxset.Polytope import Polytope


def chebyshev_centering(self: Polytope) -> tuple[Optional[np.ndarray], float]:
    r"""Computes a ball with the largest radius that fits within the polytope. The ball's center is known as the
    Chebyshev center, and its radius is the Chebyshev radius.

    Raises:
        NotImplementedError: Unable to solve chebyshev centering problem using CVXPY

    Returns:
        tuple: A tuple with two items
            #. center (numpy.ndarray): Chebyshev center of the polytope
            #. radius (float): Chebyshev radius of the polytope

    Notes:
        This function is called by the constructor for non-emptiness and boundedness check. It uses limited attributes
        (dim, A, b, Ae, be). This function requires H-Rep, and will perform a halfspace enumeration when a V-Rep
        polytope is provided.

        We solve the LP (see Section 8.5.1 in [BV04]_) for :math:`c` (for `center`) and :math:`R` (for `radius`),

        .. math ::
            \text{maximize}     &\quad R \\
            \text{subject to}   &\quad A c + R ||A||_\text{row} \leq b,\\
                                &\quad A_e c = b_e, \\
                                &\quad R \geq 0,

        where :math:`||A||_\text{row}` is a vector of dimension :attr:`n_halfspaces` with each element
        as :math:`||a_i||_2`. When (Ae, be) is non-empty, then R is forced to zero post-solve. Consequently,
        chebyshev_centering also serves a method to find a feasible point in the relative interior of the polytope.

        We can also infer the following from the Chebyshev radius R:
            1. :math:`R=\infty`, the polytope is unbounded. Note that, this is just a sufficient
               condition, and an unbounded polytope can have a finite Chebyshev radius. For example, consider a
               3-dimensional axis-aligned cuboid :math:`[-1, 1] \times [-1, 1] \times \mathbb{R}`.
            2. :math:`0 < R < \infty`, the polytope is nonempty and full-dimensional.
            3. :math:`R=0`, the polytope is nonempty and but not full-dimensional.
            4. :math:`R=-\infty`, the polytope is empty.
    """
    import cvxpy as cp

    if self.A.shape[1] == 0:
        return None, 0
    chebyshev_center = cp.Variable((self.dim,))
    chebyshev_radius = cp.Variable()
    norm_A_row_wise = np.linalg.norm(self.A, axis=1)  # Gives a row vector of norms
    if not self.in_H_rep and not self.in_V_rep:  # Empty case because no rep
        return None, -np.inf
    else:
        const: list[cp.Constraint] = [chebyshev_radius >= -PYCVXSET_ZERO]
        if self.n_halfspaces > 0:
            const += [self.A @ chebyshev_center + (chebyshev_radius * norm_A_row_wise) <= self.b]
        else:
            const = [chebyshev_radius <= 1e4]  # Aim to find a feasible point is deep enough
        if self.n_equalities > 0:
            # We do not include chebyshev_radius in tightened equality constraints to find a feasible (interior) point
            const += [self.Ae @ chebyshev_center == self.be]
        prob = cp.Problem(cp.Maximize(chebyshev_radius), const)
        try:
            prob.solve(**self.cvxpy_args_lp)
        except cp.error.SolverError as err:  # pyright: ignore[reportAttributeAccessIssue]
            raise NotImplementedError(
                f"Unable to solve for the chebyshev centering! CVXPY returned error: {str(err)}"
            ) from err
        if prob.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:  # Feasible
            center_value = chebyshev_center.value
            radius_value = chebyshev_radius.value
            if center_value is None or radius_value is None:
                raise NotImplementedError("Chebyshev centering did not return a solution.")
            if self.n_equalities == 0 and radius_value >= PYCVXSET_ZERO:  # Non-empty with no equalities case
                return center_value, float(radius_value)
            else:  # Non-empty but lower-dimensional case
                return center_value, 0
        elif prob.status in [cp.UNBOUNDED, cp.UNBOUNDED_INACCURATE]:  # Unbounded
            return None, np.inf
        elif prob.status in [cp.INFEASIBLE, cp.INFEASIBLE_INACCURATE]:  # Infeasible
            return None, -np.inf
        else:
            raise NotImplementedError(f"Did not expect to reach here in chebyshev_centering! {prob.status}")


def decompose_as_affine_transform_of_polytope_without_equalities(
    self: Polytope,
) -> tuple[Polytope, np.ndarray, np.ndarray]:
    r"""Express a polytope with equality constraints as an affine transformation of a lower-dimensional polytope without
    any equality constraints (hence full-dimensional). The affine transformation is obtained via a QR decomposition.

    If the polytope is already full-dimensional, returns a copy of itself and the identity matrix.  Otherwise, finds an
    affine transformation that maps a lower-but-full-dimensional polytope to the original polytope.

    Returns:
        tuple: A tuple with three items:
            #. full_dimensional_polytope (Polytope): A full-dimensional polytope.
            #. affine_transform_offset (numpy.ndarray): The affine offset (as a vector) to map the full-dimensional
               polytope to the original polytope.
            #. affine_transform_matrix (numpy.ndarray): A matrix whose columns form an orthonormal basis for the
               nullspace of the affine equality constraints.

    Notes:
        This function requires H-Rep, and will perform a halfspace enumeration when a V-Rep polytope is provided.
        The decomposition is such that `self = M @ full_dim_polytope_in_relative_interior_of_self + c`. The function
        uses QR decomposition to find `M` and `c`. To avoid numerical issues, this function calls
        :meth:`pycvxset.Polytope.minimize_H_rep()`.
    """
    if self.is_empty:
        raise ValueError("Can not compute decomposition for an empty polytope!")
    if self.is_full_dimensional or self.is_singleton:
        return self.copy(), np.zeros((self.dim,)), np.eye(self.dim)
    else:
        self.minimize_H_rep()
        # Make sure that Ae is full row rank
        irredundant_Ae, irredundant_be = compute_irredundant_affine_set_using_cdd(self.Ae, self.be)
        # Take QR decomposition of (Ae).T
        Q, R = np.linalg.qr(irredundant_Ae.T, mode="complete")
        rank_Ae = irredundant_Ae.shape[0]
        # {x | Ae @ x = be} = {u_star + Q2 z} where u_star = Q1 @ R^{-T} @ be and [Q1, Q2] @ R = Ae.T
        Q1, Q2 = Q[:, :rank_Ae], Q[:, rank_Ae:]
        u_star = Q1 @ np.linalg.inv(R[:rank_Ae, :]).T @ irredundant_be
        # self = {x | A @ x <= b, Ae @ x == be} = {u_star + Q2 z | A @ (u_star + Q2 z) <= b}
        # = {u_star + Q2 z | A @ Q2 z <= b - A @ u_star}
        full_dimensional_polytope = self.__class__(A=self.A @ Q2, b=self.b - (self.A @ u_star))
        affine_transform_matrix, affine_transform_offset = Q2, u_star
        if not full_dimensional_polytope.is_full_dimensional:
            raise NotImplementedError(
                "Decomposed full-dimensional polytope is not full-dimensional! "
                "This should not happen, please report this bug."
            )
        return full_dimensional_polytope, affine_transform_offset, affine_transform_matrix


def deflate_rectangle(cls: type[Polytope], set_to_be_centered: ConstrainedZonotope | Ellipsoid | Polytope) -> Polytope:
    r"""Compute the rectangle with the smallest volume that contains the given set.

    Args:
        set_to_be_centered (ConstrainedZonotope | Ellipsoid | Polytope): Set to compute the.

    Returns:
        Polytope: Minimum volume circumscribing rectangle

    Notes:
        This function is a wrapper for :meth:`minimum_volume_circumscribing_rectangle` of attr:`set_to_be_centered`.
        Please check that function for more details including raising exceptions.
    """
    lb, ub = set_to_be_centered.minimum_volume_circumscribing_rectangle()
    return cls(lb=lb, ub=ub)


def interior_point(self: Polytope, point_type: str | None = None) -> np.ndarray:
    """Compute a point in the interior of the polytope. When the polytope is not full-dimensional, the point may lie on
    the boundary.

    Args:
        point_type (str, optional): Type of interior point. Valid strings: {'centroid', 'chebyshev', 'mvie'}. Defaults
            to 'centroid' if has V-Rep and 'chebyshev' if has H-Rep.

    Raises:
        NotImplementedError: When an invalid point_type is provided.
        ValueError: When the polytope provided is empty.
        ValueError: When the polytope provided is not bounded.

    Returns:
        numpy.ndarray: A feasible point in the polytope in the interior as a 1D array

    Notes:
        - point_type is 'centroid': Computes the average of the vertices. The function requires the polytope to be in
          V-Rep, and a vertex enumeration is performed if the polytope is in H-Rep.
        - point_type is 'chebyshev': Computes the Chebyshev center. The function requires the polytope to be in H-Rep,
          and a halfspace enumeration is performed if the polytope is in V-Rep.
    """
    if point_type is None:
        point_type = "centroid" if self.in_V_rep else "chebyshev"
    # Switch based on point_type
    if point_type == "centroid":
        if self.is_empty:
            raise ValueError("Can not compute centroid for an empty polytope!")
        elif not self.is_bounded:
            raise ValueError("Can not compute centroid for an unbounded polytope!")
        else:
            return np.mean(self.V, axis=0, keepdims=False)
    elif point_type == "chebyshev":
        c, r = self.chebyshev_centering()
        if r < 0:
            raise ValueError("Can not compute chebyshev center for an empty polytope!")
        elif 0 <= r < np.inf:
            return c
        else:
            raise ValueError("Can not compute chebyshev center for an unbounded polytope!")
    elif point_type == "mvie":
        return self.maximum_volume_inscribing_ellipsoid()[0]
    else:
        raise NotImplementedError(f"Interior point of type {point_type:s} has not yet been implemented")


def maximum_volume_inscribing_ellipsoid(self: Polytope) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r"""Compute the maximum volume ellipsoid that fits within the given polytope.

    Raises:
        ValueError: When polytope is empty or has non-empty (Ae, be)
        NotImplementedError: Unable to solve convex problem using CVXPY

    Returns:
        tuple: A tuple with three items:
            #. center (numpy.ndarray): Maximum volume inscribed ellipsoid's center
            #. shape_matrix (numpy.ndarray): Maximum volume inscribed ellipsoid's shape matrix.
            #. sqrt_shape_matrix (numpy.ndarray): Maximum volume inscribed ellipsoid's square root of shape matrix.
               Returns np.zeros((self.dim, 0)) if the polytope is a singleton.

    Notes:
        This function requires H-Rep, and will perform a vertex enumeration when a V-Rep polytope is provided.

        When the polytope is full-dimensional, we can solve a second-order cone program (SOCP). Consider the
        full-dimensional ellipsoid :math:`\{Gu + c| {\|u\|}_2 \leq 1\}`, where :math:`G` is a square, lower-triangular
        matrix with positive diagonal entries. Then, we solve the following (equivalent) optimization problem:

        .. math ::
            \text{minimize}   &\quad \text{geomean}(G) \\
            \text{subject to} &\quad \text{diag}(G) \geq 0\\
                              &\quad \|G^T a_i\|_2 + a_i^T d \leq b_i,

        with decision variables :math:`G` and :math:`c`.  Here, we use the observation that :math:`\text{geomean}(G)` is
        a monotone function of :math:`\log\det(GG^T)` (which is proportional to the volume of the ellipsoid).

        When the polytope is not full-dimensional, we first compute the relative interior and then solve the SDP for a
        positive definite matrix :math:`B\in\mathbb{S}^n_{++}` and :math:`d\in\mathbb{R}^n`,

        .. math ::
            \text{maximize}   &\quad \log \text{det} B \\
            \text{subject to} &\quad \|B a_i\|_2 + a_i^T d \leq b_i,\\
                              &\quad A_e d = b_e,\\
                              &\quad B A_e^\top  = 0,

        where :math:`(a_i,b_i)` is the set of hyperplanes characterizing :math:`\mathcal{P}`, and the inscribing
        ellipsoid is given by :math:`\{Bu + d| {||u||}_2 \leq 1\}`. The center of the ellipsoid is given by :math:`c =
        d`, and the shape matrix is given by :math:`Q = (B B)^{-1}`. See [EllipsoidalTbx-Min_verticesolEll]_ and Section
        8.4.2 in [BV04]_ for more details. The last two constraints arise from requiring the inscribing ellipsoid to lie
        in the affine set :math:`\{A_e x = b_e\}`.

    """
    import cvxpy as cp

    inscribing_ellipsoid_c = None
    inscribing_ellipsoid_Q = None
    if self.is_empty:
        raise ValueError("Can not compute circumscribing ellipsoid for an empty polytope!")
    elif not self.is_bounded:
        raise ValueError("Polytope is not bounded!")
    elif self.is_singleton:
        inscribing_ellipsoid_c = self.V[0]
        inscribing_ellipsoid_G = np.zeros((self.dim, 0))
    elif self.is_full_dimensional:  # No equalities, so we can solve the simpler SOCP
        d = cp.Variable((self.dim,))
        G_full_made_ltri = cp.Variable((self.dim, self.dim))
        const: list[cp.Constraint] = []
        for row_index in range(self.dim - 1):
            const += [G_full_made_ltri[row_index, row_index + 1 :] == 0]
        const += [cp.diag(G_full_made_ltri) >= PYCVXSET_ZERO]  # Non-negative diag. entries for G_full_made_ltri
        for h in self.H:
            a = h[:-1]
            b = h[-1]
            const += [cp.norm(G_full_made_ltri.T @ a, p=2) + a.T @ d <= b]
        prob = cp.Problem(cp.Maximize(cp.geo_mean(cp.diag(G_full_made_ltri))), const)
        try:
            prob.solve(**self.cvxpy_args_socp)
        except cp.error.SolverError as err:  # pyright: ignore[reportAttributeAccessIssue]
            raise NotImplementedError(
                f"Unable to compute maximum volume inscribing ellipsoid! CVXPY returned error: {str(err)}"
            ) from err

        # Parse the solution
        if prob.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
            inscribing_ellipsoid_c = d.value
            inscribing_ellipsoid_G = G_full_made_ltri.value
            if inscribing_ellipsoid_c is None or inscribing_ellipsoid_G is None:
                raise NotImplementedError(
                    "CVXPY did not return a solution for the maximum volume inscribing ellipsoid."
                )
        else:
            raise NotImplementedError(
                f"CVXPY returned status {prob.status:s} when computing the maximum volume inscribing ellipsoid."
            )
    else:
        full_dim_polytope, decompose_c, decompose_M = (
            self.decompose_as_affine_transform_of_polytope_without_equalities()
        )
        full_dim_polytope_inscribing_ellipsoid_c, _, full_dim_polytope_inscribing_ellipsoid_G = (
            full_dim_polytope.maximum_volume_inscribing_ellipsoid()
        )
        inscribing_ellipsoid_c = decompose_M @ full_dim_polytope_inscribing_ellipsoid_c + decompose_c
        inscribing_ellipsoid_G = decompose_M @ full_dim_polytope_inscribing_ellipsoid_G
    inscribing_ellipsoid_Q = inscribing_ellipsoid_G @ inscribing_ellipsoid_G.T
    inscribing_ellipsoid_Q = (inscribing_ellipsoid_Q + inscribing_ellipsoid_Q.T) / 2
    return inscribing_ellipsoid_c, inscribing_ellipsoid_Q, inscribing_ellipsoid_G


def minimum_volume_circumscribing_ellipsoid(self: Polytope) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    r"""Compute the minimum volume ellipsoid that covers the given polytope (also known as Lowner-John Ellipsoid).

    Raises:
        ValueError: When polytope is empty
        NotImplementedError: Unable to solve convex problem using CVXPY

    Returns:
        tuple: A tuple with three items:
            #. center (numpy.ndarray): Minimum volume circumscribed ellipsoid's center
            #. shape_matrix (numpy.ndarray): Minimum volume circumscribed ellipsoid's shape matrix
            #. sqrt_shape_matrix (numpy.ndarray): Minimum volume circumscribed ellipsoid's square root of shape matrix.
               Returns None if the polytope is not full-dimensional.

    Notes:
        This function requires V-Rep, and will perform a vertex enumeration when a H-Rep polytope is provided.

        We solve the SDP for a positive definite matrix :math:`A\in\mathbb{S}^n_{++}` and :math:`b\in\mathbb{R}^n`,

        .. math ::
            \text{maximize}   &\quad \log \text{det} A^{-1} \\
            \text{subject to} &\quad \|Av_i + b\|_2 \leq 1,\ \forall \text{ vertices } v_i \text{ of } \mathcal{P}

        where the circumscribing ellipsoid is given by :math:`\{x| {||A x + b||}_2 \leq 1\}`, and we use the observation
        that :math:`\log \text{det} A^{-1}= -\log \text{det} A`. The center of the ellipsoid is given by :math:`c =
        -A^{-1}b`, and the shape matrix is given by :math:`Q = (A^T A)^{-1}`. See [EllipsoidalTbx-Min_verticesolEll]_
        and Section 8.4.1 in [BV04]_ for more details.

        When the polytope is full-dimensional, we can instead solve a second-order cone program (SOCP). Consider the
        full-dimensional ellipsoid :math:`\{Lu + c| {\|u\|}_2 \leq 1\}`, where :math:`G` is a square, lower-triangular
        matrix with positive diagonal entries. Then, we solve the following (equivalent) optimization problem:

        .. math ::
            \text{minimize}   &\quad \text{geomean}(L) \\
            \text{subject to} &\quad \text{diag}(L) > 0\\
                              &\quad v_i = Lu_i + c,\ \forall \text{ vertices } v_i \text{ of } \mathcal{P},\\
                              &\quad u_i\in\mathbb{R}^,\ \|u_i\|_2 \leq 1,

        with decision variables :math:`G`, :math:`c`, and :math:`u_i`, and :math:`\text{geomean}` is the geometric mean
        of the diagonal elements of :math:`G`. Here, we use the observation that :math:`\text{geomean}(L)` is a monotone
        function of :math:`\log\det(GG^T)` (the volume of the ellipsoid). For the sake of convexity, we solve the
        following equivalent optimization problem after a change of variables :math:`L_\text{inv}=L^{-1}` and
        :math:`c_{L_\text{inv}}=L^{-1}c`, and substituting for the variables :math:`u_i`:

        .. math ::
            \text{maximize}   &\quad \text{geomean}(L_\text{inv}) \\
            \text{subject to} &\quad \text{diag}(L_\text{inv}) > 0\\
                              &\quad \|L_\text{inv}v_i - c_{L_\text{inv}}\|_2 \leq 1,\ \forall \text{ vertices } v_i

        The center of the ellipsoid is :math:`c = Lc_{L_\text{inv}}`, and the shape matrix is :math:`Q = GG^T`, where
        :math:`L=L_\text{inv}^{-1}`.
    """
    import cvxpy as cp

    circumscribing_ellipsoid_c = None
    circumscribing_ellipsoid_Q = None
    if self.is_empty:
        raise ValueError("Can not compute circumscribing ellipsoid for an empty polytope!")
    elif self.is_full_dimensional:
        L_inv_c = cp.Variable((self.dim,))
        L_inv_full_made_ltri = cp.Variable((self.dim, self.dim))
        const: list[cp.Constraint] = []
        for row_index in range(self.dim - 1):
            const += [L_inv_full_made_ltri[row_index, row_index + 1 :] == 0]
        const += [cp.diag(L_inv_full_made_ltri) >= PYCVXSET_ZERO]  # Non-neg diag. entries for L_inv_full_made_ltri
        for v in self.V:
            const += [cp.norm(L_inv_full_made_ltri @ v - L_inv_c, p=2) <= 1]
        prob = cp.Problem(cp.Maximize(cp.geo_mean(cp.diag(L_inv_full_made_ltri))), const)
        try:
            prob.solve(**self.cvxpy_args_socp)
        except cp.error.SolverError as err:  # pyright: ignore[reportAttributeAccessIssue]
            raise NotImplementedError(
                f"Unable to compute minimum volume circumscribing ellipsoid! CVXPY returned error: {str(err)}"
            ) from err

        # Parse the solution
        if prob.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
            circumscribing_ellipsoid_c = np.linalg.lstsq(L_inv_full_made_ltri.value, L_inv_c.value, rcond=None)[0]
            circumscribing_ellipsoid_G = np.linalg.inv(L_inv_full_made_ltri.value)
            circumscribing_ellipsoid_Q = circumscribing_ellipsoid_G @ circumscribing_ellipsoid_G.T
        else:
            raise NotImplementedError(
                f"CVXPY returned status {prob.status:s} when computing the minimum volume circumscribing ellipsoid"
            )
    else:
        A = cp.Variable((self.dim, self.dim), symmetric=True)
        b = cp.Variable((self.dim,))
        const = []
        for v in self.V:
            const += [cp.norm(A @ v + b, p=2) <= 1]
        prob = cp.Problem(cp.Minimize(-cp.log_det(A)), const)
        try:
            prob.solve(**self.cvxpy_args_sdp)
        except cp.error.SolverError as err:  # pyright: ignore[reportAttributeAccessIssue]
            raise NotImplementedError(
                f"Unable to compute minimum volume circumscribing ellipsoid! CVXPY returned error: {str(err)}"
            ) from err

        # Parse the solution
        if prob.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
            if A.value is None or b.value is None:
                raise NotImplementedError(
                    "CVXPY did not return a solution for the minimum volume circumscribing ellipsoid."
                )
            circumscribing_ellipsoid_c = -np.linalg.lstsq(A.value, b.value, rcond=None)[0]
            circumscribing_ellipsoid_Q = np.linalg.inv(A.value.T @ A.value)
            circumscribing_ellipsoid_G = None
        else:
            raise NotImplementedError(
                f"CVXPY returned status {prob.status:s} when computing the minimum volume circumscribing ellipsoid"
            )
    circumscribing_ellipsoid_Q = (circumscribing_ellipsoid_Q + circumscribing_ellipsoid_Q.T) / 2
    return circumscribing_ellipsoid_c, circumscribing_ellipsoid_Q, circumscribing_ellipsoid_G


def minimum_volume_circumscribing_rectangle(self: Polytope) -> tuple[np.ndarray, np.ndarray]:
    r"""Compute the minimum volume circumscribing rectangle for a given polytope

    Raises:
        ValueError: When polytope is empty
        ValueError: When polytope is unbounded

    Returns:
        tuple: A tuple of two items
            #. lb (numpy.ndarray): Lower bound :math:`l` on the polytope,
               :math:`\mathcal{P}\subseteq\{l\}\oplus\mathbb{R}_{\geq 0}`.
            #. ub (numpy.ndarray): Upper bound :math:`u` on the polytope,
               :math:`\mathcal{P}\subseteq\{u\}\oplus(-\mathbb{R}_{\geq 0})`.

    Notes:
        This function accommodates H-Rep and V-Rep. The lower/upper bound for V-Rep is given by an element-wise
        minimization/maximization operation, while the lower/upper bound for H-Rep is given by an element-wise support
        computation (2 * self.dim linear programs). For a H-Rep polytope, the function uses :meth:`support` and is a
        wrapper for :meth:`pycvxset.common.minimum_volume_circumscribing_rectangle`.

        This function is called to check for boundedness of a polytope.
    """
    if self.is_empty:
        raise ValueError("Can not compute circumscribing rectangle for an empty polytope!")
    elif self.in_V_rep:
        lb = np.min(self.V, axis=0)
        ub = np.max(self.V, axis=0)
    else:
        lb, ub = convex_set_minimum_volume_circumscribing_rectangle(self)
    return lb, ub


def normalize(self: Polytope) -> None:
    r"""Normalize a H-Rep such that each row of A has unit :math:`\ell_2`-norm.

    Notes:
        This function requires P to be in H-Rep. If P is in V-Rep, a halfspace enumeration is performed.
    """
    if self.in_V_rep:
        self.determine_H_rep()
    norm_A_row_wise = np.linalg.norm(self.A, ord=2, axis=1, keepdims=True)
    normalized_H = self.H / norm_A_row_wise
    if self.n_equalities > 0:
        norm_Ae_row_wise = np.linalg.norm(self.Ae, ord=2, axis=1, keepdims=True)
        normalized_He = self.H / norm_Ae_row_wise
        self._set_attributes_from_Ab_Aebe(
            A=normalized_H[:, :-1],
            b=normalized_H[:, -1],
            Ae=normalized_He[:, :-1],
            be=normalized_He[:, -1],
            erase_V_rep=False,
        )
    else:
        self._set_attributes_from_Ab_Aebe(A=normalized_H[:, :-1], b=normalized_H[:, -1], erase_V_rep=False)


def volume(self: Polytope) -> float:
    """
    Compute the volume of the polytope using QHull

    Returns:
        float: Volume of the polytope

    Notes:
        - Works with V-representation: Yes
        - Works with H-representation: No
        - Performs a vertex-halfspace enumeration when H-rep is provided
        - Returns 0 when the polytope is empty
        - Requires polytope to be full-dimensional
    """
    if self.is_empty:
        return 0.0
    elif not self.is_full_dimensional:
        raise ValueError("Volume computation is available only for full-dimensional polytopes")
    else:
        return ConvexHull(points=self.V).volume
