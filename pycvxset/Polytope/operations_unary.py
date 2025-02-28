# Copyright (C) 2020-2025 Mitsubishi Electric Research Laboratories (MERL)
# Copyright (c) 2019 Tor Aksel N. Heirung
#
# SPDX-License-Identifier: AGPL-3.0-or-later
# SPDX-License-Identifier: MIT

# Code purpose:  Define the methods involving just the Polytope class

import cvxpy as cp
import numpy as np
from scipy.spatial import ConvexHull

from pycvxset.common import convex_set_minimum_volume_circumscribing_rectangle
from pycvxset.common.constants import PYCVXSET_ZERO


def chebyshev_centering(self):
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
    if self.A.shape[1] == 0:
        return None, 0
    chebyshev_center = cp.Variable((self.dim,))
    chebyshev_radius = cp.Variable()
    norm_A_row_wise = np.linalg.norm(self.A, axis=1)  # Gives a row vector of norms
    if not self.in_H_rep and not self.in_V_rep:  # Empty case because no rep
        return None, -np.inf
    else:
        const = [chebyshev_radius >= -PYCVXSET_ZERO]
        if self.n_halfspaces > 0:
            const += [self.A @ chebyshev_center + (chebyshev_radius * norm_A_row_wise) <= self.b]
        else:
            const = [chebyshev_radius <= 1e4]  # Aim to find a feasible point is deep enough
        if self.n_equalities > 0:
            const += [self.Ae @ chebyshev_center == self.be]
        prob = cp.Problem(cp.Maximize(chebyshev_radius), const)
        try:
            prob.solve(**self.cvxpy_args_lp)
        except cp.error.SolverError as err:
            raise NotImplementedError(
                f"Unable to solve for the chebyshev centering! CVXPY returned error: {str(err)}"
            ) from err
        if prob.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:  # Feasible
            if (
                self.n_equalities == 0 and chebyshev_radius.value >= PYCVXSET_ZERO
            ):  # Non-empty but full-dimensional case
                return chebyshev_center.value, float(chebyshev_radius.value)
            else:  # Non-empty but lower-dimensional case
                return chebyshev_center.value, 0
        elif prob.status in [cp.UNBOUNDED, cp.UNBOUNDED_INACCURATE]:  # Unbounded
            return None, np.inf
        elif prob.status in [cp.INFEASIBLE, cp.INFEASIBLE_INACCURATE]:  # Infeasible
            return None, -np.inf
        else:
            raise NotImplementedError(f"Did not expect to reach here in chebyshev_centering! {prob.status}")


def interior_point(self, point_type="centroid"):
    """Compute a point in the interior of the polytope. When the polytope is not full-dimensional, the point may lie on
    the boundary.

    Args:
        point_type (str): Type of interior point. Valid strings: {'centroid', 'chebyshev'}. Defaults to 'centroid'.

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
    else:
        raise NotImplementedError(f"Interior point of type {point_type:s} has not yet been implemented")


def maximum_volume_inscribing_ellipsoid(self):
    r"""Compute the maximum volume ellipsoid that fits within the given polytope.

    Raises:
        ValueError: When polytope is empty or has non-empty (Ae, be)
        NotImplementedError: Unable to solve convex problem using CVXPY

    Returns:
        tuple: A tuple with three items:
            #. center (numpy.ndarray): Maximum volume inscribed ellipsoid's center
            #. shape_matrix (numpy.ndarray): Maximum volume inscribed ellipsoid's shape matrix
            #. sqrt_shape_matrix (numpy.ndarray): Maximum volume inscribed ellipsoid's square root of shape matrix.
               Returns None if the polytope is not full-dimensional.

    Notes:
        This function requires H-Rep, and will perform a vertex enumeration when a V-Rep polytope is provided.

        We solve the SDP for a positive definite matrix :math:`B\in\mathbb{S}^n_{++}` and :math:`d\in\mathbb{R}^n`,

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

        When the polytope is full-dimensional, we can instead solve a second-order cone program (SOCP). Consider the
        full-dimensional ellipsoid :math:`\{Lu + c| {\|u\|}_2 \leq 1\}`, where :math:`G` is a square, lower-triangular
        matrix with positive diagonal entries. Then, we solve the following (equivalent) optimization problem:

        .. math ::
            \text{minimize}   &\quad \text{geomean}(L) \\
            \text{subject to} &\quad \text{diag}(L) \geq 0\\
                              &\quad \|L^T a_i\|_2 + a_i^T d \leq b_i,

        with decision variables :math:`G` and :math:`c`.  Here, we use the observation that :math:`\text{geomean}(L)` is
        a monotone function of :math:`\log\det(GG^T)` (the volume of the ellipsoid).
    """
    inscribing_ellipsoid_c = None
    inscribing_ellipsoid_Q = None
    if self.is_empty:
        raise ValueError("Can not compute circumscribing ellipsoid for an empty polytope!")
    elif not self.is_bounded:
        raise ValueError("Polytope is not bounded!")
    elif self.n_equalities == 0 and self.is_full_dimensional:
        d = cp.Variable((self.dim,))
        L_full_made_ltri = cp.Variable((self.dim, self.dim))
        const = []
        for row_index in range(self.dim - 1):
            const += [L_full_made_ltri[row_index, row_index + 1 :] == 0]
        const += [cp.diag(L_full_made_ltri) >= PYCVXSET_ZERO]  # Non-negative diag. entries for L_full_made_ltri
        for h in self.H:
            a = h[:-1]
            b = h[-1]
            const += [cp.norm(L_full_made_ltri.T @ a, p=2) + a.T @ d <= b]
        prob = cp.Problem(cp.Maximize(cp.geo_mean(cp.diag(L_full_made_ltri))), const)
        try:
            prob.solve(**self.cvxpy_args_socp)
        except cp.error.SolverError as err:
            raise NotImplementedError(
                f"Unable to compute maximum volume inscribing ellipsoid! CVXPY returned error: {str(err)}"
            ) from err

        # Parse the solution
        if prob.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
            inscribing_ellipsoid_c = d.value
            inscribing_ellipsoid_Q = L_full_made_ltri.value @ L_full_made_ltri.value.T
            inscribing_ellipsoid_G = L_full_made_ltri.value
        else:
            raise NotImplementedError(
                f"CVXPY returned status {prob.status:s} when computing the maximum volume inscribing ellipsoid."
            )
    else:
        B = cp.Variable((self.dim, self.dim), symmetric=True)
        d = cp.Variable((self.dim,))
        const = []
        for h in self.H:
            a = h[:-1]
            b = h[-1]
            const += [cp.norm(B @ a, p=2) + a.T @ d <= b]
        if self.n_equalities > 0:
            # We want self.Ae @ B @ B.T @ self.Ae.T to be zero
            const += [self.Ae @ d == self.be, self.Ae @ B == 0]
        prob = cp.Problem(cp.Maximize(cp.log_det(B)), const)
        try:
            prob.solve(**self.cvxpy_args_sdp)
        except cp.error.SolverError as err:
            raise NotImplementedError(
                f"Unable to compute maximum volume inscribing ellipsoid! CVXPY returned error: {str(err)}"
            ) from err

        # Parse the solution
        if prob.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
            inscribing_ellipsoid_c = d.value
            inscribing_ellipsoid_Q = B.value @ B.value.T
            inscribing_ellipsoid_G = None
        else:
            raise NotImplementedError(
                f"CVXPY returned status {prob.status:s} when computing the minimum volume circumscribing ellipsoid"
            )
    inscribing_ellipsoid_Q = (inscribing_ellipsoid_Q + inscribing_ellipsoid_Q.T) / 2
    return inscribing_ellipsoid_c, inscribing_ellipsoid_Q, inscribing_ellipsoid_G


def minimum_volume_circumscribing_ellipsoid(self):
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
    circumscribing_ellipsoid_c = None
    circumscribing_ellipsoid_Q = None
    if self.is_empty:
        raise ValueError("Can not compute circumscribing ellipsoid for an empty polytope!")
    elif self.is_full_dimensional:
        L_inv_c = cp.Variable((self.dim,))
        L_inv_full_made_ltri = cp.Variable((self.dim, self.dim))
        const = []
        for row_index in range(self.dim - 1):
            const += [L_inv_full_made_ltri[row_index, row_index + 1 :] == 0]
        const += [cp.diag(L_inv_full_made_ltri) >= PYCVXSET_ZERO]  # Non-negative diag. entries for L_inv_full_made_ltri
        for v in self.V:
            const += [cp.norm(L_inv_full_made_ltri @ v - L_inv_c, p=2) <= 1]
        prob = cp.Problem(cp.Maximize(cp.geo_mean(cp.diag(L_inv_full_made_ltri))), const)
        try:
            prob.solve(**self.cvxpy_args_socp)
        except cp.error.SolverError as err:
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
        except cp.error.SolverError as err:
            raise NotImplementedError(
                f"Unable to compute minimum volume circumscribing ellipsoid! CVXPY returned error: {str(err)}"
            ) from err

        # Parse the solution
        if prob.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
            circumscribing_ellipsoid_c = -np.linalg.lstsq(A.value, b.value, rcond=None)[0]
            circumscribing_ellipsoid_Q = np.linalg.inv(A.value.T @ A.value)
            circumscribing_ellipsoid_G = None
        else:
            raise NotImplementedError(
                f"CVXPY returned status {prob.status:s} when computing the minimum volume circumscribing ellipsoid"
            )
    circumscribing_ellipsoid_Q = (circumscribing_ellipsoid_Q + circumscribing_ellipsoid_Q.T) / 2
    return circumscribing_ellipsoid_c, circumscribing_ellipsoid_Q, circumscribing_ellipsoid_G


def minimum_volume_circumscribing_rectangle(self):
    r"""Compute the minimum volume circumscribing rectangle for a given polytope

    Raises:
        ValueError: When polytope is empty

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


def deflate_rectangle(cls, set_to_be_centered):
    r"""Compute the rectangle with the smallest volume that contains the given set.

    Args:
        set_to_be_centered (Polytope | ConstrainedZonotope | Ellipsoid): Set to compute the.

    Returns:
        Polytope: Minimum volume circumscribing rectangle

    Notes:
        This function is a wrapper for :meth:`minimum_volume_circumscribing_rectangle` of attr:`set_to_be_centered`.
        Please check that function for more details including raising exceptions.
    """
    lb, ub = set_to_be_centered.minimum_volume_circumscribing_rectangle()
    return cls(lb=lb, ub=ub)


def normalize(self):
    r"""Normalize a H-Rep such that each row of A has unit :math:`\ell_2`-norm.

    Notes:
        This function requires P to be in H-Rep. If P is in V-Rep, a halfspace enumeration is performed.
    """
    if self.in_V_rep:
        self.determine_H_rep()
    norm_A_row_wise = np.linalg.norm(self.A, ord=2, axis=1, keepdims=True)  # Gives a row vector of norms
    normalized_A = self.A / norm_A_row_wise
    normalized_b = self.b / norm_A_row_wise[0]  # [0] to ensure that broadcast does not create a matrix
    if self.n_equalities > 0:
        norm_Ae_row_wise = np.linalg.norm(self.Ae, ord=2, axis=1, keepdims=True)  # Gives a row vector of norms
        normalized_Ae = self.Ae / norm_Ae_row_wise
        normalized_be = self.be / norm_Ae_row_wise[0]  # [0] to ensure that broadcast does not create a matrix
        self._set_attributes_from_Ab_Aebe(
            normalized_A, normalized_b, Ae=normalized_Ae, be=normalized_be, erase_V_rep=False
        )
    else:
        self._set_attributes_from_Ab_Aebe(normalized_A, normalized_b, erase_V_rep=False)


def volume(self):
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
