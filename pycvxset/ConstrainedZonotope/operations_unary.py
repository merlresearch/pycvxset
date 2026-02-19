# Copyright (C) 2020-2026 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

# Code purpose:  Define the methods involving just the ConstrainedZonotope class
# Coverage: This file has 6 missing statements + 9 excluded statements + 2 partial branches.

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Sequence, cast

import numpy as np

from pycvxset.common import compute_irredundant_affine_set_using_cdd, sanitize_and_identify_Aebe
from pycvxset.common.constants import PYCVXSET_ZERO
from pycvxset.Ellipsoid import Ellipsoid
from pycvxset.Polytope import Polytope

if TYPE_CHECKING:
    from pycvxset.ConstrainedZonotope import ConstrainedZonotope


def interior_point(self: ConstrainedZonotope, point_type: str = "mvie", enable_warning: bool = True) -> np.ndarray:
    r"""Compute an interior point of the constrained zonotope.

    Args:
        point_type (str, optional): Type of interior point. Valid strings: {'mvie', 'chebyshev'}. Defaults to 'mvie'.
        enable_warning (bool, optional): Enables the UserWarning. May be turned off if expected. Defaults to True.

    Returns:
        numpy.ndarray: A point that lies in the (relative) interior of the constrained zonotope.

    Notes:
        This function is a wrapper for existing centering methods in ConstrainedZonotope class ---
        maximum_volume_inscribing_ellipsoid, chebyshev_centering. For the case where self is a zonotope, it returns c.
    """
    if self.n_equalities == 0:
        return self.c
    elif point_type == "mvie":
        return self.maximum_volume_inscribing_ellipsoid(enable_warning=enable_warning)[0]
    elif point_type == "chebyshev":
        return self.chebyshev_centering(enable_warning=enable_warning)[0]
    else:
        raise NotImplementedError(f"Interior point of type {point_type:s} has not yet been implemented")


def get_matrix_least_norm_solution(
    G: Sequence[Sequence[float]] | np.ndarray,
    Ae: Sequence[Sequence[float]] | np.ndarray,
    operation_name: str,
    enable_warning: bool = True,
) -> np.ndarray:
    r"""Get the least norm solution for stacked matrix for the constrained zonotope.

    Specifically, the function computes :math:`[G;A_e]^\dagger[I_n;0_{M\times n}]`, where :math:`\dagger` denotes the
    pseudo-inverse.

    Args:
        G (Sequence[Sequence[float]] | np.ndarray): The generator matrix of the zonotope.
        Ae (Sequence[Sequence[float]] | np.ndarray): The inequality constraints of the zonotope.
        operation_name (str): The name of the operation being performed.
        enable_warning (bool, optional): Enables the UserWarning. May be turned off if expected. Defaults to True.

    Raises:
        ValueError: Expected constrained zonotope with the matrix [G; Ae] to have full row-rank

    Returns:
        numpy.ndarray: The least norm solution :math:`[G; Ae]^\dagger[I_n;0_{M\times n}]`.
    """
    G_arr: np.ndarray = np.asarray(G, dtype=float)
    Ae_arr: np.ndarray = np.asarray(Ae, dtype=float)
    dim, _ = G_arr.shape
    stacked_matrix_GAe = G_arr
    stacked_matrix_b = np.eye(dim)
    n_equalities = Ae_arr.shape[0]
    if n_equalities:
        stacked_matrix_GAe = np.vstack((stacked_matrix_GAe, Ae_arr))
        stacked_matrix_b = np.vstack((stacked_matrix_b, np.zeros((n_equalities, dim))))
    try:
        matrix_least_norm_solution, _, matrix_rank_GAe, _ = np.linalg.lstsq(
            stacked_matrix_GAe, stacked_matrix_b, rcond=None
        )
    except np.linalg.LinAlgError as err:
        raise ValueError("Unable to perform least_squares") from err

    if matrix_rank_GAe != dim + n_equalities:
        raise ValueError(
            "Expected constrained zonotope with the matrix [G; Ae] to have full row-rank! If the constrained "
            "zonotope is full-dimensional, then use remove_redundancies method first and then perform "
            f" {operation_name:s}"
        )

    if enable_warning:
        warnings.warn(f"This function returns a sub-optimal but feasible solution for {operation_name:s}.", UserWarning)
    return matrix_least_norm_solution


def chebyshev_centering(self: ConstrainedZonotope, enable_warning: bool = True) -> tuple[np.ndarray, float]:
    r"""Computes a ball with the largest radius that fits within the constrained zonotope. The ball's center is known as
    the Chebyshev center, and its radius is the Chebyshev radius.

    Args:
        enable_warning (bool, optional): Enables the UserWarning. May be turned off if expected. Defaults to True.

    Raises:
        ValueError: When the constrained zonotope is not full-dimensional
        ValueError: When the constrained zonotope is empty
        NotImplementedError: Unable to solve the linear program using CVXPY

    Returns:
        tuple: A tuple with two items
            #. center (numpy.ndarray): Approximate Chebyshev radius of the constrained zonotope
            #. radius (float): Approximate Chebyshev radius of the constrained zonotope

    Notes:
        Unlike :meth:`pycvxset.Polytope.Polytope.chebyshev_centering`, this function computes an approximate Chebyshev
        center and radius. Specifically, it guarantees that a ball of the computed radius, centered at the computed
        center is contained in the constrained zonotope. However, it does not guarantee that the computed radius is the
        radius of the largest possible ball contained in the given constrained zonotope. For more details, see [VWS24]_.
    """
    import cvxpy as cp

    if self.c is None:
        raise ValueError("Expected non-empty constrained zonotope!")
    elif self.is_singleton:
        return self.c, 0.0
    elif not self.is_full_dimensional:
        B_infty = Polytope(c=np.zeros((self.latent_dim,)), h=1)
        if self.n_equalities:
            irredundant_Ae, irredundant_be = compute_irredundant_affine_set_using_cdd(self.Ae, self.be)
            B_infty = B_infty.intersection_with_affine_set(Ae=irredundant_Ae, be=irredundant_be)
            if B_infty.is_empty:
                raise ValueError("Expected non-empty constrained zonotope!")
        ellipsoid = self.G @ Ellipsoid.inflate(B_infty) + self.c
        return ellipsoid.chebyshev_centering()
    else:
        matrix_least_norm_solution = get_matrix_least_norm_solution(
            self.G, self.Ae, "chebyshev_centering", enable_warning=enable_warning
        )
        chebyshev_center = cp.Variable((self.dim,))
        chebyshev_radius = cp.Variable()
        const, xi = self.containment_constraints(chebyshev_center)
        xi = cast(cp.Variable, xi)
        rowwise_vecnorm_gamma_matrix = np.linalg.norm(matrix_least_norm_solution, ord=2, axis=1)
        const += [
            chebyshev_radius >= -PYCVXSET_ZERO,
            xi <= 1 - chebyshev_radius * rowwise_vecnorm_gamma_matrix,
            -xi <= 1 - chebyshev_radius * rowwise_vecnorm_gamma_matrix,
        ]
        prob = cp.Problem(cp.Maximize(chebyshev_radius), const)

        try:
            prob.solve(**self.cvxpy_args_lp)
        except cp.error.SolverError as err:  # pyright: ignore[reportAttributeAccessIssue]
            raise NotImplementedError(
                f"Unable to solve for the chebyshev centering! CVXPY returned error: {str(err)}"
            ) from err
        if prob.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
            center_value = chebyshev_center.value
            radius_value = chebyshev_radius.value
            if center_value is None or radius_value is None:
                raise NotImplementedError("CVXPY did not return a solution for chebyshev_centering.")
            return center_value, max(float(radius_value), 0.0)
        elif prob.status in [cp.INFEASIBLE, cp.INFEASIBLE_INACCURATE]:
            raise ValueError("Unable to solve the approximate Chebyshev centering problem. CVXPY status: {prob.status}")
        else:
            raise NotImplementedError(
                f"Did not expect to reach here in chebyshev_centering! CVXPY status: {prob.status}"
            )


def maximum_volume_inscribing_ellipsoid(
    self: ConstrainedZonotope, enable_warning: bool = True
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r"""Compute the maximum volume ellipsoid that fits within the given constrained zonotope.

    Args:
        enable_warning (bool, optional): Enables the UserWarning. May be turned off if expected. Defaults to True.

    Raises:
        ValueError: When the constrained zonotope is not full-dimensional
        ValueError: When the constrained zonotope is empty
        NotImplementedError: Unable to solve the convex program using CVXPY

    Returns:
        tuple: A tuple with three items:
            #. center (numpy.ndarray): Approximate maximum volume inscribed ellipsoid's center
            #. shape_matrix (numpy.ndarray): Approximate maximum volume inscribed ellipsoid's shape matrix
            #. sqrt_shape_matrix (numpy.ndarray): Approximate maximum volume inscribed ellipsoid's square root of shape
               matrix.

    Notes:
        Unlike :meth:`pycvxset.Polytope.Polytope.maximum_volume_inscribing_ellipsoid`, this function computes an
        approximate maximum volume inscribed ellipsoid. Specifically, it guarantees that the computed ellipsoid is
        contained in the constrained zonotope. However, it does not guarantee that the computed ellipsoid is the largest
        possible ellipsoid (in terms of volume) contained in the given constrained zonotope. For more details, see
        [VWS24]_.
    """
    import cvxpy as cp

    if self.c is None:
        raise ValueError("Expected non-empty constrained zonotope!")
    elif self.is_singleton:
        return self.c, np.zeros((self.dim, 0)), np.zeros((self.dim, self.dim))
    elif not self.is_full_dimensional:
        B_infty = Polytope(c=np.zeros((self.latent_dim,)), h=1)
        if self.n_equalities:
            irredundant_Ae, irredundant_be = compute_irredundant_affine_set_using_cdd(self.Ae, self.be)
            B_infty = B_infty.intersection_with_affine_set(Ae=irredundant_Ae, be=irredundant_be)
        ellipsoid = self.G @ Ellipsoid.inflate(B_infty) + self.c
        return ellipsoid.c, ellipsoid.Q, ellipsoid.G
    else:
        matrix_least_norm_solution = get_matrix_least_norm_solution(
            self.G, self.Ae, "maximum_volume_inscribing_ellipsoid", enable_warning=enable_warning
        )

        mvie_center = cp.Variable((self.dim,))
        mvie_G_ltri = cp.Variable((self.dim, self.dim))
        const, xi = self.containment_constraints(mvie_center)
        xi = cast(cp.Variable, xi)
        for row_index in range(self.dim - 1):
            const += [mvie_G_ltri[row_index, row_index + 1 :] == 0]
        rowwise_vecnorm_gamma_matrix = cp.norm(matrix_least_norm_solution @ mvie_G_ltri, p=2, axis=1)
        const += [
            xi <= 1 - rowwise_vecnorm_gamma_matrix,
            -xi <= 1 - rowwise_vecnorm_gamma_matrix,
        ]
        prob = cp.Problem(cp.Maximize(cp.geo_mean(cp.diag(mvie_G_ltri))), const)
        try:
            prob.solve(**self.cvxpy_args_lp)
        except cp.error.SolverError as err:  # pyright: ignore[reportAttributeAccessIssue]
            raise NotImplementedError(
                f"Unable to solve for maximum volume inscribing ellipsoid! CVXPY returned error: {str(err)}"
            ) from err
        if prob.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
            inscribing_ellipsoid_c = mvie_center.value
            inscribing_ellipsoid_G = mvie_G_ltri.value
            if inscribing_ellipsoid_c is None or inscribing_ellipsoid_G is None:
                raise NotImplementedError("CVXPY did not return a solution for maximum_volume_inscribing_ellipsoid.")
            inscribing_ellipsoid_Q = inscribing_ellipsoid_G @ inscribing_ellipsoid_G.T
            inscribing_ellipsoid_Q = (inscribing_ellipsoid_Q + inscribing_ellipsoid_Q.T) / 2
            return inscribing_ellipsoid_c, inscribing_ellipsoid_Q, inscribing_ellipsoid_G
        elif prob.status in [cp.INFEASIBLE, cp.INFEASIBLE_INACCURATE]:
            raise ValueError(
                "Unable to solve the approximate maximum volume circumscribing ellipsoid problem. "
                f"CVXPY status: {prob.status}"
            )
        else:
            raise NotImplementedError(
                f"Did not expect to reach here in maximum volume inscribing ellipsoid! CVXPY status: {prob.status}"
            )


def remove_redundancies(self: ConstrainedZonotope) -> None:
    """Remove any redundancies in CZ using pycddlib and other geometric properties.

    Updates full-dimensional flag when empty

    Updates empty flag when empty OR non-empty zonotope OR when latent set is non-empty
    """

    def set_parameters_for_empty_constrained_zonotope() -> None:
        self._is_full_dimensional, self._is_empty = (self.dim == 0), True
        self._G, self._c, self._Ae, self._be = self._get_Gc_Aebe_for_empty_constrained_zonotope(self.dim, 0)

    def set_parameters_for_nonempty_zonotope() -> None:
        self._is_empty = False
        self._G, self._c, self._Ae, self._be = self._get_Gc_Aebe_for_empty_constrained_zonotope(self.dim, 0)

    _, _, Aebe_status, solution_to_Ae_xi_eq_be = sanitize_and_identify_Aebe(self.Ae, self.be)

    if Aebe_status == "no_Ae_be":
        if self.c is None:
            set_parameters_for_empty_constrained_zonotope()
        else:
            G, c = self.G, self.c
            set_parameters_for_nonempty_zonotope()
            self._G, self._c = G, c
    elif Aebe_status == "infeasible":
        set_parameters_for_empty_constrained_zonotope()
    elif Aebe_status == "single_point":
        if np.linalg.norm(solution_to_Ae_xi_eq_be, np.inf) <= 1:
            # Singleton => Set (G, Ae, be) to empty > Overwrite G
            c = self._c + self.G @ solution_to_Ae_xi_eq_be
            set_parameters_for_nonempty_zonotope()
            self._c = c
        else:
            set_parameters_for_empty_constrained_zonotope()
    else:  # affine_set
        self._Ae, self._be = compute_irredundant_affine_set_using_cdd(self.Ae, self.be)
        if np.linalg.norm(solution_to_Ae_xi_eq_be, ord=np.inf) <= 1:  # If it is not, it is still possibly non-empty
            # Full-dimensional => Set (Ae, be) to irredundant version
            self._is_empty = self.c is None

    recurse_to_remove_redundancies = False
    # Remove any zero columns in [G; Ae] since they do not contribute to the shape of the constrained zonotope
    stacked_matrix_GAe = self.G
    if self.n_equalities:
        stacked_matrix_GAe = np.vstack((stacked_matrix_GAe, self.Ae))
    zero_columns_GAe = np.where(np.linalg.norm(stacked_matrix_GAe, axis=0) <= PYCVXSET_ZERO)[0]
    if len(zero_columns_GAe) > 0:
        self._G = np.delete(self.G, zero_columns_GAe, axis=1)
        self._Ae = np.delete(self.Ae, zero_columns_GAe, axis=1) if self.n_equalities else self.Ae
        recurse_to_remove_redundancies = True

    if recurse_to_remove_redundancies:
        self.remove_redundancies()  # Recurse to remove any new redundancies after removing zero columns
