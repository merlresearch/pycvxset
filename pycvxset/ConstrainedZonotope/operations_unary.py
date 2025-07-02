# Copyright (C) 2020-2025 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

# Code purpose:  Define the methods involving just the ConstrainedZonotope class
# Coverage: This file has 4 untested statements to handle unexpected errors from np.linalg.lstsq

import warnings

import cvxpy as cp
import numpy as np

from pycvxset.common import compute_irredundant_affine_set_using_cdd, sanitize_and_identify_Aebe
from pycvxset.common.constants import PYCVXSET_ZERO
from pycvxset.Polytope import Polytope


def interior_point(self):
    r"""Compute an interior point of the constrained zonotope.

    Returns:
        numpy.ndarray: A point that lies in the (relative) interior of the constrained zonotope.

    Notes:
        This function returns :math:`G v + c`, where :math:`v` is the Chebyshev center of the polytope
        :math:`B_\infty(A_e, b_e)= \{\xi\ |\ \| \xi \|_\infty \leq 1, A_e \xi = b_e\} \subset \mathbb{R}^{N_C}`. See
        :meth:`pycvxset.Polytope.Polytope.chebyshev_centering` for more details on Chebyshev centering.
    """
    if self.is_zonotope:
        return self.c
    else:
        B_infty_without_Ae_be = Polytope(c=np.zeros((self.latent_dim,)), h=1)
        B_infty = B_infty_without_Ae_be.intersection_with_affine_set(Ae=self.Ae, be=self.be)
        return self.G @ B_infty.interior_point("chebyshev") + self.c


def chebyshev_centering(self):
    r"""Computes a ball with the largest radius that fits within the constrained zonotope. The ball's center is known as
    the Chebyshev center, and its radius is the Chebyshev radius.

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
    if self.is_empty:
        raise ValueError("Expected non-empty constrained zonotope!")
    stacked_matrix_GAe = self.G
    stacked_matrix_b = np.eye(self.dim)
    if not self.is_zonotope:
        stacked_matrix_GAe = np.vstack((stacked_matrix_GAe, self.Ae))
        stacked_matrix_b = np.vstack((stacked_matrix_b, np.zeros((self.n_equalities, self.dim))))
    try:
        matrix_least_norm_solution, _, matrix_rank_GAe, _ = np.linalg.lstsq(
            stacked_matrix_GAe, stacked_matrix_b, rcond=None
        )
    except np.linalg.LinAlgError as err:
        raise ValueError("Unable to perform least_squares") from err

    if matrix_rank_GAe != self.dim + self.n_equalities:
        raise ValueError(
            "Expected constrained zonotope with the matrix [G; Ae] to have full row-rank! If the constrained zonotope "
            "is full-dimensional, then use remove_redundancies method first and then call this function --- "
            "chebyshev_centering"
        )

    warnings.warn("This function returns a sub-optimal but feasible solution for Chebyshev centering.", UserWarning)

    chebyshev_center = cp.Variable((self.dim,))
    chebyshev_radius = cp.Variable()
    const, xi = self.containment_constraints(chebyshev_center)
    rowwise_vecnorm_gamma_matrix = np.linalg.norm(matrix_least_norm_solution, ord=2, axis=1)
    const += [
        chebyshev_radius >= -PYCVXSET_ZERO,
        xi <= 1 - chebyshev_radius * rowwise_vecnorm_gamma_matrix,
        -xi <= 1 - chebyshev_radius * rowwise_vecnorm_gamma_matrix,
    ]
    prob = cp.Problem(cp.Maximize(chebyshev_radius), const)

    try:
        prob.solve(**self.cvxpy_args_lp)
    except cp.error.SolverError as err:
        raise NotImplementedError(
            f"Unable to solve for the chebyshev centering! CVXPY returned error: {str(err)}"
        ) from err
    if prob.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:  # Feasible
        return chebyshev_center.value, max(float(chebyshev_radius.value), 0.0)
    else:
        raise NotImplementedError(f"Did not expect to reach here in chebyshev_centering! CVXPY status: {prob.status}")


def maximum_volume_inscribing_ellipsoid(self):
    r"""Compute the maximum volume ellipsoid that fits within the given constrained zonotope.

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
    if self.is_empty:
        raise ValueError("Expected non-empty constrained zonotope!")
    stacked_matrix_GAe = self.G
    stacked_matrix_b = np.eye(self.dim)
    if not self.is_zonotope:
        stacked_matrix_GAe = np.vstack((stacked_matrix_GAe, self.Ae))
        stacked_matrix_b = np.vstack((stacked_matrix_b, np.zeros((self.n_equalities, self.dim))))
    try:
        matrix_least_norm_solution, _, matrix_rank_GAe, _ = np.linalg.lstsq(
            stacked_matrix_GAe, stacked_matrix_b, rcond=None
        )
    except np.linalg.LinAlgError as err:
        raise ValueError("Unable to perform least_squares") from err

    if matrix_rank_GAe != self.dim + self.n_equalities:
        raise ValueError(
            "Expected constrained zonotope with the matrix [G; Ae] to have full row-rank! If the constrained zonotope "
            "is full-dimensional, then use remove_redundancies method first and then call this function --- "
            "maximum_volume_inscribing_ellipsoid"
        )

    warnings.warn(
        "This function returns a sub-optimal but feasible solution for maximum volume inscribed ellipsoid-based "
        "centering.",
        UserWarning,
    )

    mvie_center = cp.Variable((self.dim,))
    mvie_G_ltri = cp.Variable((self.dim, self.dim))
    const, xi = self.containment_constraints(mvie_center)
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
    except cp.error.SolverError as err:
        raise NotImplementedError(
            f"Unable to solve for maximum volume inscribing ellipsoid! CVXPY returned error: {str(err)}"
        ) from err
    if prob.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:  # Feasible
        inscribing_ellipsoid_c = mvie_center.value
        inscribing_ellipsoid_Q = mvie_G_ltri.value @ mvie_G_ltri.value.T
        inscribing_ellipsoid_Q = (inscribing_ellipsoid_Q + inscribing_ellipsoid_Q.T) / 2
        inscribing_ellipsoid_G = mvie_G_ltri.value
        return inscribing_ellipsoid_c, inscribing_ellipsoid_Q, inscribing_ellipsoid_G
    else:
        raise NotImplementedError(
            f"Did not expect to reach here in maximum volume inscribing ellipsoid! CVXPY status: {prob.status}"
        )


def remove_redundancies(self):
    """Remove any redundancies in the equality system using pycddlib and other geometric properties"""
    _, _, Aebe_status, solution_to_Ae_x_eq_be = sanitize_and_identify_Aebe(self.Ae, self.be)
    if Aebe_status == "no_Ae_be":
        # Set only (Ae, be) to empty | Full-dimensionality depends on rank of G
        _, _, self._Ae, self._be = self._get_Gc_Aebe_for_empty_constrained_zonotope(self.dim, self.latent_dim)
        self._is_full_dimensional, self._is_empty = None, self.c is None
    elif Aebe_status == "infeasible":
        # Set (G, c, Ae, be) to empty > Overwrite G, c
        self._G, self._c, self._Ae, self._be = self._get_Gc_Aebe_for_empty_constrained_zonotope(self.dim, 0)
        self._is_full_dimensional, self._is_empty = (self.dim == 0), True
    elif Aebe_status == "single_point":
        # Set (G, Ae, be) to empty > Overwrite G | We can not have self.dim = 0 so not full-dimensional
        dim = solution_to_Ae_x_eq_be.size
        self._G, _, self._Ae, self._be = self._get_Gc_Aebe_for_empty_constrained_zonotope(dim, 0)
        self._c = solution_to_Ae_x_eq_be
        self._is_full_dimensional, self._is_empty = False, False
    else:  # affine_set
        self._Ae, self._be = compute_irredundant_affine_set_using_cdd(self.Ae, self.be)
