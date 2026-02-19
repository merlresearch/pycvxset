# Copyright (C) 2020-2026 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

# Code purpose:  Define the methods involving another set or a point used with Ellipsoid class
# Coverage: This file has 0 missing statements + 6 excluded statements + 0 partial statements.

from __future__ import annotations

from typing import TYPE_CHECKING, Sequence, cast

import numpy as np
import scipy as sp

from pycvxset.common import (
    PYCVXSET_ZERO,
    compute_irredundant_affine_set_using_cdd,
    convex_set_contains_points,
    is_constrained_zonotope,
    is_ellipsoid,
    is_polytope,
    sanitize_and_identify_Aebe,
)

if TYPE_CHECKING:
    from pycvxset.ConstrainedZonotope import ConstrainedZonotope
    from pycvxset.Ellipsoid import Ellipsoid
    from pycvxset.Polytope import Polytope

DOCSTRING_FOR_PROJECT: str = (
    "\n"
    + r"""
    Notes:

        For a point :math:`y\in\mathbb{R}^{\mathcal{P}.\text{dim}}` and an ellipsoid :math:`\mathcal{P}=\{G u + c\ |\ \|
        u \|_2 \leq 1 \}` with :math:`GG^T=Q`,  this function solves a convex program with decision variables
        :math:`x,u\in\mathbb{R}^{\mathcal{P}.\text{dim}}`,

        .. math::
                \text{minimize}    &\quad  \|x - y\|_p\\
                \text{subject to}  &\quad  x = G u + c\\
                                   &\quad  {\| u \|}_2 \leq 1
    """
)

DOCSTRING_FOR_PROJECTION: str = (
    "\n"
    + r"""
    Returns:
        Ellipsoid: m-dimensional set obtained via projection.
    """
)


DOCSTRING_FOR_SLICE: str = (
    "\n"
    + r"""
    Returns:
        Ellipsoid: Ellipsoid that has been sliced at the specified dimensions.

    Notes:
        -
    """
)

DOCSTRING_FOR_SLICE_THEN_PROJECTION: str = (
    "\n"
    + r"""
    Returns:
        Ellipsoid: m-dimensional set obtained via projection after slicing.
    """
)


def affine_map(self: Ellipsoid, M: int | float | Sequence[Sequence[float]] | np.ndarray) -> Ellipsoid:
    r"""Compute the affine transformation of the given ellipsoid based on a given scalar/matrix.

    Args:
        M (int | float | Sequence[Sequence[float]] | np.ndarray): Scalar or matrix (m times self.dim) for the affine
            map.

    Raises:
        ValueError: When M is not convertible into a 2D numpy array of float
        ValueError: When M has columns not equal to self.dim

    Returns:
        Ellipsoid: Affine transformation of the given ellipsoid
        :math:`\mathcal{R}=M\mathcal{P}=\{Mx|x\in\mathcal{P}\}`
    """
    try:
        M_arr: np.ndarray = np.atleast_2d(M).astype(float)
    except (ValueError, TypeError) as err:
        raise TypeError(f"Expected M to be convertible into 2D numpy array of float. Got {type(M)}!") from err
    if M_arr.ndim > 2:
        raise ValueError(f"M is must be convertible into a 2D numpy.ndarray. But got {M_arr.ndim:d}D array.")
    elif M_arr.shape[0] == M_arr.shape[1] and M_arr.shape[0] == 1:  # Scalar multiplication case
        m = np.squeeze(M_arr)
        # scalar * matrix is the same as (scalar * identity matrix) @ matrix
        return self.__class__(c=m * self.c, G=(m * self.G))
    elif M_arr.shape[1] != self.dim:
        raise ValueError(f"Expected M to be matrix with {self.dim:d}-dim columns. Got M: {M_arr.shape} matrix")
    else:
        new_G = M_arr @ self.G
        new_c = M_arr @ self.c
        return self.__class__(c=new_c, G=new_G)


def contains(
    self: Ellipsoid, Q: Sequence[float] | Sequence[Sequence[float]] | np.ndarray | Polytope | Ellipsoid
) -> bool | np.ndarray:
    r"""Check containment of a set or a collection of points in an ellipsoid.

    Args:
        Q (Sequence[float] | Sequence[Sequence[float]] | np.ndarray | Polytope | Ellipsoid): Polytope/Ellipsoid or a
            collection of points (each row is a point) to be tested for containment. When providing a collection of
            points, Q is a matrix (N times self.dim) with each row is a point.

    Raises:
        ValueError: Test point(s) are NOT of the same dimension
        ValueError: Test point(s) can not be converted into a 2D numpy array of floats
        ValueError: Q is a constrained zonotope
        NotImplementedError: Unable to perform ellipsoidal containment check using CVXPY

    Returns:
        bool or numpy.ndarray[bool]: An element of the array is True if the point is in the ellipsoid, with as many
            elements as the number of rows in `test_points`.

    Notes:
        - **Containment of a polytope**: This function requires the polytope Q to be in V-Rep. If Q is in H-Rep, a
          vertex enumeration is performed. This function then checks if all vertices of Q are in the given
          ellipsoid, which occurs if and only if the polytope is contained within the ellipsoid [BV04]_.
        - **Containment of an ellipsoid**: This function solves a semi-definite program (S-procedure) [BV04]_.
        - **Containment of points**: For each point :math:`v`, the function checks if there is a
          :math:`u\in\mathbb{R}^{\mathcal{P}.\text{dim}}` such that :math:`\|u\|_2 \leq 1` and :math:`Gu + c=v`
          [BV04]_. This can be efficiently done via `numpy.linalg.lstsq
          <https://numpy.org/doc/stable/reference/generated/numpy.linalg.lstsq.html>`_.
    """
    import cvxpy as cp

    if is_constrained_zonotope(Q):
        raise ValueError("Checking containment of a ConstrainedZonotope in an Ellipsoid is not supported.")
    elif is_polytope(Q):
        polytope_Q = cast("Polytope", Q)
        return cast(np.ndarray, self.contains(polytope_Q.V)).all()
    elif is_ellipsoid(Q):
        ellipsoid_Q = cast("Ellipsoid", Q)
        if ellipsoid_Q.dim != self.dim:
            raise ValueError(
                f"Containment check failed due to dimension mismatch! self.dim:{self.dim}, Q.dim:{ellipsoid_Q.dim}"
            )
        # Consider all types of Q - singleton, special not full-dimensional | not full-dimensional | full-dimensional
        # self is either full-dimensional or a singleton
        elif ellipsoid_Q.is_singleton:
            return self.contains(ellipsoid_Q.c)
        elif not self.is_full_dimensional:
            if not ellipsoid_Q.is_full_dimensional:
                raise ValueError("Checking containment between non-singleton degenerate ellipsoids is not supported!")
            else:
                return False  # Q is full dimensional but self is not, so clearly Q can not be contained in self
        else:
            # self is full-dimensional | Q may or may not be full-dimensional
            # Q_Gc_mat_T ensures that the S-Procedure checks only for x belonging to the affine hull of Q
            # Also Q_Gc_mat_T searches over u in Q's latent dimension space (possibly <= Q.dim)
            Q_Gc_mat_T = np.hstack(
                (
                    np.vstack((ellipsoid_Q.G.T, ellipsoid_Q.c)),
                    np.vstack((np.zeros((ellipsoid_Q.latent_dim, 1)), 1)),
                )
            )
            lhs_matrix = Q_Gc_mat_T @ self.quadratic_form_as_a_symmetric_matrix() @ Q_Gc_mat_T.T

            lambda_var = cp.Variable(nonneg=True)
            const = [lhs_matrix << lambda_var * sp.linalg.block_diag(np.eye(ellipsoid_Q.latent_dim), -1)]
            prob = cp.Problem(cp.Minimize(lambda_var), const)
            try:
                prob.solve(**self.cvxpy_args_sdp)
            except cp.error.SolverError as err:  # pyright: ignore[reportAttributeAccessIssue]
                raise NotImplementedError(
                    f"Unable to check containment of an ellipsoid in another! CVXPY returned error: {str(err)}"
                ) from err
            if prob.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
                return True
            elif prob.status in [cp.INFEASIBLE, cp.INFEASIBLE_INACCURATE]:
                return False
            else:
                raise NotImplementedError(
                    f"Did not expect to reach here during containment check of an ellipsoid in another. CVXPY "
                    f"returned status: {prob.status:s}."
                )
    else:
        points = np.atleast_2d(Q).astype(float)
        n_points, point_dim = points.shape
        if point_dim != self.dim:
            raise ValueError(f"Mismatch in dimensions (self.dim: {self.dim:d} and test_points.dim: {point_dim:d})")
        elif self.is_singleton:
            return np.isclose(points, self.c).all(axis=1)
        elif self.is_full_dimensional:
            self_quadratic_form = self.quadratic_form_as_a_symmetric_matrix()
            stacked_point_one_vector = np.vstack((points.T, np.ones((n_points,))))
            containment_flag = (
                np.diag(stacked_point_one_vector.T @ self_quadratic_form @ stacked_point_one_vector) <= PYCVXSET_ZERO
            )
            if n_points > 1:
                return containment_flag
            else:
                return containment_flag[0]
        else:
            return convex_set_contains_points(self, points)


def deflate(cls: type[Ellipsoid], cvx_set: Ellipsoid | Polytope) -> Ellipsoid:
    r"""Compute the minimum volume ellipsoid that covers the given set (also known as Lowner-John Ellipsoid).

    Args:
       cvx_set (Ellipsoid | Polytope): Set to be circumscribed.

    Returns:
        Ellipsoid: Minimum volume circumscribing ellipsoid

    Notes:
        This function is a wrapper for :meth:`minimum_volume_circumscribing_ellipsoid` of the set
        :attr:`set_to_be_centered`. Please check that function for more details including raising exceptions.
        [EllipsoidalTbx-Min_verticesolEll]_
    """
    c, _, G = cvx_set.minimum_volume_circumscribing_ellipsoid()
    return cls(c=c, G=G)


def inflate(cls: type[Ellipsoid], cvx_set: ConstrainedZonotope | Ellipsoid | Polytope) -> Ellipsoid:
    r"""Compute the maximum volume ellipsoid that fits within the given set.

    Args:
        cvx_set (ConstrainedZonotope | Ellipsoid | Polytope): Set to be inscribed.

    Returns:
        Ellipsoid: Maximum volume inscribing ellipsoid

    Notes:
        This function is a wrapper for :meth:`maximum_volume_inscribing_ellipsoid` of the set
        :attr:`set_to_expand_within`. Please check that function for more details including raising exceptions.
        [EllipsoidalTbx-MinVolEll]_
    """
    c, _, G = cvx_set.maximum_volume_inscribing_ellipsoid()
    return cls(c=c, G=G)


def inflate_ball(cls: type[Ellipsoid], cvx_set: ConstrainedZonotope | Ellipsoid | Polytope) -> Ellipsoid:
    r"""Compute the largest ball (Chebyshev ball) of a given set.

    Args:
        cvx_set (ConstrainedZonotope | Ellipsoid | Polytope): Set to compute Chebyshev ball for.

    Returns:
        Ellipsoid: Maximum volume inscribing ellipsoid

    Notes:
        This function is a wrapper for :meth:`chebyshev_center` of attr:`set_to_be_centered`. Please check that function
        for more details including raising exceptions.
    """
    c, r = cvx_set.chebyshev_centering()
    return cls(c=c, r=r)


def intersection_with_affine_set(
    self: Ellipsoid, Ae: Sequence[Sequence[float]] | np.ndarray, be: Sequence[float] | np.ndarray
) -> Ellipsoid:
    r"""Compute the intersection of an ellipsoid with an affine set.

    Args:
        Ae (Sequence[Sequence[float]] | np.ndarray): Equality coefficient matrix (N times self.dim) that define the
            affine set :math:`\{x|A_ex = b_e\}`.
        be (Sequence[float] | np.ndarray): Equality constant vector (N,) that define the affine set :math:`\{x| A_ex =
            b_e\}`.

    Raises:
        ValueError: When the number of columns in Ae is different from self.dim

    Returns:
        Ellipsoid: The intersection of an ellipsoid with the affine set.

    Notes:
        This function implements imposes the constraints :math:`\{A_ex = b_e\}` as constraints in the latent dimension
        of the ellipsoid --- :math:`A_e (G \xi + c) = b_e` for every feasible :math:`\xi`.
    """
    sanitized_Ae, sanitized_be, Aebe_status, solution_to_Ae_x_eq_be = sanitize_and_identify_Aebe(Ae, be)
    if Aebe_status == "no_Ae_be":
        return self.copy()
    sanitized_Ae = cast(np.ndarray, sanitized_Ae)
    sanitized_be = cast(np.ndarray, sanitized_be)
    if self.dim != sanitized_Ae.shape[1]:
        raise ValueError(f"Expected Ae to have {self.dim:d} columns. Got Ae with shape {sanitized_Ae.shape}!")
    if Aebe_status == "infeasible":
        raise ValueError("Intersection with an empty affine set!")
    elif Aebe_status == "single_point":
        solution_to_Ae_x_eq_be = cast(np.ndarray, solution_to_Ae_x_eq_be)
        if solution_to_Ae_x_eq_be in self:
            # By the check above, solution_to_Ae_x_eq_be is in self
            return self.__class__(c=solution_to_Ae_x_eq_be)
        else:
            raise ValueError("The affine set characterizes a point, and the point is outside the ellipsoid!")
    else:
        # Make sure that Ae @ G is full row rank
        irredundant_Ae_G, irredundant_be_minus_Ae_c = compute_irredundant_affine_set_using_cdd(
            sanitized_Ae @ self.G, sanitized_be - sanitized_Ae @ self.c
        )
        # Take QR decomposition of (Ae @ G).T
        Q, R = np.linalg.qr(irredundant_Ae_G.T, mode="complete")
        rank_Ae_G = irredundant_Ae_G.shape[0]
        # {u | x = G @ u +c, Ae @ x=be} = {u | Ae @ (G @ u +c) = be} = {u | Ae @ G @ u  = be - Ae @ c}
        # = {u | u_star + Q2 z} where u_star = Q1 @ R^{-T} @ (be - Ae @ c) and [Q1, Q2] @ R = (Ae @ G).T
        Q1, Q2 = Q[:, :rank_Ae_G], Q[:, rank_Ae_G:]
        u_star = Q1 @ np.linalg.inv(R[:rank_Ae_G, :]).T @ irredundant_be_minus_Ae_c
        # ||u_star + Q2 z ||^2 = u_star @ u_star + z @ z <= 1 <=> (z @ z) <= 1 - u_star @ u_star
        scaling_sqr = 1 - u_star @ u_star
        # Set of interest is {G u + c | ||u|| <= 1} \cap {x | Ae x = be}
        # = G {u | x = G @ u +c, Ae @ x=be, || u || <= 1} + c
        # = G {u | Ae (G @ u +c)=be, || u || <= 1} + c
        # = G {u_star + Q2 z | || u_star + Q2 z || <= 1} + c
        # = {G Q2 z | z^T z <= 1 - u_star @ u_star} + G u_star + c
        # = {G Q2 z | || z ||/np.sqrt(scaling_sqr) <= 1}  + G u_star + c
        # = {np.sqrt(scaling_sqr) G Q2 w | ||w|| <= 1} + G u_star + c
        new_c = self.c + self.G @ u_star
        if scaling_sqr < -PYCVXSET_ZERO:
            raise ValueError("pycvxset does not support empty ellipsoids, but the intersection resulted in one!")
        elif scaling_sqr <= PYCVXSET_ZERO:
            return self.__class__(c=new_c)
        else:
            new_G = (np.sqrt(scaling_sqr) * np.eye(self.G.shape[0])) @ self.G @ Q2
            return self.__class__(c=new_c, G=new_G)


def inverse_affine_map_under_invertible_matrix(
    self: Ellipsoid, M: int | float | Sequence[Sequence[float]] | np.ndarray
) -> Ellipsoid:
    r"""Compute the inverse affine transformation of an ellipsoid based on a given scalar/matrix.

    Args:
        M (int | float | Sequence[Sequence[float]] | np.ndarray): Scalar or invertible square matrix for the affine map

    Raises:
        TypeError: When M is not convertible into a 2D square numpy matrix
        TypeError: When M is not invertible

    Returns:
        Ellipsoid: Inverse affine transformation of the given ellipsoid
        :math:`\mathcal{R}=\mathcal{P}M=\{x|Mx\in\mathcal{P}\}`

    Notes:
        1. Since M is invertible, :math:`\mathcal{R}` is also a bounded ellipsoid.
    """
    try:
        M_arr: np.ndarray = np.atleast_2d(M).astype(float)
    except (TypeError, ValueError) as err:
        raise TypeError(f"Multiplication of Ellipsoid with {type(M)} is not supported!") from err
    if M_arr.shape != (self.dim, self.dim):
        raise TypeError(f"Expected M to be a square matrix of shape ({self.dim:d},{self.dim:d}). Got {M_arr.shape}!")
    try:
        M_inv = np.linalg.inv(M_arr)
    except np.linalg.LinAlgError as err:
        raise TypeError("Expected M to be invertible!") from err
    return affine_map(self, M_inv)


def plus(self: Ellipsoid, point: Sequence[float] | np.ndarray) -> Ellipsoid:
    """Add a point to an ellipsoid

    Args:
        point (Sequence[float] | np.ndarray): Vector (self.dim,) that describes the point to be added.

    Raises:
        ValueError: point is a set (ConstrainedZonotope or Ellipsoid or Polytope)
        TypeError: point can not be converted into a numpy array of float
        ValueError: point can not be converted into a 1D numpy array of float
        ValueError: Mismatch in dimension

    Returns:
        Ellipsoid: Sum of the ellipsoid and the point.
    """
    if is_constrained_zonotope(point) or is_ellipsoid(point) or is_polytope(point):
        raise ValueError(f"Ellipsoid and {type(point)} can not be added exactly!")
    else:
        try:
            point_arr: np.ndarray = np.atleast_1d(np.squeeze(point)).astype(float)
        except (TypeError, ValueError) as err:
            raise TypeError(f"Unsupported operation Ellipsoid + {type(point)}!") from err
        if point_arr.ndim != 1:
            raise ValueError(f"Expected 1D sequence or numpy array. Got point with shape: {point_arr.shape}!")
        if point_arr.size != self.dim:
            raise ValueError(f"Expected point to be {self.dim:d}-dimensional. Got point with shape: {point_arr.shape}!")
        new_c = point_arr + self.c
        return self.__class__(c=new_c, G=self.G)
