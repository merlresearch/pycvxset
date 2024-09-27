# Copyright (C) 2020-2024 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

# Code purpose:  Define the methods involving another set or a point used with Ellipsoid class

import cvxpy as cp
import numpy as np

from pycvxset.common import (
    PYCVXSET_ZERO,
    convex_set_contains_points,
    is_constrained_zonotope,
    is_ellipsoid,
    is_polytope,
)


def _compute_support_function_single_eta(self, eta):
    """Private function to compute the support function"""
    support_vector = self.c + ((self.Q @ eta) / np.linalg.norm(self.G.T @ eta))
    return eta @ support_vector, support_vector


def affine_map(self, M):
    r"""Compute the affine transformation of the given ellipsoid based on a given scalar/matrix.

    Args:
        M (int | float | array_like): Scalar not equal to zero or matrix (m times self.dim) for the affine map where
            :math:`1 \leq m \leq \text{self.dim}`.

    Raises:
        ValueError: When M is not convertible into a 2D numpy array of float
        ValueError: When M has more rows than self.dim or M has columns not equal to self.dim
        ValueError: When M is not full row-rank
        ValueError: When M is a scalar but not positive enough

    Returns:
        Ellipsoid: Affine transformation of the given ellipsoid
        :math:`\mathcal{R}=M\mathcal{P}=\{Mx|x\in\mathcal{P}\}`

    Notes:
        The matrix M needs to have M has self.dim columns, and full row-rank to preserve full-dimensionality of
        ellipsoids. See [HJ13rank]_ for the non-degeneracy sufficient condition on M.
    """
    try:
        M = np.atleast_2d(M).astype(float)
    except (ValueError, TypeError) as err:
        raise TypeError(f"Expected M to be convertible into 2D numpy array of float. Got {type(M)}!") from err
    if M.ndim > 2:
        raise ValueError(f"M is must be convertible into a 2D numpy.ndarray. But got {M.ndim:d}D array.")
    elif M.shape[0] == M.shape[1] and M.shape[0] == 1:  # Scalar multiplication case
        m = np.squeeze(M)
        if abs(m) > PYCVXSET_ZERO:
            return self.__class__(m * self.c, G=(m * np.eye(self.dim)) @ self.G)
        else:
            raise ValueError(f"Expected scalar M to be positive (greater than {PYCVXSET_ZERO:1.2e}! Got {m}")
    elif M.shape[1] != self.dim and M.shape[0] > self.dim:
        raise ValueError(
            f"Expected M to be matrix with at most {self.dim:d}-dim rows and {self.dim:d}-dim columns. "
            f"Got M: {M.shape} matrix"
        )
    else:
        rank_M = np.linalg.matrix_rank(M)
        if rank_M != M.shape[0]:
            raise ValueError(f"Expected M to be matrix with full row-rank. Got: {rank_M} matrix")
        new_Q = M @ self.Q @ M.T
        new_c = M @ self.c
        return self.__class__(c=new_c, Q=new_Q)


def contains(self, Q):
    r"""Check containment of a set or a collection of points in an ellipsoid.

    Args:
        Q (array_like | Polytope | Ellipsoid): Polytope/Ellipsoid or a collection of points (each row is a point) to
            be tested for containment. When providing a collection of points, Q is a matrix (N times self.dim) with
            each row is a point.

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
        - **Containment of an ellipsoid**: This function solves a feasibility semi-definite program and requires the
          ellipsoids to be full-dimensional [BV04]_.
        - **Containment of points**: For each point :math:`v`, the function checks if there is a
          :math:`u\in\mathbb{R}^{\mathcal{P}.\text{dim}}` such that :math:`\|u\|_2` \leq 1 and :math:`Gu + c=v`
          [BV04]_. This can be efficiently done via `numpy.linalg.lstsq
          <https://numpy.org/doc/stable/reference/generated/numpy.linalg.lstsq.html>`_.
    """
    if is_constrained_zonotope(Q):
        raise ValueError("Checking containment of a ConstrainedZonotope in an Ellipsoid is not supported.")
    elif is_polytope(Q):
        return self.contains(Q.V).all()
    elif is_ellipsoid(Q):
        # From https://systemanalysisdpt-cmc-msu.github.io/ellipsoids/doc/chap_ellcalc.html
        # checking-if-one-ellipsoid-contains-another
        lambda_var = cp.Variable(nonneg=True)

        def matrix_definer(c, Q):
            Q_inv = np.linalg.inv(Q)
            return np.vstack(
                (np.hstack((Q_inv, -Q_inv @ c[:, None])), np.hstack((c.T @ -Q_inv.T, c.T @ Q_inv @ c - 1)))
            )

        lhs_matrix = matrix_definer(self.c, self.Q)
        rhs_matrix = matrix_definer(Q.c, Q.Q)
        const = [lhs_matrix << lambda_var * rhs_matrix]
        prob = cp.Problem(cp.Minimize(0), const)
        try:
            prob.solve(**self.cvxpy_args_sdp)
        except cp.error.SolverError as err:
            raise NotImplementedError(
                f"Unable to check containment of an ellipsoid in another! CVXPY returned error: {str(err)}"
            ) from err
        if prob.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
            return True
        elif prob.status in [cp.INFEASIBLE, cp.INFEASIBLE_INACCURATE]:
            return False
        else:
            raise NotImplementedError(
                f"Did not expect to reach here during containment check of an ellipsoid in another. CVXPY returned "
                f"status: {prob.status:s}."
            )
    else:
        return convex_set_contains_points(self, Q)


def deflate(cls, set_to_be_centered):
    r"""Compute the minimum volume ellipsoid that covers the given set (also known as Lowner-John Ellipsoid).

    Args:
        set_to_be_centered (Polytope | ConstrainedZonotope): Set to be circumscribed.

    Returns:
        Ellipsoid: Minimum volume circumscribing ellipsoid

    Notes:
        This function is a wrapper for :meth:`minimum_volume_circumscribing_ellipsoid` of the set
        :attr:`set_to_be_centered`. Please check that function for more details including raising exceptions.
        [EllipsoidalTbx-Min_verticesolEll]_
    """
    c, Q, _ = set_to_be_centered.minimum_volume_circumscribing_ellipsoid()
    return cls(c, Q=Q)


def inflate(cls, set_to_be_centered):
    r"""Compute the maximum volume ellipsoid that fits within the given set.

    Args:
        set_to_be_centered (Polytope | ConstrainedZonotope): Set to be inscribed.

    Returns:
        Ellipsoid: Maximum volume inscribing ellipsoid

    Notes:
        This function is a wrapper for :meth:`maximum_volume_inscribing_ellipsoid` of the set
        :attr:`set_to_expand_within`. Please check that function for more details including raising exceptions.
        [EllipsoidalTbx-MinVolEll]_
    """
    c, Q, _ = set_to_be_centered.maximum_volume_inscribing_ellipsoid()
    return cls(c, Q=Q)


def inverse_affine_map_under_invertible_matrix(self, M):
    r"""Compute the inverse affine transformation of an ellipsoid based on a given scalar/matrix.

    Args:
        M (int | float | array_like): Scalar or invertible square matrix for the affine map

    Raises:
        TypeError: When M is not convertible into a 2D square numpy matrix
        TypeError: When M is not invertible

    Returns:
        Ellipsoid: Inverse affine transformation of the given ellipsoid
        :math:`\mathcal{R}=\mathcal{P}M=\{x|Mx\in\mathcal{P}\}`
    """
    try:
        M = np.atleast_2d(M).astype(float)
    except (TypeError, ValueError) as err:
        raise TypeError(f"Multiplication of Ellipsoid with {type(M)} is not supported!") from err
    if M.shape != (self.dim, self.dim):
        raise TypeError(f"Expected M to be a square matrix of shape ({self.dim:d},{self.dim:d}). Got {M.shape}!")
    try:
        M_inv = np.linalg.inv(M)
    except np.linalg.LinAlgError as err:
        raise TypeError("Expected M to be invertible!") from err
    return self.affine_map(M_inv)


def inflate_ball(cls, set_to_be_centered):
    r"""Compute the largest ball (Chebyshev ball) of a given set.

    Args:
        set_to_be_centered (Polytope | ConstrainedZonotope): Set to compute Chebyshev ball for.

    Returns:
        Ellipsoid: Maximum volume inscribing ellipsoid

    Notes:
        This function is a wrapper for :meth:`chebyshev_center` of attr:`set_to_be_centered`. Please check that function
        for more details including raising exceptions.
    """
    c, r = set_to_be_centered.chebyshev_centering()
    return cls(c, r=r)


def plus(self, point):
    """Add a point to an ellipsoid

    Args:
        point (array_like): Vector (self.dim,) that describes the point to be added.

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
            point = np.atleast_1d(np.squeeze(point)).astype(float)
        except (TypeError, ValueError) as err:
            raise TypeError(f"Unsupported operation Ellipsoid + {type(point)}!") from err
        if point.ndim != 1:
            raise ValueError(f"Expected 1D array_like object. Got point with shape: {point.shape}!")
        if point.size != self.dim:
            raise ValueError("Expected point to be {self.dim:d}-dimensional. Got point with shape: {point.shape}!")
        new_c = point + self.c
        return self.__class__(c=new_c, Q=self.Q)
