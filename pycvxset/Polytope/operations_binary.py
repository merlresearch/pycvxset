# Copyright (C) 2020-2025 Mitsubishi Electric Research Laboratories (MERL)
# Copyright (c) 2019 Tor Aksel N. Heirung
#
# SPDX-License-Identifier: AGPL-3.0-or-later
# SPDX-License-Identifier: MIT

# Code purpose:  Define the methods involving another set or a point used with Polytope class

import numpy as np

from pycvxset.common import (
    convex_set_contains_points,
    is_constrained_zonotope,
    is_ellipsoid,
    is_polytope,
    sanitize_Ab,
    sanitize_and_identify_Aebe,
)
from pycvxset.common.constants import PYCVXSET_ZERO

DOCSTRING_FOR_PROJECT = (
    "\n"
    + r"""
    Notes:
        - This function allows for :math:`\mathcal{P}` to be in V-Rep or in H-Rep.
        - Given a polytope :math:`\mathcal{P}` in V-Rep and a test point
          :math:`y\in\mathbb{R}^{\mathcal{P}.\text{dim}}`, this function solves a convex program with decision variables
          :math:`x\in\mathbb{R}^{\mathcal{P}.\text{dim}}` and
          :math:`\theta\in\mathbb{R}^{\mathcal{P}.\text{n\_vertices}}`,

            .. math::
                \text{minimize}    &\quad  \|x - y\|_p\\
                \text{subject to}  &\quad  x = \sum_i \theta_i v_i\\
                                   &\quad  \sum_i \theta_i = 1, \theta_i \geq 0
        - Given a polytope :math:`\mathcal{P}` in H-Rep and a test point
          :math:`y\in\mathbb{R}^{\mathcal{P}.\text{dim}}`, this function solves a convex program with a decision
          variable :math:`x\in\mathbb{R}^{\mathcal{P}.\text{dim}}`,

            .. math::
                \text{minimize}    &\quad  \|x - y\|_p\\
                \text{subject to}  &\quad  A x \leq b\\
                                   &\quad  A_e x = b_e
    """
)

DOCSTRING_FOR_PROJECTION = (
    "\n"
    + r"""
    Returns:
        Polytope: m-dimensional set obtained via projection.

    Notes:
        This function requires P to be in V-Rep, and performs a vertex enumeration when P is in H-Rep.
    """
)


DOCSTRING_FOR_SLICE = (
    "\n"
    + r"""
    Returns:
        Polytope: Polytope that has been sliced at the specified dimensions.

    Notes:
        This function requires :math:`\mathcal{P}` to be in H-Rep, and performs a vertex halfspace when
        :math:`\mathcal{P}` is in V-Rep.
    """
)

DOCSTRING_FOR_SLICE_THEN_PROJECTION = (
    "\n"
    + r"""
    Returns:
        Polytope: m-dimensional set obtained via projection after slicing.

    Notes:
        This function requires :math:`\mathcal{P}` to be in H-Rep, and performs a vertex halfspace when
        :math:`\mathcal{P}` is in V-Rep.
    """
)

DOCSTRING_FOR_SUPPORT = (
    "\n"
    + r"""
    Notes:
        - This function allows for :math:\mathcal{P} to be in V-Rep or in H-Rep.
        - Given a polytope :math:`\mathcal{P}` in V-Rep and a support direction
          :math:`\eta\in\mathbb{R}^{\mathcal{P}.\text{dim}}`, this function solves a convex program with decision
          variables :math:`x\in\mathbb{R}^{\mathcal{P}.\text{dim}}` and
          :math:`\theta\in\mathbb{R}^{\mathcal{P}.\text{n\_vertices}}`,

            .. math::
                \text{maximize}    &\quad  \eta^\top x\\
                \text{subject to}  &\quad  x = \sum_i \theta_i v_i\\
                                   &\quad  \sum_i \theta_i = 1, \theta_i \geq 0
        - Given a polytope :math:`\mathcal{P}` in H-Rep and a support direction
          :math:`\eta\in\mathbb{R}^{\mathcal{P}.\text{dim}}`, this function solves a convex program with a decision
          variable :math:`x\in\mathbb{R}^{\mathcal{P}.\text{dim}}`,

            .. math::
                \text{maximize}    &\quad  \eta^\top x\\
                \text{subject to}  &\quad  A x \leq b\\
                                   &\quad  A_e x = b_e
    """
)


def affine_map(self, M):
    r"""Compute the matrix times set.

    Args:
        M (array_like): A vector or array (0, 1, or 2-D numpy.ndarray-like object) with self.dim columns

    Raises:
        ValueError: When M can not be converted EXACTLY into a 2D array of float
        ValueError: When M does not have self.dim columns

    Returns:
        Polytope: The scaled polytope :math:`\mathcal{R} = M \mathcal{P} = \{M x: x\in \mathcal{P}\}`

    Notes:
        This function requires :math:`\mathcal{P}` to be in V-Rep, and performs a vertex enumeration when
        :math:`\mathcal{P}` is in H-Rep.
    """
    try:
        M = np.atleast_2d(M).astype(float)
    except (TypeError, ValueError) as err:
        raise TypeError("Expected M to be 2D array_like that can be converted to float!") from err
    if M.ndim > 2:
        raise ValueError(f"M is must be convertible into a 2D numpy.ndarray. But got {M.ndim:d}D array.")
    elif self.dim != 1 and M.shape[0] == M.shape[1] and M.shape[0] == 1:
        # Scalar multiplication
        m = M[0][0]
        if self.is_empty:
            return self.__class__(dim=self.dim)
        elif abs(m) <= PYCVXSET_ZERO:
            # m=0 multiplier => product is polytope that only has origin
            return self.__class__(V=np.zeros((1, self.dim)))
        elif self.in_H_rep:
            # In H-rep, {m x: Ax <= b} = {z: A(z/m) <= b} = {z: sign(m) * A(z/|m|) <= b}={z: sign(m) Az <= b|m|}
            if self.n_equalities > 0:
                return self.__class__(
                    A=(np.sign(m) * self.A), b=(self.b * np.abs(m)), Ae=(np.sign(m) * self.Ae), be=(self.be * np.abs(m))
                )
            else:
                return self.__class__(A=(np.sign(m) * self.A), b=(self.b * np.abs(m)))
        else:
            # Scale the V-rep: m conv(v_1,...,v_n) = conv(m v_1, ..., m v_n)
            return self.__class__(V=(m * self.V))
    elif M.shape[1] != self.dim:
        raise ValueError(f"Expected M upon promotion to 2D array to have {self.dim:d} columns. M: {M.shape} matrix")
    elif self.is_empty:
        return self.__class__(dim=M.shape[0])
    else:
        transposed_new_vertices = M @ self.V.T
        return self.__class__(V=transposed_new_vertices.T)


def contains(self, Q):
    r"""Check containment of a set :math:`\mathcal{Q}` (could be a polytope, an ellipsoid, or a constrained zonotope),
    or a collection of points :math:`Q \in \mathbb{R}^{n_Q \times \mathcal{P}.\text{dim}}` in the polytope
    :math:`\mathcal{P}`.

    Args:
        Q (array_like | Polytope | ConstrainedZonotope | Ellipsoid): Set or a collection of points (each row is a point)
            to be tested for containment within :math:`\mathcal{P}`. When providing a collection of points, Q is a
            matrix (N times self.dim) with each row is a point.

    Raises:
        ValueError: When two polytopes | the polytope and the test point(s) are NOT of the same dimension

    Returns:
        bool | numpy.ndarray[bool]: When `Q` is a polytope, a bool is returned which is True if and only if
        :math:`Q\subseteq P`. On the other hand, if `Q` is array_like (Q is a point or a collection of points), then an
        numpy.ndarray of bool is returned, with as many elements as the number of rows in `Q`.

    Notes:
        - *Containment of polytopes*: This function accommodates the following combinations of representations for
          :math:`\mathcal{P}` and :math:`\mathcal{Q}`. It eliminates the need for enumeration by comparing support
          function evaluations almost always. When P is H-Rep and Q is V-Rep, then we perform a quick check for
          containment of vertices of Q in P. Otherwise,
          * Q is empty: Return True
          * P is empty: Return False

        - *Containment of points*: This function accommodates :math:`\mathcal{P}` to be in H-Rep or V-Rep. When testing
          if a point is in a polytope where only V-Rep is available for self, we solve a collection of second-order cone
          programs for each point using :meth:`project` (check if distance between v and :math:`\mathcal{P}` is nearly
          zero).  Otherwise, we use the observation that for :math:`\mathcal{P}` with (A, b, Ae, be), then
          :math:`v\in\mathcal{P}` if and only if :math:`Av\leq b` and :math:`A_ev=b_e`. For numerical precision
          considerations, we use :meth:`numpy.isclose`.
    """
    if is_constrained_zonotope(Q) or is_ellipsoid(Q) or is_polytope(Q):
        # Q is a polytope or a constrained zonotope
        if self.dim != Q.dim:
            raise ValueError(f"Mismatch in dimensions (self.dim: {self.dim:d} and Q.dim: {Q.dim:d})")
        elif self.is_empty:
            return False
        elif Q.is_empty:
            return True
        elif is_constrained_zonotope(Q) or is_ellipsoid(Q) or not (self.in_H_rep and Q.in_V_rep):
            # Compute inclusion via support function of Q
            containment_in_Ab = True
            containment_in_Aebe = True
            if self.n_halfspaces > 0:
                containment_in_Ab = np.all(Q.support(self.A)[0] <= self.b + PYCVXSET_ZERO)
            if self.n_equalities > 0:
                containment_in_Aebe = bool(np.all(Q.support(self.Ae)[0] <= self.be + PYCVXSET_ZERO)) and bool(
                    np.all(-Q.support(-self.Ae)[0] >= self.be - PYCVXSET_ZERO)
                )
            return containment_in_Aebe and containment_in_Ab
        else:
            # Compute inclusion via vertices of Q, which is most useful when (self.in_H_rep and Q.in_V_rep) is True
            containment_of_vertices = self.contains(Q.V)
            return bool(np.all(containment_of_vertices))
    else:
        # Q is a set of test_points
        try:
            test_points = np.atleast_2d(Q).astype(float)
        except (TypeError, ValueError) as err:
            raise ValueError(f"Expected Q to be a convertible into a float 2D numpy array. Got {type(Q)}!") from err
        n_test_points, test_point_dim = test_points.shape

        if self.is_empty:
            if n_test_points > 1:
                return np.zeros((n_test_points,), dtype="bool")
            else:
                return False
        elif test_point_dim != self.dim:
            # Could transpose here if test_points.shape[1] == self.dim, but better to
            # specify that points must be columns:
            raise ValueError(f"Mismatch in dimensions (self.dim: {self.dim:d} and test_points.dim: {test_point_dim:d})")

        if self.in_H_rep:
            # Check if A test_points <= b
            AxT_minus_b = self.A @ test_points.T - np.tile(np.array([self.b]).T, (1, test_points.shape[0]))
            test_points_contained_in_Ab = AxT_minus_b <= PYCVXSET_ZERO
            if self.n_equalities > 0:
                AexT_minus_be = self.Ae @ test_points.T - np.tile(np.array([self.be]).T, (1, test_points.shape[0]))
                test_points_contained_in_Aebe = np.abs(AexT_minus_be) <= PYCVXSET_ZERO
                if n_test_points > 1:
                    return np.bitwise_and(
                        np.all(test_points_contained_in_Ab, axis=0), np.all(test_points_contained_in_Aebe, axis=0)
                    )
                else:
                    if self.n_halfspaces > 0:
                        return test_points_contained_in_Ab[0] and test_points_contained_in_Aebe[0]
                    else:
                        return test_points_contained_in_Aebe[0]
            else:
                return np.all(test_points_contained_in_Ab, axis=0)
        else:  # in_V_rep alone is available!
            return convex_set_contains_points(self, test_points)


def intersection(self, Q):
    r"""Intersect the polytope :math:`\mathcal{P}` with another polytope :math:`\mathcal{Q}`.

    Args:
        Q (Polytope): Polytope to be intersected with self

    Raises:
        ValueError: Q is not a polytope | Mismatch in dimensions

    Returns:
        Polytope: The intersection of `P` and `Q`

    Notes:
        This function requires :math:`\mathcal{P}` and Q to be of the same dimension and in H-Rep, and performs
        halfspace enumerations when P or Q are in V-Rep.
    """
    if not is_polytope(Q):
        if is_constrained_zonotope(Q):
            raise ValueError(
                "Intersection between a polytope and constrained zonotope is possible with pycvxset. However, to avoid "
                "confusion, such an intersection must be invoked from the constrained zonotope. Currently, the "
                "intersection method was invoked from the polytope."
            )
        else:
            raise ValueError("Intersection must be between two polytope objects")
    elif self.dim != Q.dim:
        raise ValueError(f"Mismatch in dimensions (self.dim: {self.dim:d} and Q.dim: {Q.dim:d})")
    elif self.is_empty or Q.is_empty:
        # The intersection of an (empty) set with any set is an empty set, so return an empty polytope.
        return self.__class__(dim=self.dim)
    else:
        P_cap_Q_A = np.vstack((self.A, Q.A))
        P_cap_Q_b = np.hstack((self.b, Q.b))
        P_cap_Q_Ae = np.vstack((self.Ae, Q.Ae))
        P_cap_Q_be = np.hstack((self.be, Q.be))
        return self.__class__(A=P_cap_Q_A, b=P_cap_Q_b, Ae=P_cap_Q_Ae, be=P_cap_Q_be)


def intersection_with_halfspaces(self, A, b):
    r"""Intersect the polytope with a collection of halfspaces.

    Args:
        A (array_like): Inequality coefficient vectors (H-Rep). The vectors are stacked vertically.
        b (array_like): Inequality constants (H-Rep). The constants are expected to be in a 1D numpy array.

    Raises:
        ValueError: Mismatch in dimensions | (A, b) is not a valid collection of halfspaces

    Returns:
        Polytope: The intersection of :math:`P` and :math:`\{x\in\mathbb{R}^n\ |\ Ax\leq b\}`

    Notes:
        This function requires :math:`\mathcal{P}.\text{dim}` = `A.shape[1]` and :math:`\mathcal{P}` should be in H-Rep,
        and performs halfspace enumeration if :math:`\mathcal{P}` in V-Rep.
    """
    A, b = sanitize_Ab(A, b)
    if self.dim != A.shape[1]:
        raise ValueError(f"Mismatch in dimensions (self.dim: {self.dim:d} and A.shape[1]: {A.shape[1]:d})")
    elif self.is_empty:
        # The intersection of an (empty) set with any set is an empty set, so return an empty polytope.
        return self.__class__(dim=self.dim)
    else:
        # Combine the H-representation of both polytopes:
        P_cap_Ab_A = np.vstack((self.A, A))
        P_cap_Ab_b = np.hstack((self.b, b))
        return self.__class__(A=P_cap_Ab_A, b=P_cap_Ab_b, Ae=self.Ae, be=self.be)


def intersection_with_affine_set(self, Ae, be):
    r"""Intersect the polytope with an affine set.

    Args:
        Ae (array_like): Equality coefficient vectors (H-Rep). The vectors are stacked vertically.
        be (array_like): Equality constants (H-Rep). The constants are expected to be in a 1D numpy array.

    Raises:
        ValueError: Mismatch in dimensions | (Ae, be) is not a valid collection of equality constraints.

    Returns:
        Polytope: The intersection of :math:`P` and :math:`\{x:A_e x = b_e\}`

    Notes:
        This function requires :math:`\mathcal{P}.\text{dim}` = `Ae.shape[1]` and :math:`\mathcal{P}` should be in
        H-Rep, and performs halfspace enumeration if :math:`\mathcal{P}` in V-Rep.
    """
    Ae, be, Aebe_status, solution_to_Ae_x_eq_be = sanitize_and_identify_Aebe(Ae, be)
    if Aebe_status != "no_Ae_be" and self.dim != Ae.shape[1]:
        raise ValueError(f"Mismatch in dimensions (self.dim: {self.dim:d} and Ae.shape[1]: {Ae.shape[1]:d})")
    elif Aebe_status == "no_Ae_be":
        return self.copy()
    elif (
        self.is_empty
        or Aebe_status == "infeasible"
        or (Aebe_status == "single_point" and solution_to_Ae_x_eq_be not in self)
    ):
        return self.__class__(dim=self.dim)
    elif Aebe_status == "single_point":
        return self.__class__(V=np.array([solution_to_Ae_x_eq_be]))
    else:
        # Combine the H-representation of both polytopes:
        P_cap_Ab_Ae = np.vstack((self.Ae, Ae))
        P_cap_Ab_be = np.hstack((self.be, be))
        return self.__class__(A=self.A, b=self.b, Ae=P_cap_Ab_Ae, be=P_cap_Ab_be)


def intersection_under_inverse_affine_map(self, Q, R):
    r"""Compute the intersection of constrained zonotope under an inverse affine map

    Args:
        Q (Polytope): Set to intersect with
        R (array_like): Matrix of dimension Y.dim times self.dim

    Raises:
        ValueError: When Q is not a Polytope
        ValueError: When R is not of correct dimension
        ValueError: When self is not bounded

    Returns:
        Polytope: The intersection of a polytope with another polytope under an inverse affine map. Specifically, given
        polytopes :math:`\mathcal{P}` (self) and :math:`\mathcal{Q}`, and a matrix :math:`R`, we compute the set
        :math:`\{x \in \mathcal{P}| Rx \in \mathcal{Q}\}`.

    Notes:
        This function requires both polytopes to be in H-Rep. Halfspace enumeration is performed when necessary.

        Unlike :meth:`inverse_affine_map_under_invertible_matrix`, this function does not require R to be invertible or
        square and also accommodates Q to be an unbounded polytope. However, self must be a bounded set.
    """
    if not is_polytope(Q):
        raise ValueError(f"Expected Q to be a polytope. Got {type(Q)}!")
    if not self.is_bounded:
        raise ValueError("Expected self to be bounded!")
    try:
        R = np.atleast_2d(R).astype(float)
    except (TypeError, ValueError) as err:
        raise ValueError(f"Invalid R obtained! Got R {np.array2string(np.array(R))}") from err

    try:
        if self.n_equalities > 0:
            Q_at_R = Q.__class__(A=Q.A @ R, b=Q.b, Ae=Q.Ae @ R, be=Q.be)
        else:
            Q_at_R = self.__class__(A=Q.A @ R, b=Q.b)
    except ValueError as err:
        raise ValueError(f"Could not compute Q @ R! Q {repr(Q)} and R {np.array2string(np.array(R))}") from err
    return self.intersection(Q_at_R)


def inverse_affine_map_under_invertible_matrix(self, M):
    r"""Compute the set times matrix, the inverse affine map of the set under an invertible matrix M.

    Args:
        M (array_like): A invertible array of size self.dim times self.dim

    Raises:
        ValueError: When self is empty
        TypeError: When M is not convertible into a 2D numpy array of float
        ValueError: When M is not a square matrix
        ValueError: When M is not invertible

    Returns:
        Polytope: The inverse-scaled polytope :math:`\mathcal{R} = \mathcal{P} \times M = \{x: M x\in \mathcal{P}\}`

    Notes:
        * This function accommodates :math:`\mathcal{P}` to be in H-Rep or in V-Rep. When :math:`\mathcal{P}` is in
          H-Rep, :math:`\mathcal{P} M=\{x|Mx\in\mathcal{P}\}=\{x|AMx\leq b, A_eMx = b_e\}`. On the other hand, when
          :math:`\mathcal{P}` is in V-Rep, :math:`\mathcal{P} M=ConvexHull(M^{-1}v_i)`.
        * We require M to be invertible in order to ensure that the resulting set is representable as a polytope.
    """
    if self.is_empty:
        raise ValueError("Expected polytope be non-empty!")
    else:
        try:
            M = np.atleast_2d(M).astype(float)
        except (TypeError, ValueError) as err:
            raise TypeError(f"Multiplication of Polytope with {type(M)} is not supported!") from err
        if M.shape != (self.dim, self.dim):
            raise ValueError("Expected M to be a square matrix of shape ({self.dim:d},{self.dim:d}). Got {M.shape}!")
        try:
            M_inv = np.linalg.inv(M)
        except np.linalg.LinAlgError as err:
            raise ValueError("Expected M to be invertible!") from err
        if self.in_H_rep:
            if self.n_equalities > 0:
                return self.__class__(A=self.A @ M, b=self.b, Ae=self.Ae @ M, be=self.be)
            else:
                return self.__class__(A=self.A @ M, b=self.b)
        else:
            return affine_map(self, M_inv)


def plus(self, Q):
    r"""Add a point or a set to the polytope

    Args:
        Q (array_like | Polytope): Point or set to add to the polytope.

    Raises:
        ValueError: When the point dimension does not match the polytope dimension.

    Returns:
        Polytope: Polytope which is the sum of self and Q.

    Notes:
        - Given a polytope :math:`\mathcal{P}`, and a set :math:`Q`, this function computes the Minkowski sum of Q and
          the polytope, defined as :math:`\mathcal{R}=\{x + q|x\in\mathcal{P}, q\in\mathcal{Q}\}`. On the other hand,
          when Q is a point, this function computes the polytope :math:`\mathcal{R}=\{x + Q|x\in\mathcal{P}\}`.
        - *Addition with a point*: This function allows for :math:`\mathcal{P}` to be in V-Rep or in H-Rep. For
          :math:`\mathcal{P}` in V-rep, the polytope :math:`\mathcal{R}` is the convex hull of the vertices of
          :math:`\mathcal{P}` shifted by :math:`\text{point}`. Given :math:`\{v_i\}` as the collection of vertices of
          :math:`\mathcal{P}`, :math:`\mathcal{R}=\mathrm{convexHull}(v_i + \text{point})`.  For :math:`\mathcal{P}` in
          H-rep, the polytope :math:`\mathcal{R}` is defined by all points :math:`r = \text{point} + x` with
          :math:`Ax\leq b, A_ex=b_e`.  Thus, :math:`\mathcal{R}=\{x: A x \leq b + A \text{point}, A_e x = b_e + A_e
          \text{point}\}`.
        - *Addition with a polytope*: This function requires self and Q to be in V-Rep, and performs a vertex
          enumeration when self or Q are in H-Rep.. In vertex representation, :math:`\mathcal{R}` is the convex hull of
          the pairwise sum of all combinations of points in :math:`\mathcal{P}` and :math:`\mathcal{Q}`.
    """
    if is_polytope(Q):
        if self.dim != Q.dim:
            raise ValueError(f"Mismatch in dimensions (self.dim: {self.dim:d} and Q.dim: {Q.dim:d})")
        elif Q.is_empty:
            return self.copy()
        elif self.is_empty:
            return Q.copy()
        else:  # both P.is_empty and Q.is_empty are False:
            # Vertices of the Minkowski sum:
            minkowski_sum_V = np.array([p + q for p in self.V for q in Q.V])
            return self.__class__(V=minkowski_sum_V)
    elif is_constrained_zonotope(Q):
        # Q is a constrained zonotope ====> Polytope + ConstrainedZonotope case
        return NotImplemented
    else:
        try:
            Q = np.atleast_1d(np.squeeze(Q)).astype(float)
        except (TypeError, ValueError) as err:
            raise TypeError(f"Unsupported operation Polytope + {type(Q)}!") from err
        if Q.ndim != 1:
            raise ValueError(f"Expected a single R^{self.dim:d} point, but got {np.array2string(np.array(Q)):s}")
        elif Q.size != self.dim:  # ensure point is n times 1
            raise ValueError(f"points dim. ({Q.size:d}) is different from set dim. ({self.dim:d})")
        elif self.in_V_rep:
            # V-rep: The sum is all vertices of :math:\mathcal{P} shifted by point.
            V_shifted = self.V + np.repeat(np.array([Q]), self.n_vertices, axis=0)
            return self.__class__(V=V_shifted)
        elif self.in_H_rep:
            # H-rep: Shift P.b by P.A @ point (and P.be by P.Ae @ point)
            b_shifted = self.b + np.squeeze(self.A @ Q)
            if self.n_equalities > 0:
                be_shifted = self.be + np.squeeze(self.Ae @ Q)
                return self.__class__(A=self.A, b=b_shifted, Ae=self.Ae, be=be_shifted)
            else:
                return self.__class__(A=self.A, b=b_shifted)
        else:
            # Singleton polytope at Q
            singleton_vertex = np.atleast_2d(Q)  # skip astype since we have already converted into float
            return self.__class__(V=singleton_vertex)


def minus(self, Q):
    r"""Implement - operation (Pontryagin difference when Q is a polytope, translation by -Q when Q is a point)

    Args:
        Q (array_like | Polytope | ConstrainedZonotope | Ellipsoid): Polytope to be subtracted in Pontryagin
        difference sense from self or a vector for negative translation of the polytope

    Raises:
        TypeError: When Q is neither a Polytope or a point
        ValueError: When Q is not of the same dimension as self

    Returns:
        Polytope: The polytope :math:`\mathcal{R}` that is the Pontryagin difference of `P` and `Q` or a negative
        translation of `P` by `Q`.

    Notes:
        - *Subtraction with a point*: This function accommodates :math:`\mathcal{P}` in H-Rep and V-Rep. See
          :meth:`plus` for more details.
        - *Subtraction with a polytope*: This function requires :math:`\mathcal{P}` and Q to be of the same dimension
          and :math:\mathcal{P} in H-Rep, and performs halfspace enumerations when :math:`\mathcal{P}` is in V-Rep. The
          H-rep of the polytope :math:`\mathcal{R}` is, :math:`H_{\mathcal{R}} = [\mathcal{P}.A, \mathcal{P}.b - \rho_{
          \mathcal{Q}}(\mathcal{P}.A)]` where :math:`\rho_{\mathcal{Q}}` is the support function (see
          :meth:`support`). [KG98]_
    """
    if is_polytope(Q) or is_constrained_zonotope(Q) or is_ellipsoid(Q):
        if self.dim != Q.dim:  # Throw error if both :math:\mathcal{P} and Q are not of same dimensions
            raise ValueError(f"Mismatch in dimensions (self.dim: {self.dim:d} and Q.dim: {Q.dim:d})")
        elif Q.is_empty:  # Return P
            return self.copy()
        elif self.is_empty:  # P is empty
            return self.__class__(dim=self.dim)
        elif is_polytope(Q) and self.in_H_rep and Q.in_V_rep:  # P - Q = \cap_{v \in V(Q)} (P - v)
            P_diff_Q = self - Q.V[0, :]
            for v in Q.V[1:, :]:
                P_shifted_by_vertex = self + (-v)
                P_diff_Q = P_diff_Q.intersection(P_shifted_by_vertex)
            return P_diff_Q
        else:  # P - Q = {P.A * x <= P.b - Q.support(A)}
            self.minimize_H_rep()
            if self.n_equalities > 0:
                if Q.is_full_dimensional or not np.all(
                    np.isclose(Q.support(np.vstack((self.Ae, -self.Ae)))[0], np.vstack((self.be, -self.be)))
                ):
                    return self.__class__(dim=self.dim)
            P_diff_Q_b = self.b - Q.support(self.A)[0]
            return self.__class__(A=self.A, b=P_diff_Q_b, Ae=self.Ae, be=self.be)
    else:
        try:
            Q = np.atleast_1d(Q).astype(float)
        except (TypeError, ValueError) as err:
            raise TypeError(f"Unsupported operation Polytope - {type(Q)}!") from err
        return self.plus(-Q)
