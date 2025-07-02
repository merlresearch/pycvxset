# Copyright (C) 2020-2025 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

# Code purpose:  Define the Ellipsoid class
# Coverage: This file has 2 untested statements to handle unexpected errors from np.linalg.eig

import math

import cvxpy as cp
import numpy as np
import scipy as sp

from pycvxset.common import (
    _compute_project_single_point,
    convex_set_closest_point,
    convex_set_distance,
    convex_set_extreme,
    convex_set_minimum_volume_circumscribing_rectangle,
    convex_set_project,
    convex_set_projection,
    convex_set_slice,
    convex_set_slice_then_projection,
    convex_set_support,
    minimize,
    plot,
)
from pycvxset.common.constants import (
    DEFAULT_CVXPY_ARGS_LP,
    DEFAULT_CVXPY_ARGS_SDP,
    DEFAULT_CVXPY_ARGS_SOCP,
    PYCVXSET_ZERO,
)
from pycvxset.common.polytope_approximations import polytopic_inner_approximation, polytopic_outer_approximation
from pycvxset.Ellipsoid.operations_binary import (
    DOCSTRING_FOR_PROJECT,
    DOCSTRING_FOR_PROJECTION,
    DOCSTRING_FOR_SLICE,
    DOCSTRING_FOR_SLICE_THEN_PROJECTION,
    DOCSTRING_FOR_SUPPORT,
    _compute_support_function_single_eta,
    affine_map,
    contains,
    deflate,
    inflate,
    inflate_ball,
    intersection_with_affine_set,
    inverse_affine_map_under_invertible_matrix,
    plus,
)


class Ellipsoid:
    r"""Ellipsoid class.

    We can define a bounded, non-empty ellipsoid :math:`\mathcal{P}` using **one** of the following combinations:

    #. :math:`(c, Q)` for a full-dimensional ellipsoid in the **quadratic form**
       :math:`\mathcal{P}=\{x \in \mathbb{R}^n\ |\  (x - c)^T Q^{-1} (x - c) \leq 1\}` with a n-dimensional
       positive-definite matrix :math:`Q` and a n-dimensional vector :math:`c`. Here, pycvxset computes a n-dimensional
       lower-triangular, square matrix :math:`G` that satisfies :math:`GG^T=Q`.
    #. :math:`(c, G)` for a full-dimensional or a degenerate ellipsoid as an **affine transformation of a unit-ball**
       :math:`\mathcal{P} = \{x \in \mathbb{R}^n\ |\ \exists u\in\mathbb{R}^N,\ x = c + G u,\ {\|u\|}_2 \leq 1\}` with a
       n x N matrix :math:`G`. Here, pycvxset computes :math:`Q=GG^T`.

    Args:
        c (array_like): Center of the ellipsoid c. Vector of length (self.dim,)
        Q (array_like, optional): Shape matrix of the ellipsoid Q. Q must be a positive definite matrix
            (self.dim times self.dim).
        G (array_like, optional): Square root of the shape matrix of the ellipsoid G that satisfies :math:`GG^T=Q`.
            Need not be a square matrix, but must have self.dim rows.
        r (scalar, optional): Non-negative scalar that provides the radius of the self.dim-dimensional ball.

    Raises:
        ValueError: When more than one of Q, G, r was provided
        ValueError: When c or Q or G or r does not satisfy implicit properties

    Notes:
        1. Empty ellipsoids are not permitted (c is a required keyword argument).
        2. When provided G is such that :math:`Q=GG^T` is positive definite, we overwrite :math:`G` with a
           lower-triangular, square, n-dimensional matrix  for consistency. Here, :math:`G` has strictly positive
           diagonal elements, and its determinant is the product of its diagonal elements (see volume computation).
        3. We use the eigenvalues of :math:`Q` to determine the radii of the maximum volume inscribing ball (Chebyshev
           radius :math:`R^-`) and the minimum volume circumscribing ball :math:`R^+\geq R^-`.

           a. The ellipsoid represents a singleton when :math:`R^+` is negligible.
           b. The ellipsoid is full-dimensional when :math:`R^-` is non-trivial.
    """

    def __init__(self, **kwargs):
        """Constructor for Ellipsoid"""
        self._c, self._Q, self._G, self._cheby_radius, self._outer_approx_radius = None, None, None, None, None

        # Set up c
        try:
            self._c = np.atleast_1d(np.squeeze(kwargs.pop("c"))).astype(float)
            if self._c.ndim != 1:
                raise ValueError("Expected c to be convertible to a 1D vector")
        except KeyError as err:
            raise ValueError("c is a required argument!") from err
        except (TypeError, ValueError) as err:
            raise ValueError("Expected c to be convertible into a numpy 1D array of float") from err

        # Define Q, G, cheby_radius, outer_approx_radius (when possible)
        if len(kwargs) >= 2:
            # Set only one of Q, G, or r
            raise ValueError("Expected only Q or G or r to be provided.")
        elif len(kwargs) == 0:
            self._set_ellipsoid_to_singleton()
        elif "r" in kwargs:
            r = float(kwargs.pop("r"))
            if r < 0:
                raise ValueError("Expected r to be a positive scalar")
            elif r <= PYCVXSET_ZERO:
                self._set_ellipsoid_to_singleton()
            else:
                self._Q = (r**2) * np.eye(self.dim)
                self._G = r * np.eye(self.dim)
                self._outer_approx_radius, self._cheby_radius = r, r
        elif "Q" not in kwargs and "G" not in kwargs:
            raise ValueError(f"Invalid kwarg provided! Got {kwargs}. Expected either Q, G, or r!")
        else:
            # Compute Q = G @ G.T given G or Compute G = cholesky(Q) given Q
            if "G" in kwargs:
                self._G = np.atleast_2d(kwargs.pop("G")).astype(float)
                if np.isnan(self.G[0, 0]) or self.G.shape == (1, 0):
                    self._set_ellipsoid_to_singleton()
                elif self.G.shape[0] != self.dim:
                    raise ValueError(f"Expected G to have {self.dim:d} rows.")
                else:
                    self._Q = self.G @ self.G.T
            else:  # "Q" in kwargs
                # Check if Q is indeed a 2D square matrix of correct dimension
                self._Q = np.atleast_2d(kwargs.pop("Q")).astype(float)
                n_rows, n_cols = self.Q.shape
                if n_rows != n_cols or n_rows != self.dim:
                    raise ValueError(f"Expected square Q of dimension {self.dim:d}")
                if not np.isclose(2 * self.Q, self.Q + self.Q.T).all():
                    raise ValueError("Expected Q to be symmetric!")
                try:
                    self._G = np.linalg.cholesky(self.Q)
                except np.linalg.LinAlgError as err:
                    raise ValueError(
                        "Expected Q to be positive definite! Use (c, G) to define degenerate ellipsoids."
                    ) from err

            # Get self._outer_approx_radius, self._cheby_radius from Q
            self._set_radii_from_Q()

            # For consistent behavior across (c, G), (c, Q), or (c, r)
            if self.is_singleton:  # sqrt of all eigenvalues of Q are below PYCVXSET_ZERO
                self._set_ellipsoid_to_singleton()
            elif self.is_full_dimensional and not np.isclose(np.triu(self.G), 0).all():
                self._G = np.linalg.cholesky(self.Q)  # Overwrite provided G with Cholesky decomposition for consistency

        # These attributes are used by CVXPY to solve problems
        self._cvxpy_args_lp = DEFAULT_CVXPY_ARGS_LP
        self._cvxpy_args_socp = DEFAULT_CVXPY_ARGS_SOCP
        self._cvxpy_args_sdp = DEFAULT_CVXPY_ARGS_SDP

    def _set_ellipsoid_to_singleton(self):
        """Protected method to set G, Q, _outer_approx_radius, _cheby_radius when ellipsoid is a singleton"""
        self._G, self._Q = np.zeros((self.dim, 0)), np.zeros((self.dim, self.dim))
        self._outer_approx_radius, self._cheby_radius = 0.0, 0.0

    def _set_radii_from_Q(self):
        """Protected method to set _outer_approx_radius, _cheby_radius using the eigenvalue_Q vector, and compute the
        elementwise square root of the vector"""
        try:
            eigenvalue_Q = np.linalg.eigvals(self.Q)
        except np.linalg.LinAlgError as eigenvalue_error:
            raise ValueError("Eigenvalue computation for Q failed!") from eigenvalue_error
        eigenvalue_Q[eigenvalue_Q <= PYCVXSET_ZERO**2] = 0.0
        sqrt_eigenvalue_Q = np.sqrt(eigenvalue_Q)
        self._outer_approx_radius, self._cheby_radius = np.max(sqrt_eigenvalue_Q), np.min(sqrt_eigenvalue_Q)

    @property
    def dim(self):
        """Dimension of the ellipsoid :math:`dim`"""
        return self._c.shape[0]

    @property
    def latent_dim(self):
        """Latent dimension of the ellipsoid :math:`dim`"""
        return self._G.shape[1]

    @property
    def c(self):
        """Center of the ellipsoid :math:`c`"""
        return self._c

    @property
    def Q(self):
        """Shape matrix of the ellipsoid :math:`Q`"""
        return self._Q

    @property
    def G(self):
        r"""Affine transformation matrix :math:`G` that satisfies :math:`GG^T=Q`"""
        return self._G

    @property
    def is_empty(self):
        """Check if the ellipsoid is empty. Always False by construction."""
        return False

    @property
    def is_full_dimensional(self):
        """Check if the ellipsoid is full-dimensional, i.e., sqrt of all eigenvalues of Q are above PYCVXSET_ZERO"""
        return self._cheby_radius > PYCVXSET_ZERO

    @property
    def is_singleton(self):
        """Check if the ellipsoid is a singleton, i.e., sqrt of all eigenvalues of Q are below PYCVXSET_ZERO"""
        return self._outer_approx_radius <= PYCVXSET_ZERO

    @property
    def cvxpy_args_lp(self):
        """CVXPY arguments in use when solving a linear program

        Returns:
            dict: CVXPY arguments in use when solving a linear program. Defaults to dictionary in
            `pycvxset.common.DEFAULT_CVXPY_ARGS_LP`.

        """
        return self._cvxpy_args_lp

    @cvxpy_args_lp.setter
    def cvxpy_args_lp(self, value):
        """Update CVXPY arguments in use when solving a linear program

        Args:
            value: Dictionary with new CVXPY arguments in use when solving a linear program.
        """
        self._cvxpy_args_lp = value

    @property
    def cvxpy_args_socp(self):
        """CVXPY arguments in use when solving a second-order cone program

        Returns:
            dict: CVXPY arguments in use when solving a second-order cone program. Defaults to dictionary in
            `pycvxset.common.DEFAULT_CVXPY_ARGS_SOCP`.
        """
        return self._cvxpy_args_socp

    @cvxpy_args_socp.setter
    def cvxpy_args_socp(self, value):
        """Update CVXPY arguments in use when solving a second-order cone program

        Args:
            value: Dictionary with new CVXPY arguments in use when solving a second-order cone program.
        """
        self._cvxpy_args_socp = value

    @property
    def cvxpy_args_sdp(self):
        """CVXPY arguments in use when solving a semi-definite program

        Returns:
            dict: CVXPY arguments in use when solving a semi-definite program. Defaults to dictionary in
            `pycvxset.common.DEFAULT_CVXPY_ARGS_SDP`.
        """
        return self._cvxpy_args_sdp

    @cvxpy_args_sdp.setter
    def cvxpy_args_sdp(self, value):
        """Update CVXPY arguments in use when solving a semi-definite program

        Args:
            value: Dictionary with new CVXPY arguments in use when solving a semi-definite program.
        """
        self._cvxpy_args_sdp = value

    ##################################
    # Polytope and plotting operations
    ##################################
    deflate = classmethod(deflate)
    inflate = classmethod(inflate)
    inflate_ball = classmethod(inflate_ball)
    polytopic_inner_approximation = polytopic_inner_approximation
    polytopic_outer_approximation = polytopic_outer_approximation
    plot = plot

    ###########
    # Auxiliary
    ###########
    def containment_constraints(self, x, flatten_order="F"):
        """Get CVXPY constraints for containment of x (a cvxpy.Variable) in an ellipsoid.

        Args:
            x (cvxpy.Variable): CVXPY variable to be optimized
            flatten_order (char): Order to use for flatten (choose between "F", "C"). Defaults to "F", which
                implements column-major flatten. In 2D, column-major flatten results in stacking rows horizontally to
                achieve a single horizontal row.

        Returns:
            tuple: A tuple with two items:

            #. constraint_list (list): CVXPY constraints for the containment of x in the ellipsoid.
            #. xi (cvxpy.Variable): CVXPY variable representing the latent dimension variable with length G.shape[1]
        """
        x = x.flatten(order=flatten_order)
        if self.is_singleton:
            return [x == self.c], None
        else:
            xi = cp.Variable((self.G.shape[1],))
            return [x == self.G @ xi + self.c, cp.norm(xi, p=2) <= 1], xi

    minimize = minimize

    ##################
    # Unary operations
    ##################
    def copy(self):
        """Create a copy of the ellipsoid. Copy (c, G) to preserve degenerate ellipsoids."""
        return self.__class__(c=self.c, G=self.G)

    def quadratic_form_as_a_symmetric_matrix(self):
        """Define a (self.dim + 1)-dimensional symmetric matrix M where self = {x | [x, 1] @ M @ [x, 1] <= 0}. Here,
        when Q is not positive definite, we use pseudo-inverse of Q."""
        try:
            F = np.linalg.inv(self.Q)
        except np.linalg.LinAlgError:
            F = np.linalg.pinv(self.Q)
        g = -F.T @ self.c
        h = self.c @ F @ self.c - 1
        return np.hstack((np.vstack((F, g)), np.array([[*g, h]]).T))

    def affine_hull(self):
        """Compute the left null space of self.G to identify the affine hull. Affine hull is the entire
        self.dim-dimensional space when self is full-dimensional."""
        if self.is_full_dimensional:
            return None, None
        else:
            null_G = sp.linalg.null_space(self.G.T, rcond=PYCVXSET_ZERO).T
            Ae, be = null_G, null_G @ self.c
            return Ae, be

    def volume(self):
        """Compute the volume of the ellipsoid.

        Notes:
            Volume of the ellipsoid is zero if it is not full-dimensional. For full-dimensional ellipsoid, we used the
            following observations:

                1. from [BV04]_ , the volume of an ellipsoid is proportional to :math:`det(G)`.
                2. Square-root of the determinant of the shape matrix coincides with the determinant of G
                3. Since G is lower-triangular, its determinant is the product of its diagonal elements.
        """
        if self.is_full_dimensional:
            return math.pi ** (self.dim / 2) / math.gamma(self.dim / 2 + 1) * np.prod(np.diag(self.G))
        else:
            return 0.0

    def chebyshev_centering(self):
        """Compute the Chebyshev center and radius of the ellipsoid."""
        return self.c, self._cheby_radius

    def interior_point(self):
        """Compute an interior point to the Ellipsoid"""
        return self.c

    def maximum_volume_inscribing_ellipsoid(self):
        """Compute the maximum volume inscribing ellipsoid for a given ellipsoid."""
        return self.c, self.Q, self.G

    def minimum_volume_circumscribing_ellipsoid(self):
        """Compute the minimum volume circumscribing ellipsoid for a given ellipsoid."""
        return self.c, self.Q, self.G

    def minimum_volume_circumscribing_ball(self):
        """Compute the radius of the ball that circumscribes the ellipsoid and has the minimum volume."""
        return self.c, self._outer_approx_radius

    minimum_volume_circumscribing_rectangle = convex_set_minimum_volume_circumscribing_rectangle

    ######################
    # Comparison operators
    ######################
    contains = contains
    __contains__ = contains

    def __le__(self, Q):
        """Overload <= operator for containment. self <= Q is equivalent to Q.contains(self)"""
        return Q.contains(self)

    def __ge__(self, Q):
        """Overload >= operator for containment. self >= Q is equivalent to P.contains(Q)"""
        return self.contains(Q)

    def __eq__(self, Q):
        """Overload == operator with equality check. P == Q is equivalent to Q.contains(P) and P.contains(Q)"""
        return self <= Q and self >= Q

    __lt__ = __le__
    __gt__ = __ge__

    ###################
    # Binary operations
    ###################
    __array_ufunc__ = None  # Allows for numpy matrix times Ellipsoid

    plus = plus
    __add__ = plus
    __radd__ = plus

    def __sub__(self, y):
        r"""Implement - operator for ellipsoid,

        Args:
            y (array_like): Point to subtract from ellipsoid

        Raises:
            TypeError: When y is not convertible into a 1D numpy array of float

        Returns:
            Ellipsoid: Sum of given ellipsoid :math:`\mathcal{P}` and the negation of the point.
        """
        try:
            y = np.atleast_1d(np.squeeze(y)).astype(float)
            return self.plus(-y)
        except (TypeError, ValueError) as err:
            raise TypeError(f"Unsupported operation: Ellipsoid - {type(y)}") from err

    # Ellipsoid times Matrix
    inverse_affine_map_under_invertible_matrix = inverse_affine_map_under_invertible_matrix
    __matmul__ = inverse_affine_map_under_invertible_matrix

    def __mul__(self, x):
        """Do not allow Ellipsoid * anything"""
        return NotImplemented

    def __neg__(self):
        return self.affine_map(-1)

    # Scalar/Matrix times Ellipsoid (called when left operand does not support multiplication)
    affine_map = affine_map

    def __rmatmul__(self, M):
        """Overload @ operator for affine map (matrix times Ellipsoid)."""
        return self.affine_map(M)

    def __rmul__(self, m):
        """Overload * operator for multiplication."""
        try:
            m = np.squeeze(m).astype(float)
        except (TypeError, ValueError) as err:
            raise TypeError(f"Unsupported operation: {type(m)} * Ellipsoid!") from err
        return self.affine_map(m)

    closest_point = convex_set_closest_point
    distance = convex_set_distance
    extreme = convex_set_extreme
    _compute_project_single_point = _compute_project_single_point

    def project(self, x, p=2):
        return convex_set_project(self, x, p=p)

    project.__doc__ = convex_set_project.__doc__ + DOCSTRING_FOR_PROJECT

    def projection(self, project_away_dims):
        return convex_set_projection(self, project_away_dims=project_away_dims)

    projection.__doc__ = convex_set_projection.__doc__ + DOCSTRING_FOR_PROJECTION

    def slice(self, dims, constants):
        return convex_set_slice(self, dims=dims, constants=constants)

    slice.__doc__ = convex_set_slice.__doc__ + DOCSTRING_FOR_SLICE

    def slice_then_projection(self, dims, constants):
        return convex_set_slice_then_projection(self, dims=dims, constants=constants)

    slice_then_projection.__doc__ = convex_set_slice_then_projection.__doc__ + DOCSTRING_FOR_SLICE_THEN_PROJECTION

    def support(self, eta):
        return convex_set_support(self, eta)

    support.__doc__ = convex_set_support.__doc__ + DOCSTRING_FOR_SUPPORT

    _compute_project_single_point = _compute_project_single_point
    _compute_support_function_single_eta = _compute_support_function_single_eta

    intersection_with_affine_set = intersection_with_affine_set

    ##########################
    # Ellipsoid representation
    ##########################
    def __str__(self):
        return f"Ellipsoid in R^{self.dim:d}"

    def __repr__(self):
        base_string = str(self)
        if self.is_full_dimensional:
            return base_string + ", and the ellipsoid is full-dimensional"
        elif self.is_singleton:
            return base_string + ", and the ellipsoid is not full-dimensional (it is a singleton)"
        else:
            return base_string + ", and the ellipsoid is not full-dimensional (it is a degenerate ellipsoid)"
