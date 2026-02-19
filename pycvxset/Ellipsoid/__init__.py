# Copyright (C) 2020-2026 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

# Code purpose:  Define the Ellipsoid class
# Coverage: This file has 2 missing statements + 12 excluded statements + 0 partial statements.

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, Literal, Optional, Sequence, cast, overload

if TYPE_CHECKING:
    import cvxpy
    from pycvxset.Polytope import Polytope

import numpy as np
import scipy as sp

from pycvxset.common import (
    _compute_project_multiple_points,
    convex_set_closest_point,
    convex_set_distance,
    convex_set_extreme,
    convex_set_minimum_volume_circumscribing_rectangle,
    convex_set_project,
    convex_set_projection,
    convex_set_slice,
    convex_set_slice_then_projection,
    convex_set_support,
    is_ellipsoid,
    is_polytope,
    minimize,
    plot_polytopic_approximation,
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
    #. :math:`(c, r)` for a ball of radius :math:`r \geq 0`
       :math:`\mathcal{P} = \{x \in \mathbb{R}^n\ |\ {\|x - c\|}_2 \leq r\}`.
    #. :math:`(c)` for a singleton ellipsoid :math:`\mathcal{P} = \{c\}`.

    Args:
        c (Sequence[float] | numpy.ndarray): Center of the ellipsoid c. Vector of length (self.dim,)
        Q (Sequence[Sequence[float]] | numpy.ndarray, optional): Shape matrix of the ellipsoid Q. Q must be a positive
            definite matrix (self.dim times self.dim).
            (self.dim times self.dim).
        G (Sequence[Sequence[float]] | numpy.ndarray, optional): Square root of the shape matrix of the ellipsoid G that
            satisfies :math:`GG^T=Q`.  Need not be a square matrix, but must have self.dim rows. If a singleton must be
            specified, G must have zero columns.
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

    if TYPE_CHECKING:

        @overload
        def __init__(self, *, c: Sequence[float] | np.ndarray, Q: Sequence[Sequence[float]] | np.ndarray) -> None: ...

        @overload
        def __init__(self, *, c: Sequence[float] | np.ndarray, G: Sequence[Sequence[float]] | np.ndarray) -> None: ...

        @overload
        def __init__(self, *, c: Sequence[float] | np.ndarray, r: float) -> None: ...

        @overload
        def __init__(self, *, c: Sequence[float] | np.ndarray) -> None: ...

    def __init__(self, **kwargs: Any) -> None:
        """Constructor for Ellipsoid"""
        self._type_of_set: str = "Ellipsoid"
        self._c: np.ndarray = np.empty((0,), dtype=float)

        # Set up c
        try:
            self._c = np.atleast_1d(np.squeeze(kwargs.pop("c"))).astype(float)
            if self._c.ndim != 1:
                raise ValueError("Expected c to be convertible to a 1D vector")
        except KeyError as err:
            raise ValueError("c is a required argument!") from err
        except (TypeError, ValueError) as err:
            raise ValueError("Expected c to be convertible into a numpy 1D array of float") from err
        if np.isnan(self._c).any() or np.isinf(self._c).any():
            raise ValueError("Expected c to be finite; got NaNs or infs.")

        self._Q: np.ndarray = np.zeros((self.dim, self.dim), dtype=float)
        self._G: np.ndarray = np.zeros((self.dim, 0), dtype=float)
        self._cheby_radius: float = 0.0
        self._outer_approx_radius: float = 0.0

        # Define Q, G, cheby_radius, outer_approx_radius (when possible)
        if len(kwargs) >= 2:
            # Set only one of Q, G, or r
            raise ValueError("Expected only Q or G or r to be provided.")
        elif len(kwargs) == 0:
            self._set_ellipsoid_to_singleton()
        elif "r" in kwargs:
            r = float(kwargs.pop("r"))
            if not np.isfinite(r):
                raise ValueError("Expected r to be finite.")
            elif r < 0:
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
                if np.isnan(self.G).any() or np.isinf(self.G).any():
                    raise ValueError("Expected G to be finite; got NaNs or infs.")
                if self.G.shape[1] == 0:
                    self._set_ellipsoid_to_singleton()
                elif self.G.shape[0] != self.dim:
                    raise ValueError(f"Expected G to have {self.dim:d} rows.")
                else:
                    self._Q = self.G @ self.G.T
            else:  # "Q" in kwargs
                # Check if Q is indeed a 2D square matrix of correct dimension
                self._Q = np.atleast_2d(kwargs.pop("Q")).astype(float)
                if np.isnan(self.Q).any() or np.isinf(self.Q).any():
                    raise ValueError("Expected Q to be finite; got NaNs or infs.")
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

    def _set_ellipsoid_to_singleton(self: Ellipsoid) -> None:
        """Protected method to set G, Q, _outer_approx_radius, _cheby_radius when ellipsoid is a singleton"""
        self._G, self._Q = np.zeros((self.dim, 0)), np.zeros((self.dim, self.dim))
        self._outer_approx_radius, self._cheby_radius = 0.0, 0.0

    def _set_radii_from_Q(self: Ellipsoid) -> None:
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
    def dim(self: Ellipsoid) -> int:
        """Dimension of the ellipsoid :math:`dim`.

        Returns:
            int: Dimension of the ellipsoid.
        """
        return self._c.shape[0]

    @property
    def latent_dim(self: Ellipsoid) -> int:
        """Latent dimension of the ellipsoid :math:`dim`.

        Returns:
            int: Latent dimension of the ellipsoid.
        """
        return self._G.shape[1]

    @property
    def c(self: Ellipsoid) -> np.ndarray:
        """Center of the ellipsoid :math:`c`.

        Returns:
            numpy.ndarray: Center vector.
        """
        return self._c

    @property
    def Q(self: Ellipsoid) -> np.ndarray:
        """Shape matrix of the ellipsoid :math:`Q`.

        Returns:
            numpy.ndarray: Shape matrix.
        """
        return self._Q

    @property
    def G(self: Ellipsoid) -> np.ndarray:
        r"""Affine transformation matrix :math:`G` that satisfies :math:`GG^T=Q`.

        Returns:
            numpy.ndarray: Generator matrix.
        """
        return self._G

    @property
    def is_empty(self: Ellipsoid) -> bool:
        """Check if the ellipsoid is empty. Always False by construction.

        Returns:
            bool: Always False for ellipsoids since we require non-empty ellipsoids.
        """
        return False

    @property
    def is_full_dimensional(self: Ellipsoid) -> bool:
        """Check if the ellipsoid is full-dimensional.

        Returns:
            bool: True if full-dimensional.
        """
        return self._cheby_radius > PYCVXSET_ZERO

    @property
    def is_singleton(self: Ellipsoid) -> bool:
        """Check if the ellipsoid is a singleton.

        Returns:
            bool: True if the ellipsoid is a singleton.
        """
        return self._outer_approx_radius <= PYCVXSET_ZERO

    @property
    def is_bounded(self: Ellipsoid) -> bool:
        """Check if the ellipsoid is bounded. Always True by construction."""
        return True

    @property
    def cvxpy_args_lp(self: Ellipsoid) -> dict[str, Any]:
        """CVXPY arguments in use when solving a linear program

        Returns:
            dict: CVXPY arguments in use when solving a linear program. Defaults to dictionary in
            `pycvxset.common.DEFAULT_CVXPY_ARGS_LP`.

        """
        return self._cvxpy_args_lp

    @cvxpy_args_lp.setter
    def cvxpy_args_lp(self: Ellipsoid, value: dict[str, Any]) -> None:
        """Update CVXPY arguments in use when solving a linear program

        Args:
            value: Dictionary with new CVXPY arguments in use when solving a linear program.
        """
        self._cvxpy_args_lp = value

    @property
    def cvxpy_args_socp(self: Ellipsoid) -> dict[str, Any]:
        """CVXPY arguments in use when solving a second-order cone program

        Returns:
            dict: CVXPY arguments in use when solving a second-order cone program. Defaults to dictionary in
            `pycvxset.common.DEFAULT_CVXPY_ARGS_SOCP`.
        """
        return self._cvxpy_args_socp

    @cvxpy_args_socp.setter
    def cvxpy_args_socp(self: Ellipsoid, value: dict[str, Any]) -> None:
        """Update CVXPY arguments in use when solving a second-order cone program

        Args:
            value: Dictionary with new CVXPY arguments in use when solving a second-order cone program.
        """
        self._cvxpy_args_socp = value

    @property
    def cvxpy_args_sdp(self: Ellipsoid) -> dict[str, Any]:
        """CVXPY arguments in use when solving a semi-definite program

        Returns:
            dict: CVXPY arguments in use when solving a semi-definite program. Defaults to dictionary in
            `pycvxset.common.DEFAULT_CVXPY_ARGS_SDP`.
        """
        return self._cvxpy_args_sdp

    @cvxpy_args_sdp.setter
    def cvxpy_args_sdp(self: Ellipsoid, value: dict[str, Any]) -> None:
        """Update CVXPY arguments in use when solving a semi-definite program

        Args:
            value: Dictionary with new CVXPY arguments in use when solving a semi-definite program.
        """
        self._cvxpy_args_sdp = value

    @property
    def type_of_set(self: Ellipsoid) -> str:
        """Return the type of set

        Returns:
            str: Type of the set
        """
        return self._type_of_set

    ##################################
    # Polytope and plotting operations
    ##################################
    deflate = classmethod(deflate)
    inflate = classmethod(inflate)
    inflate_ball = classmethod(inflate_ball)
    polytopic_inner_approximation = polytopic_inner_approximation
    polytopic_outer_approximation = polytopic_outer_approximation
    plot = plot_polytopic_approximation

    ###########
    # Auxiliary
    ###########
    def containment_constraints(
        self, x: cvxpy.Variable, flatten_order: Literal["F", "C"] = "F"
    ) -> tuple[list[cvxpy.Constraint], Optional[cvxpy.Variable]]:
        """Get CVXPY constraints for containment of x (a cvxpy.Variable) in an ellipsoid.

        Args:
            x (cvxpy.Variable): CVXPY variable to be optimized
            flatten_order (Literal["F", "C"]): Order to use for flatten (choose between "F", "C"). Defaults to "F",
                which implements column-major flatten. In 2D, column-major flatten results in stacking rows horizontally
                to achieve a single horizontal row.

        Returns:
            tuple: A tuple with two items:

            #. constraint_list (list[cvxpy.Constraint]): CVXPY constraints for the containment of x in the ellipsoid.
            #. xi (cvxpy.Variable | None): CVXPY variable representing the latent dimension variable with length
               G.shape[1]. It is None when the ellipsoid is a singleton.
        """
        from cvxpy import Variable, norm, reshape  # pyright: ignore[reportUnknownVariableType]

        x_reshaped = reshape(x, (x.size,), order=flatten_order)
        if self.is_singleton:
            return [x_reshaped == self.c], None
        else:
            xi = Variable((self.G.shape[1],))
            return [x_reshaped == self.G @ xi + self.c, norm(xi, p=2) <= 1], xi

    minimize = minimize

    ##################
    # Unary operations
    ##################
    def copy(self: Ellipsoid) -> Ellipsoid:
        """Create a copy of the ellipsoid. Copy (c, G) to preserve degenerate ellipsoids."""
        return self.__class__(c=self.c, G=self.G)

    def quadratic_form_as_a_symmetric_matrix(self: Ellipsoid) -> np.ndarray:
        """Define a (self.dim + 1)-dimensional symmetric matrix M where self = {x | [x, 1] @ M @ [x, 1] <= 0}. Here,
        when Q is not positive definite, we use pseudo-inverse of Q.

        Returns:
            numpy.ndarray: (self.dim + 1) x (self.dim + 1) symmetric matrix defining the quadratic form.
        """
        try:
            F = np.linalg.inv(self.Q)
        except np.linalg.LinAlgError:
            F = np.linalg.pinv(self.Q)
        g = -F.T @ self.c
        h = self.c @ F @ self.c - 1
        return np.hstack((np.vstack((F, g)), np.array([[*g, h]]).T))

    def affine_hull(self: Ellipsoid) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Compute the left null space of self.G to identify the affine hull. Affine hull is the entire
        self.dim-dimensional space when self is full-dimensional.

        Returns:
            tuple: (Ae, be) defining the affine set {x | Ae x = be}, or (None, None) when full-dimensional.
        """
        if self.is_full_dimensional:
            return None, None
        else:
            null_G = sp.linalg.null_space(self.G.T, rcond=PYCVXSET_ZERO).T
            Ae, be = null_G, null_G @ self.c
            return Ae, be

    def volume(self: Ellipsoid) -> float:
        """Compute the volume of the ellipsoid.

        Returns:
            float: Volume of the ellipsoid.

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

    def chebyshev_centering(self: Ellipsoid) -> tuple[np.ndarray, float]:
        """Compute the Chebyshev center and radius of the ellipsoid.

        Returns:
            tuple: (center, radius) of the maximum volume inscribed ball.
        """
        return self.c, self._cheby_radius

    def interior_point(self: Ellipsoid) -> np.ndarray:
        """Compute an interior point to the Ellipsoid

        Returns:
            np.ndarray: center of the ellipsoid, which is an interior point.
        """
        return self.c

    def maximum_volume_inscribing_ellipsoid(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute the parameters of the maximum volume inscribing ellipsoid for a given ellipsoid.

        Returns:
            tuple: (center, Q, G) describing the ellipsoid.
        """
        return self.c, self.Q, self.G

    def minimum_volume_circumscribing_ellipsoid(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute the parameters of the minimum volume circumscribing ellipsoid for a given ellipsoid.

        Returns:
            tuple: (center, Q, G) describing the ellipsoid.
        """
        return self.c, self.Q, self.G

    def minimum_volume_circumscribing_ball(self: Ellipsoid) -> tuple[np.ndarray, float]:
        """Compute the parameters of a minimum volume circumscribing ball.

        Returns:
            tuple: (center, radius) for the minimum-volume circumscribing ball.
        """
        return self.c, self._outer_approx_radius

    minimum_volume_circumscribing_rectangle = convex_set_minimum_volume_circumscribing_rectangle

    ######################
    # Comparison operators
    ######################
    contains = contains
    __contains__ = contains

    def __le__(self: Ellipsoid, Q: "Polytope | Ellipsoid") -> bool:
        """Overload <= operator for containment. self <= Q is equivalent to Q.contains(self)."""
        if is_polytope(Q) or is_ellipsoid(Q):
            return Q.contains(self)
        else:
            return NotImplemented

    def __ge__(self: Ellipsoid, Q: "Sequence[float] | np.ndarray | Polytope | Ellipsoid") -> bool:
        """Overload >= operator for containment. self >= Q is equivalent to P.contains(Q)."""
        if is_polytope(Q) or is_ellipsoid(Q):
            return self.contains(Q)
        else:
            try:
                Q_arr = np.atleast_2d(Q).astype(float)
            except (TypeError, ValueError):
                return NotImplemented
            if Q_arr.size == self.dim and Q_arr.shape[0] == 1:
                # Q is a point ====> Ellipsoid >= Point case
                return self.contains(Q_arr)
            else:
                return NotImplemented

    def __eq__(self: Ellipsoid, Q: object) -> bool:
        """Overload == operator with equality check. P == Q is equivalent to Q.contains(P) and P.contains(Q)"""
        if is_ellipsoid(Q):
            return self <= cast(Ellipsoid, Q) and self >= cast(Ellipsoid, Q)
        else:
            return NotImplemented

    __lt__ = __le__
    __gt__ = __ge__

    ###################
    # Binary operations
    ###################
    __array_ufunc__ = None  # Allows for numpy matrix times Ellipsoid

    plus = plus
    __add__ = plus
    __radd__ = plus

    def __sub__(self, y: Sequence[float] | np.ndarray) -> Ellipsoid:
        r"""Implement - operator for ellipsoid,

        Args:
            y (Sequence[float] | np.ndarray): Point to subtract from ellipsoid

        Raises:
            TypeError: When y is not convertible into a 1D numpy array of float

        Returns:
            Ellipsoid: Sum of given ellipsoid :math:`\mathcal{P}` and the negation of the point.
        """
        try:
            y_arr = np.atleast_1d(np.squeeze(y)).astype(float)
            return self.plus(-y_arr)
        except (TypeError, ValueError) as err:
            raise TypeError(f"Unsupported operation: Ellipsoid - {type(y)}") from err

    # Ellipsoid times Matrix
    inverse_affine_map_under_invertible_matrix = inverse_affine_map_under_invertible_matrix
    __matmul__ = inverse_affine_map_under_invertible_matrix

    def __mul__(self, x: Any) -> Any:
        """Do not allow Ellipsoid * anything.

        Args:
            x: Right operand.

        Returns:
            NotImplemented: Always not supported.
        """
        return NotImplemented

    def __neg__(self: Ellipsoid) -> Ellipsoid:
        """Negate the ellipsoid.

        Returns:
            Ellipsoid: Negated ellipsoid.
        """
        return affine_map(self, -1)

    # Scalar/Matrix times Ellipsoid (called when left operand does not support multiplication)
    affine_map = affine_map

    def __rmatmul__(self: Ellipsoid, M: Any) -> Ellipsoid:
        """Overload @ operator for affine map (matrix times Ellipsoid)."""
        return affine_map(self, M)

    def __rmul__(self: Ellipsoid, m: int | float) -> Ellipsoid:
        """Overload * operator for multiplication with scalar."""
        try:
            m = np.squeeze(m).astype(float)
        except (TypeError, ValueError) as err:
            raise TypeError(f"Unsupported operation: {type(m)} * Ellipsoid!") from err
        return affine_map(self, m)

    closest_point = convex_set_closest_point
    distance = convex_set_distance
    extreme = convex_set_extreme

    def project(
        self: Ellipsoid, x: Sequence[float] | Sequence[Sequence[float]] | np.ndarray, p: int | str = 2
    ) -> tuple[np.ndarray, np.ndarray]:
        return convex_set_project(self, x, p=p)  # pyright: ignore[reportArgumentType]

    project.__doc__ = (convex_set_project.__doc__ or "") + DOCSTRING_FOR_PROJECT

    def projection(self: Ellipsoid, project_away_dims: int | Sequence[int]) -> Ellipsoid:
        return convex_set_projection(self, project_away_dims=project_away_dims)

    projection.__doc__ = (convex_set_projection.__doc__ or "") + DOCSTRING_FOR_PROJECTION

    def slice(
        self: Ellipsoid, dims: int | Sequence[int] | np.ndarray, constants: float | Sequence[float] | np.ndarray
    ) -> Ellipsoid:
        return convex_set_slice(self, dims=dims, constants=constants)

    slice.__doc__ = (convex_set_slice.__doc__ or "") + DOCSTRING_FOR_SLICE

    def slice_then_projection(self: Ellipsoid, dims: int | Sequence[int], constants: int | Sequence[int]) -> Ellipsoid:
        return convex_set_slice_then_projection(self, dims=dims, constants=constants)

    slice_then_projection.__doc__ = (
        convex_set_slice_then_projection.__doc__ or ""
    ) + DOCSTRING_FOR_SLICE_THEN_PROJECTION

    def support(
        self: Ellipsoid, eta: np.ndarray | Sequence[float] | Sequence[Sequence[float]]
    ) -> tuple[np.ndarray, np.ndarray]:
        # Ellipsoid is non-empty by construction, so we do not need to check for empty set case.
        eta_arr: np.ndarray = np.atleast_2d(eta).astype(float)
        if eta_arr.ndim > 2:
            raise ValueError("Expected eta to be a 1D/2D numpy array")
        elif eta_arr.shape[1] != self.dim:
            raise ValueError(
                f"eta dim. ({eta_arr.shape[1]:d}), no. of columns, is different from set dimension ({self.dim:d})"
            )
        else:  # eta.ndim == 2:
            # sqrt(eta^T Q eta) = ||G^T eta|| = ||eta G|| per row
            norm_eta_arr_G = np.linalg.norm(eta_arr @ self.G, ord=2, axis=1)  # (m,)
            eta_arr_Q = eta_arr @ self.Q  # (m, dim)
            # avoid divide-by-zero (np.where evaluates both branches)
            support_vector = np.empty_like(eta_arr_Q)
            support_vector[:] = self.c[None, :]
            # eta normal to the affine hull of the ellipsoid > non-trivial support vectors
            nonzero = norm_eta_arr_G > PYCVXSET_ZERO
            if nonzero.any():
                scaled_eta_arr_Q_by_norm_eta_arr_G = np.divide(
                    eta_arr_Q[nonzero],
                    norm_eta_arr_G[nonzero, None],
                    out=np.zeros_like(eta_arr_Q[nonzero]),
                    where=norm_eta_arr_G[nonzero, None] > PYCVXSET_ZERO,
                )
                support_vector[nonzero] = self.c[None, :] + scaled_eta_arr_Q_by_norm_eta_arr_G
            support_value = np.sum(eta_arr * support_vector, axis=1)  # (m,) | Both are arranged as rows
            return support_value, support_vector

    support.__doc__ = (convex_set_support.__doc__ or "") + (
        "\n"
        + r"""
        Notes:
            Using duality, the support function and vector of an ellipsoid  has a closed-form expressions. For a support
            direction :math:`\eta\in\mathbb{R}^{\mathcal{P}.\text{dim}}` and an ellipsoid :math:`\mathcal{P}=\{G u + c |
            \| u \|_2 \leq 1 \}` with :math:`GG^T=Q`,

            .. math ::
                \rho_{\mathcal{P}}(\eta) &= \eta^\top c + \sqrt{\eta^\top Q \eta} = \eta^\top c + \|G^T \eta\|_2\\
                \nu_{\mathcal{P}}(\eta) &= c + \frac{G G^\top \eta}{\|G^T \eta\|_2} = c + \frac{Q \eta}{\|G^T \eta\|_2}

            For degenerate (not full-dimensional) ellipsoids and :math:`\eta` not in the low-dimensional affine hull
            containing the ellipsoid,

            .. math ::
                \rho_{\mathcal{P}}(\eta) &= \eta^\top c \\
                \nu_{\mathcal{P}}(\eta) &= c
        """
    )

    _compute_project_multiple_points = _compute_project_multiple_points

    intersection_with_affine_set = intersection_with_affine_set

    ##########################
    # Ellipsoid representation
    ##########################
    def __str__(self: Ellipsoid) -> str:
        return f"Ellipsoid in R^{self.dim:d}"

    def __repr__(self: Ellipsoid) -> str:
        base_string = str(self)
        if self.is_full_dimensional:
            return base_string + ", and the ellipsoid is full-dimensional"
        elif self.is_singleton:
            return base_string + ", and the ellipsoid is not full-dimensional (it is a singleton)"
        else:
            return base_string + ", and the ellipsoid is not full-dimensional (it is a degenerate ellipsoid)"
