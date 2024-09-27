# Copyright (C) 2020-2024 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

# Code purpose:  Define the Ellipsoid class

import math

import cvxpy as cp
import numpy as np

from pycvxset.common import (
    _compute_project_single_point,
    convex_set_closest_point,
    convex_set_distance,
    convex_set_extreme,
    convex_set_minimum_volume_circumscribing_rectangle,
    convex_set_project,
    convex_set_projection,
    convex_set_support,
    plot,
    solve_convex_program_with_containment_constraints,
)
from pycvxset.common.constants import (
    DEFAULT_CVXPY_ARGS_LP,
    DEFAULT_CVXPY_ARGS_SDP,
    DEFAULT_CVXPY_ARGS_SOCP,
    PYCVXSET_ZERO,
)
from pycvxset.common.polytope_approximations import polytopic_inner_approximation, polytopic_outer_approximation
from pycvxset.Ellipsoid.operations_binary import (
    _compute_support_function_single_eta,
    affine_map,
    contains,
    deflate,
    inflate,
    inflate_ball,
    inverse_affine_map_under_invertible_matrix,
    plus,
)


class Ellipsoid:
    r"""Ellipsoid class.

    We can define a full-dimensional set :math:`\mathcal{P}` using **one** of the following combinations:

    #. :math:`(c, Q)` for ellipsoid in the **quadratic form**
       :math:`\mathcal{P}=\{x \in \mathbb{R}^n\ |\  (x - c)^T Q^{-1} (x - c) \leq 1\}` with a n-dimensional
       positive-definite matrix :math:`Q` and a n-dimensional vector :math:`c`, and
    #. :math:`(c, G)` for ellipsoid as an **affine transformation of a unit-ball**
       :math:`\mathcal{P} = \{c + G u \in \mathbb{R}^n\ |\  {\|u\|}_2 \leq 1\}` with a n-dimensional square
       lower-triangular :math:`G` that satisfies :math:`GG^T=Q`, and a n-dimensional vector :math:`c`.

    Args:
        c (array_like): Center of the ellipsoid c. Vector of length (self.dim,)
        Q (array_like, optional): Shape matrix of the ellipsoid Q. Symmetric, positive-definite matrix (self.dim
            times self.dim). If provided, G must not be provided.
        G (array_like, optional): Square root of the shape matrix of the ellipsoid G that satisfies :math:`GG^T=Q`.
            Lower-triangular square matrix (self.dim times self.dim) with non-zero diagonal. If provided, Q must not be
            provided.

    Raises:
        ValueError: When c or Q or G does not satisfy implicit properties
    """

    def __init__(self, c, **kwargs):
        """Constructor for Ellipsoid"""
        try:
            self._c = np.atleast_1d(np.squeeze(c)).astype(float)
        except (TypeError, ValueError) as err:
            raise ValueError("Expected c to be convertible into a numpy 1D array of float") from err
        # Check if c is indeed a 1D
        if self._c.ndim != 1:
            raise ValueError("Expected c to be convertible to a 1D vector")
        # Set Q, G, or both
        if len(kwargs) == 0:
            raise ValueError("Expected either Q or G or r to be provided.")
        elif len(kwargs) >= 2:
            raise ValueError("Expected only Q or G or r to be provided.")
        elif "r" in kwargs:
            r = kwargs.pop("r")
            if r <= 0:
                raise ValueError("Expected r to be a positive scalar")
            self._Q = (r**2) * np.eye(self.dim)
            self._G = r * np.eye(self.dim)
        elif "Q" in kwargs:
            # Check if Q is indeed a 2D square matrix of correct dimension
            Q = kwargs.pop("Q")
            Q = np.atleast_2d(Q).astype(float)
            n_rows, n_cols = Q.shape
            if n_rows != n_cols or n_rows != self.dim:
                raise ValueError(f"Expected square Q of dimension {self.dim:d}")
            # Make the matrix symmetric
            Q = (Q + Q.T) / 2
            try:
                # Check for positive semi-definiteness
                self._G = np.linalg.cholesky(Q)
                self._Q = Q
            except np.linalg.LinAlgError as err:
                raise ValueError("Expected Q to be positive-semidefinite") from err
        else:  # The last case is "G" in kwargs:
            G = kwargs.pop("G")
            G = np.atleast_2d(G).astype(float)
            n_rows, n_cols = G.shape
            if (
                n_rows != n_cols
                or n_rows != self.dim
                or np.triu(G, k=1).any()
                or np.any(np.abs(np.diag(G)) <= PYCVXSET_ZERO)
            ):
                raise ValueError(
                    f"Expected lower-triangular square G of dimension {self.dim:d} with non-zero " + "diagonal entries."
                )
            self._G = G
            self._Q = G @ G.T
        # These attributes are used by CVXPY to solve problems
        self._cvxpy_args_lp = DEFAULT_CVXPY_ARGS_LP
        self._cvxpy_args_socp = DEFAULT_CVXPY_ARGS_SOCP
        self._cvxpy_args_sdp = DEFAULT_CVXPY_ARGS_SDP

    @property
    def dim(self):
        """Dimension of the ellipsoid :math:`dim`"""
        return self._c.shape[0]

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
        r"""Lower-triangular square root :math:`G` of the shape matrix :math:`Q` that satisfies :math:`GG^T=Q`"""
        return self._G

    @property
    def is_empty(self):
        """Check if the ellipsoid is empty. Always False by construction."""
        return False

    @property
    def is_full_dimensional(self):
        """Check if the ellipsoid is full-dimensional. Always True by construction."""
        return True

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
    def get_cvxpy_containment_constraints(self, x):
        """Get CVXPY constraints for containment of x (a cvxpy.Variable) in an ellipsoid.

        Args:
            x (cvxpy.Variable): CVXPY variable to be optimized

        Returns:
            tuple: A tuple with two items:

            #. constraint_list (list): CVXPY constraints for the containment of x in the ellipsoid.
            #. xi (cvxpy.Variable): CVXPY variable representing the latent dimension variable.
        """
        xi = cp.Variable((self.dim,))
        return [x == self.G @ xi + self.c, cp.norm(xi, p=2) <= 1], xi

    solve_convex_program_with_containment_constraints = solve_convex_program_with_containment_constraints

    ##################
    # Unary operations
    ##################
    def copy(self):
        """Create a copy of the ellipsoid"""
        return self.__class__(c=self.c, Q=self.Q)

    def volume(self):
        """Compute the volume of the ellipsoid.

        Notes:
            We used the following observations:
            1. from [BV04]_ , the volume of an ellipsoid is proportional to :math:`det(G)`.
            2. Square-root of the determinant of the shape matrix coincides with the determinant of G
            3. Since G is lower-triangular, its determinant is the product of its diagonal elements.
        """
        return math.pi ** (self.dim / 2) / math.gamma(self.dim / 2 + 1) * np.prod(np.diag(self.G))

    def chebyshev_centering(self):
        """Compute the Chebyshev center and radius of the ellipsoid."""
        eigenvalues, _ = np.linalg.eig(self.Q)
        return self.c, float(np.min(eigenvalues))

    def interior_point(self):
        """Compute an interior point to the Ellipsoid"""
        return self.c

    def maximum_volume_inscribing_ellipsoid(self):
        """Compute the maximum volume inscribing ellipsoid for a given ellipsoid."""
        return self.c, self.Q, self.G

    def minimum_volume_circumscribing_ellipsoid(self):
        """Compute the minimum volume circumscribing ellipsoid for a given ellipsoid."""
        return self.c, self.Q, self.G

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

    project.__doc__ = (
        convex_set_project.__doc__
        + "\n"
        + r"""
    Notes:

        For a point :math:`y\in\mathbb{R}^{\mathcal{P}.\text{dim}}` and an ellipsoid :math:`\mathcal{P}=\{G u + c\ |\ \|
        u \|_2 \leq 1 \}` with :math:`GG^T=Q`,  this function solves a convex program with decision variables
        :math:`x,u\in\mathbb{R}^{\mathcal{P}.\text{dim}}`,

            .. math::
                \begin{align}
                    \text{minimize}    &\quad  \|x - y\|_p\\
                    \text{subject to}  &\quad  x = G u + c\\
                                       &\quad  {\| u \|}_2 \leq 1
                \end{align}
    """
    )

    def projection(self, project_away_dim):
        return convex_set_projection(self, project_away_dim=project_away_dim)

    projection.__doc__ = (
        convex_set_projection.__doc__
        + "\n"
        + r"""
    Returns:
        Ellipsoid: m-dimensional set obtained via projection.
    """
    )

    def support(self, eta):
        return convex_set_support(self, eta)

    support.__doc__ = (
        convex_set_support.__doc__
        + "\n"
        + r"""
    Notes:
        Using duality, the support function and vector of an ellipsoid  has a closed-form expressions. For a support
        direction :math:`\eta\in\mathbb{R}^{\mathcal{P}.\text{dim}}` and an ellipsoid :math:`\mathcal{P}=\{G u + c |
        \| u \|_2 \leq 1 \}` with :math:`GG^T=Q`,

        .. math ::
            \begin{align}
                \rho_{\mathcal{P}}(\eta) &= \eta^\top c + \sqrt{\eta^\top Q \eta} = \eta^\top c + \|G^T \eta\|_2\\
                \nu_{\mathcal{P}}(\eta) &= c + \frac{G G^\top \eta}{\|G^T \eta\|_2} = c + \frac{Q \eta}{\|G^T \eta\|_2}
            \end{align}
    """
    )
    _compute_project_single_point = _compute_project_single_point
    _compute_support_function_single_eta = _compute_support_function_single_eta

    ##########################
    # Ellipsoid representation
    ##########################
    def __str__(self):
        return f"Ellipsoid in R^{self.dim:d}"

    __repr__ = __str__
