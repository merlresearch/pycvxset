# Copyright (C) 2020-2025 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

# Code purpose:  Define the ConstrainedZonotope class

import cvxpy as cp
import numpy as np

from pycvxset.common import (
    _compute_project_single_point,
    _compute_support_function_single_eta,
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
    sanitize_Aebe,
    sanitize_and_identify_Aebe,
    sanitize_Gc,
)
from pycvxset.common.constants import DEFAULT_CVXPY_ARGS_LP, DEFAULT_CVXPY_ARGS_SOCP, PYCVXSET_ZERO
from pycvxset.common.polytope_approximations import polytopic_inner_approximation, polytopic_outer_approximation
from pycvxset.ConstrainedZonotope.operations_binary import (
    DOCSTRING_FOR_PROJECT,
    DOCSTRING_FOR_PROJECTION,
    DOCSTRING_FOR_SLICE,
    DOCSTRING_FOR_SLICE_THEN_PROJECTION,
    DOCSTRING_FOR_SUPPORT,
    affine_map,
    approximate_pontryagin_difference,
    cartesian_product,
    contains,
    intersection,
    intersection_under_inverse_affine_map,
    intersection_with_affine_set,
    intersection_with_halfspaces,
    inverse_affine_map_under_invertible_matrix,
    minus,
    plus,
)
from pycvxset.ConstrainedZonotope.operations_unary import (
    chebyshev_centering,
    interior_point,
    maximum_volume_inscribing_ellipsoid,
    remove_redundancies,
)


class ConstrainedZonotope:
    r"""Constrained zonotope class

    Constrained zonotope defines a polytope in the working dimension :math:`\mathbb{R}^n` as an affine transformation of
    a polytope defined in latent space :math:`B_\infty(A_e, b_e)\subset \mathbb{R}^{N_C}`. Here, :math:`B_\infty(A_e,
    b_e)` is defined as the intersection of a unit :math:`\ell_\infty`-norm ball and a collection of :math:`M_C` linear
    constraints :math:`\{\xi\in\mathbb{R}^{N_C}|A_e \xi = b_e\}.`

    Formally, a **constrained zonotope** is defined as follows,

    .. math::
            \mathcal{C} = \{G \xi + c\ |\ \xi \in B_\infty(A_e, b_e)\} \subset \mathbb{R}^n,

    where

    .. math::
            B_\infty(A_e, b_e)= \{\xi\ |\ \| \xi \|_\infty \leq 1, A_e \xi = b_e\} \subset \mathbb{R}^{N_C},

    with :math:`G\in\mathbb{R}^{n\times N_C}`, :math:`c\in\mathbb{R}^{n}`, :math:`A_e\in\mathbb{R}^{M_C\times N_C}`, and
    :math:`b\in\mathbb{R}^{M_C}`.

    A constrained zonotope provide an alternative and equivalent representation of any
    convex and compact polytope.  Furthermore, a constrained zonotope admits closed-form expressions for several set
    manipulations that can often be accomplished without invoking any optimization solvers. See [SDGR16]_ [RK22]_
    [VWD24]_ for more details.

    A **zonotope** is a special class of constrained zonotopes, and are defined as

    .. math::
            \mathcal{Z} = \{G \xi + c\ |\ \|\xi\|_\infty \leq 1\} \subset \mathbb{R}^n.

    In other words, a zonotope is a constrained zonotope with no equality constraints in the latent dimension space. In
    :class:`ConstrainedZonotope`, we model zonotopes by having (Ae,be) be empty (n\_equalities is zero).

    Constrained zonotope object construction admits **one** of the following combinations (as keyword arguments):

    #. dim for an **empty** constrained zonotope of dimension dim,
    #. (G, c, Ae, be) for a **constrained zonotope**,
    #. (G, c) for a **zonotope**,
    #. (lb, ub) for a **zonotope** equivalent to an **axis-aligned cuboid** with appropriate bounds :math:`\{x\ |\
       lb\leq x \leq ub\}`, and
    #. (c, h) for a **zonotope** equivalent to an **axis-aligned cuboid** centered at c with specified
       scalar/vector half-sides :math:`h`, :math:`\{x\ |\ \forall i\in\{1,2,...,n\}, |x_i - c_i| \leq h_i\}`.
    #. (c=p, G=None) for a **zonotope** equivalent to a **single point** p,
    #. P for a **constrained zonotope** equivalent to the :class:`pycvxset.Polytope.Polytope` object P,

    Args:
        dim (int, optional): Dimension of the empty constrained zonotope. If NOTHING is provided, dim=0 is assumed.
        c (array_like, optional): Affine transformation translation vector. Must be 1D array, and the constrained
            zonotope dimension is determined by number of elements in c. When c is provided, either (G) or (G, Ae, be)
            or (h) must be provided additionally. When h is provided, c is the centroid of the resulting zonotope.
        G (array_like):  Affine transformation matrix. The vectors are stacked vertically with matching number of
            rows as c. When G is provided, (c, Ae, be) OR (c) must also be provided. To define a constrained zonotope
            with a single point, set c to the point AND G to None (do not set (Ae, be) or set them to (None, None)).
        Ae (array_like):  Equality coefficient vectors. The vectors are stacked vertically with matching number of
            columns as G. When Ae is provided, (G, c, be) must also be provided.
        be (array_like):  Equality coefficient constants. The constants are expected to be in a 1D numpy array. When be
            is provided, (G, c, Ae) must also be provided.
        lb (array_like, optional): Lower bounds of the axis-aligned cuboid. Must be 1D array, and the constrained
            zonotope dimension is determined by number of elements in lb. When lb is provided, ub must also be provided.
        ub (array_like, optional): Upper bounds of the axis-aligned cuboid. Must be 1D array of length as same as lb.
            When ub is provided, lb must also be provided.
        h (array_like, optional): Half-side length of the axis-aligned cuboid. Can be a scalar or a vector of length as
            same as c. When h is provided, c must also be provided.
        polytope (Polytope, optional): Polytope to use to construct constrained zonotope.

    Raises:
        ValueError: (G, c) is not compatible.
        ValueError: (G, c, Ae, be) is not compatible.
        ValueError: (lb, ub) is not valid
        ValueError: (c, h) is not valid
        ValueError: Provided polytope is not bounded.
        UserWarning: When a row with all zeros in Ae and be.
    """

    def __init__(self, **kwargs):
        """Constructor for ConstrainedZonotope class"""
        self._G, self._c, self._Ae, self._be = None, None, None, None
        self._is_full_dimensional, self._is_empty = None, None
        # These attributes are used by CVXPY to solve problems
        self._cvxpy_args_lp = DEFAULT_CVXPY_ARGS_LP
        self._cvxpy_args_socp = DEFAULT_CVXPY_ARGS_SOCP

        # Check how the constructor was called.
        empty_constrained_zonotope_passed = ("dim" in kwargs) or not kwargs
        lb_and_ub_passed = all(kw in kwargs for kw in ("lb", "ub"))
        c_and_h_passed = all(kw in kwargs for kw in ("c", "h"))
        G_and_c_passed = all(k in kwargs for k in ("G", "c"))
        Ae_and_be_passed = all(k in kwargs for k in ("Ae", "be"))
        polytope_passed = "polytope" in kwargs

        # Set _G, _c, _Ae, _be, _is_empty, _is_full_dimensional
        if empty_constrained_zonotope_passed:
            if len(kwargs) > 1:
                raise ValueError("Cannot set dimension dim with other arguments")
            else:
                dim = kwargs.get("dim", 0)
                self._G, self._c, self._Ae, self._be = self._get_Gc_Aebe_for_empty_constrained_zonotope(dim, 0)
                self._is_full_dimensional, self._is_empty = (dim == 0), True
        elif lb_and_ub_passed or c_and_h_passed:
            if lb_and_ub_passed:
                if len(kwargs) != 2:
                    raise ValueError("Cannot set bounds (lb, ub) with other arguments")
                lb, ub = kwargs.get("lb"), kwargs.get("ub")
            else:  # We have c_and_h_passed is True
                if len(kwargs) != 2:
                    raise ValueError("Cannot set up constrained zonotope from (c, h) with other arguments")
                try:
                    c = np.atleast_1d(np.squeeze(kwargs.get("c"))).astype(float)
                    h = np.squeeze(kwargs.get("h")).astype(float)
                except (TypeError, ValueError) as err:
                    raise ValueError(
                        "Expected c and h to be convertible into a numpy 1D array and scalar/1D array of "
                        f"float respectively! Got c: {np.array2string(np.array(kwargs.get('c')))} and "
                        f"h: {np.array2string(np.array(kwargs.get('h')))}"
                    ) from err
                if c.ndim >= 2:
                    raise ValueError(
                        f"Expected c to be a 1-dimensional array-like object! Got c: {np.array2string(np.array(c))}."
                    )
                if h.ndim >= 2:
                    raise ValueError(
                        "Expected h to be a 0-dimensional or 1-dimensional array-like object "
                        f"Got {np.array2string(np.array(h))}."
                    )
                elif h.ndim == 1:
                    if h.shape != c.shape:
                        raise ValueError(
                            "Expected c and 1-dimensional h to match in dimensions. Got {c.shape} and {h.shape}"
                        )
                    lb = c - h
                    ub = c + h
                else:
                    lb = c - h * np.ones_like(c)
                    ub = c + h * np.ones_like(c)
            self._G, self._c, self._Ae, self._be, lb, ub = self._get_Gc_Aebe_from_bounds(lb, ub)
            self._is_full_dimensional = self.dim == 1 or (np.abs(ub - lb) > PYCVXSET_ZERO).all()
            self._is_empty = self.c is None
        elif G_and_c_passed:
            # Either it is a zonotope (G, c) or a constrained zonotope (G, c, Ae, be)
            if len(kwargs) != 2 and (len(kwargs) != 4 and not Ae_and_be_passed):
                raise ValueError(
                    "Cannot set zonotope (G, c) or constrained zonotope (G, c, Ae, be) with other arguments"
                )
            self._G, self._c = sanitize_Gc(kwargs.get("G"), kwargs.get("c"))
            if Ae_and_be_passed:
                sanitized_Ae, sanitized_be = sanitize_Aebe(kwargs.get("Ae"), kwargs.get("be"))
                if self.G.size == 0 and (sanitized_Ae is not None or sanitized_be is not None):
                    raise ValueError("When G is None and (Ae, be) was passed, then (Ae, be) must be (None, None)!")
                else:
                    if sanitized_Ae is None:
                        self._Ae, self._be = np.empty((0, self.dim)), np.empty((0,))
                    else:
                        self._Ae, self._be = sanitized_Ae, sanitized_be
                        if self.Ae.shape[1] != self.latent_dim:  # Check if (Ae, be) and (A, b) can go together?
                            raise ValueError(
                                f"Expected Ae to have {self.latent_dim:d} number of columns. Got Ae.shape: "
                                f"{self.Ae.shape}!"
                            )
                    # ConstrainedZonotope emptiness and full-dimensionality needs to be confirmed
                    self._is_full_dimensional, self._is_empty = None, None
            else:
                # Set only (Ae, be) to empty | Full-dimensionality depends on rank of G
                _, _, self._Ae, self._be = self._get_Gc_Aebe_for_empty_constrained_zonotope(self.dim, self.latent_dim)
                if self.G.size == 0 and self.c is not None:
                    self._is_full_dimensional, self._is_empty = (self.dim == 1), False
                else:
                    self._is_full_dimensional, self._is_empty = None, self.c is None
        elif polytope_passed:
            if len(kwargs) != 1:
                raise ValueError("Cannot construct constrained zonotope from polytope if other arguments are provided")
            polytope = kwargs.get("polytope")
            self._is_full_dimensional, self._is_empty = polytope.is_full_dimensional, polytope.is_empty
            if not polytope.is_bounded:
                raise ValueError("Expected a convex and compact polytope!")
            elif polytope.is_empty:
                self._G, self._c, self._Ae, self._be = self._get_Gc_Aebe_for_empty_constrained_zonotope(polytope.dim, 0)
            else:
                if polytope.in_H_rep:
                    # Compute zonotope Z_0 ={G \xi + c| ||\xi||_\infty \leq 1, \xi\in R^n_g} so that polytope \subseteq
                    # Z_0.
                    lb, ub = polytope.minimum_volume_circumscribing_rectangle()
                    Z_0 = self.__class__(lb=lb, ub=ub)

                    # sigma satisfies sigma <= H z <= k for all z \in polytope where H is polytope.A (using Scott's
                    # notation in the paper [SDGR16]_)
                    H = polytope.A
                    sigma, k = -polytope.support(-H)[0], polytope.b

                    # Implement (21) from Scott's paper (with additional equality constraints when needed)
                    latent_dim = Z_0.dim + polytope.n_halfspaces
                    G = np.hstack((Z_0.G, np.zeros((Z_0.dim, latent_dim - Z_0.dim))))
                    c = Z_0.c
                    Ae = np.hstack((H @ Z_0.G, np.diag((sigma - k) / 2)))
                    be = (sigma + k) / 2 - (H @ Z_0.c)

                    if polytope.n_equalities > 0:
                        # Define CZ and unpack relevant members
                        CZ = self.__class__(G=G, c=c, Ae=Ae, be=be).intersection_with_affine_set(
                            Ae=polytope.Ae, be=polytope.be
                        )
                        self._G, self._c, self._Ae, self._be = CZ.G, CZ.c, CZ.Ae, CZ.be
                    else:
                        self._G, self._c, self._Ae, self._be = G, c, Ae, be
                else:
                    if polytope.n_vertices == 1:
                        self._G, _, self._Ae, self._be = self._get_Gc_Aebe_for_empty_constrained_zonotope(
                            polytope.dim, 0
                        )
                        self._c = np.squeeze(polytope.V)
                    else:
                        # Define a ConstrainedZonotope object corresponding to the polytope.n_vertices-dimension simplex
                        CZ_simplex = self.__class__(
                            lb=np.zeros((polytope.n_vertices, 1)), ub=np.ones((polytope.n_vertices, 1))
                        ).intersection_with_affine_set(Ae=np.ones((1, polytope.n_vertices)), be=1)
                        # CZ of interest is the affine transformation of the simplex
                        CZ = polytope.V.T @ CZ_simplex
                        # Unpack relevant members
                        self._G, self._c, self._Ae, self._be = CZ.G, CZ.c, CZ.Ae, CZ.be
        else:
            raise ValueError(
                "Got invalid arguments while defining a constrained zonotope. Please specify either (G, c) or "
                "(G, c, Ae, be) or (lb, ub) or (c, h) or polytope or dim or NOTHING."
            )

    @staticmethod
    def _get_Gc_Aebe_for_empty_constrained_zonotope(dim, latent_dim):
        return np.empty((dim, latent_dim)), None, np.empty((0, latent_dim)), np.empty((0,))

    def _get_Gc_Aebe_from_bounds(self, lb, ub):
        r"""Define a zonotope from bounds (lb, ub), i.e., a zonotope that is equivalent to the polytope defined from the
        bounds (lb, ub).

        Args:
            lb (array_like): Lower bound of the constrained zonotope.
            ub (array_like): Upper bound of the constrained zonotope.

        Raises:
            ValueError: Mismatch in lb, ub shape
            ValueError: lb, ub is not convertible into 1D numpy float arrays

        Notes:
            When :math:`lb_i > ub_i` for any :math:`i`, then the zonotope is empty. Otherwise, we uses the following
            simple manipulations to define a zonotope from the bounds lb, ub:

            .. math ::
                newobj  &= {x\ |\ lb \leq x \leq ub}\\
                        &= {x\ |\ - (ub - lb)/2 \leq x - (ub + lb)/2 \leq + (ub - lb)/2}\\
                        &= {x\ |\ - d \leq x - c \leq d}\\
                        &= {x\ |\ -1 \leq diag(1./d)(x - c) \leq 1}\\
                        &= {diag(d)z + c\ |\ -1 \leq z \leq 1}

            Embedded zonotopes (some dimension is held constant) also have their latent dimension equal to set
            dimension.
        """
        try:
            lb = np.atleast_1d(np.squeeze(lb)).astype(float)
            ub = np.atleast_1d(np.squeeze(ub)).astype(float)
        except (TypeError, ValueError) as err:
            raise ValueError("Expected lb, ub to convertible into 1D float numpy arrays") from err
        if lb.shape != ub.shape or lb.ndim != 1:
            raise ValueError("Expected lb, ub to 1D numpy arrays of same shape")
        elif np.any(ub < lb):
            return *self._get_Gc_Aebe_for_empty_constrained_zonotope(lb.shape[0], 0), lb, ub
        elif np.allclose(lb, ub):
            G, _, Ae, be = self._get_Gc_Aebe_for_empty_constrained_zonotope(lb.shape[0], 0)
            c = lb
            return G, c, Ae, be, lb, ub
        else:
            dim, latent_dim = lb.shape[0], lb.shape[0]
            _, _, Ae, be = self._get_Gc_Aebe_for_empty_constrained_zonotope(dim, latent_dim)
            d = (ub - lb) / 2
            G = np.diag(d)
            c = (lb + ub) / 2
            return G, c, Ae, be, lb, ub

    @property
    def dim(self):
        """Dimension of the constrained zonotope.

        Returns:
            int: Dimension of the constrained zonotope.

        Notes:
            We determine dimension from G, since c is set to None in case of empty (constrained) zonotope.
        """
        return self.G.shape[0]

    @property
    def c(self):
        """Affine transformation vector c for the constrained zonotope.

        Returns:
            numpy.ndarray: Affine transformation vector c.
        """
        return self._c

    @property
    def G(self):
        """Affine transformation matrix G for the constrained zonotope.

        Returns:
            numpy.ndarray: Affine transformation matrix G.
        """
        return self._G

    @property
    def latent_dim(self):
        """Latent dimension of the constrained zonotope.

        Returns:
            int: Latent dimension of the constrained zonotope.
        """
        return self.G.shape[1]

    @property
    def Ae(self):
        """Equality coefficient vectors Ae for the constrained zonotope.

        Returns:
            numpy.ndarray: Equality coefficient vectors Ae for the constrained zonotope. Ae is np.empty((0,
            self.latent_dim)) for a zonotope.
        """
        return self._Ae

    @property
    def be(self):
        """Equality constants be for the constrained zonotope.

        Returns:
            numpy.ndarray: Equality constants be for the constrained zonotope. be is np.empty((0,)) for a zonotope.
        """
        return self._be

    @property
    def He(self):
        r"""Equality constraints `He=[Ae, be]` for the constrained zonotope.

        Returns:
            numpy.ndarray: H-Rep in [Ae, be]. He is np.empty((0, self.latent_dim + 1)) for a zonotope.
        """
        return np.hstack((self.Ae, np.array([self.be]).T))

    @property
    def n_equalities(self):
        """Number of equality constraints used when defining the constrained zonotope.

        Returns:
            int: Number of equality constraints used when defining the constrained zonotope
        """
        return self.Ae.shape[0]

    @property
    def is_bounded(self):
        """Check if the constrained zonotope is bounded (which is always True)"""
        return True

    @property
    def is_empty(self):
        """Check if the constrained zonotope is empty

        Raises:
            NotImplementedError: Unable to solve the feasibility problem using CVXPY

        Returns:
            bool: When True, the polytope is empty
        """
        if self._is_empty is None:
            x = cp.Variable((self.dim,))
            _, feasibility_value, _ = self.minimize(
                x,
                objective_to_minimize=cp.Constant(0),
                cvxpy_args=self.cvxpy_args_lp,
                task_str="emptiness check for the constrained zonotope",
            )
            self._is_empty = feasibility_value == np.inf
        return self._is_empty

    @property
    def is_full_dimensional(self):
        r"""
        Check if the affine dimension of the constrained zonotope is the same as the constrained zonotope dimension

        Returns:
            bool: True when the affine hull containing the constrained zonotope has the dimension `self.dim`

        Notes:
            An empty polytope is full dimensional if dim=0, otherwise it is not full-dimensional. See Sec. 2.1.3 of
            [BV04] for discussion on affine dimension.

            A non-empty zonotope is full-dimensional if and only if G has full row rank.

            A non-empty constrained zonotope is full-dimensional if and only if [G; A] has full row rank.
        """
        if self._is_full_dimensional is None:
            if self.is_empty:
                self._is_full_dimensional = self.dim == 0
            elif self.is_zonotope:
                self._is_full_dimensional = np.linalg.matrix_rank(self.G) == self.dim
            else:
                # Affine transformation of a ball in latent_dim remains full-dimensional
                stacked_G_A = np.vstack((self.G, self.Ae))
                self._is_full_dimensional = np.linalg.matrix_rank(stacked_G_A) == stacked_G_A.shape[0]
        return self._is_full_dimensional

    @property
    def is_singleton(self):
        """Check if the constrained zonotope is a singleton"""
        if self.is_zonotope:
            return (self.G is None or self.G.size == 0) and (self.c is not None)
        else:
            _, _, Aebe_status, _ = sanitize_and_identify_Aebe(self.Ae, self.be)
            return Aebe_status == "single_point"

    @property
    def is_zonotope(self):
        """Check if the constrained zonotope is a zonotope"""
        return self.n_equalities == 0

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

    ##################################
    # Plotting and polytope operations
    ##################################
    plot = plot
    polytopic_inner_approximation = polytopic_inner_approximation
    polytopic_outer_approximation = polytopic_outer_approximation

    ###########
    # Auxiliary
    ###########
    def containment_constraints(self, x, flatten_order="F"):
        """Get CVXPY constraints for containment of x (a cvxpy.Variable) in a constrained zonotope.

        Args:
            x (cvxpy.Variable): CVXPY variable to be optimized
            flatten_order (char): Order to use for flatten (choose between "F", "C"). Defaults to "F", which
                implements column-major flatten. In 2D, column-major flatten results in stacking rows horizontally to
                achieve a single horizontal row.

        Raises:
            ValueError: When constrained zonotope is empty

        Returns:
            tuple: A tuple with two items:

            #. constraint_list (list): CVXPY constraints for the containment of x in the constrained zonotope.
            #. xi (cvxpy.Variable | None): CVXPY variable representing the latent dimension variable. It is None,
               when the constrained zonotope is a single point.
        """
        x = x.flatten(order=flatten_order)
        if self.c is None:
            raise ValueError("Containment constraints can not be generated for an empty constrained zonotope!")
        elif self.is_singleton:
            return [x == self.c], None
        else:
            xi = cp.Variable((self.latent_dim,))
            if self.is_zonotope:
                return [x == self.G @ xi + self.c, cp.norm(xi, p="inf") <= 1], xi
            else:
                return [x == self.G @ xi + self.c, cp.norm(xi, p="inf") <= 1, self.Ae @ xi == self.be], xi

    minimize = minimize

    def __pow__(self, power):
        r"""Compute the Cartesian product with itself.

        Args:
            power (int): Number of times the polytope is multiplied with itself

        Returns:
            Polytope: The polytope :math:`\mathcal{R}` corresponding to P`^N`.
        """
        concatenated_G = np.kron(np.eye(power), self.G)
        concatenated_c = np.tile(self.c, (power,))
        concatenated_Ae = np.kron(np.eye(power), self.Ae)
        concatenated_be = np.tile(self.be, (power,))
        return self.__class__(G=concatenated_G, c=concatenated_c, Ae=concatenated_Ae, be=concatenated_be)

    ##################
    # Unary operations
    ##################
    def copy(self):
        """Get a copy of the constrained zonotope"""
        return self.__class__(G=self.G, c=self.c, Ae=self.Ae, be=self.be)

    chebyshev_centering = chebyshev_centering
    interior_point = interior_point
    maximum_volume_inscribing_ellipsoid = maximum_volume_inscribing_ellipsoid
    minimum_volume_circumscribing_rectangle = convex_set_minimum_volume_circumscribing_rectangle
    remove_redundancies = remove_redundancies

    ######################
    # Comparison operators
    ######################
    contains = contains
    __contains__ = contains

    def __le__(self, Q):
        """Overload <= operator for containment. self <= Q is equivalent to Q.contains(self)."""
        return Q.contains(self)

    def __ge__(self, Q):
        """Overload >= operator for containment. self >= Q is equivalent to P.contains(Q)."""
        return self.contains(Q)

    def __eq__(self, Q):
        """Overload == operator with equality check. P == Q is equivalent to Q.contains(P) and P.contains(Q)"""
        return self <= Q and self >= Q

    __lt__ = __le__
    __gt__ = __ge__

    ####################
    # Binary operations
    ####################
    plus = plus
    __add__ = plus
    __radd__ = plus
    minus = minus
    __sub__ = minus

    def __rsub__(self, Q):
        raise TypeError(f"Unsupported operation: {type(Q)} - ConstrainedZonotope!")

    __array_ufunc__ = None  # Allows for numpy matrix times Polytope
    affine_map = affine_map
    inverse_affine_map_under_invertible_matrix = inverse_affine_map_under_invertible_matrix

    # ConstrainedZonotope times Matrix
    __matmul__ = inverse_affine_map_under_invertible_matrix

    def __mul__(self, x):
        """Do not allow ConstrainedZonotope * anything"""
        return NotImplemented

    def __neg__(self):
        return affine_map(self, -1)

    # Scalar/Matrix times ConstrainedZonotope (called when left operand does not support multiplication)
    def __rmatmul__(self, M):
        """Overload @ operator for affine map (matrix times ConstrainedZonotope)."""
        return affine_map(self, M)

    def __rmul__(self, m):
        """Overload * operator for multiplication."""
        try:
            m = np.squeeze(m).astype(float)
        except (TypeError, ValueError) as err:
            raise TypeError(f"Unsupported operation: {type(m)} * ConstrainedZonotope!") from err
        return affine_map(self, m)

    approximate_pontryagin_difference = approximate_pontryagin_difference
    cartesian_product = cartesian_product
    closest_point = convex_set_closest_point
    distance = convex_set_distance
    extreme = convex_set_extreme
    intersection = intersection
    intersection_with_halfspaces = intersection_with_halfspaces
    intersection_with_affine_set = intersection_with_affine_set
    intersection_under_inverse_affine_map = intersection_under_inverse_affine_map

    def project(self, x, p=2):
        return convex_set_project(self, x, p=p)

    project.__doc__ = convex_set_project.__doc__ + DOCSTRING_FOR_PROJECT

    def projection(self, project_away_dims):
        return convex_set_projection(self, project_away_dims=project_away_dims)

    projection.__doc__ = convex_set_projection.__doc__ + DOCSTRING_FOR_PROJECTION

    def slice(self, dims, constants):
        return convex_set_slice(self, dims, constants)

    slice.__doc__ = convex_set_slice.__doc__ + DOCSTRING_FOR_SLICE

    def slice_then_projection(self, dims, constants):
        return convex_set_slice_then_projection(self, dims=dims, constants=constants)

    slice_then_projection.__doc__ = convex_set_slice_then_projection.__doc__ + DOCSTRING_FOR_SLICE_THEN_PROJECTION

    def support(self, eta):
        return convex_set_support(self, eta)

    support.__doc__ = convex_set_support.__doc__ + DOCSTRING_FOR_SUPPORT

    _compute_support_function_single_eta = _compute_support_function_single_eta
    _compute_project_single_point = _compute_project_single_point

    #####################################
    # Constrained zonotope representation
    #####################################
    def __str__(self):
        if self.is_empty:
            short_str = f"(empty) in R^{self.dim:d}"
        else:
            short_str = f"in R^{self.dim:d}"
        return f"Constrained Zonotope {short_str:s}"

    def __repr__(self):
        long_str = [str(self)]
        if self.is_empty:
            pass
        elif self.is_zonotope and self.G.size == 0:
            long_str += ["\n\tthat is a zonotope representing a single point"]
        elif self.is_zonotope:
            long_str += [f"\n\tthat is a zonotope with latent dimension {self.latent_dim:d}"]
        elif self.n_equalities == 1:
            long_str += [f"\n\twith latent dimension {self.latent_dim:d} and 1 equality constraint"]
        else:
            long_str += [
                f"\n\twith latent dimension {self.latent_dim:d} and {self.n_equalities:d} equality constraints"
            ]
        return "".join(long_str)
