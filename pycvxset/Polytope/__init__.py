# Copyright (C) 2020-2026 Mitsubishi Electric Research Laboratories (MERL)
# Copyright (c) 2019 Tor Aksel N. Heirung
#
# SPDX-License-Identifier: AGPL-3.0-or-later
# SPDX-License-Identifier: MIT

# Code purpose:  Define the Polytope class
# Coverage: This file has 2 missing statements + 46 excluded statements + 0 partial branches.

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any, Literal, Optional, Sequence, cast, overload

from pycvxset import ConstrainedZonotope

if TYPE_CHECKING:
    import cvxpy
    from pycvxset.Ellipsoid import Ellipsoid

import numpy as np

from pycvxset.common import (
    _compute_project_multiple_points,
    _compute_support_function_multiple_eta,
    convex_set_closest_point,
    convex_set_distance,
    convex_set_extreme,
    convex_set_project,
    convex_set_projection,
    convex_set_slice,
    convex_set_slice_then_projection,
    convex_set_support,
    is_constrained_zonotope,
    is_ellipsoid,
    is_polytope,
    minimize,
    sanitize_Ab,
    sanitize_and_identify_Aebe,
)
from pycvxset.common.constants import (
    DEFAULT_CVXPY_ARGS_LP,
    DEFAULT_CVXPY_ARGS_SDP,
    DEFAULT_CVXPY_ARGS_SOCP,
    PYCVXSET_ZERO,
)
from pycvxset.Polytope.operations_binary import (
    DOCSTRING_FOR_PROJECT,
    DOCSTRING_FOR_PROJECTION,
    DOCSTRING_FOR_SLICE,
    DOCSTRING_FOR_SLICE_THEN_PROJECTION,
    DOCSTRING_FOR_SUPPORT,
    affine_map,
    contains,
    intersection,
    intersection_under_inverse_affine_map,
    intersection_with_affine_set,
    intersection_with_halfspaces,
    inverse_affine_map_under_invertible_matrix,
    minus,
    plus,
)
from pycvxset.Polytope.operations_unary import (
    chebyshev_centering,
    decompose_as_affine_transform_of_polytope_without_equalities,
    deflate_rectangle,
    interior_point,
    maximum_volume_inscribing_ellipsoid,
    minimum_volume_circumscribing_ellipsoid,
    minimum_volume_circumscribing_rectangle,
    normalize,
    volume,
)
from pycvxset.Polytope.plotting_scripts import plot, plot2d, plot3d
from pycvxset.Polytope.vertex_halfspace_enumeration import (
    determine_H_rep,
    determine_V_rep,
    minimize_H_rep,
    minimize_V_rep,
    valid_rows_with_not_all_zeros_in_A_and_no_inf_in_b,
)


class Polytope:
    r"""Polytope class.

    Polytope object construction admits **one** of the following combinations (as keyword arguments):

    #. (A, b) for a polytope in **halfspace representation (H-rep)**  :math:`\{x\ |\ Ax \leq b\}`,
    #. (A, b, Ae, be) for a polytope in **halfspace representation (H-rep) with equality constraints** :math:`\{x\ |\
       Ax \leq b, A_e x = b_e\}`,
    #. V for a polytope in **vertex representation (V-rep)** --- :math:`\text{ConvexHull}(v_i)` where :math:`v_i` are
       rows of matrix V,
    #. (lb, ub) for an **axis-aligned cuboid** with appropriate bounds :math:`\{x\ |\ lb\leq x \leq ub\}`, and
    #. (c, h) for an **axis-aligned cuboid** centered at c with specified scalar/vector half-sides :math:`h`,
       :math:`\{x\ |\ \forall i\in\{1,\cdots,n\}, |x_i - c_i| \leq h_i\}.`
    #. dim for an **empty** Polytope of dimension dim (no argument generates a zero-dimensional **empty** Polytope),

    Args:
        dim (int, optional): Dimension of the empty polytope. If NOTHING is provided, dim=0 is assumed.
        V (Sequence[Sequence[float]] | np.ndarray, optional): List of vertices of the polytope (V-Rep). The list must be
            2-dimensional with vertices arranged row-wise and the polytope dimension determined by the column count.
        A (Sequence[Sequence[float]] | np.ndarray, optional): Inequality coefficient vectors (H-Rep). The vectors are
            stacked vertically with the polytope dimension determined by the column count. When A is provided, b must
            also be provided.
        b (Sequence[float] | np.ndarray, optional): Inequality constants (H-Rep). The constants are expected to be in a
            1D numpy array.  When b is provided, A must also be provided.
        Ae (Sequence[Sequence[float]] | np.ndarray): Equality coefficient vectors (H-Rep). The vectors are stacked
            vertically with matching number of columns as A. When Ae is provided, A, b, and be must also be provided.
        be (Sequence[float] | np.ndarray): Equality coefficient constants (H-Rep). The constants are expected to be in a
            1D numpy array.  When be is provided, A, b, and Ae must also be provided.
        lb (Sequence[float] | np.ndarray, optional): Lower bounds of the axis-aligned cuboid. Must be 1D array, and the
            polytope dimension is determined by number of elements in lb. When lb is provided, ub must also be provided.
        ub (Sequence[float] | np.ndarray, optional): Upper bounds of the axis-aligned cuboid. Must be 1D array of length
            as same as lb.  When ub is provided, lb must also be provided.
        c (Sequence[float] | np.ndarray, optional): Center of the axis-aligned cuboid. Must be 1D array, and the
            polytope dimension is determined by number of elements in c. When c is provided, h must also be provided.
        h (float | Sequence[float] | np.ndarray, optional): Half-side length of the axis-aligned cuboid. Can be a scalar
            or a vector of length as same as c. When h is provided, c must also be provided.

    Raises:
        ValueError: When arguments provided is not one of [(A, b), (A, b, Ae, be), (lb, ub), (c, h), V, dim,
            NOTHING]
        ValueError: Errors raised by issues with (A, b) and (Ae, be) --- mismatch in dimensions, not convertible to
            appropriately-dimensioned numpy arrays, incompatible systems (A, b) and (Ae, be) etc.
        ValueError: Errors raised by issues with V --- not convertible to a 2-D numpy array, etc.
        ValueError: Errors raised by issues with lb, ub --- mismatch in dimensions, not convertible to
            1D numpy arrays, etc.
        ValueError: Errors raised by issues with c, h --- mismatch in dimensions, not convertible to
            1D numpy arrays, etc.
        ValueError: Polytope is not bounded in any direction. We use a sufficient condition for speed, and therefore the
            detection may not be exhaustive.
        ValueError: Polytope is not bounded in some directions. We use a sufficient condition for speed, and therefore
            the detection may not be exhaustive.
        UserWarning: If some rows are removed from (A, b) due to all zeros in A or np.inf in b | (Ae, be) when all
            zeros in Ae and be.
    """

    if TYPE_CHECKING:

        @overload
        def __init__(self) -> None: ...

        @overload
        def __init__(self, *, dim: int) -> None: ...

        @overload
        def __init__(
            self,
            *,
            V: Sequence[Sequence[float]] | np.ndarray,
        ) -> None: ...

        @overload
        def __init__(
            self,
            *,
            A: Sequence[Sequence[float]] | np.ndarray,
            b: Sequence[float] | np.ndarray,
        ) -> None: ...

        @overload
        def __init__(
            self,
            *,
            A: Sequence[Sequence[float]] | np.ndarray,
            b: Sequence[float] | np.ndarray,
            Ae: Sequence[Sequence[float]] | np.ndarray,
            be: Sequence[float] | np.ndarray,
        ) -> None: ...

        @overload
        def __init__(
            self,
            *,
            lb: Sequence[float] | np.ndarray,
            ub: Sequence[float] | np.ndarray,
        ) -> None: ...

        @overload
        def __init__(
            self,
            *,
            c: Sequence[float] | np.ndarray,
            h: float | Sequence[float] | np.ndarray,
        ) -> None: ...

    def __init__(self, **kwargs: Any) -> None:
        """Constructor for Polytope class."""
        # The following attributes are accessible to the user via a getter to avoid end-user interactions
        ##########################################
        # Attributes set by _set_attributes_from_X
        ##########################################
        self._type_of_set: str = "Polytope"
        self._dim: int = 0
        self._A, self._b = np.empty((0, 1)), np.empty((0,))
        self._Ae, self._be = np.empty((0, 1)), np.empty((0,))
        self._V = np.empty((0, 1))
        self._in_H_rep: bool = False
        self._in_V_rep: bool = False
        # These attributes are computed if necessary within _set_attributes_from_X
        self._is_full_dimensional: Optional[bool] = None
        self._is_empty: Optional[bool] = None
        self._is_bounded: Optional[bool] = None
        self._is_singleton: Optional[bool] = None
        # These attributes are used by CVXPY to solve problems
        self._cvxpy_args_lp = DEFAULT_CVXPY_ARGS_LP
        self._cvxpy_args_socp = DEFAULT_CVXPY_ARGS_SOCP
        self._cvxpy_args_sdp = DEFAULT_CVXPY_ARGS_SDP
        # Check how the constructor was called.
        empty_polytope_via_n = ("dim" in kwargs) or not kwargs
        V_passed = any(kw in kwargs for kw in ("V"))
        A_and_b_passed = all(k in kwargs for k in ("A", "b"))
        Ae_and_be_passed = all(k in kwargs for k in ("Ae", "be"))
        lb_and_ub_passed = all(kw in kwargs for kw in ("lb", "ub"))
        c_and_h_passed = all(kw in kwargs for kw in ("c", "h"))

        # Check valid combination of inputs
        if empty_polytope_via_n:
            if len(kwargs) <= 1:
                # Calling Polytope() or without n as a kwarg:
                self._set_polytope_to_empty(kwargs.get("dim", 0))
            else:
                raise ValueError("Cannot set dimension dim with other arguments")
        elif V_passed:
            if len(kwargs) == 1:
                # Calling Polytope(V=?)
                self._set_attributes_from_V(kwargs.get("V"))
            else:
                raise ValueError("Cannot set vertices V with other arguments")
        elif lb_and_ub_passed:
            if len(kwargs) == 2:
                # Parse lower and upper bounds.
                self._set_attributes_from_bounds(kwargs.get("lb"), kwargs.get("ub"))
            else:
                raise ValueError("Cannot set bounds (lb, ub) with other arguments")
        elif c_and_h_passed:
            if len(kwargs) != 2:
                raise ValueError("Cannot set up Polytope from (c, h) with other arguments")
            else:
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
                self._set_attributes_from_bounds(lb, ub)
        elif A_and_b_passed:
            if len(kwargs) == 2:
                # Calling Polytope(A=?, b=?)
                self._set_attributes_from_Ab_Aebe(kwargs.get("A"), kwargs.get("b"))
            elif Ae_and_be_passed and len(kwargs) == 4:
                # Calling Polytope(A=?, b=?, Ae=?, be=?)
                self._set_attributes_from_Ab_Aebe(kwargs.get("A"), kwargs.get("b"), kwargs.get("Ae"), kwargs.get("be"))
            else:
                raise ValueError("Cannot set H-Rep (A, b) or (A, b, Ae, be) with other arguments")
        else:
            raise ValueError(
                "Got invalid arguments while defining a polytope. Please specify either (A, b) or (A, b, Ae, be) or "
                "(lb, ub) or (c, h) or V or dim or NOTHING."
            )

    @property
    def type_of_set(self) -> str:
        """Return the type of set

        Returns:
            str: Type of the set
        """
        return self._type_of_set

    def _set_attributes_from_Ab_Aebe(
        self,
        A: Optional[Sequence[Sequence[float]] | np.ndarray | None] = None,
        b: Optional[Sequence[float] | np.ndarray | None] = None,
        Ae: Optional[Sequence[Sequence[float]] | np.ndarray | None] = None,
        be: Optional[Sequence[float] | np.ndarray | None] = None,
        erase_V_rep: bool = True,
        enable_warning: bool = True,
    ) -> None:
        r"""Protected method to set various attributes given (A, b, Ae, be) --- _dim,  _A, _b, _Ae, _be, _in_H_rep,
        _V, _in_V_rep, _is_empty, _is_full_dimensional, _is_bounded.

        Args:
            A (Sequence[Sequence[float]] | np.ndarray | None, optional): Inequality coefficient vectors (H-Rep). The
                vectors are stacked vertically.
            b (Sequence[float] | np.ndarray | None, optional): Inequality constants (H-Rep). The constants are expected
                to be in a 1D numpy array.
            Ae (Sequence[Sequence[float]] | np.ndarray | None, optional): Equality coefficient vectors (H-Rep). The
                vectors are stacked vertically.  Defaults to None.
            be (Sequence[float] | np.ndarray | None, optional): Equality constants (H-Rep). The constants are expected
                to be in a 1D numpy array. Defaults to None.
            erase_V_rep (bool, optional): When set to True, we erase V-rep. Defaults to True.
            enable_warning (bool, optional): Enables the UserWarning. May be turned off if expected. Defaults to True.

        Raises:
            ValueError: When (A, b) and (Ae, be) are not a valid system of linear inequalities and equations
            ValueError: Polytope is not bounded in any direction
            ValueError: Polytope is not bounded in some directions
            UserWarning: If some rows are removed

        Notes:
            We first check if (Ae, be) is a valid system of linear equations. The set :math:`\{A_e x = b_e\}` can be
            empty, a single point, or an affine set of dimension :math:`\leq \mathcal{P}.\text{dim}`.
        """
        self._in_H_rep = True  # To allow for querying self.n_equalities
        sanitized_A, sanitized_b = sanitize_Ab(A, b)
        sanitized_Ae, sanitized_be, Aebe_status, solution_to_Ae_x_eq_be = sanitize_and_identify_Aebe(Ae, be)

        # Set polytope dimension and check for compatibility of (A, b) and (Ae, be)
        inequalities_present = sanitized_A is not None
        if inequalities_present:
            self._dim = sanitized_A.shape[1]
        elif sanitized_Ae is not None:
            self._dim = sanitized_Ae.shape[1]
        else:
            raise ValueError("Polytope is unbounded in all directions!")

        # Check for compatibility of (A, b) and (Ae, be) if both are present.
        if inequalities_present:
            self._A, self._b = sanitized_A, sanitized_b
        else:
            self._A, self._b = np.empty((0, self.dim)), np.empty((0,))
        if Aebe_status == "no_Ae_be":
            self._Ae, self._be = np.empty((0, self.dim)), np.empty((0,))
        else:
            self._Ae, self._be = cast(np.ndarray, sanitized_Ae), cast(np.ndarray, sanitized_be)
            if self._Ae.shape[1] != self.dim:
                raise ValueError(
                    "Expected A and Ae to have same number of columns. A: {self._A.shape}, Ae: {self._Ae.shape}"
                )

        # Process equalities and inequalities together to check if the polytope is empty, single point, affine set, or
        # not full-dimensional, and to set up attributes accordingly.
        if Aebe_status == "infeasible" or (
            Aebe_status == "single_point"
            and inequalities_present
            and (self._A @ solution_to_Ae_x_eq_be - self._b > PYCVXSET_ZERO).any()
        ):
            # Infeasible or empty with single point only if (A, b) exists and excludes the point.
            self._set_polytope_to_empty(self.dim)
        elif Aebe_status == "single_point":
            # Single point case (either inequalities are not present or they are present but do not exclude the point)
            self._is_empty, self._is_full_dimensional, self._is_bounded, self._is_singleton = (
                False,
                self.dim == 1,
                True,
                True,
            )
            # Omit assignment of inequalities! They are redundant.
            self._A, self._b = np.empty((0, self.dim)), np.empty((0,))
            # Also update V-Rep
            self._V = np.array([solution_to_Ae_x_eq_be])
            self._in_V_rep = True
            erase_V_rep = False
        elif not inequalities_present:
            if Aebe_status == "affine_set":
                raise ValueError("Polytope is not bounded in some directions!")
            else:  # pragma: no cover
                # no_Ae_be will error out, infeasible and single_point are handled above.
                pass  # Already handled above. This is just for code clarity.
        else:  # inequalities_present is True and Aebe_status \in ['affine_set', 'no_Ae_be'].
            if (self._b == -np.inf).any():
                self._set_polytope_to_empty(self.dim)
            elif (self._b == np.inf).all():
                raise ValueError("Polytope is not bounded in any direction!")
            else:
                valid_rows_Ab = valid_rows_with_not_all_zeros_in_A_and_no_inf_in_b(self._A, self._b)
                if sum(valid_rows_Ab) < 2:
                    # Expected non-singleton H-Rep polytope to have at least 2 inequalities!")
                    if Aebe_status == "no_Ae_be":
                        raise ValueError("Polytope is not bounded in any direction!")
                    else:
                        raise ValueError("Polytope is not bounded in some directions!")
                elif enable_warning and sum(valid_rows_Ab) != self._A.shape[0]:
                    warnings.warn("Removed some rows in A that had all zeros | b that had np.inf!", UserWarning)
                self._A, self._b = self.A[valid_rows_Ab, :], self.b[valid_rows_Ab]
                # We have not assigned self._is_empty, self._is_full_dimensional, self._is_bounded OR
                # Keep the previous assignments

        if erase_V_rep:
            self._V = np.empty((0, self.dim))
            self._in_V_rep = False

    def _set_attributes_from_bounds(self: Polytope, lb: np.ndarray, ub: np.ndarray) -> None:
        """Protected method to set various attributes given (lb, ub) --- _dim,  _A, _b, _Ae, _be, _in_H_rep, _V,
        _in_V_rep, _is_empty, _is_full_dimensional, _is_bounded.

        Args:
            lb (np.ndarray): Lower bounds of the axis-aligned box
            ub (np.ndarray): Upper bounds of the axis-aligned box

        Raises:
            ValueError: When lb, ub is not 1D array
            ValueError: Mismatched dimensions

        Notes:
            When lb = ub, this function generates a polytope in V-Rep. Otherwise, it generates a polytope in H-Rep.
        """
        try:
            lb = np.atleast_1d(np.squeeze(lb)).astype(float)
            ub = np.atleast_1d(np.squeeze(ub)).astype(float)
        except (TypeError, ValueError) as err:
            raise ValueError("Expected lb, ub to convertible into 1D float numpy arrays") from err
        if lb.shape != ub.shape or lb.ndim != 1:
            raise ValueError(
                f"Expected lb, ub to 1D numpy arrays of same shape. Got lb: {np.array2string(np.array(lb)):s} "
                f"and ub: {np.array2string(np.array(ub)):s}!"
            )
        else:
            n = lb.size
            if np.any(lb > ub):
                self._set_polytope_to_empty(dim=n)
            elif np.isinf(lb).any() or np.isinf(ub).any():
                raise ValueError("Polytope is not bounded in some directions!")
            else:  # lb <= ub
                lb_ub_indices_for_equality = np.where(np.abs(ub - lb) <= PYCVXSET_ZERO)[0]
                if lb_ub_indices_for_equality.size == n:  # lb = ub. So a single vertex!
                    self._set_attributes_from_V(np.array([(lb + ub) / 2]), erase_H_rep=True)
                elif lb_ub_indices_for_equality.size == 0:
                    # No equality constraints: lb <= x <= ub with lb < ub <===> x <= ub, -x <= -lb
                    A_bound = np.vstack((np.eye(n), -np.eye(n)))
                    b_bound = np.hstack((ub, -lb))
                    # Help _set_attributes_from_Ab_Aebe by setting discernable attributes
                    self._is_full_dimensional, self._is_empty, self._is_bounded, self._is_singleton = (
                        True,
                        False,
                        True,
                        False,
                    )
                    self._set_attributes_from_Ab_Aebe(A_bound, b_bound, erase_V_rep=True)
                else:
                    # Some equality and some inequality constraints
                    n_equalities = lb_ub_indices_for_equality.size
                    n_inequalities = n - n_equalities
                    # Populate Ae, be
                    Ae = np.zeros((n_equalities, n))
                    be = np.zeros((n_equalities,))
                    for Ae_index, dim in enumerate(lb_ub_indices_for_equality):
                        Ae[Ae_index, dim] = 1
                        be[Ae_index] = (lb[dim] + ub[dim]) / 2
                    # Populate A, b
                    lb_ub_indices_for_inequality = np.where(np.abs(ub - lb) > PYCVXSET_ZERO)[0]
                    A = np.zeros((2 * n_inequalities, n))
                    b = np.zeros((2 * n_inequalities,))
                    for A_index, dim in enumerate(lb_ub_indices_for_inequality):
                        A[A_index, dim] = 1
                        A[n_inequalities + A_index, dim] = -1
                        b[A_index] = ub[dim]
                        b[n_inequalities + A_index] = -lb[dim]
                    # Help _set_attributes_from_Ab_Aebe by setting discernable attributes
                    self._is_full_dimensional, self._is_empty, self._is_bounded, self._is_singleton = (
                        False,
                        False,
                        True,
                        False,
                    )
                    self._set_attributes_from_Ab_Aebe(A, b, Ae=Ae, be=be, erase_V_rep=True)

    def _set_attributes_from_V(self: Polytope, V: np.ndarray, erase_H_rep: bool = True) -> None:
        """Protected method to set various attributes given (V) --- _dim,  _A, _b, _Ae, _be, _in_H_rep, _V,
        _in_V_rep, _is_empty, _is_full_dimensional, _is_bounded.

        Args:
            V (np.ndarray): List of vertices of the polytope. The list must be 2-dimensional, to avoid the risk of
                mistaking 1D polytopes as a self.n_vertices-dimensional polytope with 1 vertex. The vertices are
                arranged row-wise.
            erase_H_rep (bool): When set to True, we erase H-rep. Defaults to True.

        Raises:
            ValueError: When V is not a 2-D numpy array.

        Notes:
            This function automatically sets the polytope to be nonempty and bounded.
        """
        V = np.asarray(V, dtype=float)
        if V.ndim != 2:
            raise ValueError("Expected V to be a 2-D numpy array.")
        n = V.shape[1]
        if n == 0:
            self._set_polytope_to_empty(n)
        else:  # in_V_rep if V is not an empty array
            self._in_V_rep = True
            self._V = V
            self._dim = n
            self._is_empty, self._is_bounded = False, True
            self._is_singleton = self.n_vertices == 1
            # We have not assigned self._is_full_dimensional
        if erase_H_rep:
            self._A, self._b = np.empty((0, self.dim)), np.empty((0,))
            self._Ae, self._be = np.empty((0, self.dim)), np.empty((0,))
            self._in_H_rep = False

    def _set_polytope_to_empty(self: Polytope, dim: int) -> None:
        """Protected method to set A, b, dim, no_AbV, V members for an empty polytope"""
        self._dim = dim
        self._A, self._b = np.empty((0, self.dim)), np.empty((0,))
        self._Ae, self._be = np.empty((0, self.dim)), np.empty((0,))
        self._V = np.empty((0, self.dim))
        self._in_H_rep, self._in_V_rep = False, False
        self._is_full_dimensional, self._is_empty, self._is_bounded, self._is_singleton = (
            self.dim == 0,
            True,
            True,
            False,
        )

    def _update_emptiness_full_dimensionality_for_h_rep_polytope(self) -> None:
        r"""Update self._is_empty and self._is_full_dimensional using Chebyshev centering results.

        Raises:
            NotImplementedError: Unable to solve chebyshev centering problem using CVXPY

        Notes:
            - When in H-Rep, we use Chebyshev_centering to determine if the polytope is nonempty and full-dimensional.
              Specifically,
              #. Chebyshev radius == - :math:`infty`, the polytope is empty. In this case, it is full-dimensional, only
                 if dim=0.
              #. 0 <= Chebyshev radius <= :math:`\infty`, the polytope is always nonempty. It is full-dimensional when
                 `self.n_equalities` is 0 or `self.dim` is 1.
        """
        _, chebyshev_radius = self.chebyshev_centering()
        if chebyshev_radius == -np.inf:
            # Set is empty chebyshev_radius < 0 when infeasible
            self._is_empty, self._is_full_dimensional = True, self.dim == 0
        else:
            # Set is non-empty since chebyshev_radius >= 0
            # Set is full-dimensional if self.n_equalities == 0 or dim == 1
            self._is_empty, self._is_full_dimensional = False, (chebyshev_radius > 0 or self.dim == 1)

    def _update_boundedness_singleton_for_h_rep_polytope(self) -> None:
        r"""Update self._is_empty and self._is_full_dimensional using minimum_volume_circumscribing_rectangle.

        Raises:
            ValueError: Unable to solve minimum_volume_circumscribing_rectangle using CVXPY
        """
        try:
            lb, ub = self.minimum_volume_circumscribing_rectangle()
        except ValueError as err:
            raise ValueError(
                "Check for is_bounded and/or is_singleton using minimum_volume_circumscribing_rectangle failed!"
                "If the set is_bounded and/or is_singleton, try using a different solver."
            ) from err
        self._is_singleton = bool(np.isclose(lb, ub).all())
        self._is_bounded = bool((np.abs(np.hstack((ub, lb))) < np.inf).all())

    @property
    def dim(self) -> int:
        """Dimension of the polytope. In H-Rep polytope (A, b), this is the number of columns of A, while in V-Rep,
        this is the number of components of the vertices.

        Returns:
            int: Dimension of the polytope
        """
        return self._dim

    @property
    def A(self) -> np.ndarray:
        r"""Inequality coefficient vectors `A` for the polytope :math:`\{Ax \leq b, A_e x = b_e\}`.

        Returns:
            numpy.ndarray: Inequality coefficient vector (H-Rep). A is np.empty((0, self.dim)) for empty polytope.

        Notes:
            This function requires the polytope to be in H-Rep, and performs a halfspace enumeration if required.
        """
        if not self.in_H_rep:
            self.determine_H_rep()
        return self._A

    @property
    def b(self) -> np.ndarray:
        r"""Inequality constants `b` for the polytope :math:`\{Ax \leq b, A_e x = b_e\}`.

        Returns:
            numpy.ndarray: Inequality constants (H-Rep). b is np.empty((0,)) for empty polytope.

        Notes:
            This function requires the polytope to be in H-Rep, and performs a halfspace enumeration if required.
        """
        if not self.in_H_rep:
            self.determine_H_rep()
        return cast(np.ndarray, self._b)

    @property
    def H(self) -> np.ndarray:
        r"""Inequality constraints in halfspace representation `H=[A, b]` for the polytope
        :math:`\{Ax \leq b, A_e x = b_e\}`.

        Returns:
            numpy.ndarray: H-Rep in [A, b]. H is np.empty((0, self.dim + 1)) for empty polytope.

        Notes:
            This function requires the polytope to be in H-Rep, and performs a halfspace enumeration if required.
        """
        return np.hstack((self.A, np.array([self.b]).T))

    @property
    def n_halfspaces(self) -> int:
        r"""Number of halfspaces used to define the polytope :math:`\{Ax \leq b, A_e x = b_e\}`

        Returns:
            int: Number of halfspaces

        Notes:
            A call to this property performs a halfspace enumeration if the polytope is in V-Rep.
        """
        return self.A.shape[0]  # determines H-rep if not determined

    @property
    def Ae(self) -> np.ndarray:
        r"""Equality coefficient vectors `Ae` for the polytope :math:`\{Ax \leq b, A_e x = b_e\}`.

        Returns:
            numpy.ndarray: Equality coefficient vector (H-Rep). Ae is np.empty((0, self.dim)) for empty or
                full-dimensional polytope.

        Notes:
            This function requires the polytope to be in H-Rep, and performs a halfspace enumeration if required.
        """
        if not self.in_H_rep:
            self.determine_H_rep()
        return self._Ae

    @property
    def be(self) -> np.ndarray:
        r"""Equality constants `be` for the polytope :math:`\{Ax \leq b, A_e x = b_e\}`.

        Returns:
            numpy.ndarray: Equality constants (H-Rep). be is np.empty((0,)) for empty or full-dimensional polytope.

        Notes:
            This function requires the polytope to be in H-Rep, and performs a halfspace enumeration if required.
        """
        if not self.in_H_rep:
            self.determine_H_rep()
        return self._be

    @property
    def He(self) -> np.ndarray:
        r"""Equality constraints in halfspace representation `He=[Ae, be]` for the polytope
        :math:`\{Ax \leq b, A_e x = b_e\}`.

        Returns:
            numpy.ndarray: H-Rep in [Ae, be]. He is np.empty((0, self.dim + 1)) for empty or full-dimensional polytope.

        Notes:
            This function requires the polytope to be in H-Rep, and performs a halfspace enumeration if required.
        """
        return np.hstack((self.Ae, np.array([self.be]).T))

    @property
    def n_equalities(self) -> int:
        r"""Number of linear equality constraints used to define the polytope :math:`\{Ax \leq b, A_e x = b_e\}`

        Returns:
            int: Number of linear equality constraints

        Notes:
            A call to this property performs a halfspace enumeration if the polytope is in V-Rep.
        """
        return self.Ae.shape[0]  # determines H-rep if not determined

    @property
    def V(self) -> np.ndarray:
        r"""Vertex representation (`V`) where the polytope is given by :math:`\text{ConvexHull}(v_i)` with
        :math:`v_i` as the rows of :math:`V=[v_1;v_2;\ldots;v_{n_vertices}]`.

        Returns:
            numpy.ndarray: Vertices of the polytope, arranged row-wise. V is np.empty((0, self.dim)) if polytope is
            empty.

        Notes:
            This function requires the polytope to be in V-Rep, and performs a vertex enumeration if required.
        """
        if not self.in_V_rep and self.in_H_rep:
            self.determine_V_rep()  # Call only if no V-rep and available H-rep
        return self._V

    @property
    def n_vertices(self) -> int:
        """Number of vertices

        Returns:
            int: Number of vertices

        Notes:
            A call to this property performs a vertex enumeration if the polytope is in H-Rep.
        """
        return self.V.shape[0]  # determines V-rep if not determined

    @property
    def is_full_dimensional(self) -> bool:
        """
        Check if the affine dimension of the polytope is the same as the polytope dimension

        Returns:
            bool: True when the affine hull containing the polytope has the dimension `self.dim`

        Notes:
            This function can have self to be in V-Rep or H-Rep. See Sec. 2.1.3 of [BV04] for discussion on affine
            dimension.

            An empty polytope is full dimensional if dim=0, otherwise it is not full-dimensional.

            When the n-dimensional polytope is in V-rep, it is full-dimensional when its affine dimension is n. Recall
            that, the affine dimension is the dimension of the affine hull of the polytope is the linear subspace
            spanned by the vectors formed by subtracting the vertices with one of the vertices. Consequently, its
            dimension is given by the rank of the matrix P.V[1:] - P.V[0]. When there are fewer than self.dim + 1
            vertices, we know it is coplanar without checking for matrix rank (simplex needs at least self.dim + 1
            vertices and that is the polytope with the fewest vertices). For numerical stability, we zero-out all delta
            vertices below PYCVXSET_ZERO.

            When the n-dimensional polytope is in H-Rep, it is full-dimensional if it can fit a n-dimensional ball of
            appropriate center and radius inside it (Chebyshev radius).
        """
        if self._is_full_dimensional is None:
            if self.in_V_rep:
                if self.dim == 1:
                    self._is_full_dimensional = True
                elif self.n_vertices <= self.dim:
                    # Simplex of n-dim with n > 1 needs at least self.dim + 1 vertices
                    self._is_full_dimensional = False
                else:
                    delta_vertices = self.V[1:] - self.V[0]
                    delta_vertices[abs(delta_vertices) <= PYCVXSET_ZERO] = 0
                    self._is_full_dimensional = bool(np.linalg.matrix_rank(delta_vertices) == self.dim)
            else:
                self._update_emptiness_full_dimensionality_for_h_rep_polytope()
        return cast(bool, self._is_full_dimensional)

    @property
    def is_empty(self) -> bool:
        """Check if the polytope is empty.

        Returns:
            bool: When True, the polytope is empty

        Notes:
            This property is well-defined (`self.n_vertices == 0`) when the polytope is in V-Rep and initialized in the
            constructor. For H-Rep, it solves a Chebyshev centering problem.
        """
        if self._is_empty is None:
            self._update_emptiness_full_dimensionality_for_h_rep_polytope()
        return cast(bool, self._is_empty)

    @property
    def is_singleton(self) -> bool:
        """Check if the polytope is singleton.

        Returns:
            bool: When True, the polytope is singleton.

        Notes:
            This property is well-defined (`self.n_vertices == 1`) when the polytope is in V-Rep and initialized in the
            constructor. For H-Rep, it solves a minimum_volume_circumscribing_rectangle problem.
        """
        if self._is_singleton is None:
            self._update_boundedness_singleton_for_h_rep_polytope()
        return cast(bool, self._is_singleton)

    @property
    def is_bounded(self) -> bool:
        """Check if the polytope is bounded.

        Returns:
            bool: True if the polytope is bounded, and False otherwise.
        """
        if self._is_bounded is None:
            self._update_boundedness_singleton_for_h_rep_polytope()
        return cast(bool, self._is_bounded)

    @property
    def in_H_rep(self) -> bool:
        """Check if the polytope have a halfspace representation (H-Rep).

        Returns:
            bool: When True, the polytope has halfspace representation (H-Rep). Otherwise, False.
        """
        return self._in_H_rep

    @property
    def in_V_rep(self) -> bool:
        """Check if the polytope have a vertex representation (V-Rep).

        Returns:
           bool: When True, the polytope has vertex representation (V-Rep). Otherwise, False.
        """
        return self._in_V_rep

    @property
    def cvxpy_args_lp(self) -> dict[str, Any]:
        """CVXPY arguments in use when solving a linear program

        Returns:
            dict: CVXPY arguments in use when solving a linear program. Defaults to dictionary in
            `pycvxset.common.DEFAULT_CVXPY_ARGS_LP`.

        """
        return self._cvxpy_args_lp

    @cvxpy_args_lp.setter
    def cvxpy_args_lp(self: Polytope, value: dict[str, Any]) -> None:
        """Update CVXPY arguments in use when solving a linear program

        Args:
            value: Dictionary with new CVXPY arguments in use when solving a linear program.
        """
        self._cvxpy_args_lp = value

    @property
    def cvxpy_args_socp(self: Polytope) -> dict[str, Any]:
        """CVXPY arguments in use when solving a second-order cone program

        Returns:
            dict: CVXPY arguments in use when solving a second-order cone program. Defaults to dictionary in
            `pycvxset.common.DEFAULT_CVXPY_ARGS_SOCP`.
        """
        return self._cvxpy_args_socp

    @cvxpy_args_socp.setter
    def cvxpy_args_socp(self: Polytope, value: dict[str, Any]) -> None:
        """Update CVXPY arguments in use when solving a second-order cone program

        Args:
            value: Dictionary with new CVXPY arguments in use when solving a second-order cone program.
        """
        self._cvxpy_args_socp = value

    @property
    def cvxpy_args_sdp(self: Polytope) -> dict[str, Any]:
        """CVXPY arguments in use when solving a semi-definite program

        Returns:
            dict: CVXPY arguments in use when solving a semi-definite program. Defaults to dictionary in
            `pycvxset.common.DEFAULT_CVXPY_ARGS_SDP`.
        """
        return self._cvxpy_args_sdp

    @cvxpy_args_sdp.setter
    def cvxpy_args_sdp(self: Polytope, value: dict[str, Any]) -> None:
        """Update CVXPY arguments in use when solving a semi-definite program

        Args:
            value: Dictionary with new CVXPY arguments in use when solving a semi-definite program.
        """
        self._cvxpy_args_sdp = value

    ###########
    # Plotting
    ###########
    plot = plot
    plot2d = plot2d
    plot3d = plot3d

    ################
    # CVXPY-focussed
    ################
    def containment_constraints(
        self, x: cvxpy.Variable, flatten_order: Literal["F", "C"] = "F"
    ) -> tuple[list[cvxpy.Constraint], Optional[cvxpy.Variable]]:
        """Get CVXPY constraints for containment of x (a cvxpy.Variable) in a polytope.

        Args:
            x (cvxpy.Variable): CVXPY variable to be optimized
            flatten_order (Literal["F", "C"]): Order to use for flatten (choose between "F", "C"). Defaults to "F",
                which implements column-major flatten. In 2D, column-major flatten results in stacking rows horizontally
                to achieve a single horizontal row.

        Raises:
            ValueError: When polytope is empty

        Returns:
            tuple: A tuple with two items:

            #. constraint_list (list): CVXPY constraints for the containment of x in the polytope.
            #. theta (cvxpy.Variable | None): CVXPY variable representing the convex combination coefficient when
               polytope is in V-Rep. It is None when the polytope is in H-Rep or empty.
        """
        import cvxpy as cp

        x_reshaped = cp.reshape(x, (x.size,), order=flatten_order)
        if self.in_V_rep:
            theta = cp.Variable((self.n_vertices,), nonneg=True)
            constraints: list[cp.Constraint] = [x_reshaped == self.V.T @ theta, cast(cp.Constraint, cp.sum(theta) == 1)]
            return constraints, theta
        elif self.in_H_rep:
            polytope_containment_constraints: list[cp.Constraint] = [self.A @ x_reshaped <= self.b]
            if self.n_equalities > 0:
                polytope_containment_constraints += [self.Ae @ x_reshaped == self.be]
            return polytope_containment_constraints, None
        else:
            raise ValueError("Containment constraints can not be generated for an empty polytope!")

    minimize = minimize

    ##################
    # Unary operations
    ##################
    def copy(self) -> Polytope:
        """Create a copy of the polytope"""
        if self.in_H_rep:
            copy_polytope = self.__class__(A=self.A, b=self.b, Ae=self.Ae, be=self.be)
            if self.in_V_rep:
                copy_polytope._set_attributes_from_V(V=self.V, erase_H_rep=False)
            return copy_polytope
        elif self.in_V_rep:
            return self.__class__(V=self.V)
        else:
            return self.__class__(dim=self.dim)

    chebyshev_centering = chebyshev_centering
    decompose_as_affine_transform_of_polytope_without_equalities = (
        decompose_as_affine_transform_of_polytope_without_equalities
    )
    interior_point = interior_point
    maximum_volume_inscribing_ellipsoid = maximum_volume_inscribing_ellipsoid
    minimum_volume_circumscribing_ellipsoid = minimum_volume_circumscribing_ellipsoid
    minimum_volume_circumscribing_rectangle = minimum_volume_circumscribing_rectangle
    normalize = normalize
    deflate_rectangle = classmethod(deflate_rectangle)
    volume = volume

    def __pow__(self, power: int) -> Any:
        r"""Compute the Cartesian product with itself.

        Args:
            power (int): Number of times the polytope is multiplied with itself

        Returns:
            Polytope: The polytope :math:`\mathcal{R}` corresponding to P`^N`.

        Notes:
            This function requires the polytope to be in H-Rep, and performs a halfspace enumeration if polytope is in
            V-Rep.
        """
        concatenated_polytope_A = np.kron(np.eye(power), self.A)
        concatenated_polytope_b = np.tile(self.b, (power,))
        if self.n_equalities > 0:
            concatenated_polytope_Ae = np.kron(np.eye(power), self.Ae)
            concatenated_polytope_be = np.tile(self.be, (power,))
            return self.__class__(
                A=concatenated_polytope_A,
                b=concatenated_polytope_b,
                Ae=concatenated_polytope_Ae,
                be=concatenated_polytope_be,
            )
        else:
            return self.__class__(A=concatenated_polytope_A, b=concatenated_polytope_b)

    ######################
    # Comparison operators
    ######################
    contains = contains
    __contains__ = contains

    def __le__(self: Polytope, Q: ConstrainedZonotope | Polytope) -> Any:
        """Overload <= operator for containment. self <= Q is equivalent to Q.contains(self)."""
        if is_polytope(Q) or is_constrained_zonotope(Q):
            return Q.contains(self)
        else:
            # Q is a constrained zonotope ====> Polytope <= ConstrainedZonotope case
            return NotImplemented

    def __ge__(
        self: Polytope,
        Q: "Sequence[float] | np.ndarray | ConstrainedZonotope | Ellipsoid | Polytope",
    ) -> bool:
        """Overload >= operator for containment. self >= Q is equivalent to P.contains(Q)."""
        if is_polytope(Q) or is_constrained_zonotope(Q) or is_ellipsoid(Q):
            return self.contains(Q)
        else:
            try:
                Q_arr = np.atleast_2d(Q).astype(float)
            except (TypeError, ValueError):
                return NotImplemented
            if Q_arr.size == self.dim and Q_arr.shape[0] == 1:
                # Q is a point ====> Polytope >= Point case
                return self.contains(Q_arr)
            else:
                return NotImplemented

    def __eq__(self: Polytope, Q: object) -> Any:
        """Overload == operator with equality check. P == Q is equivalent to Q.contains(P) and P.contains(Q)"""
        return self <= cast("ConstrainedZonotope | Polytope", Q) and self >= Q

    __lt__ = __le__
    __gt__ = __ge__

    ####################
    # Binary operations
    ####################
    # Since it is symmetric
    plus = plus
    __add__ = plus
    __radd__ = plus
    minus = minus
    __sub__ = minus

    def __rsub__(self, Q: Any):
        raise TypeError(f"Unsupported operation: {type(Q)} - Polytope!")

    __array_ufunc__ = None  # Allows for numpy matrix times Polytope
    affine_map = affine_map
    inverse_affine_map_under_invertible_matrix = inverse_affine_map_under_invertible_matrix

    # Polytope times Matrix
    __matmul__ = inverse_affine_map_under_invertible_matrix

    def __mul__(self, x: Any):
        """Do not allow Polytope * anything"""
        return NotImplemented

    def __neg__(self) -> Polytope:
        r"""Negation of the polytope :math:`\mathcal{R}=\{-p: p\in\mathcal{ P}\}`.

        Returns:
            Polytope: The polytope :math:`\mathcal{R}` that flips the polytope :math:`\mathcal{P}` about origin.
        """
        return affine_map(self, -1)

    # Scalar/Matrix times Polytope (called when left operand does not support multiplication)
    def __rmatmul__(self: Polytope, M: Sequence[float] | Sequence[Sequence[float]] | np.ndarray) -> Polytope:
        """Overload @ operator for affine map (matrix times Polytope)."""
        return affine_map(self, M)

    def __rmul__(self: Polytope, m: int | float | Sequence[float] | np.ndarray) -> Polytope:
        """Overload * operator for multiplication."""
        try:
            m = np.squeeze(m).astype(float)
        except (TypeError, ValueError) as err:
            raise TypeError(f"Unsupported operation: {type(m)} * Polytope!") from err
        return affine_map(self, m)

    closest_point = convex_set_closest_point
    distance = convex_set_distance
    extreme = convex_set_extreme
    intersection = intersection
    intersection_with_halfspaces = intersection_with_halfspaces
    intersection_with_affine_set = intersection_with_affine_set
    intersection_under_inverse_affine_map = intersection_under_inverse_affine_map

    def project(
        self: Polytope, x: Sequence[float] | Sequence[Sequence[float]] | np.ndarray, p: int | str = 2
    ) -> tuple[np.ndarray, np.ndarray]:
        return convex_set_project(self, x, p=p)

    project.__doc__ = (convex_set_project.__doc__ or "") + DOCSTRING_FOR_PROJECT

    def projection(self: Polytope, project_away_dims: int | Sequence[int] | np.ndarray) -> Polytope:
        return convex_set_projection(self, project_away_dims=project_away_dims)

    projection.__doc__ = (convex_set_projection.__doc__ or "") + DOCSTRING_FOR_PROJECTION

    def slice(
        self: Polytope, dims: int | Sequence[int] | np.ndarray, constants: float | Sequence[float] | np.ndarray
    ) -> Polytope:
        return convex_set_slice(self, dims, constants)

    slice.__doc__ = (convex_set_slice.__doc__ or "") + DOCSTRING_FOR_SLICE

    def slice_then_projection(
        self: Polytope, dims: int | Sequence[int] | np.ndarray, constants: float | Sequence[float] | np.ndarray
    ) -> Polytope:
        return convex_set_slice_then_projection(self, dims=dims, constants=constants)

    slice_then_projection.__doc__ = (
        convex_set_slice_then_projection.__doc__ or ""
    ) + DOCSTRING_FOR_SLICE_THEN_PROJECTION

    def support(
        self: Polytope, eta: Sequence[float] | Sequence[Sequence[float]] | np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        return convex_set_support(self, eta)

    support.__doc__ = (convex_set_support.__doc__ or "") + DOCSTRING_FOR_SUPPORT

    _compute_support_function_multiple_eta = _compute_support_function_multiple_eta
    _compute_project_multiple_points = _compute_project_multiple_points

    ###########################
    # Vertex-halfspace enumeration
    ###########################
    determine_H_rep = determine_H_rep
    determine_V_rep = determine_V_rep
    minimize_H_rep = minimize_H_rep
    minimize_V_rep = minimize_V_rep

    #########################
    # Polytope representation
    #########################
    def __str__(self) -> str:
        if self.is_empty:
            repr_str = f"(empty) in R^{self.dim:d}"
        else:
            if self.in_H_rep and self.in_V_rep:
                repr_str = f"in R^{self.dim:d} in H-Rep and V-Rep"
            elif self.in_H_rep:
                repr_str = f"in R^{self.dim:d} in only H-Rep"
            else:
                repr_str = f"in R^{self.dim:d} in only V-Rep"
        return f"Polytope {repr_str:s}"

    def __repr__(self) -> str:
        repr_str = [str(self)]
        if self.in_H_rep:
            inequality_str = f"{self.n_halfspaces:d} inequalities"
            if self.n_equalities > 1:
                equality_str = f"{self.n_equalities:d} equality constraints"
            elif self.n_equalities == 1:
                equality_str = "1 equality constraint"
            else:
                equality_str = "no equality constraints"
            repr_str += [f"\n\tIn H-rep: {inequality_str:s} and {equality_str:s}"]
        if self.in_V_rep:
            if self.n_vertices > 1:
                vertex_str = "vertices"
            else:
                vertex_str = "vertex"
            repr_str += [f"\n\tIn V-rep: {self.n_vertices:d} {vertex_str:s}"]
        return "".join(repr_str)
