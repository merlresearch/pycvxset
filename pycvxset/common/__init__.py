# Copyright (C) 2020-2026 Mitsubishi Electric Research Laboratories (MERL)
# Copyright (c) 2019 Tor Aksel N. Heirung
#
# SPDX-License-Identifier: AGPL-3.0-or-later
# SPDX-License-Identifier: MIT

# Code purpose: Describe various constants and methods that are common to different set representations.
# Coverage: This file has 5 missing statements + 31 excluded statements + 0 partial branches.

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any, Optional, Sequence, cast

if TYPE_CHECKING:
    from pycvxset.ConstrainedZonotope import ConstrainedZonotope
    from pycvxset.Ellipsoid import Ellipsoid
    from pycvxset.Polytope import Polytope
    import cvxpy

import itertools
import os

import cdd
import numpy as np
from matplotlib.axes import Axes
from mpl_toolkits.mplot3d.axes3d import Axes3D

from pycvxset.common.constants import (
    DEFAULT_CVXPY_ARGS_SOCP,
    PLOTTING_DECIMAL_PRECISION_CDD,
    PYCVXSET_ZERO,
    PYCVXSET_ZERO_CDD,
    SPOAUS_COST_TOLERANCE,
    SPOAUS_DIRECTIONS_PER_QUADRANT,
    SPOAUS_INITIAL_TAU,
    SPOAUS_ITERATIONS_AT_TAU_MAX,
    SPOAUS_MINIMUM_NORM_VALUE_SQR,
    SPOAUS_SAVE_POINTS_ON_A_UNIT_SPHERE,
    SPOAUS_SCALING_TAU,
    SPOAUS_SLACK_TOLERANCE,
    SPOAUS_TAU_MAX,
)

DOCSTRING_FOR_PROJECT = ""


def approximate_volume_from_grid(
    cvx_set: "ConstrainedZonotope | Ellipsoid | Polytope", area_grid_step_size: int | float
) -> float:
    r"""Estimate area of a two-dimensional set using a grid of given step size

    Args:
        cvx_set (ConstrainedZonotope | Ellipsoid | Polytope): Set for which area is to be computed
        area_grid_step_size (int | float): Scalar step size that is constant in both dimensions

    Raises:
        ValueError: When set is not 2-dimensional

    Returns:
        float: Approximate area of the set

    Notes:
        This function creates a 2D grid of points with a given step size, and computes the fraction of points that lie
        in the set. Consequently, the computed area is an approximation of the actual area of the set. The area of the
        bounding box is computed using the :meth:`minimum_volume_circumscribing_rectangle` method associated with the
        set.
    """
    if cvx_set.dim != 2:
        raise ValueError("Expected 2-dimensional set")
    elif cvx_set.is_empty:
        return 0.0
    elif area_grid_step_size <= 0:
        raise ValueError("Expected area_grid_step_size to be a positive float!")

    lb, ub = cvx_set.minimum_volume_circumscribing_rectangle()
    if not (np.isfinite(lb).all() and np.isfinite(ub).all()):
        raise ValueError("Expected the set to be bounded for approximate_volume_from_grid!")
    elif (lb > ub).any() or np.max(np.abs(ub - lb)) <= PYCVXSET_ZERO:
        return 0.0
    else:
        area_of_bounding_box = (ub[0] - lb[0]) * (ub[1] - lb[1])
        corrected_lb = lb + area_grid_step_size / 2
        corrected_ub = ub - area_grid_step_size / 2
        x1 = np.arange(corrected_lb[0], corrected_ub[0] + area_grid_step_size, area_grid_step_size)
        x2 = np.arange(corrected_lb[1], corrected_ub[1] + area_grid_step_size, area_grid_step_size)
        X1, X2 = np.meshgrid(x1, x2)
        n_area_boxes = len(x1) * len(x2)
        points_to_test = np.vstack((X1.flatten(), X2.flatten())).T
        containment_flag = cvx_set.contains(points_to_test)
        set_area = float(np.count_nonzero(containment_flag) / n_area_boxes * area_of_bounding_box)
        return set_area


def check_matrices_are_equal_ignoring_row_order(
    A: Sequence[Sequence[float]] | np.ndarray, B: Sequence[Sequence[float]] | np.ndarray
) -> bool:
    """Check matrices are equal while ignoring row order

    Args:
        A (Sequence[Sequence[float]] | np.ndarray): Matrix 1
        B (Sequence[Sequence[float]] | np.ndarray): Matrix 2

    Returns:
        bool: A == B

    Notes:
        isclose does element-wise comparison, all with axis=1, provides a row-wise test, and finally any checks for some
        row where row-wise match is true
    """
    A_arr: np.ndarray = np.array(A, dtype=float)
    B_arr: np.ndarray = np.array(B, dtype=float)
    return (
        A_arr.shape == B_arr.shape
        and sum([np.any(np.all(np.isclose(row, B_arr), axis=1)) for row in A_arr]) == B_arr.shape[0]
    )


def check_vectors_are_equal_ignoring_row_order(
    A: Sequence[float] | np.ndarray, B: Sequence[float] | np.ndarray
) -> bool:
    """Check vectors are equal while ignoring row order

    Args:
        A (Sequence[float] | np.ndarray): Vector 1
        B (Sequence[float] | np.ndarray): Vector 2

    Returns:
        bool: A == B

    Notes:
        isclose does element-wise comparison and we sort
    """
    A_arr: np.ndarray = np.squeeze(A).astype(float)
    B_arr: np.ndarray = np.squeeze(B).astype(float)
    return cast(
        bool,
        A_arr.ndim == 1 and A_arr.shape == B_arr.shape and np.all(np.isclose(np.sort(A_arr), np.sort(B_arr))),
    )


def convex_set_closest_point(
    self: "ConstrainedZonotope | Ellipsoid | Polytope", points: Sequence[Sequence[float]] | np.ndarray, p: int | str = 2
) -> np.ndarray:
    """Wrapper for :meth:`project` to compute the point in the convex set closest to the given point.

    Args:
        points (Sequence[Sequence[float]] | np.ndarray): Points to project. Matrix (N times self.dim), where each row
            is a point.
        p (str | int): Norm-type. It can be 1, 2, or 'inf'. Defaults to 2.

    Returns:
        numpy.ndarray: Projection of points to the set as a 2D numpy.ndarray. These arrays have as many rows as points.

    Notes:
        For more detailed description, see documentation for :meth:`project` function.
    """
    return self.project(points, p=p)[0]


def convex_set_contains_points(
    self: "ConstrainedZonotope | Ellipsoid | Polytope", points: Sequence[Sequence[float]] | np.ndarray
) -> bool | np.ndarray:
    """Wrapper for :meth:`distance` to compute containment from a given collection of points.

    Args:
        points (Sequence[Sequence[float]] | np.ndarray): Points to project (N times self.dim) with each row as a point.

    Returns:
        bool | numpy.ndarray[bool]: Logical array of containment of points in the set.

    Notes:
        For more detailed description, see documentation for :meth:`distance` function.
    """
    points_arr: np.ndarray = np.atleast_2d(points).astype(float)
    n_points, point_dim = points_arr.shape
    if point_dim != self.dim:
        # Could transpose here if test_points.shape[1] == self.dim, but better to
        # specify that points must be columns:
        raise ValueError(f"Mismatch in dimensions (self.dim: {self.dim:d} and test_points.dim: {point_dim:d})")
    elif self.is_empty:
        if n_points > 1:
            return np.zeros((n_points,), dtype="bool")
        else:
            return False
    else:
        distance_to_test_points = self.distance(points_arr, p="inf")
        if n_points > 1:
            return distance_to_test_points <= PYCVXSET_ZERO
        else:
            return bool(distance_to_test_points[0] <= PYCVXSET_ZERO)


def convex_set_distance(
    self: "ConstrainedZonotope | Ellipsoid | Polytope", points: Sequence[Sequence[float]] | np.ndarray, p: int | str = 2
) -> np.ndarray:
    """Wrapper for :meth:`project` to compute distance of a point to a convex set.

    Args:
        points (Sequence[Sequence[float]] | np.ndarray): Points to project. Matrix (N times self.dim), where each row
            is a point.
        p (int | str): Norm-type. It can be 1, 2, or 'inf'. Defaults to 2.

    Returns:
        numpy.ndarray: Distance of points to the set as a 1D numpy.ndarray. These arrays have as many rows as points.

    Notes:
        For more detailed description, see documentation for :meth:`project` function.
    """
    return self.project(points, p=p)[1]


def convex_set_extreme(
    self: "ConstrainedZonotope | Ellipsoid | Polytope", eta: Sequence[Sequence[float]] | np.ndarray
) -> np.ndarray:
    """Wrapper for :meth:`support` to compute the extreme point.

    Args:
        eta (Sequence[Sequence[float]] | np.ndarray): Support directions. Matrix (N times self.dim), where each row is
            a support direction.

    Returns:
        numpy.ndarray: Support vector evaluation(s) as a 2D numpy.ndarray. The array has as many rows as eta.

    Notes:
        For more detailed description, see documentation for :meth:`support` function.
    """
    return self.support(eta)[1]


def convex_set_minimum_volume_circumscribing_rectangle(
    self: "ConstrainedZonotope | Ellipsoid | Polytope",
) -> tuple[np.ndarray, np.ndarray]:
    r"""Compute the minimum volume circumscribing rectangle for a set.

    Raises:
        ValueError: Solver error or set is empty!

    Returns:
        tuple: A tuple of two elements
            - lb (numpy.ndarray): Lower bound :math:`l` on the set,
              :math:`\mathcal{P}\subseteq\{l\}\oplus\mathbb{R}_{\geq 0}`.
            - ub (numpy.ndarray): Upper bound :math:`u` on the set,
              :math:`\mathcal{P}\subseteq\{u\}\oplus(-\mathbb{R}_{\geq 0})`.

    Notes:
        This function computes the lower/upper bound by an element-wise support computation (2n linear programs), where
        n is attr:`self.dim`. To reuse the :meth:`support` function for the lower bound computation, we solve the
        optimization for each :math:`i\in\{1,2,...,n\}`,

        .. math::
            \inf_{x\in\mathcal{P}} e_i^\top x=-\sup_{x\in\mathcal{P}} -e_i^\top x=-\rho_{\mathcal{P}}(-e_i),

        where :math:`e_i\in\mathbb{R}^n` denotes the standard coordinate vector, and :math:`\rho_{\mathcal{P}}` is the
        support function of :math:`\mathcal{P}`.
    """
    try:
        lb = -self.support(-np.eye(self.dim))[0]
        ub = self.support(np.eye(self.dim))[0]
    except ValueError as err:
        raise ValueError("Can not compute minimum_volume_circumscribing_rectangle. Is set empty?") from err
    except NotImplementedError as err:
        raise ValueError("Computation of minimum_volume_circumscribing_rectangle failed!") from err
    return lb, ub


def convex_set_project(
    cvx_set: "ConstrainedZonotope | Ellipsoid | Polytope",
    points: Sequence[Sequence[float]] | np.ndarray,
    p: int | str = 2,
) -> tuple[np.ndarray, np.ndarray]:
    r"""Project a point or a collection of points on to a set.

    Given a set :math:`\mathcal{P}` and a test point :math:`y\in\mathbb{R}^{\mathcal{P}.\text{dim}}`, this function
    solves a convex program,

        .. math::
            \text{minimize}    &\quad  \|x - y\|_p\\
            \text{subject to}  &\quad  x \in \mathcal{P}\\

    Args:
        points (Sequence[Sequence[float]] | np.ndarray): Points to project (N times self.dim) with each row as a point.
        p (str | int): Norm-type. It can be 1, 2, or 'inf'. Defaults to 2, which is the Euclidean norm.

    Raises:
        ValueError: Set is empty
        ValueError: Dimension mismatch --- no. of columns in points is different from self.dim.
        ValueError: Points is not convertible into a 2D array
        NotImplementedError: Unable to solve problem using CVXPY

    Returns:
        tuple: A tuple with two items:
            1. projected_point (numpy.ndarray): Projection point(s) as a 2D numpy.ndarray. Matrix (N times self.dim),
               where each row is a projection of the point in points to the set :math:`\mathcal{P}`.
            2. distance (numpy.ndarray): Distance(s) as a 1D numpy.ndarray. Vector (N,), where each row is a projection
               of the point in points to the set :math:`\mathcal{P}`.
    """
    if cvx_set.is_empty:
        raise ValueError("Set must be non-empty for project.")
    else:
        points_arr: np.ndarray = np.atleast_2d(points).astype(float)
        if points_arr.ndim > 2:
            raise ValueError("Expected points to be a 1D/2D numpy array")
        elif points_arr.shape[1] != cvx_set.dim:
            raise ValueError(
                f"points dim. ({points_arr.shape[1]:d}), no. of columns, is different from set dim. ({cvx_set.dim:d})"
            )
        else:  # points.ndim == 2
            return cvx_set._compute_project_multiple_points(points_arr, p)


def convex_set_projection(
    self: "ConstrainedZonotope | Ellipsoid | Polytope", project_away_dims: int | Sequence[int] | np.ndarray
) -> Any:
    r"""Orthogonal projection of a set :math:`\mathcal{P}` after removing some user-specified dimensions.

    .. math::
        \mathcal{R} = \{r \in \mathbb{R}^{m}\ |\  \exists v \in \mathbb{R}^{n - m},\ \text{Lift}(r,v)\in \mathcal{P}\}

    Here, :math:`m = \mathcal{P}.\text{dim} - \text{length}(\text{project\_away\_dim})`, and
    :math:`\text{Lift}(r,v)` lifts ("undo"s the projection) using the appropriate components of `v`. This function uses
    :meth:`affine_map` to implement the projection by designing an appropriate affine map :math:`M \in
    \{0,1\}^{m\times\mathcal{P}.\text{dim}}` with each row of :math:`M` corresponding to some standard axis vector
    :math:`e_i\in\mathbb{R}^m`.

    Args:
        project_away_dims (Sequence[int] | np.ndarray): Dimensions to projected away in integer interval
            [0, 1, ..., n - 1].

    Raises:
        ValueError: When project_away_dims are not in the integer interval | All dimensions are projected away

    Returns:
        object: Set obtained via projection.
    """
    project_away_dims_arr: np.ndarray = np.atleast_1d(np.squeeze(project_away_dims)).astype(int)
    if project_away_dims_arr.size == 0:
        return self.copy()
    else:
        if np.min(project_away_dims_arr) < 0 or np.max(project_away_dims_arr) >= self.dim:
            raise ValueError(
                f"Expected project_away_dims to be in the integer interval [0:{self.dim-1:d}]!"
                f"Got {np.array2string(np.array(project_away_dims_arr)):s}"
            )
        project_away_dims_list = project_away_dims_arr.tolist()
        corrected_retain_dimensions = [d for d in range(self.dim) if d not in project_away_dims_list]
        n_dimensions_to_retain = len(corrected_retain_dimensions)
        if n_dimensions_to_retain == 0:
            raise ValueError("Can not project away all dimensions!")
        else:
            # Perform the projection using projection_matrix @ V
            projection_matrix = np.zeros((n_dimensions_to_retain, self.dim))
            for row_index, d in enumerate(corrected_retain_dimensions):
                projection_matrix[row_index, d] = 1
            return projection_matrix @ self


def convex_set_support(
    self: "ConstrainedZonotope | Polytope", eta: Sequence[float] | Sequence[Sequence[float]] | np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    r"""Evaluates the support function and support vector of a set.

    The support function of a set :math:`\mathcal{P}` is defined as :math:`\rho_{\mathcal{P}}(\eta) =
    \max_{x\in\mathcal{P}} \eta^\top x`. The support vector of a set :math:`\mathcal{P}` is defined as
    :math:`\nu_{\mathcal{P}}(\eta) = \arg\max_{x\in\mathcal{P}} \eta^\top x`.

    Args:
        eta (Sequence[float] | Sequence[Sequence[float]] | np.ndarray): Support directions. Matrix (N times self.dim),
            where each row is a support direction.

    Raises:
        ValueError: Set is empty
        ValueError: Mismatch in eta dimension
        ValueError: eta is not convertible into a 2D array
        NotImplementedError: Unable to solve problem using CVXPY

    Returns:
        tuple: A tuple with two items:
            1. support_function_evaluations (numpy.ndarray): Support function evaluation(s) as a 2D numpy.ndarray.
               Vector (N,) with as many rows as eta.
            2. support_vectors (numpy.ndarray): Support vectors as a 2D numpy.ndarray. Matrix N x self.dim with as many
               rows as eta.
    """
    eta_arr: np.ndarray = np.atleast_2d(eta).astype(float)
    if eta_arr.ndim > 2:
        raise ValueError("Expected eta to be a 1D/2D numpy array")
    elif eta_arr.shape[1] != self.dim:
        raise ValueError(
            f"eta dim. ({eta_arr.shape[1]:d}), no. of columns, is different from set dimension ({self.dim:d})"
        )
    else:  # eta.ndim == 2:
        return self._compute_support_function_multiple_eta(eta_arr)


def convex_set_slice(
    self: "ConstrainedZonotope | Ellipsoid | Polytope",
    dims: int | Sequence[int] | np.ndarray,
    constants: float | Sequence[float] | np.ndarray,
) -> Any:
    """Slice a set restricting certain dimensions to constants.

    This function uses :meth:`intersection_with_affine_set` to implement the slicing by designing an appropriate affine
    set from dims and constants.

    Args:
        dims (Sequence[int] | np.ndarray): List of dims to restrict to a constant in the integer interval
            [0, 1, ..., n - 1].
        constants (float | Sequence[float] | np.ndarray): List of constants

    Raises:
        ValueError: dims has entries beyond n
        ValueError: dims and constants are not 1D arrays of same size

    Returns:
        object: Sliced set.
    """
    dims_arr_raw: np.ndarray = np.atleast_1d(np.squeeze(dims))
    if not np.issubdtype(dims_arr_raw.dtype, np.integer):
        raise ValueError("Expected dims to be finite integers.")
    dims_arr: np.ndarray = np.round(dims_arr_raw).astype(int)
    constants_arr: np.ndarray = np.atleast_1d(constants).astype(float)
    if dims_arr.ndim != 1 or constants_arr.shape != dims_arr.shape:
        raise ValueError("Expected dims and constants to be 1D sequences or numpy arrays of same shape.")
    Ae = np.zeros((len(dims_arr), self.dim))
    be = np.zeros((len(dims_arr),))
    for index, (dim, value) in enumerate(zip(dims_arr, constants_arr)):
        try:
            Ae[index, dim] = 1
        except IndexError as err:
            raise ValueError(f"dims has an entry {dim} that is not in the integer interval [1:{self.dim:d}]") from err
        be[index] = value
    return self.intersection_with_affine_set(Ae, be)


def convex_set_slice_then_projection(
    self: "ConstrainedZonotope | Ellipsoid | Polytope",
    dims: int | Sequence[int] | np.ndarray,
    constants: float | Sequence[float] | np.ndarray,
) -> Any:
    """Wrapper for :meth:`slice` and :meth:`projection`.

    The function first restricts a set at certain dimensions to constants, and then projects away those dimensions.
    Useful for visual inspection of higher dimensional sets.

    Args:
        dims (Sequence[int] | np.ndarray): List of dims to restrict to a constant in the integer interval
            [0, 1, ..., dim - 1], and then project away.
        constants (float | Sequence[float] | np.ndarray): List of constants

    Raises:
        ValueError: dims has entries beyond n
        ValueError: dims and constants are not 1D arrays of same size
        ValueError: When dims are not in the integer interval | All dimensions are projected away

    Returns:
        object: Sliced then projected set.
    """
    return self.slice(dims=dims, constants=constants).projection(project_away_dims=dims)


def _compute_project_multiple_points(  # pyright: ignore[reportUnusedFunction]
    self: "ConstrainedZonotope | Ellipsoid | Polytope", points: Sequence[Sequence[float]] | np.ndarray, p: int | str
) -> tuple[np.ndarray, float]:
    """Private function to project a point on to a set with distance characterized an appropriate p-norm. This function
    is not to be called directly. Instead, call `project` method."""
    import cvxpy as cp

    if p not in [1, 2] and str(p).lower() != "inf":
        raise ValueError(f"Unhandled p norm: {p}!")
    if p == 2:
        cvxpy_args = self.cvxpy_args_socp
    else:
        cvxpy_args = self.cvxpy_args_lp

    points_arr: np.ndarray = np.asarray(points, dtype=float)
    projected_points_on_the_polytope: np.ndarray = np.nan * np.ones((points_arr.shape[0], self.dim), dtype=float)
    projection_distances: np.ndarray = np.inf * np.ones((points_arr.shape[0],), dtype=float)
    point = cp.Parameter((self.dim,))
    x = cp.Variable((self.dim,))
    try:
        containment_constraints, _ = self.containment_constraints(x)
    except ValueError as err:  # pragma: no cover
        raise ValueError("Can not compute project points on to an empty set!") from err
    else:
        problem = cp.Problem(cp.Minimize(cast(cp.Expression, cp.norm(point - x, p=p))), containment_constraints)

        for index, point_row in enumerate(points_arr):
            point.value = point_row
            try:
                problem.solve(**cvxpy_args)
            except cp.error.SolverError as err:  # pyright: ignore[reportAttributeAccessIssue]
                raise NotImplementedError(
                    f"Unable to project the point {np.array2string(np.array(point_row)):s} to the set. "
                    f"CVXPY returned error: {str(err)}"
                ) from err
            if problem.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
                projected_points_on_the_polytope[index] = x.value  # type: ignore[assignment]
                projection_distances[index] = problem.value  # type: ignore[assignment]
            else:
                raise NotImplementedError(
                    f"Unable to project the point {np.array2string(np.array(point_row)):s} to the set. "
                    f"CVXPY returned an unhandled status: {problem.status:s}"
                )
    return projected_points_on_the_polytope, projection_distances


def _compute_support_function_multiple_eta(  # pyright: ignore[reportUnusedFunction]
    self: "ConstrainedZonotope | Polytope", eta: Sequence[float] | np.ndarray
) -> tuple[float, np.ndarray]:
    """Private function to compute the support function of a set along the support direction eta. Instead, call
    `support` method. Relies on the solver to detect unbounded/empty sets."""
    import cvxpy as cp

    eta_arr: np.ndarray = np.asarray(eta, dtype=float)
    support_vector: np.ndarray = np.nan * np.ones((eta_arr.shape[0], self.dim), dtype=float)
    support_value: np.ndarray = np.inf * np.ones((eta_arr.shape[0],), dtype=float)
    eta = cp.Parameter((self.dim,))
    x = cp.Variable((self.dim,))
    try:
        containment_constraints, _ = self.containment_constraints(x)
    except ValueError as err:  # pragma: no cover
        raise ValueError("Can not compute support function for an empty set!") from err
    else:
        problem = cp.Problem(cp.Maximize(eta @ x), containment_constraints)

        for index, eta_row in enumerate(eta_arr):
            eta.value = eta_row
            try:
                problem.solve(**self.cvxpy_args_lp)
            except cp.error.SolverError as err:  # pyright: ignore[reportAttributeAccessIssue]
                raise NotImplementedError(
                    f"Unable to solve the task (support function evaluation of the set at )"
                    f"eta = {np.array2string(np.array(eta_row)):s}). CVXPY returned error: {str(err)}"
                ) from err
            if problem.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
                support_vector[index] = x.value  # type: ignore[assignment]
                support_value[index] = problem.value  # type: ignore[assignment]
            elif problem.status in [cp.UNBOUNDED, cp.UNBOUNDED_INACCURATE]:
                support_vector[index] = np.nan * np.ones((self.dim,), dtype=float)
                support_value[index] = np.inf
            else:
                raise NotImplementedError(
                    f"Could not solve the task (support function evaluation of the set at eta = "
                    f"{np.array2string(np.array(eta_row)):s}), due to an unhandled status: {problem.status:s}."
                )
    return support_value, support_vector


def compute_irredundant_affine_set_using_cdd(
    Ae: Sequence[Sequence[float]] | np.ndarray, be: Sequence[float] | np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Given an affine set (Ae, be), compute an irredundant set (irredundant_Ae, irredundant_be) using cdd

    Args:
        Ae (Sequence[Sequence[float]] | np.ndarray): Equality coefficient matrix (N times self.dim) that define the
            affine set
            :math:`\\{x|A_ex = b_e\\}`.
        be (Sequence[float] | np.ndarray): Equality constant vector (N,) that define the affine set
            :math:`\\{x| A_ex = b_e\\}`.

    Returns:
        tuple: A tuple containing two elements:
            1. irredundant_Ae (numpy.ndarray): Irredundant Ae describing the given affine set :math:`{x|A_e x = b_e}`
            2. irredundant_be (numpy.ndarray): Irredundant be describing the given affine set :math:`{x|A_e x = b_e}`
    """
    Ae_arr: np.ndarray = np.array(Ae, dtype=float)
    be_arr: np.ndarray = np.array(be, dtype=float)
    be_mAe = np.hstack((np.array([be_arr]).T, -Ae_arr))
    He_cdd = cdd.matrix_from_array(be_mAe, lin_set=set(range(len(be_arr))), rep_type=cdd.RepType.INEQUALITY)
    cdd.matrix_canonicalize(He_cdd)
    He_cdd_values: np.ndarray = np.array(He_cdd.array)
    irredundant_be, irredundant__Ae = He_cdd_values[:, 0], -He_cdd_values[:, 1:]
    return irredundant__Ae, irredundant_be


def is_ellipsoid(Q: Any) -> bool:
    """Check if the set is an ellipsoid

    Args:
        Q (object): Set to check

    Returns:
        bool: Returns True if the set is an ellipsoid, False otherwise
    """
    return hasattr(Q, "_type_of_set") and Q.type_of_set == "Ellipsoid"


def is_constrained_zonotope(Q: Any) -> bool:
    """Check if the set is a constrained zonotope

    Args:
        Q (object): Set to check

    Returns:
        bool: Returns True if the set is a constrained zonotope, False otherwise
    """
    return hasattr(Q, "_type_of_set") and Q.type_of_set == "ConstrainedZonotope"


def is_polytope(Q: Any) -> bool:
    """Check if the set is a polytope

    Args:
        Q (object): Set to check

    Returns:
        bool: Returns True if the set is a polytope, False otherwise
    """
    return hasattr(Q, "_type_of_set") and Q.type_of_set == "Polytope"


def make_aspect_ratio_equal(
    set_with_unequal_aspect_ratio: Polytope | Ellipsoid | ConstrainedZonotope,
) -> tuple[Polytope | Ellipsoid | ConstrainedZonotope, np.ndarray, np.ndarray]:
    r"""
    Compute the set whose minimum_volume_circumscribing_rectangle is the unit box, and find the center and scaling
    matrix to undo the affine transformation.

    Args:
        set_with_unequal_aspect_ratio (Polytope | Ellipsoid | ConstrainedZonotope): Set to make aspect ratio equal

    Returns:
        tuple: A tuple with two items
            #. set_with_equal_aspect_ratio (Polytope | Ellipsoid | ConstrainedZonotope): Set with equal aspect ratio.
               Obtained after a specific affine transformation. The minimum_volume_circumscribing_rectangle of
               set_with_equal_aspect_ratio is the unit box.
            #. shift_to_undo_the_affine_transform (numpy.ndarray): Shift for the affine transformation
            #. scaling_matrix_to_undo_the_affine_transform  (numpy.ndarray): Scaling matrix for the affine
               transformation

    Notes:
        The function expects the set to be full-dimensional. Otherwise, scaling_matrix_to_make_aspect_equal may be
        undefined. To recover set_with_unequal_aspect_ratio from set_with_equal_aspect_ratio, perform the affine
        transform `(scaling_matrix_to_undo_the_affine_transform @ set_with_equal_aspect_ratio) +
        shift_to_under_the_affine_transform`.
    """
    lb, ub = set_with_unequal_aspect_ratio.minimum_volume_circumscribing_rectangle()
    if np.min(np.abs(ub - lb)) <= PYCVXSET_ZERO:
        # At least one dimension has zero width in the minimum_volume_circumscribing_rectangle, so the set is not
        # full-dimensional and we can not make aspect ratio equal.
        raise ValueError(f"Expected the set {repr(set_with_unequal_aspect_ratio):s} to be full-dimensional!")
    shift_to_undo_the_affine_transform = (lb + ub) / 2
    scaling_matrix_to_make_aspect_equal = np.diag(2 / (ub - lb))
    scaling_matrix_to_undo_the_affine_transform = np.diag((ub - lb) / 2)
    set_with_equal_aspect_ratio = scaling_matrix_to_make_aspect_equal @ (
        set_with_unequal_aspect_ratio - shift_to_undo_the_affine_transform
    )
    return set_with_equal_aspect_ratio, shift_to_undo_the_affine_transform, scaling_matrix_to_undo_the_affine_transform


def plot_polytopic_approximation(
    self: "ConstrainedZonotope | Ellipsoid",
    method: str = "inner",
    ax: Optional[Axes | Axes3D | None] = None,
    direction_vectors: Optional[Sequence[Sequence[float]] | np.ndarray] = None,
    n_vertices: Optional[int] = None,
    n_halfspaces: Optional[int] = None,
    patch_args: Optional[dict[str, Any]] = None,
    vertex_args: Optional[dict[str, Any]] = None,
    center_args: Optional[dict[str, Any]] = None,
    autoscale_enable: bool = True,
    decimal_precision: int = PLOTTING_DECIMAL_PRECISION_CDD,
    enable_warning: bool = True,
) -> tuple[Any, ...]:
    """Plot a polytopic approximation of the set.

    Args:
        method (str, optional): Type of polytopic approximation to use. Can be ["inner" or "outer"]. Defaults to
            "inner".
        ax (Axes | Axes3D | None, optional): Axis on which the patch is to be plotted
        direction_vectors (Sequence[Sequence[float]] | np.ndarray, optional): Directions to use when performing ray
            shooting. Matrix (N times self.dim) for some N >= 1. Defaults to None, in which case we use
            :meth:`pycvxset.common.spread_points_on_a_unit_sphere` to compute the direction vectors.
        n_vertices (int, optional): Number of vertices to use when computing the polytopic inner-approximation. Ignored
            if method is "outer" or when direction_vectors are provided. More than n_vertices may be used in some cases
            (see notes). Defaults to None.
        n_halfspaces (int, optional): Number of halfspaces to use when computing the polytopic outer-approximation.
            Ignored if method is "outer" or when direction_vectors are provided. More than n_halfspaces may be used in
            some cases (see notes). Defaults to None.
        patch_args (dict, optional): Arguments to pass for plotting faces and edges. See
            :meth:`pycvxset.Polytope.Polytope.plot` for more details. Defaults to None.
        vertex_args (dict, optional): Arguments to pass for plotting vertices. See
            :meth:`pycvxset.Polytope.Polytope.plot` for more details. Defaults to None.
        center_args (dict, optional): For ellipsoidal set, arguments to pass to scatter plot for the center. If a label
            is desired, pass it in center_args.
        autoscale_enable (bool, optional): When set to True, matplotlib adjusts axes to view full polytope. See
            :meth:`pycvxset.Polytope.Polytope.plot` for more details. Defaults to True.
        decimal_precision (int, optional): When plotting a 3D polytope that is in V-Rep and not in H-Rep, we round
            vertex to the specified precision to avoid numerical issues. Defaults to PLOTTING_DECIMAL_PRECISION_CDD
            specified in pycvxset.common.constants.
        enable_warning (bool, optional): Enables the UserWarning. May be turned off if expected. Defaults to True.

    Returns:
        tuple: See :meth:`pycvxset.Polytope.Polytope.plot` for details.

    Notes:
        This function is a wrapper for :meth:`polytopic_inner_approximation` and :meth:`polytopic_outer_approximation`
        for more details in polytope construction.
    """
    if (is_constrained_zonotope(self) or is_polytope(self)) and center_args is not None:
        raise NotImplementedError("center_args is only for ellipsoidal set plotting!")
    if method == "inner":
        polytopic_inner_approximation = self.polytopic_inner_approximation(
            direction_vectors, n_vertices, enable_warning=enable_warning
        )
        tuple_to_return = polytopic_inner_approximation.plot(
            ax=ax,
            patch_args=patch_args,
            vertex_args=vertex_args,
            autoscale_enable=autoscale_enable,
            decimal_precision=decimal_precision,
            enable_warning=enable_warning,
        )
    else:  # "outer" case
        polytopic_outer_approximation = self.polytopic_outer_approximation(
            direction_vectors, n_halfspaces, enable_warning=enable_warning
        )
        tuple_to_return = polytopic_outer_approximation.plot(
            ax=ax,
            patch_args=patch_args,
            vertex_args=vertex_args,
            autoscale_enable=autoscale_enable,
            enable_warning=enable_warning,
        )
    # In case of ellipsoids, plot the center when needed.
    if is_ellipsoid(self) and center_args is not None:
        if "color" not in center_args:
            center_args["color"] = "k"
        ax = cast("Axes | Axes3D", tuple_to_return[0])
        ax.scatter(*cast("Ellipsoid", self).c, **center_args)
    return tuple_to_return


def prune_and_round_vertices(
    V: Sequence[Sequence[float]] | np.ndarray, decimal_precision: int = PLOTTING_DECIMAL_PRECISION_CDD
) -> np.ndarray:
    """Filter through the vertices to skip any point that has another point (down in the list) that is close to it in
    the list.

    Args:
        V (Sequence[Sequence[float]] | np.ndarray): Matrix of vertices (N times self.dim)
        decimal_precision (int, optional): The decimal precision for rounding the vertices. Defaults to
            PLOTTING_DECIMAL_PRECISION_CDD from pycvxset.common.constants.

    Returns:
        numpy.ndarray: The pruned and rounded array of vertices.
    """
    V_arr: np.ndarray = np.asarray(V, dtype=float)
    n_vertices = V_arr.shape[0]
    new_vertex_list: list[np.ndarray] = []
    for ind_1 in range(n_vertices):
        no_point_is_close_to_this_point = True
        for ind_2 in range(ind_1 + 1, n_vertices):
            distances = np.linalg.norm(V_arr[ind_1, :] - V_arr[ind_2, :])
            if distances <= PYCVXSET_ZERO_CDD:
                no_point_is_close_to_this_point = False
                break
        if no_point_is_close_to_this_point:
            new_vertex_list += [V_arr[ind_1, :]]
    new_vertex_list = np.round(np.array(new_vertex_list), decimal_precision)
    return new_vertex_list


def sanitize_Ab(
    A: Sequence[Sequence[float]] | np.ndarray | None, b: Sequence[float] | np.ndarray | None
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Sanitize and check if (`A`, `b`) to make a valid halfspace combination

    Args:
        A (Sequence[Sequence[float]] | np.ndarray | None): Can be numpy arrays, list, or tuples
        b (Sequence[float] | np.ndarray | None): Can be numpy arrays, list, or tuples

    Raises:
        ValueError: A is not 2D numpy array free from NaNs and inf
        ValueError: b is not 1D numpy array free from NaNs

    Returns:
        tuple: A tuple with two items:
            #. A (numpy.ndarray | None): 2D Inequality coefficient matrix. None when no valid rows for A, b.
            #. b (numpy.ndarray | None): 1D Inequality constant vector. None when no valid rows for A, b.

    Notes:
        This function is used in the constructor of Polytope to check if (A, b) is compatible.
        sanitize_Aebe also uses this function remove any NaNs as well check for dimensionality.
    """
    A_value: np.ndarray | None = None
    b_value: np.ndarray | None = None
    try:
        if A is not None:
            A_value = np.atleast_2d(A).astype(float)
    except ValueError as err:
        raise ValueError(f"Can not convert A into a float array. Got {np.array2string(np.array(A)):s}") from err
    try:
        if b is not None:
            b_value = np.atleast_1d(np.squeeze(b)).astype(float)
    except ValueError as err:
        raise ValueError(f"Can not convert b into a float array. Got {np.array2string(np.array(b)):s}") from err
    if (A_value is None or A_value.size == 0) and (b_value is None or b_value.size == 0):
        return None, None
    elif A_value is None or b_value is None:
        raise ValueError("Both A and b must be provided or both must be None.")
    elif A_value.ndim != 2 or b_value.ndim != 1:
        raise ValueError(
            "Expected A, b to be a 2D, 1D arrays! "
            f"Got A: {np.array2string(np.array(A_value)):s} and b: {np.array2string(np.array(b_value)):s}"
        )
    elif np.any(np.isnan(A_value)) or np.any(np.isnan(b_value)):
        raise ValueError(
            "Expected A, b to be from NaNs. "
            f"Got {np.array2string(np.array(A_value)):s}, {np.array2string(np.array(b_value)):s}"
        )
    elif np.any(np.isinf(A_value)):
        raise ValueError(f"Expected A to be from inf. Got {np.array2string(np.array(A_value)):s}!")
    elif A_value.shape[0] != b_value.shape[0] and not (A_value.shape[0] == 1 and b_value.shape[0] == 0):
        raise ValueError(f"A and b has different number of rows! A: {A_value.shape[0]:d} and b: {b_value.shape[0]:d}.")
    return A_value, b_value


def sanitize_Gc(
    G: Sequence[Sequence[float]] | np.ndarray | None, c: Sequence[float] | np.ndarray | None
) -> tuple[np.ndarray, Optional[np.ndarray]]:
    """Sanitize and check if (`G`, `c`) to make a valid zonotope generator combination

    Args:
        G (Sequence[Sequence[float]] | np.ndarray | None): Can be numpy arrays, list, or tuples or None
        c (Sequence[float] | np.ndarray | None): Can be numpy arrays, list, or tuples

    Raises:
        ValueError: G is not 2D numpy array free from NaNs
        ValueError: c is not 2D numpy array free from NaNs
        ValueError: Both G and c cannot be None

    Returns:
        tuple: A tuple with two items:
            # G (numpy.ndarray): 2D generator coefficient matrix.
            # c (numpy.ndarray): 1D center vector.

    Notes:
        This function is used in the constructor of Constrained Zonotope to check if ((G, c) is compatible.
    """
    G_arr: np.ndarray = np.empty((0, 0))
    c_arr: np.ndarray = np.empty((0, 0))
    if G is not None:
        try:
            G_arr = np.atleast_2d(G).astype(float)
        except ValueError as err:
            raise ValueError(f"Expected G to be a 2D float matrix. Got {np.array2string(np.array(G)):s}") from err
        if G_arr.ndim != 2:
            raise ValueError("G is not a two-dimensional array!")
        if c is None:
            # (G not None, c None) Empty set with dimension provided by G
            return np.empty((G_arr.shape[0], 0)), None
    if c is not None:
        try:
            c_arr = np.atleast_1d(np.squeeze(c)).astype(float)
        except ValueError as err:
            raise ValueError(f"Expected c to be a 1D float array. Got {np.array2string(np.array(c)):s}") from err
        if c_arr.ndim != 1:
            raise ValueError("c is not a one-dimensional array!")
        if G is None:
            # (G None, c not None) Singleton set with dimension provided by c
            return np.empty((c_arr.size, 0)), c_arr
    else:
        # (G None, c None)
        raise ValueError("Both G and c cannot be None!")
    if np.isinf(G_arr).any() or np.isinf(c_arr).any() or np.isnan(G_arr).any() or np.isnan(c_arr).any():
        raise ValueError(
            f"Expected G, c to be 2D and 1D array free from NaNs and infs. "
            f"Got {np.array2string(np.array(G_arr)):s}, {np.array2string(np.array(c_arr)):s}"
        )
    elif G_arr.shape[0] != c_arr.shape[0]:
        raise ValueError(f"G and c has different number of rows! G: {G_arr.shape[0]:d} and c: {c_arr.shape[0]:d}.")
    return G_arr, c_arr


def sanitize_Aebe(
    Ae: Sequence[Sequence[float]] | np.ndarray | None,
    be: Sequence[float] | np.ndarray | None,
    enable_warning: bool = True,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Sanitize and return (`Ae`, `be`) to make a valid affine set combination

    Args:
        Ae (Sequence[Sequence[float]] | np.ndarray | None): Equality coefficient matrix specifying the affine set
        be (Sequence[float] | np.ndarray | None): Equality constant vector specifying the affine set
        enable_warning (bool, optional): Enables the UserWarning. May be turned off if expected. Defaults to True.

    Raises:
        ValueError: Ae is not 2D numpy array free from NaNs and infs
        ValueError: be is not 1D numpy array free from NaNs and infs
        ValueError: Provided (Ae, be) is not a valid system of linear equations
        UserWarning: When Ae, be has a row of zeros

    Returns:
        tuple: A tuple with two items:
            #. Ae (numpy.ndarray | None): 2D Equality coefficient matrix. None when no valid rows for Ae, be.
            #. be (numpy.ndarray | None): 1D Equality constant vector. None when no valid rows for Ae, be.

    Notes:
        This function is used in the constructor of Polytope as well as Constrained Zonotope to check if
        (Ae, be) is compatible.
    """
    try:
        Ae, be = sanitize_Ab(Ae, be)
    except ValueError as err:
        raise ValueError(
            f"Invalid linear system (Ae, be)! Got {np.array2string(np.array(Ae)):s}, {np.array2string(np.array(be)):s}"
        ) from err
    if Ae is None or be is None:
        return None, None
    Ae_arr: np.ndarray = Ae
    be_arr: np.ndarray = be
    if np.isinf(be_arr).any():
        raise ValueError(f"Expected be to free from NaNs and infs. Got {np.array2string(np.array(be_arr)):s}")
    else:
        valid_rows_Ae_be = (np.abs(np.hstack((Ae_arr, np.array([be_arr]).T))) > PYCVXSET_ZERO).any(axis=1)
        if sum(valid_rows_Ae_be) == 0:
            return None, None
        else:
            if enable_warning and sum(valid_rows_Ae_be) != Ae_arr.shape[0]:
                warnings.warn("Removed some rows in (Ae, be) that had all zeros!", UserWarning)
            return Ae_arr[valid_rows_Ae_be, :], be_arr[valid_rows_Ae_be]


def sanitize_and_identify_Aebe(
    Ae: Sequence[Sequence[float]] | np.ndarray | None, be: Sequence[float] | np.ndarray | None
) -> tuple[np.ndarray | None, np.ndarray | None, str, np.ndarray | None]:
    """_summary_

    Args:
        Ae (Sequence[Sequence[float]] | np.ndarray): Equality coefficient matrix specifying the affine set after
            sanitize_Aebe
        be (Sequence[float] | np.ndarray): Equality constant vector specifying the affine set after sanitize_Aebe

    Raises:
        ValueError: _description_

    Returns:
        tuple: A tuple with two items:
            #. sanitize_Ae (numpy.ndarray): 2D Equality coefficient matrix. None when no valid rows for Ae, be.
            #. sanitize_be (numpy.ndarray): 1D Equality constant vector. None when no valid rows for Ae, be.
            #. Aebe_status (str): Can be one of ["no_Ae_be", "affine_set", "single_point", "infeasible"]
            #. solution_to_Ae_x_eq_be (numpy.ndarray): 1D feasible solution to the affine set, when it exists. Set to
               None when Aebe_status is no_Ae_be or infeasible or unidentified.
    """
    sanitized_Ae, sanitized_be = sanitize_Aebe(Ae, be)
    if sanitized_Ae is None or sanitized_be is None:
        solution_to_Ae_x_eq_be = None
        Aebe_status = "no_Ae_be"
    else:
        try:
            solution_to_Ae_x_eq_be, _, matrix_rank_Ae, _ = np.linalg.lstsq(sanitized_Ae, sanitized_be, rcond=None)
            max_abs_residual_Ax_x_eq_be = np.max(np.abs(sanitized_be - sanitized_Ae @ solution_to_Ae_x_eq_be))
        except np.linalg.LinAlgError as err:
            raise ValueError("Provided (Ae, be) is not a valid system of linear equations.") from err

        if max_abs_residual_Ax_x_eq_be > PYCVXSET_ZERO:  # At least one equality has large residual
            Aebe_status = "infeasible"
        elif matrix_rank_Ae == sanitized_Ae.shape[1]:
            # There are as many linearly independent equality constraints as dimension. So (Ae, be) is
            # solved by a single point.
            Aebe_status = "single_point"
        else:
            Aebe_status = "affine_set"
    return sanitized_Ae, sanitized_be, Aebe_status, solution_to_Ae_x_eq_be


def minimize(
    self: "ConstrainedZonotope | Ellipsoid | Polytope",
    x: cvxpy.Variable,
    objective_to_minimize: cvxpy.Expression,
    cvxpy_args: dict[str, Any],
    task_str: str = "",
) -> tuple[np.ndarray, float, str]:
    """Solve a convex program with CVXPY objective subject to containment constraints.

    Args:
        x (cvxpy.Variable): CVXPY variable to be optimized
        objective_to_minimize (cvxpy.Expression): CVXPY expression to be minimized
        cvxpy_args (dict): CVXPY arguments to be passed to the solver
        task_str (str, optional): Task string to be used in error messages. Defaults to ''.

    Raises:
        NotImplementedError: Unable to solve problem using CVXPY

    Returns:
        tuple: A tuple with three items:
            #. x.value (numpy.ndarray): Optimal value of x. np.nan * np.ones((self.dim,)) if the problem is not solved.
            #. problem.value (float): Optimal value of the convex program. np.inf if the problem is infeasible, -np.inf
               if problem is unbounded, and finite otherwise.
            #. problem_status (str): Status of the problem

    Notes:
        This function uses :meth:`containment_constraints` to obtain the list of CVXPY expressions that form the
        containment constraints on x.

    Warning:
        Please pay attention to the NotImplementedError generated by this function. It may be possible to get CVXPY to
        solve the same problem by switching the solver. For example, consider the following code block.

        .. code-block:: python

            from pycvxset import Polytope
            P = Polytope(A=[[1, 1], [-1, -1]], b=[1, 1])
            P.cvxpy_args_lp = {'solver': 'CLARABEL'}    # Default solver used in pycvxset
            try:
                print('Is polytope bounded?', P.is_bounded)
            except NotImplementedError as err:
                print(str(err))
            P.cvxpy_args_lp = {'solver': 'OSQP'}
            print('Is polytope bounded?', P.is_bounded)

        This code block produces the following output::

            Unable to solve the task (support function evaluation of the set at eta = [-0. -1.]). CVXPY returned error:
            Solver 'CLARABEL' failed. Try another solver, or solve with verbose=True for more information.

            Is polytope bounded? False

    """
    import cvxpy as cp

    try:
        containment_constraints, _ = self.containment_constraints(x)
    except ValueError:
        return np.nan * np.ones((self.dim,)), np.inf, cp.INFEASIBLE
    problem = cp.Problem(cp.Minimize(objective_to_minimize), containment_constraints)
    try:
        problem.solve(**cvxpy_args)
    except cp.error.SolverError as err:  # pyright: ignore[reportAttributeAccessIssue]
        raise NotImplementedError(f"Unable to solve the task ({task_str:s}). CVXPY returned error: {str(err)}") from err
    if problem.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
        x_value: np.ndarray = x.value  # type: ignore[assignment]
        problem_value: float = problem.value  # type: ignore[assignment]
        return x_value, problem_value, problem.status
    elif problem.status in [cp.UNBOUNDED, cp.UNBOUNDED_INACCURATE]:
        return np.nan * np.ones((self.dim,)), -np.inf, problem.status
    elif problem.status in [cp.INFEASIBLE, cp.INFEASIBLE_INACCURATE]:
        # Infeasible is possible when containment_constraints returns infeasible constraints but without detection
        return np.nan * np.ones((self.dim,)), np.inf, problem.status
    else:
        # Should never happen!
        raise NotImplementedError(
            f"Could not solve the task ({task_str:s}), due to an unhandled status: {problem.status:s}."
        )


def spread_points_on_a_unit_sphere(
    dim: int,
    n_points: Optional[int] = None,
    cvxpy_socp_args: Optional[dict[str, Any]] = None,
    verbose: bool = False,
    enable_warning: bool = True,
    save_points_on_a_unit_sphere: bool = SPOAUS_SAVE_POINTS_ON_A_UNIT_SPHERE,
) -> tuple[np.ndarray, float, np.ndarray]:
    r"""Spread points on a unit sphere in n-dimensional space.

    Args:
        dim (int): The dimension of the sphere.
        n_points (int): The number of points to be spread on the unit sphere. Defaults to None, in which case, we choose
        n_points = (2 * self.dim) + (2**self.dim) * SPOAUS_DIRECTIONS_PER_QUADRANT.
        cvxpy_socp_args (dict): Additional arguments to be passed to the CVXPY solver. Defaults to None, in which case
            the function uses DEFAULT_CVXPY_ARGS_SOCP from pycvxset.common.constants.
        verbose (bool, optional): Whether to print verbose output. Defaults to False.
        enable_warning (bool, optional): Enables the UserWarning. May be turned off if expected. Defaults to True.
        save_points_on_a_unit_sphere (bool, optional): Whether to save the computed points and minimum separation to a
            file. Defaults to SPOAUS_SAVE_POINTS_ON_A_UNIT_SPHERE.

    Returns:
        tuple: A tuple containing three items:
            # opt_locations (ndarray): The spread points on the unit sphere.
            # minimum_separation (float): The minimum separation between the points.
            # opt_locations_first_quad (ndarray): The spread points in the first quadrant.

    Raises:
        ValueError: If n_points is less than 2 * dim.
        UserWarning: If n_points - 2 * dim is not a multiple of 2^dim.
        NotImplementedError: Unable to solve the convexified problem using CVXPY
        NotImplementedError: Convex-concave procedure did not converge!

    Notes:
        This function uses the CVXPY library to solve a convex optimization problem to spread the points on the unit
        sphere. The spread points are returned as opt_locations, the separation between the points is returned as
        separation, and the spread points in the first quadrant are returned as opt_locations_first_quad.

        For dim in [1, 2], the points are available in closed-form. For dim >= 3, we solve the following non-convex
        optimization problem using convex-concave procedure:

        .. math ::
            \begin{array}{rlrl}
                \text{maximize}     &\quad R \\
                \text{subject to}   &\quad R \geq 0\\
                                    &\quad x \geq R/2\\
                                    &\quad \|x_i - x_j\| \geq R, && 1 \leq i < j \leq n_\text{points}\\
                                    &\quad \|x_i - e_j\| \geq R, && 1 \leq i \leq n_\text{points},
                                                                    1 \leq j \leq n_\text{dim}\\
                                    &\quad \|x_i \| \leq 1,      && 1 \leq i \leq n_\text{points}\\
                                    &\quad \|x_i \| \geq 0.8,    && 1 \leq i \leq n_\text{points}\\
            \end{array}

        The optimization problem seeks to spread points (apart from the standard axes) in the first quadrant so that
        their pairwise separation is maximized, while they have a norm close to 1. The second constraint is motivated to
        ensure that the reflections of the points about the quadrant plane is also separated by R.

        When `save_points_on_a_unit_sphere` is True, the function saves the computed points (all), the computed
        points in the first quadrant, and the minimum separation to a file named
        `tmp/spoaus_dim_{dim}_n_points_{n_points_int}.npz`. If the file already exists, the function loads the
        points and minimum separation from the file instead of recomputing them. This can save time when calling
        `spread_points_on_a_unit_sphere` with default values.
    """
    import cvxpy as cp

    if n_points is None:
        n_points_int = (2 * dim) + (2**dim) * SPOAUS_DIRECTIONS_PER_QUADRANT
    elif n_points <= 0:
        raise ValueError(f"Expected n_points to be positive! Got {n_points}!")
    else:
        n_points_int = n_points
    if dim == 1:
        opt_locations = np.array([[1], [-1]])
        opt_locations_first_quad = np.array([[1]])
        minimum_separation = 2.0
    elif dim == 2:
        theta_vec = np.linspace(0, 2 * np.pi, n_points_int + 1)[:-1]
        opt_locations = np.vstack((np.cos(theta_vec), np.sin(theta_vec))).T
        minimum_separation = float(np.linalg.norm(opt_locations[-2] - opt_locations[-1], ord=2))
        opt_locations_first_quad = opt_locations[theta_vec <= (np.pi / 2), :]
    else:
        # Ensure that n_points - (2 * dim) to be a multiple of 2**dim
        if n_points_int < 2 * dim:
            # Need to allow at least the standard vectors and their reflections
            raise ValueError("Expected n_points >= 2*dim")
        elif (
            cvxpy_socp_args is not None
            and "solver" in cvxpy_socp_args
            and cvxpy_socp_args["solver"] not in cp.installed_solvers()
        ):
            raise ValueError(f"Solver {cvxpy_socp_args['solver']:s} is not installed!")
        opt_locations = np.vstack((np.eye(dim), -np.eye(dim)))
        # Guaranteed to be positive since n_points_int >= 2 * dim
        n_points_first_quad_old = (n_points_int - 2 * dim) / (2**dim)
        n_points_old = n_points_int
        n_points_first_quad = int(np.ceil(n_points_first_quad_old))
        n_points_int = (2**dim) * n_points_first_quad + 2 * dim
        if enable_warning and not np.isclose(n_points_first_quad_old, n_points_first_quad):
            warnings.warn(
                f"Invalid combination of (dim={dim}, n_points={n_points_old}). "
                f"spread_points_on_a_unit_sphere requires n_points - 2*dim "
                f"(Got: {n_points_old - (2 * dim):d}) to be a multiple of 2^dim"
                f"(Got: {2 ** dim:d}). Will return {n_points_int} points instead of requested {n_points_old} "
                f"points.",
                UserWarning,
            )
        if save_points_on_a_unit_sphere:  # pragma: no cover
            try:
                os.makedirs("tmp", exist_ok=True)
            except PermissionError as err:
                raise PermissionError(
                    "Cannot create tmp/ — no write permissions. Set save_points_on_a_unit_sphere to False or ensure "
                    "write permissions."
                ) from err
            save_filename = f"tmp/spoaus_dim_{dim}_n_points_{n_points_int}.npz"
        else:  # pragma: no cover
            save_filename = ""

        if n_points_first_quad == 0:
            # Return the standard axes and their reflections
            opt_locations_first_quad = np.eye(dim)
            minimum_separation = float(np.sqrt(2))
            if verbose:
                print("Skipped spacing vectors! Returned the standard axes!")
        elif save_points_on_a_unit_sphere and os.path.exists(save_filename):
            loaded_data = np.load(save_filename)
            opt_locations = loaded_data["opt_locations"]
            minimum_separation = float(loaded_data["minimum_separation"])
            opt_locations_first_quad = loaded_data["opt_locations_first_quad"]
            if verbose:
                print(
                    f"Loaded pre-computed spread points on a unit sphere for dim={dim} and n_points={n_points_int}!"
                    " This saves time. If you want to re-run the convex-concave procedure, please delete the file "
                    f"{save_filename}, and run the function again."
                )
            return opt_locations, minimum_separation, opt_locations_first_quad
        else:
            if verbose:
                print(f"Spreading {n_points_int} unit-length vectors in {dim}-dim space")
                print(f"Analyzing {n_points_first_quad} unit-length vectors in first quadrant")

            if cvxpy_socp_args is None:
                cvxpy_socp_args = DEFAULT_CVXPY_ARGS_SOCP

            # Difference of convex approach
            # INITIAL_TAU times SCALING_TAU ^ MAX_ITERATIONS = TAU_MAX
            MAX_ITERATIONS = SPOAUS_ITERATIONS_AT_TAU_MAX + int(
                np.log(SPOAUS_TAU_MAX) / (np.log(SPOAUS_SCALING_TAU) + np.log(SPOAUS_INITIAL_TAU))
            )

            # Initialize counter for iterations
            continue_condition = True
            tau_iter = SPOAUS_INITIAL_TAU
            iter_count = 1

            # Initialize vectors by assigning all along the unit-norm vector equidistant from all standard vectors
            x_iter = np.ones((n_points_first_quad, dim)) / np.sqrt(dim)

            # For DC approach, initialize the previous costs
            separation_prev = 0.0
            sum_slack_var_prev = 0.0
            objective_value_prev = 0.0
            cost_prev_with_new_tau_iter = 0.0
            while continue_condition:
                if verbose:
                    print(f"{iter_count}/{MAX_ITERATIONS}. Setting up the CVXPY problem...")

                x = cp.Variable((n_points_first_quad, dim))
                separation = cp.Variable(nonneg=True)
                pairwise_separation_slack_var = cp.Variable((n_points_first_quad, n_points_first_quad), nonneg=True)
                standard_vector_separation_slack_var = cp.Variable((n_points_first_quad, dim), nonneg=True)
                minimum_norm_slack_var = cp.Variable((n_points_first_quad,), nonneg=True)
                # Constraint 1: Enforces the separation constraint is satisfied by reflections about quadrant planes
                constraints: list[cp.Constraint] = [
                    np.power(2, 1 / dim) * x >= separation,
                ]
                # Constraint 2: Set the unused slack variables to zero
                constraints += [pairwise_separation_slack_var[np.triu_indices(n_points_first_quad, -1)] == 0]
                for pt_index_1 in range(n_points_first_quad):
                    if verbose:
                        print(f"{pt_index_1:2d} | ", end="")
                        if ((pt_index_1 - 9) % 10 == 0) or pt_index_1 == n_points_first_quad - 1:
                            print("")

                    # Constraint 3: Enforces the pairwise separation constraint ||x_i - x_j||^2 >= r^2
                    x_i_prev = x_iter[pt_index_1, :]
                    for pt_index_2 in range(pt_index_1):
                        x_j_prev = x_iter[pt_index_2, :]
                        constraints.append(
                            (x_i_prev - x_j_prev) @ (x_i_prev - x_j_prev)
                            + 2 * ((x_i_prev - x_j_prev) @ (x[pt_index_1, :] - x_i_prev))
                            - 2 * ((x_i_prev - x_j_prev) @ (x[pt_index_2, :] - x_j_prev))
                            + pairwise_separation_slack_var[pt_index_1, pt_index_2]
                            >= cp.square(separation)
                        )

                    # Constraint 4: Enforces the pairwise separation constraint ||x_i - e_j||^2 >= r^2
                    for dim_index in range(dim):
                        e_i_vector = np.zeros(dim)
                        e_i_vector[dim_index] = 1
                        constraints.append(
                            (x_i_prev - e_i_vector) @ (x_i_prev - e_i_vector)
                            + 2 * (x_i_prev - e_i_vector) @ (x[pt_index_1, :] - x_i_prev)
                            + standard_vector_separation_slack_var[pt_index_1, dim_index]
                            >= cp.square(separation)
                        )

                    # Constraint 5: Enforces the maximum norm constraint ||x_i ||_2 <= 1
                    constraints.append(cast(cp.Constraint, cp.norm(x[pt_index_1, :], p=2) <= 1))

                    # Constraint 6: Enforces the minimum norm constraint ||x_i ||_2 >= 0.8
                    constraints.append(
                        x_i_prev @ x_i_prev
                        + 2 * x_i_prev @ (x[pt_index_1, :] - x_i_prev)
                        + minimum_norm_slack_var[pt_index_1]
                        >= SPOAUS_MINIMUM_NORM_VALUE_SQR
                    )

                if verbose:
                    print("Solving the CVXPY problem...", end="")

                sum_slack_var = cast(
                    cp.Expression,
                    cp.sum(minimum_norm_slack_var)
                    + cp.sum(standard_vector_separation_slack_var)
                    + cp.sum(pairwise_separation_slack_var),
                )
                objective = -separation + tau_iter * sum_slack_var

                problem = cp.Problem(cp.Minimize(objective), constraints)

                try:
                    problem.solve(**cvxpy_socp_args)
                except cp.error.SolverError as err:  # pyright: ignore[reportAttributeAccessIssue]
                    raise NotImplementedError(
                        f"Unable to spread points on a unit sphere! CVXPY returned error: {str(err)}"
                    ) from err

                if problem.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
                    cost_prev_with_new_tau_iter = -separation_prev + tau_iter * sum_slack_var_prev
                    sum_slack_value = cast(float, sum_slack_var.value)
                    separation_value = cast(float, separation.value)
                    objective_value = cast(float, objective.value)
                    x_value = cast(np.ndarray, x.value)
                    if verbose:
                        print("done! Successfully solved.")
                        print(f"Tau: {tau_iter:1.2f} (< {SPOAUS_TAU_MAX:1.2f})")
                        print(f"Sum of slack: {sum_slack_value:.3e} (< {SPOAUS_SLACK_TOLERANCE:.3e})")
                        print(
                            f"Change in opt cost: {np.abs(objective_value - cost_prev_with_new_tau_iter):.3e} "
                            f"(< {SPOAUS_COST_TOLERANCE:.3e})"
                        )
                        print(f"Max. change in ||x_i||_2: {max(np.linalg.norm(x_value - x_iter, ord=2, axis=1)):.3e}\n")
                    continue_condition = (
                        sum_slack_value > SPOAUS_SLACK_TOLERANCE
                        or np.abs(objective_value - cost_prev_with_new_tau_iter) > SPOAUS_COST_TOLERANCE
                    ) and iter_count < MAX_ITERATIONS
                    # Update the values for the next iteration
                    x_iter = x_value
                    objective_value_prev = objective_value
                    separation_prev = separation_value
                    sum_slack_var_prev = sum_slack_value
                    iter_count += 1
                    tau_iter = min(tau_iter * SPOAUS_SCALING_TAU, SPOAUS_TAU_MAX)
                else:
                    raise NotImplementedError(
                        f"Use of slack variables should have prevented problem status: {problem.status}"
                    )

            if (
                iter_count >= MAX_ITERATIONS
                or sum_slack_var_prev > SPOAUS_SLACK_TOLERANCE
                or np.abs(objective_value_prev - cost_prev_with_new_tau_iter) > SPOAUS_COST_TOLERANCE
            ):
                raise NotImplementedError(
                    f"Difference of convex programming did not converge while spreading {n_points_int} points on a "
                    f"{dim}-dimensional unit sphere! The convex-concave procedure parameters may need tuning.\n"
                    f"iter_count >= {MAX_ITERATIONS:d}: {iter_count >= MAX_ITERATIONS}\n"
                    f"sum_slack_var.value > {SPOAUS_SLACK_TOLERANCE:1.2f}: "
                    f"{sum_slack_var_prev > SPOAUS_SLACK_TOLERANCE}\n"
                    f"np.abs(objective.value - cost_prev_with_new_tau_iter) > {SPOAUS_COST_TOLERANCE:1.2f}: "
                    f"{np.abs(objective_value_prev - cost_prev_with_new_tau_iter) > SPOAUS_COST_TOLERANCE}"
                )
            norm_val = np.linalg.norm(x_iter, ord=2, axis=1, keepdims=True)
            opt_locations_first_quad_without_axes = x_iter / norm_val
            for sign_vector in itertools.product([-1, 1], repeat=dim):
                opt_locations = np.vstack((opt_locations, sign_vector * opt_locations_first_quad_without_axes))
            opt_locations_first_quad = np.vstack((np.eye(dim), opt_locations_first_quad_without_axes))
            minimum_separation = separation_prev

            if verbose:
                print("Completed spreading the vectors!")
                print(f"Minimum separation among the points: {minimum_separation:1.4f}")
            if save_points_on_a_unit_sphere:  # pragma: no cover
                np.savez(
                    save_filename,
                    opt_locations=opt_locations,
                    minimum_separation=minimum_separation,
                    opt_locations_first_quad=opt_locations_first_quad,
                )
                if verbose:
                    print(f"Saved the computed points on a unit sphere for dim={dim} and n_points={n_points_int:d}")
    return opt_locations, minimum_separation, opt_locations_first_quad
