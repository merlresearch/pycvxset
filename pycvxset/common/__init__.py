# Copyright (C) 2020-2025 Mitsubishi Electric Research Laboratories (MERL)
# Copyright (c) 2019 Tor Aksel N. Heirung
#
# SPDX-License-Identifier: AGPL-3.0-or-later
# SPDX-License-Identifier: MIT

# Code purpose: Describe various constants and methods that are common to different set representations.
# Coverage: This file has 2 untested statements to handle errors from np.linalg.lstsq.

import itertools
import warnings

import cvxpy as cp
import numpy as np

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
    SPOAUS_SCALING_TAU,
    SPOAUS_SLACK_TOLERANCE,
    SPOAUS_TAU_MAX,
)

DOCSTRING_FOR_PROJECT = ""


def approximate_volume_from_grid(set_to_compute_area_for, area_grid_step_size):
    r"""Estimate area of a two-dimensional set using a grid of given step size

    Args:
        set_to_compute_area_for (Polytope | ConstrainedZonotope | Ellipsoid): Set for which area is to be computed
        area_grid_step_size (float): Scalar step size that is constant in both dimensions

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
    if set_to_compute_area_for.dim != 2:
        raise ValueError("Expected 2-dimensional set")
    area_grid_step_size = np.squeeze(area_grid_step_size).astype(float)
    if area_grid_step_size.ndim != 0 or area_grid_step_size <= 0:
        raise ValueError("Expected area_grid_step_size to be a positive float!")

    try:
        lb, ub = set_to_compute_area_for.minimum_volume_circumscribing_rectangle()
    except ValueError:
        return 0.0  # Empty set
    if (lb > ub).any() or np.max(np.abs(ub - lb)) <= PYCVXSET_ZERO:
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
        containment_flag = set_to_compute_area_for.contains(points_to_test)
        set_area = float(np.count_nonzero(containment_flag) / n_area_boxes * area_of_bounding_box)
        return set_area


def check_matrices_are_equal_ignoring_row_order(A, B):
    """Check matrices are equal while ignoring row order

    Args:
        A (array_like): Matrix 1
        B (array_like): Matrix 2

    Returns:
        bool: A == B

    Notes:
        isclose does element-wise comparison, all with axis=1, provides a row-wise test, and finally any checks for some
        row where row-wise match is true
    """
    A = np.array(A).astype(float)
    B = np.array(B).astype(float)
    return A.shape == B.shape and sum([np.any(np.all(np.isclose(row, B), axis=1)) for row in A]) == B.shape[0]


def check_vectors_are_equal_ignoring_row_order(A, B):
    """Check vectors are equal while ignoring row order

    Args:
        A (array_like): Vector 1
        B (array_like): Vector 2

    Returns:
        bool: A == B

    Notes:
        isclose does element-wise comparison and we sort
    """
    A = np.squeeze(A).astype(float)
    B = np.squeeze(B).astype(float)
    return A.ndim == 1 and A.shape == B.shape and np.all(np.isclose(np.sort(A), np.sort(B)))


def convex_set_closest_point(self, points, p=2):
    """Wrapper for :meth:`project` to compute the point in the convex set closest to the given point.

    Args:
        points (array_like): Points to project. Matrix (N times self.dim), where each row is a point.
        p (str | int): Norm-type. It can be 1, 2, or 'inf'. Defaults to 2.

    Returns:
        numpy.ndarray: Projection of points to the set as a 2D numpy.ndarray. These arrays have as many rows as points.

    Notes:
        For more detailed description, see documentation for :meth:`project` function.
    """
    return self.project(points, p=p)[0]


def convex_set_contains_points(self, points):
    """Wrapper for :meth:`distance` to compute containment from a given collection of points.

    Args:
        points (array_like): Points to project. Matrix (N times self.dim), where each row is a point.

    Returns:
        bool | numpy.ndarray[bool]: Logical array of containment of points in the set.

    Notes:
        For more detailed description, see documentation for :meth:`distance` function.
    """
    points = np.atleast_2d(points).astype(float)
    n_points, point_dim = points.shape
    if self.is_empty and n_points > 1:
        return np.zeros((n_points,), dtype="bool")
    elif self.is_empty and n_points == 1:
        return False
    elif point_dim != self.dim:
        # Could transpose here if test_points.shape[1] == self.dim, but better to
        # specify that points must be columns:
        raise ValueError(f"Mismatch in dimensions (self.dim: {self.dim:d} and test_points.dim: {point_dim:d})")
    distance_to_test_points = self.distance(points, p="inf")
    if n_points > 1:
        return distance_to_test_points <= PYCVXSET_ZERO
    else:
        return distance_to_test_points[0] <= PYCVXSET_ZERO


def convex_set_distance(self, points, p=2):
    """Wrapper for :meth:`project` to compute distance of a point to a convex set.

    Args:
        points (array_like): Points to project. Matrix (N times self.dim), where each row is a point.
        p (int | str): Norm-type. It can be 1, 2, or 'inf'. Defaults to 2.

    Returns:
        numpy.ndarray: Distance of points to the set as a 1D numpy.ndarray. These arrays have as many rows as points.

    Notes:
        For more detailed description, see documentation for :meth:`project` function.
    """
    return self.project(points, p=p)[1]


def convex_set_extreme(self, eta):
    """Wrapper for :meth:`support` to compute the extreme point.

    Args:
        eta (array_like): Support directions. Matrix (N times self.dim), where each row is a support direction.

    Returns:
        numpy.ndarray: Support vector evaluation(s) as a 2D numpy.ndarray. The array has as many rows as eta.

    Notes:
        For more detailed description, see documentation for :meth:`support` function.
    """
    return self.support(eta)[1]


def convex_set_project(cvx_set, points, p=2):
    r"""Project a point or a collection of points on to a set.

    Given a set :math:`\mathcal{P}` and a test point :math:`y\in\mathbb{R}^{\mathcal{P}.\text{dim}}`, this function
    solves a convex program,

        .. math::
            \text{minimize}    &\quad  \|x - y\|_p\\
            \text{subject to}  &\quad  x \in \mathcal{P}\\

    Args:
        points (array_like): Points to project. Matrix (N times self.dim), where each row is a point.
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
        points = np.atleast_2d(points).astype(float)
        if points.ndim > 2:
            raise ValueError("Expected points to be a 1D/2D numpy array")
        else:  # points.ndim == 2
            project_evaluation_list = []
            distance_evaluation_list = []
            if points.shape[1] != cvx_set.dim:
                raise ValueError(
                    f"points dim. ({points.shape[1]:d}), no. of columns, is different from set dim. ({cvx_set.dim:d})"
                )
            for point in points:
                project_evaluation, distance_evaluation = cvx_set._compute_project_single_point(point, p)
                project_evaluation_list.append(project_evaluation)
                distance_evaluation_list.append(distance_evaluation)
            return np.array(project_evaluation_list), np.array(distance_evaluation_list)


def convex_set_projection(self, project_away_dim):
    r"""Orthogonal projection of a set :math:`\mathcal{P}` after removing some user-specified dimensions.

    .. math::
        \mathcal{R} = \{r \in \mathbb{R}^{m}\ |\  \exists v \in \mathbb{R}^{n - m},\ \text{Lift}(r,v)\in \mathcal{P}\}

    Here, :math:`m = \mathcal{P}.\text{dim} - \text{length}(\text{project\_away\_dim})`, and
    :math:`\text{Lift}(r,v)` lifts ("undo"s the projection) using the appropriate components of `v`. This function uses
    :meth:`affine_map` to implement the projection by designing an appropriate affine map :math:`M \in
    \{0,1\}^{m\times\mathcal{P}.\text{dim}}` with each row of :math:`M` corresponding to some standard axis vector
    :math:`e_i\in\mathbb{R}^m`.

    Args:
        project_away_dim (array_like): Dimensions to projected away in integer interval [0, 1, ..., n - 1].
    """
    project_away_dim = np.atleast_1d(np.squeeze(project_away_dim)).astype(float)
    if project_away_dim.size == 0:
        return self.copy()
    else:
        if np.min(project_away_dim) < 0 or np.max(project_away_dim) >= self.dim:
            raise ValueError(
                f"Expected project_away_dim to be in the integer interval [0:{self.dim-1:d}]!"
                f"Got {np.array2string(np.array(project_away_dim)):s}"
            )
        corrected_retain_dimensions = [d for d in range(self.dim) if d not in project_away_dim]
        n_dimensions_to_retain = len(corrected_retain_dimensions)
        if n_dimensions_to_retain == 0:
            raise ValueError("Can not project away all dimensions!")
        else:
            # Perform the projection using projection_matrix @ V
            projection_matrix = np.zeros((n_dimensions_to_retain, self.dim))
            for row_index, d in enumerate(corrected_retain_dimensions):
                projection_matrix[row_index, d] = 1
            return projection_matrix @ self


def convex_set_support(self, eta):
    r"""Evaluates the support function and support vector of a set.

    The support function of a set :math:`\mathcal{P}` is defined as :math:`\rho_{\mathcal{P}}(\eta) =
    \max_{x\in\mathcal{P}} \eta^\top x`. The support vector of a set :math:`\mathcal{P}` is defined as
    :math:`\nu_{\mathcal{P}}(\eta) = \arg\max_{x\in\mathcal{P}} \eta^\top x`.

    Args:
        eta (array_like): Support directions. Matrix (N times self.dim), where each row is a support direction.

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
    if self.is_empty:
        raise ValueError("Set must be non-empty for support function evaluation.")
    else:
        eta = np.atleast_2d(eta).astype(float)
        if eta.ndim > 2:
            raise ValueError("Expected eta to be a 1D/2D numpy array")
        elif eta.shape[1] != self.dim:
            raise ValueError(
                f"eta dim. ({eta.shape[1]:d}), no. of columns, is different from set dimension ({self.dim:d})"
            )
        else:  # eta.ndim == 2:
            support_function_list = []
            support_vector_list = []
            for single_eta in eta:
                support_function, support_vector = self._compute_support_function_single_eta(single_eta)
                support_function_list.append(support_function)
                support_vector_list.append(support_vector)
            return np.array(support_function_list), np.array(support_vector_list)


def convex_set_slice(self, dims, constants):
    """Slice a set restricting certain dimensions to constants.

    Args:
        dims (array_like): List of dims to restrict to a constant in the integer interval [0, 1, ..., n - 1].
        constants (array_like): List of constants

    Raises:
        ValueError: dims has entries beyond n
        ValueError: dims and constants are not 1D arrays of same size
    """
    dims = np.atleast_1d(dims)  # skip astype since we want an IndexError when enumerating over non-integers
    constants = np.atleast_1d(constants).astype(float)
    if dims.ndim != 1 or constants.shape != dims.shape:
        raise ValueError("Expected dims and constants to be 1D array_like of same shape.")
    Ae = np.zeros((len(dims), self.dim))
    be = np.zeros((len(dims),))
    for index, (dim, value) in enumerate(zip(dims, constants)):
        try:
            Ae[index, dim] = 1
        except IndexError as err:
            raise ValueError(f"dims has an entry {dim} that is not in the integer interval [1:{self.dim:d}]") from err
        be[index] = value
    return self.intersection_with_affine_set(Ae, be)


def _compute_project_single_point(self, point, p):
    """Private function to project a point on to a set with distance characterized an appropriate p-norm. This function
    is not to be called directly. Instead, call `project` method. This functions uses :meth:`minimize`."""
    if p not in [1, 2] and str(p).lower() != "inf":
        raise ValueError(f"Unhandled p norm: {p}!")
    if p == 2:
        cvxpy_args = self.cvxpy_args_socp
    else:
        cvxpy_args = self.cvxpy_args_lp
    x = cp.Variable((self.dim,))
    projected_point_on_the_polytope, projection_distance, _ = self.minimize(
        x,
        objective_to_minimize=cp.norm(point - x, p=p),
        cvxpy_args=cvxpy_args,
        task_str=f"project the point {np.array2string(np.array(point)):s} on to the set",
    )
    return projected_point_on_the_polytope, projection_distance


def _compute_support_function_single_eta(self, eta):
    """Private function to compute the support function of a set along the support direction eta. Instead, call
    `support` method. This functions uses :meth:`minimize`."""
    x = cp.Variable((self.dim,))
    support_vector, negative_support_function_evaluation, _ = self.minimize(
        x,
        objective_to_minimize=-eta @ x,
        cvxpy_args=self.cvxpy_args_lp,
        task_str=f"support function evaluation of the set at eta = {np.array2string(np.array(eta)):s}",
    )
    return -negative_support_function_evaluation, support_vector


def is_ellipsoid(Q):
    """Check if the set is an ellipsoid

    Args:
        Q (object): Set to check

    Returns:
        bool: Returns True if the set is an ellipsoid, False otherwise
    """
    return hasattr(Q, "inflate")


def is_constrained_zonotope(Q):
    """Check if the set is a constrained zonotope

    Args:
        Q (object): Set to check

    Returns:
        bool: Returns True if the set is a constrained zonotope, False otherwise
    """
    return hasattr(Q, "is_zonotope")


def is_polytope(Q):
    """Check if the set is a polytope

    Args:
        Q (object): Set to check

    Returns:
        bool: Returns True if the set is a polytope, False otherwise
    """
    return hasattr(Q, "in_H_rep")


def plot(
    self,
    method="inner",
    ax=None,
    direction_vectors=None,
    n_vertices=None,
    n_halfspaces=None,
    patch_args=None,
    vertex_args=None,
    center_args=None,
    autoscale_enable=True,
    decimal_precision=PLOTTING_DECIMAL_PRECISION_CDD,
):
    """Plot a polytopic approximation of the set.

    Args:
        method (str, optional): Type of polytopic approximation to use. Can be ["inner" or "outer"]. Defaults to
            "inner".
        ax (axis object, optional): Axis on which the patch is to be plotted
        direction_vectors (array_like, optional): Directions to use when performing ray shooting. Matrix (N times
            self.dim) for some N >= 1. Defaults to None, in which case we use
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

    Returns:
        tuple: See :meth:`pycvxset.Polytope.Polytope.plot` for details.

    Notes:
        This function is a wrapper for :meth:`polytopic_inner_approximation` and :meth:`polytopic_outer_approximation`
        for more details in polytope construction.
    """
    if (is_constrained_zonotope(self) or is_polytope(self)) and center_args is not None:
        raise NotImplementedError("center_args is only for ellipsoidal set plotting!")
    if method == "inner":
        polytopic_inner_approximation = self.polytopic_inner_approximation(direction_vectors, n_vertices)
        tuple_to_return = polytopic_inner_approximation.plot(
            ax=ax,
            patch_args=patch_args,
            vertex_args=vertex_args,
            autoscale_enable=autoscale_enable,
            decimal_precision=decimal_precision,
        )
    else:  # "outer" case
        polytopic_outer_approximation = self.polytopic_outer_approximation(direction_vectors, n_halfspaces)
        tuple_to_return = polytopic_outer_approximation.plot(
            ax=ax, patch_args=patch_args, vertex_args=vertex_args, autoscale_enable=autoscale_enable
        )
    # In case of ellipsoids, plot the center when needed.
    if is_ellipsoid(self) and center_args is not None:
        if "color" not in center_args:
            center_args["color"] = "k"
        ax = tuple_to_return[0]
        ax.scatter(*self.c, **center_args)
    return tuple_to_return


def prune_and_round_vertices(V, decimal_precision=PLOTTING_DECIMAL_PRECISION_CDD):
    """Filter through the vertices to skip any point that has another point (down in the list) that is close to it in
    the list.

    Args:
        V (array_like): Matrix of vertices (N times self.dim)
        decimal_precision (int, optional): The decimal precision for rounding the vertices. Defaults to
            PLOTTING_DECIMAL_PRECISION_CDD from pycvxset.common.constants.

    Returns:
        numpy.ndarray: The pruned and rounded array of vertices.
    """
    n_vertices = V.shape[0]
    new_vertex_list = []
    for ind_1 in range(n_vertices):
        found_at_least_one_point_in_future_that_is_close_to_this_point = False
        for ind_2 in range(ind_1 + 1, n_vertices):
            distances = np.linalg.norm(V[ind_1, :] - V[ind_2, :])
            if distances <= PYCVXSET_ZERO_CDD:
                found_at_least_one_point_in_future_that_is_close_to_this_point = True
                break
        if not found_at_least_one_point_in_future_that_is_close_to_this_point:
            new_vertex_list += [V[ind_1, :]]
    # Last vertex is always skipped, so add it. This also ensures that the new_vertex_list is never empty.
    new_vertex_list += [V[-1, :]]
    new_vertex_list = np.round(np.array(new_vertex_list), decimal_precision)
    return new_vertex_list


def sanitize_Ab(A, b):
    """Sanitize and check if (`A`, `b`) to make a valid halfspace combination

    Args:
        A (array_like): Can be numpy arrays, list, or tuples
        b (array_like): Can be numpy arrays, list, or tuples

    Raises:
        ValueError: A is not 2D numpy array free from NaNs and inf
        ValueError: b is not 1D numpy array free from NaNs

    Returns:
        (numpy.ndarray, numpy.ndarray): 2D numpy arrays that is sanitized for `A`, and 1D numpy array that is sanitized
            for `b`

    Notes:
        This function is used in the constructor of Polytope to check if (A, b) is compatible.
        sanitize_Aebe also uses this function remove any NaNs as well check for dimensionality.
    """
    try:
        A = np.atleast_2d(A).astype(float)
    except ValueError as err:
        raise ValueError(f"Can not convert A into a float array. Got {np.array2string(np.array(A)):s}") from err
    try:
        b = np.atleast_1d(np.squeeze(b)).astype(float)
    except ValueError as err:
        raise ValueError(f"Can not convert b into a float array. Got {np.array2string(np.array(b)):s}") from err
    if A.ndim != 2 or b.ndim != 1:
        raise ValueError(
            "Expected A, b to be a 2D, 1D arrays! "
            f"Got A: {np.array2string(np.array(A)):s} and b: {np.array2string(np.array(b)):s}"
        )
    elif np.any(np.isnan(A)) or np.any(np.isnan(b)):
        raise ValueError(
            f"Expected A, b to be from NaNs. Got {np.array2string(np.array(A)):s}, {np.array2string(np.array(b)):s}"
        )
    elif np.any(np.isinf(A)):
        raise ValueError(f"Expected A to be from inf. Got {np.array2string(np.array(A)):s}!")
    elif A.shape[0] != b.shape[0] and not (A.shape[0] == 1 and b.shape[0] == 0):
        raise ValueError(f"A and b has different number of rows! A: {A.shape[0]:d} and b: {b.shape[0]:d}.")
    return A, b


def sanitize_Gc(G, c):
    """Sanitize and check if (`G`, `c`) to make a valid zonotope generator combination

    Args:
        G (array_like): Can be numpy arrays, list, or tuples or None
        c (array_like): Can be numpy arrays, list, or tuples

    Raises:
        ValueError: G is not 2D numpy array free from NaNs
        ValueError: c is not 2D numpy array free from NaNs

    Returns:
        tuple: A tuple with two items:
            # G (numpy.ndarray): 2D generator coefficient matrix.
            # c (numpy.ndarray): 1D center vector.

    Notes:
        This function is used in the constructor of Constrained Zonotope to check if ((G, c) is compatible.
    """
    if G is not None:
        try:
            G = np.atleast_2d(G).astype(float)
            if G.ndim != 2:
                raise ValueError("G is not a two-dimensional array!")
        except ValueError as err:
            raise ValueError(f"Expected G to be a 2D float matrix. Got {np.array2string(np.array(G)):s}") from err
        if c is None:
            # (G not None, c None) Empty set with dimension provided by G
            return np.empty((G.shape[0], 0)), None
    # At this point, either G is None | (c is not None and G is not None)
    if c is not None:
        try:
            c = np.atleast_1d(np.squeeze(c)).astype(float)
            if c.ndim != 1:
                raise ValueError("c is not a one-dimensional array!")
        except ValueError as err:
            raise ValueError(f"Expected c to be a 1D float array. Got {np.array2string(np.array(c)):s}") from err
        if G is None:
            # (G None, c not None) Singleton set with dimension provided by c
            return np.empty((c.size, 0)), c
    else:
        # (G None, c None)
        raise ValueError("Both G and c cannot be None!")
    # The last case is (G is not None, c is not None)
    if np.any(np.isinf(G)) or np.any(np.isinf(c)) or np.any(np.isnan(G)) or np.any(np.isnan(c)):
        raise ValueError(
            f"Expected G, c to be 2D and 1D array free from NaNs and infs. "
            f"Got {np.array2string(np.array(G)):s}, {np.array2string(np.array(c)):s}"
        )
    elif G.shape[0] != c.shape[0]:
        raise ValueError(f"G and c has different number of rows! G: {G.shape[0]:d} and c: {c.shape[0]:d}.")
    return G, c


def sanitize_Aebe(Ae, be):
    """Sanitize and return (`Ae`, `be`) to make a valid affine set combination

    Args:
        Ae (array_like): Equality coefficient matrix specifying the affine set
        be (array_like): Equality constant vector specifying the affine set

    Raises:
        ValueError: Ae is not 2D numpy array free from NaNs and infs
        ValueError: be is not 1D numpy array free from NaNs and infs
        ValueError: Provided (Ae, be) is not a valid system of linear equations
        UserWarning: When Ae, be has a row of zeros

    Returns:
        tuple: A tuple with four items:
            #. Ae (numpy.ndarray): 2D Equality coefficient matrix. None when no valid rows for Ae, be.
            #. be (numpy.ndarray): 1D Equality constant vector. None when no valid rows for Ae, be.

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
    if Ae.size == 0 and be.size == 0:
        return None, None
    elif np.isinf(be).any():
        raise ValueError(f"Expected be to free from NaNs and infs. Got {np.array2string(np.array(be)):s}")
    else:
        valid_rows_Ae_be = (np.abs(np.hstack((Ae, np.array([be]).T))) > PYCVXSET_ZERO).any(axis=1)
        if sum(valid_rows_Ae_be) == 0:
            return None, None
        else:
            if sum(valid_rows_Ae_be) != Ae.shape[0]:
                warnings.warn("Removed some rows in (Ae, be) that had all zeros!", UserWarning)
            return Ae[valid_rows_Ae_be, :], be[valid_rows_Ae_be]


def sanitize_and_identify_Aebe(Ae, be):
    """_summary_

    Args:
        Ae (array_like): Equality coefficient matrix specifying the affine set after sanitize_Aebe
        be (array_like): Equality constant vector specifying the affine set after sanitize_Aebe

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
    if sanitized_Ae is None:
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


def minimize(self, x, objective_to_minimize, cvxpy_args, task_str=""):
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
    try:
        containment_constraints, _ = self.containment_constraints(x)
    except ValueError:
        return np.nan * np.ones((self.dim,)), np.inf, cp.INFEASIBLE
    problem = cp.Problem(cp.Minimize(objective_to_minimize), containment_constraints)
    try:
        problem.solve(**cvxpy_args)
    except cp.error.SolverError as err:
        raise NotImplementedError(f"Unable to solve the task ({task_str:s}). CVXPY returned error: {str(err)}") from err
    if problem.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
        return x.value, problem.value, problem.status
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


def spread_points_on_a_unit_sphere(n_dim, n_points=None, cvxpy_socp_args=None, verbose=False):
    r"""Spread points on a unit sphere in n-dimensional space.

    Args:
        n_dim (int): The dimension of the sphere.
        n_points (int): The number of points to be spread on the unit sphere. Defaults to None, in which case, we choose
        n_points = (2 * self.dim) + (2**self.dim) * SPOAUS_DIRECTIONS_PER_QUADRANT.
        cvxpy_socp_args (dict): Additional arguments to be passed to the CVXPY solver. Defaults to None, in which case
            the function uses DEFAULT_CVXPY_ARGS_SOCP from pycvxset.common.constants.
        verbose (bool, optional): Whether to print verbose output. Defaults to False.

    Returns:
        tuple: A tuple containing three items:
            # opt_locations (ndarray): The spread points on the unit sphere.
            # minimum_separation (float): The minimum separation between the points.
            # opt_locations_first_quad (ndarray): The spread points in the first quadrant.

    Raises:
        ValueError: If n_points is less than 2 * n_dim.
        UserWarning: If n_points - 2 * n_dim is not a multiple of 2^n_dim.
        NotImplementedError: Unable to solve the convexified problem using CVXPY
        NotImplementedError: Convex-concave procedure did not converge!

    Notes:
        This function uses the CVXPY library to solve a convex optimization problem to spread the points on the unit
        sphere. The spread points are returned as opt_locations, the separation between the points is returned as
        separation, and the spread points in the first quadrant are returned as opt_locations_first_quad.

        For n_dim in [1, 2], the points are available in closed-form. For n_dim >= 3, we solve the following non-convex
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
    """
    if n_points is None:
        n_points = (2 * n_dim) + (2**n_dim) * SPOAUS_DIRECTIONS_PER_QUADRANT
    if n_points <= 0:
        raise ValueError(f"Expected n_points to be positive! Got {n_points}!")
    if n_dim == 1:
        opt_locations = np.array([[1], [-1]])
        opt_locations_first_quad = np.array([[1]])
        minimum_separation = 2.0
    elif n_dim == 2:
        theta_vec = np.linspace(0, 2 * np.pi, n_points + 1)[:-1]
        opt_locations = np.vstack((np.cos(theta_vec), np.sin(theta_vec))).T
        minimum_separation = float(np.linalg.norm(opt_locations[-2] - opt_locations[-1], ord=2))
        opt_locations_first_quad = opt_locations[theta_vec <= (np.pi / 2), :]
    else:
        # Ensure that n_points - (2 * n_dim) to be a multiple of 2**n_dim
        if n_points < 2 * n_dim:
            # Need to allow at least the standard vectors and their reflections
            raise ValueError("Expected n_points >= 2*n_dim")
        else:
            opt_locations = np.vstack((np.eye(n_dim), -np.eye(n_dim)))
            if (n_points - (2 * n_dim)) % (2**n_dim) > 0:
                n_points_first_quad = int(np.ceil((n_points - 2 * n_dim) / (2**n_dim)))
                n_points_old = n_points
                n_points = (2**n_dim) * n_points_first_quad + 2 * n_dim
                warnings.warn(
                    f"Invalid combination of (n_dim={n_dim}, n_points={n_points}). spread_points_on_a_unit_sphere "
                    f"requires n_points - 2*n_dim (Got: {n_points_old - (2 * n_dim):d}) to be a multiple of 2^n_dim "
                    f"(Got: {2 ** n_dim:d}). Will return {n_points} points instead of requested {n_points_old} points.",
                    UserWarning,
                )
            else:
                n_points_first_quad = (n_points - 2 * n_dim) // (2**n_dim)

        if verbose:
            print(f"Spreading {n_points} unit-length vectors in {n_dim}-dim space")
            print(f"Analyzing {n_points_first_quad} unit-length vectors in first quadrant")

        if n_points_first_quad == 0:
            opt_locations_first_quad = np.eye(n_dim)
            minimum_separation = float(np.sqrt(2))
            if verbose:
                print("Skipped spacing vectors! Returned the standard axes!")
        else:
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
            x_iter = np.ones((n_points_first_quad, n_dim)) / np.sqrt(n_dim)

            # For DC approach, initialize the previous costs
            separation_prev = 0
            sum_slack_var_prev = 0

            while continue_condition:
                if verbose:
                    print(f"{iter_count}/{MAX_ITERATIONS}. Setting up the CVXPY problem...")

                x = cp.Variable((n_points_first_quad, n_dim))
                separation = cp.Variable(nonneg=True)
                pairwise_separation_slack_var = cp.Variable((n_points_first_quad, n_points_first_quad), nonneg=True)
                standard_vector_separation_slack_var = cp.Variable((n_points_first_quad, n_dim), nonneg=True)
                minimum_norm_slack_var = cp.Variable((n_points_first_quad,), nonneg=True)
                # Constraint 1: Enforces the separation constraint is satisfied by reflections about quadrant planes
                constraints = [
                    2 * x >= separation,
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
                    for dim_index in range(n_dim):
                        e_i_vector = np.zeros(n_dim)
                        e_i_vector[dim_index] = 1
                        constraints.append(
                            (x_i_prev - e_i_vector) @ (x_i_prev - e_i_vector)
                            + 2 * (x_i_prev - e_i_vector) @ (x[pt_index_1, :] - x_i_prev)
                            + standard_vector_separation_slack_var[pt_index_1, dim_index]
                            >= cp.square(separation)
                        )

                    # Constraint 5: Enforces the maximum norm constraint ||x_i ||_2 <= 1
                    constraints.append(cp.norm(x[pt_index_1, :], p=2) <= 1)

                    # Constraint 6: Enforces the minimum norm constraint ||x_i ||_2 >= 0.8
                    constraints.append(
                        x_i_prev @ x_i_prev
                        + 2 * x_i_prev @ (x[pt_index_1, :] - x_i_prev)
                        + minimum_norm_slack_var[pt_index_1]
                        >= SPOAUS_MINIMUM_NORM_VALUE_SQR
                    )

                if verbose:
                    print("Solving the CVXPY problem...", end="")

                sum_slack_var = (
                    cp.sum(minimum_norm_slack_var)
                    + cp.sum(standard_vector_separation_slack_var)
                    + cp.sum(pairwise_separation_slack_var)
                )
                objective = -separation + tau_iter * sum_slack_var

                problem = cp.Problem(cp.Minimize(objective), constraints)

                try:
                    problem.solve(**cvxpy_socp_args)
                except cp.error.SolverError as err:
                    raise NotImplementedError(
                        f"Unable to spread points on a unit sphere! CVXPY returned error: {str(err)}"
                    ) from err

                if problem.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
                    cost_prev_with_new_tau_iter = -separation_prev + tau_iter * sum_slack_var_prev
                    if verbose:
                        print("done! Successfully solved.")
                        print(f"Tau: {tau_iter:1.2f} (< {SPOAUS_TAU_MAX:1.2f})")
                        print(f"Sum of slack: {sum_slack_var.value:.3e} (< {SPOAUS_SLACK_TOLERANCE:.3e})")
                        print(
                            f"Change in opt cost: {np.abs(objective.value - cost_prev_with_new_tau_iter):.3e} "
                            f"(< {SPOAUS_COST_TOLERANCE:.3e})\n"
                        )
                    x_iter = x.value
                    continue_condition = (
                        sum_slack_var.value > SPOAUS_SLACK_TOLERANCE
                        or np.abs(objective.value - cost_prev_with_new_tau_iter) > SPOAUS_COST_TOLERANCE
                    ) and iter_count < MAX_ITERATIONS
                else:
                    raise NotImplementedError(
                        f"Use of slack variables should have prevented problem status: {problem.status}"
                    )

                iter_count += 1
                tau_iter = min(tau_iter * SPOAUS_SCALING_TAU, SPOAUS_TAU_MAX)
                separation_prev = separation.value
                sum_slack_var_prev = sum_slack_var.value

            if (
                iter_count >= MAX_ITERATIONS
                or sum_slack_var.value > SPOAUS_SLACK_TOLERANCE
                or np.abs(objective.value - cost_prev_with_new_tau_iter) > SPOAUS_COST_TOLERANCE
            ):
                raise NotImplementedError(
                    f"Difference of convex programming did not converge while spreading {n_points} points on a "
                    f"{n_dim}-dimensional unit sphere! The convex-concave procedure parameters may need tuning.\n"
                    f"iter_count >= {MAX_ITERATIONS:d}: {iter_count >= MAX_ITERATIONS}\n"
                    f"sum_slack_var.value > {SPOAUS_SLACK_TOLERANCE:1.2f}: "
                    f"{sum_slack_var.value > SPOAUS_SLACK_TOLERANCE}\n"
                    f"np.abs(objective.value - cost_prev_with_new_tau_iter) > {SPOAUS_COST_TOLERANCE:1.2f}: "
                    f"{np.abs(objective.value - cost_prev_with_new_tau_iter) > SPOAUS_COST_TOLERANCE}"
                )
            norm_val = np.linalg.norm(x.value, ord=2, axis=1, keepdims=True)
            opt_locations_first_quad_without_axes = x.value / norm_val
            for sign_vector in itertools.product([-1, 1], repeat=n_dim):
                opt_locations = np.vstack((opt_locations, sign_vector * opt_locations_first_quad_without_axes))
            opt_locations_first_quad = np.vstack((np.eye(n_dim), opt_locations_first_quad_without_axes))
            minimum_separation = float(separation.value)

            if verbose:
                print("Completed spreading the vectors!")
                print(f"Minimum separation among the points: {minimum_separation:1.4f}")
    return opt_locations, minimum_separation, opt_locations_first_quad


def convex_set_minimum_volume_circumscribing_rectangle(self):
    r"""Compute the minimum volume circumscribing rectangle for a set.

    Raises:
        ValueError: When set is empty

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
    if self.is_empty:
        raise ValueError("Can not compute circumscribing rectangle for an empty set!")
    else:
        lb = -self.support(-np.eye(self.dim))[0]
        ub = self.support(np.eye(self.dim))[0]
    return lb, ub
