# Copyright (C) 2020-2025 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

# Code purpose: Test methods common to all sets

import itertools

import cvxpy as cp
import matplotlib.pyplot as plt
import numpy as np
import pytest

from pycvxset import ConstrainedZonotope, Polytope
from pycvxset.common import check_matrices_are_equal_ignoring_row_order, spread_points_on_a_unit_sphere
from pycvxset.common.constants import DEFAULT_CVXPY_ARGS_LP, SPOAUS_DIRECTIONS_PER_QUADRANT

PLOT_SHOW = False
PERFORM_3D_DEFAULT = False


def test_spread_points_on_a_unit_sphere():
    # 1D case
    V, min_sep, _ = spread_points_on_a_unit_sphere(1, 2)
    assert np.isclose(min_sep, 2)
    # 2D case
    V, min_sep, _ = spread_points_on_a_unit_sphere(2, 30)
    assert np.isclose(min_sep, np.linalg.norm(V[0] - V[1]))
    assert np.isclose(np.linalg.norm(V[-2] - V[-1]), np.linalg.norm(V[0] - V[1]))
    # 3D case
    V, _, _ = spread_points_on_a_unit_sphere(3, 6, verbose=True)
    assert check_matrices_are_equal_ignoring_row_order(V, np.vstack((np.eye(3), -np.eye(3))))
    spread_points_on_a_unit_sphere(3, 6, verbose=False)
    # 3D case default no warning from pycvxset (but cvxpy warns about solution)
    if PERFORM_3D_DEFAULT:
        with pytest.warns(UserWarning, match="Solution may be inaccurate.*"):
            V, min_sep, _ = spread_points_on_a_unit_sphere(3, verbose=True)
        assert V.shape[0] == (2 * 3) + (2**3) * SPOAUS_DIRECTIONS_PER_QUADRANT
        min_sep_via_iteration = np.inf
        for i, j in itertools.combinations(range(V.shape[0]), 2):
            min_sep_via_iteration = min(np.linalg.norm(V[i] - V[j]), min_sep_via_iteration)
        assert np.isclose(min_sep, min_sep_via_iteration)
    # 3D case small no warning
    n_directions = (2 * 3) + (2**3) * 2
    V, min_sep, _ = spread_points_on_a_unit_sphere(3, n_directions, verbose=False)
    assert V.shape[0] == n_directions
    min_sep_via_iteration = np.inf
    for i, j in itertools.combinations(range(V.shape[0]), 2):
        min_sep_via_iteration = min(np.linalg.norm(V[i] - V[j]), min_sep_via_iteration)
    assert np.isclose(min_sep, min_sep_via_iteration)
    # 3D case small warning
    n_directions = (2 * 3) + (2**3) * 2 + 1
    with pytest.warns(UserWarning, match="Invalid combination*"):
        V, min_sep, _ = spread_points_on_a_unit_sphere(3, n_directions, verbose=True)
    assert V.shape[0] == (2 * 3) + (2**3) * 3
    min_sep_via_iteration = np.inf
    for i, j in itertools.combinations(range(V.shape[0]), 2):
        min_sep_via_iteration = min(np.linalg.norm(V[i] - V[j]), min_sep_via_iteration)
    assert np.isclose(min_sep, min_sep_via_iteration)

    with pytest.raises(ValueError):
        spread_points_on_a_unit_sphere(3, 2)
    with pytest.raises(NotImplementedError):
        spread_points_on_a_unit_sphere(3, cvxpy_socp_args={"solver": "WRONG_SOLVER"})

    if PLOT_SHOW:
        opt_locations, _, opt_locations_first_quad = spread_points_on_a_unit_sphere(
            3, cvxpy_socp_args={"solver": "CLARABEL"}, verbose=False
        )
        plt.figure()
        ax = plt.axes(projection="3d")
        ax.scatter(opt_locations[:, 0], opt_locations[:, 1], opt_locations[:, 2])
        ax.set_aspect("equal")
        plt.figure()
        ax = plt.axes(projection="3d")
        ax.scatter(opt_locations_first_quad[:, 0], opt_locations_first_quad[:, 1], opt_locations_first_quad[:, 2])
        ax.set_aspect("equal")
        plt.show()


def test_solve_convex_program_with_constrained_zonotope_and_polytope_containment_constraints():
    # Empty constrained zonotope
    x = cp.Variable((2,))
    x_value, problem_value, problem_status = ConstrainedZonotope(dim=2).minimize(
        x=x,
        objective_to_minimize=np.ones((2,)) @ x,
        cvxpy_args=DEFAULT_CVXPY_ARGS_LP,
        task_str="",
    )
    assert np.isnan(x_value).all()
    assert problem_value == np.inf
    assert problem_status == cp.INFEASIBLE  # We assign this!

    # Normal zonotope
    Z = ConstrainedZonotope(G=np.eye(2), c=np.zeros((2,)))
    x = cp.Variable((2,))
    x_solution, problem_value, problem_status = Z.minimize(
        x=x,
        objective_to_minimize=np.ones((2,)) @ x,
        cvxpy_args=DEFAULT_CVXPY_ARGS_LP,
        task_str="",
    )
    assert np.allclose(x_solution, -np.ones((2,)))
    assert np.isclose(problem_value, -2)
    assert problem_status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]

    # Normal constrained zonotope (and a V-Rep polytope)
    P = Polytope(V=spread_points_on_a_unit_sphere(2, 7)[0])
    C = ConstrainedZonotope(polytope=P)
    x = cp.Variable((2,))
    x_solution_constrained_zonotope, _, _ = C.minimize(
        x=x,
        objective_to_minimize=np.ones((2,)) @ x,
        cvxpy_args=DEFAULT_CVXPY_ARGS_LP,
        task_str="",
    )
    x_solution_polytope, _, _ = P.minimize(
        x=x,
        objective_to_minimize=np.ones((2,)) @ x,
        cvxpy_args=DEFAULT_CVXPY_ARGS_LP,
        task_str="",
    )
    assert np.allclose(x_solution_polytope, x_solution_constrained_zonotope)

    # Unbounded polytope
    P = Polytope(A=[[1, 0], [0, 1]], b=[1, 1])
    x = cp.Variable((2,))
    x_solution_polytope, problem_value, problem_status = P.minimize(
        x=x,
        objective_to_minimize=np.ones((2,)) @ x,
        cvxpy_args=DEFAULT_CVXPY_ARGS_LP,
        task_str="",
    )
    assert np.isnan(x_solution_polytope).all()
    assert problem_value == -np.inf
    assert problem_status in [cp.UNBOUNDED, cp.UNBOUNDED_INACCURATE]

    # Infeasible polytope
    P = Polytope(c=[0, 0], h=0.5).intersection_with_halfspaces(A=[-1, 0], b=-2)
    x = cp.Variable((2,))
    x_solution_polytope, problem_value, problem_status = P.minimize(
        x=x,
        objective_to_minimize=np.ones((2,)) @ x,
        cvxpy_args=DEFAULT_CVXPY_ARGS_LP,
        task_str="",
    )
    assert np.isnan(x_value).all()
    assert problem_value == np.inf
    assert problem_status == cp.INFEASIBLE  # We assign this!

    # Case for V-Rep (is tackled in the ConstrainedZonotope class)
