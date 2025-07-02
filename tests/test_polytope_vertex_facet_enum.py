# Copyright (C) 2020-2025 Mitsubishi Electric Research Laboratories (MERL)
# Copyright (c) 2019 Tor Aksel N. Heirung
#
# SPDX-License-Identifier: AGPL-3.0-or-later
# SPDX-License-Identifier: MIT

# Code purpose: Test the Polytope class methods for vertex-facet enumeration

import numpy as np
import pytest

from pycvxset import Polytope
from pycvxset.common import check_matrices_are_equal_ignoring_row_order


def test_determine_H_rep():
    # Create a polytope from a vertex list, determine its H-rep, use that H-rep
    # to create a new polytope, determine the vertices of the new polytope, and
    # ascertain that the vertex lists are the same.
    V1 = np.array(
        [
            [-1.42, -1.87, -1.53, -1.38, -0.80, 1.88, 1.93, 1.90, 1.59, 0.28],
            [1.96, -0.26, -1.53, -1.78, -1.76, -1.48, -0.49, 1.18, 1.79, 1.89],
        ]
    ).T
    P1 = Polytope(V=V1)
    P1.determine_H_rep()
    P2 = Polytope(A=P1.A, b=P1.b)
    P2.determine_V_rep()
    assert check_matrices_are_equal_ignoring_row_order(P1.V, P2.V)
    P3 = Polytope(V=V1)
    assert len(P3.b) == len(P1.b)
    P4 = Polytope(dim=3)
    P4.determine_H_rep()
    assert P4.H.shape == (0, 3 + 1)
    P5 = Polytope(V=[[-1], [1], [0.5]])
    assert P5.n_halfspaces == 2
    P5_true_A = np.array([[-1, 1]]).T
    P5_true_b = np.ones((2, 1))
    P5_true_H = np.hstack((P5_true_A, P5_true_b))
    assert check_matrices_are_equal_ignoring_row_order(P5.H, P5_true_H)
    P6 = Polytope(dim=2)
    P6.determine_H_rep()
    P7 = Polytope(V=[[1]])
    assert P7.n_halfspaces == 0
    assert P7.n_equalities == 1


def test_determine_V_rep():
    # Create a polytope from a vertex list, determine its H-rep, use that H-rep
    # to create a new polytope, determine the vertices of the new polytope, and
    # ascertain that the vertex lists are the same.
    P1 = Polytope(lb=[-1, -1], ub=[1, 1])
    H_original = np.array([[1, 0, 1], [0, 1, 1], [-1, 0, 1], [0, -1, 1]])
    V_original = np.array([[1, 1], [-1, 1], [-1, -1], [1, -1]])
    # Make sure that the constructor worked
    assert check_matrices_are_equal_ignoring_row_order(P1.H, H_original)
    P1.determine_H_rep()
    # Make sure that the deterimine_H_rep did not destroy it
    assert check_matrices_are_equal_ignoring_row_order(P1.H, H_original)
    P1.determine_V_rep()
    assert check_matrices_are_equal_ignoring_row_order(P1.V, V_original)
    P2 = Polytope(V=V_original)
    P2.determine_V_rep()
    P2.determine_H_rep()
    assert check_matrices_are_equal_ignoring_row_order(P1.H, P2.H)

    P3 = Polytope(dim=3)
    P3.determine_V_rep()
    assert P3.V.shape == (0, 3)

    P7 = Polytope(A=[[1, 1], [-1, -1]], b=[1, 1])  # Unbounded polytope
    with pytest.raises(ValueError):
        P7.determine_V_rep()


def test_minimal_V_rep():
    # Create a polytope from a minimal set of vertices, vertices on the convex
    # hull of those vertices, and random vertices in the interior of the convex
    # hull. Compute the minimal V-representation and test whether it matches the
    # minimal vertex list.
    x_lb = (-3, 0.9)
    x_ub = (0.6, 4)
    # Set of vertices that form the convex hull:
    V_minimal = np.array([[x_lb[0], x_lb[1]], [x_lb[0], x_ub[1]], [x_ub[0], x_ub[1]], [x_ub[0], x_lb[1]]])
    # Points that are redundant in the sense they are on simplices of the
    # convex hull but they are not vertices:
    V_redundant = np.array([[(x_ub[0] + x_lb[0]) / 2, x_lb[1]], [x_ub[0], (x_ub[1] + x_lb[1]) / 2]])
    # Random points in the interior of the convex hull:
    V_random = np.random.uniform(x_lb, x_ub, (40, len(x_lb)))
    V = np.vstack((V_minimal, V_redundant, V_random))
    P_min = Polytope(V=V_minimal)
    P = Polytope(V=V)
    assert P.n_vertices == V.shape[0]
    P.minimize_V_rep()
    assert P.n_vertices == V_minimal.shape[0]
    assert check_matrices_are_equal_ignoring_row_order(P.V, P_min.V)

    # Empty polytope
    P1 = Polytope(lb=[1, 1], ub=[-1, -1])
    P1.minimize_V_rep()
    assert P1.V.shape == (0, 2)

    # Not full-dimensional 2D plot
    P = Polytope(V=[[1, 0], [0, 1], [0.5, 0.5]])
    P.minimize_V_rep()
    assert P.V.shape == (2, 2)

    P7 = Polytope(A=[[1, 0], [0, 1]], b=[1, 1])  # Unbounded polytope
    with pytest.raises(ValueError):
        P7.minimize_V_rep()

    P = Polytope(V=[[1, 0, 0], [1, 1, 1]])
    P.minimize_V_rep()
    assert P.n_vertices == 2

    P = Polytope(V=[[1, 0, 0], [1, 0, 0]])
    P.minimize_V_rep()
    assert P.n_vertices == 1

    P = Polytope(V=[[1, 0], [1, 1]])
    P.minimize_V_rep()
    assert P.n_vertices == 2

    P = Polytope(V=[[1, 0], [1, 0]])
    P.minimize_V_rep()
    assert P.n_vertices == 1

    P = Polytope(V=[[1, 0, 0]])
    P.minimize_V_rep()
    assert P.n_vertices == 1

    P = Polytope(V=[[1], [0], [-1]])
    P.minimize_V_rep()
    assert P.n_vertices == 2


def test_minimize_H_rep():
    # Create a polytope from with four redundant constraints. The non-redundant
    # constraints specify a square with vertices (+- 1, -+ 1), the redundant
    # constraints an outer square with vertices (+- 2, -+ 2). Check that the
    # polytope first is specified with eight halfspaces, minimize the H-rep and
    # verify the number goes down to four (corresponding to the inner square),
    # and finally check that the correct (outer) halfspaces are removed from the
    # inequality set.
    n = 2
    lb_inner = np.array([[-1, -1]]).T
    ub_inner = -lb_inner
    A_inner = np.vstack((-np.eye(n), np.eye(n)))
    b_inner = np.vstack((-lb_inner, ub_inner))
    H_inner = np.hstack((A_inner, b_inner))
    A_outer = A_inner
    b_outer = 2 * b_inner
    A = np.vstack((A_inner, A_outer))
    b = np.vstack((b_inner, b_outer))
    P = Polytope(A=A, b=b)
    assert P.H.shape[0] == 8
    P.minimize_H_rep()
    assert P.H.shape[0] == H_inner.shape[0] == 4
    assert all(h in H_inner for h in P.H)

    P1 = Polytope(c=[0, 0], h=1)
    P1a = Polytope(A=np.vstack((P1.A, [0, 1])), b=np.hstack((P1.b, 5)))
    P1a.minimize_H_rep()
    assert P1a.H.shape[0] == 4

    # Empty polytope
    P2 = Polytope(A=[[1, 0], [0, 1], [0, -1]], b=[2, 10, -11])
    P2.minimize_H_rep()
    assert P2.H.shape[0] == 0
    assert P2.is_empty

    P3 = Polytope(c=[0, 0], h=3)
    P3a = P3.intersection_with_halfspaces(A=[[1, 0], [2, 0]], b=[2, 2])
    P3a.minimize_H_rep()
    assert P3a.H.shape[0] == 4

    P7 = Polytope(A=[[1, 1], [-1, -1]], b=[1, 1])  # Unbounded polytope
    try:
        P.is_bounded
        code_works_for_is_bounded = True
    except ValueError:
        code_works_for_is_bounded = False
    if not code_works_for_is_bounded:
        # Could be Solver not matching OR Can not plot an unbounded polytope
        P.cvxpy_args_lp = {"solver": "OSQP"}
    with pytest.raises(ValueError):
        P7.minimize_H_rep()
    P7_lb_ub = Polytope(lb=[-1, -1], ub=[1, 1])
    P7a = Polytope(A=P7_lb_ub.A[:-1, :], b=P7_lb_ub.b[:-1])
    with pytest.raises(ValueError):
        P7a.minimize_H_rep()

    P8 = Polytope(V=[[1, 1]])
    P8.minimize_V_rep()
    assert P8.n_vertices == 1

    P9 = Polytope(V=[[1, 1], [1, 1]])
    P9.minimize_V_rep()
    assert P9.n_vertices == 1
