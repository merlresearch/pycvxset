# Copyright (C) 2020-2024 Mitsubishi Electric Research Laboratories (MERL)
# Copyright (c) 2019 Tor Aksel N. Heirung
#
# SPDX-License-Identifier: AGPL-3.0-or-later
# SPDX-License-Identifier: MIT

# Code purpose: Test the Polytope class methods for constructor

import cvxpy as cp
import numpy as np
import pytest

from pycvxset import Polytope
from pycvxset.common import check_matrices_are_equal_ignoring_row_order, check_vectors_are_equal_ignoring_row_order
from pycvxset.common.constants import DEFAULT_CVXPY_ARGS_LP, DEFAULT_CVXPY_ARGS_SDP, DEFAULT_CVXPY_ARGS_SOCP


def test___init__():
    # Create R^2 Polytope (empty)
    P = Polytope(dim=2)
    assert not P.in_H_rep
    assert not P.in_V_rep
    assert P.dim == 2
    assert not P.is_full_dimensional
    assert P.is_empty
    assert P.is_bounded

    # Create R^2 Polytope in H-representation from upper and lower bounds, and check that dimension dim and the matrices
    # A, b, and H = [A b] are all set correctly.
    lb1 = (1, -4)
    ub1 = (3, -2)
    n1 = len(ub1)
    A1 = np.vstack((-np.eye(n1), np.eye(n1)))
    # b as a 2D array is also ok
    b1 = np.array([np.hstack((-np.asarray(lb1), ub1))]).T
    V1 = [[1, -4], [1, -2], [3, -4], [3, -2]]
    P1 = Polytope(lb=lb1, ub=ub1)
    assert P1.in_H_rep
    assert not P1.in_V_rep
    assert P1.dim == n1
    assert check_matrices_are_equal_ignoring_row_order(P1.A, A1)
    assert check_vectors_are_equal_ignoring_row_order(P1.b, b1)
    assert check_matrices_are_equal_ignoring_row_order(P1.H, np.hstack((A1, b1)))
    assert all(v in P1.V.tolist() for v in V1)
    assert P1.in_V_rep
    assert np.issubdtype(P1.A.dtype, float)
    assert np.issubdtype(P1.b.dtype, float)
    assert np.issubdtype(P1.H.dtype, float)
    assert np.issubdtype(P1.V.dtype, float)

    # Create R^2 Polytope in V-representation from a list of four vertices, and check that dimension dim and vertex list
    # V are set correctly
    V2 = np.array([[1, 1], [-1, 1], [-1, -1], [1, -1]])
    n2 = V2.shape[1]

    P2 = Polytope(V=V2)
    assert P2.in_V_rep
    assert not P2.in_H_rep
    assert P2.dim == n2
    assert all(v in P2.V.tolist() for v in V2.tolist())
    assert np.issubdtype(P2.V.dtype, float)

    # Create an R^2 Polytope in H-representation by specifying A and b in Ax <= b. Check that dimension dim and vertex
    # list V are set correctly
    A3 = [[-1, 0], [0, -1], [1, 1]]
    b3 = (0, 0, 2)  # b as a tuple is also ok
    n3 = 2
    H3 = np.hstack((A3, np.array([[*b3]]).T))
    V3 = [[0, 0], [0, 2], [2, 0]]

    P3 = Polytope(A=A3, b=b3)

    assert P3.in_H_rep
    assert not P3.in_V_rep
    assert P3.dim == n3
    assert np.all(P3.A == np.asarray(A3))
    assert np.all(P3.b == np.asarray(b3))
    assert np.all(P3.H == H3)
    assert all(v in P3.V.tolist() for v in V3)
    assert P3.in_V_rep
    assert np.issubdtype(P3.A.dtype, float)
    assert np.issubdtype(P3.b.dtype, float)
    assert np.issubdtype(P3.H.dtype, float)
    assert np.issubdtype(P3.V.dtype, float)

    # Empty polytope
    P4 = Polytope(A=[[1], [-1]], b=[-1, -1])
    assert P4.H.shape == (2, P4.dim + 1)
    P4.minimize_H_rep()
    assert P4.is_empty
    assert P4.He.shape == (0, P4.dim + 1)
    assert P4.H.shape == (0, P4.dim + 1)
    P4.minimize_V_rep()
    assert P4.V.shape == (0, P4.dim)

    P4a = Polytope(V=[[-1, -1], [-1, -1], [-1, 2]])
    assert P4a.V.shape == (3, 2)
    assert not P4a.is_empty
    P4a.minimize_H_rep()
    assert P4a.V.shape == (3, 2)  # Does not affect V
    P4a.minimize_V_rep()
    assert P4a.V.shape == (2, 2)

    # Low-dim polytope
    P4b = Polytope(A=[[1], [-1]], b=[1, -1])
    assert P4b.H.shape == (2, P4b.dim + 1)
    P4b.minimize_H_rep()
    assert not P4b.is_empty
    assert P4b.He.shape == (1, P4b.dim + 1)
    assert P4b.H.shape == (0, P4b.dim + 1)
    P4b.minimize_V_rep()
    assert P4b.V.shape == (1, P4b.dim)

    # Ensure illegal use of the constructor raises an error.
    with pytest.raises(ValueError):
        Polytope(A="char", b="char")
    with pytest.raises(ValueError):
        Polytope(A="char", b=1)
    with pytest.raises(ValueError):
        Polytope(A=[[1], [-1]], b="char")
    with pytest.raises(ValueError):
        Polytope(V=V2, A=A3, b=b3)
    with pytest.raises(ValueError):
        Polytope(A=A3)
    with pytest.raises(ValueError):
        Polytope(A=np.array([np.nan, 1]), b=np.nan)
    with pytest.raises(ValueError):
        Polytope(b=b3)
    with pytest.raises(ValueError):
        Polytope(V=V2, lb=lb1, ub=ub1)
    with pytest.raises(ValueError):
        Polytope(lb="char", ub=[1, 1])
    with pytest.raises(ValueError):
        Polytope(lb=[1, 1], ub="char")
    with pytest.raises(ValueError):
        Polytope(A=A3, b=b3, lb=lb1, ub=ub1)
    with pytest.raises(ValueError):
        Polytope(dim=2, lb=-1, ub=1)
    with pytest.raises(ValueError):
        Polytope(lb=-1)
    with pytest.raises(ValueError):
        Polytope(lb=[[-1, 2], [1, 3]], ub=[[-1], [-1]])
    with pytest.raises(ValueError):
        Polytope(ub=[[-1, 2], [1, 3]], lb=[[-1], [-1]])
    with pytest.raises(ValueError):
        Polytope(ub=[-1, 2, 1, 3], lb=[-1, -1])
    with pytest.raises(ValueError):
        Polytope(R=[[1, 1]])
    with pytest.raises(ValueError):
        Polytope(V=[[[-1, 1], [-1, 1]], [[-2, 2], [-2, 2]]])
    with pytest.raises(ValueError):
        Polytope(A=[[[-1, 1], [-1, 1]], [[-2, 2], [-2, 2]]], b=[1, 2])
    with pytest.raises(ValueError):
        Polytope(b=[[[-1, 1], [-1, 1]], [[-2, 2], [-2, 2]]], A=[[-1, 1], [1, -1]])
    with pytest.raises(ValueError):
        Polytope(A=[[-1, 1], [1, -1]], b=1)
    with pytest.raises(ValueError):
        Polytope(W=[[-1, 1]])
    with pytest.raises(TypeError):
        Polytope(dim=3) == "s"
    with pytest.raises(TypeError):
        Polytope(A=[[1], [-1]], b=[1, 1]) == "s"
    with pytest.raises(ValueError):
        Polytope(A=[0, 0, 0], b=[1])
    with pytest.raises(ValueError):
        Polytope(A=[[1], [-1]], b=[1, 1], blah="a")

    P_unbounded_which_is_missed = Polytope(A=[[-1, 0, 0], [0, -1, 0], [1, 1, 0]], b=[0, 0, 1])
    with pytest.raises(ValueError):
        print(P_unbounded_which_is_missed.V)

    # Create polytope from c and h
    P = Polytope(c=(0, 0, 0), h=(1, 2, 1))

    with pytest.raises(ValueError):
        Polytope(c=((0, 0), (0, 0)), h=1)

    with pytest.raises(ValueError):
        Polytope(c=(0, 0, 0), h=(1, 1, 1, 1))

    with pytest.raises(ValueError):
        Polytope(c=(0, 0, 0), h=((1, 1, 0), (1, 1, 0)))

    with pytest.raises(ValueError):
        Polytope(c=((0, 0), (0, 0)), h=1, lb=1)

    with pytest.raises(ValueError):
        Polytope(c="char", h=1)

    with pytest.raises(ValueError):
        Polytope(c=[0, 0], h="char")


def test_is_bounded():
    assert Polytope(dim=3).is_bounded
    assert Polytope(V=[[3]]).is_bounded
    with pytest.raises(ValueError):
        Polytope(A=[[]], b=[])
    P = Polytope(A=np.empty((1, 1)), b=[], Ae=[3], be=[1])
    assert P.in_V_rep
    assert np.isclose(P.V, 1 / 3)
    with pytest.raises(ValueError):
        P = Polytope(A=np.empty((0, 2)), b=np.empty((0,)), Ae=[3, 2], be=[1])
    P = Polytope(A=[[1, 0], [0, 1]], b=[2, 3])
    assert not P.is_bounded
    P = Polytope(A=[[1, 0], [0, 1]], b=[2, 3])
    assert P.is_full_dimensional
    P = Polytope(A=[[1, 0], [0, 1], [0, -1]], b=[2, 3, -3])
    assert not P.is_bounded
    P = Polytope(A=[[1, 0], [0, 1], [0, -1]], b=[2, 3, -3])
    assert not P.is_full_dimensional


def test_embedded_polytope_and_intersect_with_affine_set():
    P_hrep_3D = Polytope(c=[0, 0, 0], h=[2, 3, 1])
    equality_constrained_P_hrep_3D = Polytope(A=P_hrep_3D.A, b=P_hrep_3D.b, Ae=[0.03, 0, 0.2], be=[0])
    assert not equality_constrained_P_hrep_3D.is_full_dimensional
    assert P_hrep_3D.is_full_dimensional
    P = Polytope(c=[0, 0, 0], h=0.5)
    P_sliced = P.intersection_with_affine_set(Ae=[0, 0, 1], be=0.25)
    assert P_sliced <= P
    assert P.dim == 3
    P_equality_constrained_Projection = P_sliced.projection(2)
    assert P_equality_constrained_Projection.dim == 2
    assert np.isclose(P_equality_constrained_Projection.volume(), 1)
    assert P.intersection_with_affine_set(Ae=[0, 0, 1], be=1).is_empty
    # Incorrect Ae
    with pytest.raises(ValueError):
        Polytope(c=[0, 0, 0], h=0.5).intersection_with_affine_set(Ae=[1, 1], be=1)

    P = Polytope(lb=[1, 1], ub=[1, 1])
    assert P.n_vertices == 1

    P = Polytope(lb=[-1, -1], ub=[1, 1])
    with pytest.warns(UserWarning, match="Removed some rows in A *"):
        Pnew = Polytope(A=np.vstack((P.A, np.zeros((2,)))), b=np.hstack((P.b, 1)))
    assert Pnew.n_halfspaces == 4

    Pnew = Polytope(A=P.A, b=P.b, Ae=[[0, 0]], be=1)
    assert Pnew.is_empty
    with pytest.warns(UserWarning, match="Removed some rows in (Ae, be)*"):
        Pnew = Polytope(A=P.A, b=P.b, Ae=[[0, 0], [0, 1]], be=[0, 1])
    assert Pnew.n_equalities == 1

    with pytest.raises(ValueError):
        Polytope(A=P.A, b=P.b, Ae=[[0, 1, 0]], be=1)

    Pnew = Polytope(A=P.A, b=P.b, Ae=[[0, 1], [0, -1]], be=[1, 1])
    assert Pnew.is_empty

    with pytest.raises(ValueError):
        Pnew = Polytope(A=[[0, 0]], b=1, Ae=[[0, 1]], be=[1])

    Pnew = Polytope(A=[[0, 0]], b=1, Ae=[[0, 0]], be=[1])
    assert Pnew.is_empty

    Pnew = Polytope(A=[1], b=1, Ae=[1], be=[1])
    assert Pnew.n_halfspaces == 0
    assert Pnew.n_equalities == 1

    # No He
    Pnew = Polytope(A=P.A, b=P.b, Ae=[[0, 0]], be=[0])
    assert Pnew.n_equalities == 0

    # np.inf/-np.inf
    with pytest.warns(UserWarning, match="Removed some rows in A *"):
        Pnew = Polytope(A=np.vstack((np.eye(2), -np.eye(2))), b=[1, 1, 1, np.inf])
    assert Pnew.n_halfspaces == 3
    Pnew = Polytope(A=np.vstack((P.A, [1, 0])), b=np.hstack((P.b, -np.inf)))
    assert Pnew.is_empty
    assert Polytope(c=[0, 0], h=-np.inf).is_empty
    with pytest.raises(ValueError):
        Polytope(lb=[-np.inf, -np.inf], ub=[1, 1])
    with pytest.raises(ValueError):
        Polytope(lb=[0, 0], ub=[np.inf, np.inf])
    with pytest.raises(ValueError):
        Polytope(A=np.vstack((np.eye(2), -np.eye(2))), b=[np.inf, np.inf, np.inf, np.inf])
    assert Polytope(lb=[0, 0], ub=[-np.inf, -np.inf]).is_empty
    with pytest.raises(ValueError):
        Polytope(c=[0, 0], h=np.inf)
    with pytest.raises(ValueError):
        Polytope(c=[0, np.inf], h=1)
    with pytest.raises(ValueError):
        Polytope(c=[0, 0], h=1).intersection_with_affine_set(Ae=[1, np.inf], be=0)
    with pytest.raises(ValueError):
        Polytope(c=[0, 0], h=1).intersection_with_affine_set(Ae=[1, 0], be=np.inf)
    with pytest.raises(ValueError):
        Polytope(c=[0, 0], h=1).intersection_with_affine_set(Ae=[1, 0], be=-np.inf)
    with pytest.raises(ValueError):
        Polytope(c=[0, 0], h=1).intersection_with_halfspaces(A=[1, np.inf], b=0)
    with pytest.warns(UserWarning, match="Removed some rows in A *"):
        assert Polytope(c=[0, 0], h=1).intersection_with_halfspaces(A=[1, 0], b=np.inf) == Polytope(c=[0, 0], h=1)
    assert Polytope(c=[0, 0], h=1).intersection_with_halfspaces(A=[1, 0], b=-np.inf).is_empty


def test_is_full_dimensional():
    P = Polytope(lb=[1, 1], ub=[-1, -1])
    assert not P.is_full_dimensional  # Empty but not full dimensional
    P = Polytope(dim=0)
    assert P.is_full_dimensional  # Empty but full-dimensional
    P = Polytope(lb=[-1, -1], ub=[-1, 1])
    assert not P.is_full_dimensional  # Restricted to a line
    P = Polytope(lb=[-1, -1], ub=[1, 1])
    assert P.is_full_dimensional  # Unit box
    P = Polytope(A=P.A, b=P.b)
    P.cvxpy_args_lp = {"solver": "WRONG_SOLVER"}
    with pytest.raises(NotImplementedError):
        P.is_full_dimensional
    P.cvxpy_args_lp = {"solver": "CLARABEL"}
    assert P.is_full_dimensional  # Unit box but (A, b)


def test_bool_and_empty_sets():
    P = Polytope(lb=[1, 1], ub=[-1, -1])
    assert P.is_empty
    P = Polytope(lb=[-1, -1], ub=[1, 1])
    assert not P.is_empty
    P = Polytope(V=[[-1, 1]])
    assert not P.is_empty
    P1 = Polytope()
    assert P1.H.shape == (0, 1)
    assert P1.A.shape == (0, 0)
    assert P1.b.shape == (0,)
    assert P1.V.shape == (0, 0)
    assert P1.is_empty
    P2 = Polytope(dim=5)
    assert P2.H.shape == (0, 6)
    assert P2.A.shape == (0, 5)
    assert P2.b.shape == (0,)
    assert P2.V.shape == (0, 5)
    assert P2.is_empty
    with pytest.raises(ValueError):
        Polytope(A=[[]], b=[])
    with pytest.raises(ValueError):
        Polytope(A=[], b=[])
    with pytest.raises(ValueError):
        Polytope(A=[[1, 2, 3]], b=[])
    P5 = Polytope(A=[[1, 0], [-1, 0]], b=[-10, -10])
    assert P5.is_empty
    with pytest.raises(ValueError):
        Polytope(A=[[]], b=[[]])
    P7 = Polytope(V=[[]])
    assert not P7.in_H_rep
    assert not P7.in_V_rep
    assert P7.is_empty
    with pytest.raises(ValueError):
        Polytope(V=[])
    with pytest.raises(ValueError):
        Polytope(V=[[[]]])
    with pytest.raises(ValueError):
        Polytope(A=[[[]]], b=[])
    with pytest.raises(ValueError):
        Polytope(A=[[[]]], b=[])


def test_setters_error():
    P1 = Polytope()
    with pytest.raises(AttributeError):
        P1.A = np.zeros((0, 1))
    with pytest.raises(AttributeError):
        P1.b = np.zeros((0, 1))
    with pytest.raises(AttributeError):
        P1.V = np.zeros((0, 1))


def test_power_and_axis_aligned():
    power = 3
    P = Polytope(lb=-1, ub=1)
    P_raised_by_power = P**power
    assert P_raised_by_power.is_full_dimensional
    assert P_raised_by_power.is_full_dimensional == 1
    assert P_raised_by_power == Polytope(c=(0, 0, 0), h=1)

    # [-1, 1] to [0, 0] => [-1, 1] times [0] times [-1, 1] times [0]
    P = Polytope(lb=[-1, -1], ub=[1, 1]).intersection_with_affine_set(Ae=[0, 1], be=[0])
    P_4D = P**2
    P_4D.minimize_V_rep()
    assert P_4D.n_vertices == 4
    assert np.allclose(P_4D.V[:, (1, 3)], 0)
    assert np.allclose(np.abs(P_4D.V[:, (0, 2)]), 1)


def test_le_ge():
    P = Polytope(c=[0, 0], h=2)
    Q = Polytope(c=[0, 0], h=3)
    assert P <= Q
    assert not P >= Q
    P = Polytope(c=[10, 0], h=2)
    Q = Polytope(c=[0, 0], h=3)
    assert not P <= Q
    assert not P < Q
    assert not P >= Q
    assert not P > Q
    R = Polytope(dim=2)
    assert R <= P
    assert R < P
    assert not R >= P
    assert not R >= P


def test_cvxpy_args():
    P = Polytope(c=[0, 0], h=2)
    assert P.cvxpy_args_lp == DEFAULT_CVXPY_ARGS_LP
    assert P.cvxpy_args_socp == DEFAULT_CVXPY_ARGS_SOCP
    assert P.cvxpy_args_sdp == DEFAULT_CVXPY_ARGS_SDP
    NEW_SOLVER = {"solver": "SOLVER_TO_TEST"}
    P.cvxpy_args_lp = NEW_SOLVER
    P.cvxpy_args_socp = NEW_SOLVER
    P.cvxpy_args_sdp = NEW_SOLVER
    assert P.cvxpy_args_lp == NEW_SOLVER
    assert P.cvxpy_args_socp == NEW_SOLVER
    assert P.cvxpy_args_sdp == NEW_SOLVER


def test_copy():
    P = Polytope(c=[0, 0], h=2).intersection_with_affine_set(Ae=[0, 1], be=1)
    assert P.copy() == P


def test_str_repr():
    P0 = Polytope()
    print(P0)
    print(P0.__doc__)
    print(P0.__str__)
    print(P0.__repr__)
    P1 = Polytope(lb=[-1, -1], ub=[1, 1])
    print(P1)
    print(P1.__str__)
    print(P1.__repr__)
    P1.determine_V_rep()
    print(P1)
    print(P1.__str__)
    print(P1.__repr__)
    P1a = P1.intersection_with_affine_set(Ae=[1, 0], be=1)
    print(P1a)
    print(P1a.__str__)
    print(P1a.__repr__)
    P1b = P1.intersection_with_affine_set(Ae=np.eye(2), be=[1, 1])
    print(P1b)
    print(P1b.__str__)
    print(P1b.__repr__)
    P1c = Polytope(lb=[-1, -1, -1], ub=[1, 1, 1]).intersection_with_affine_set(
        Ae=np.array([[1, 0, 0], [0, 1, 0]]), be=[1, 1]
    )
    print(P1c)
    print(P1c.__str__)
    print(P1c.__repr__)
    P2 = Polytope(V=P1.V)
    print(P2)
    print(P2.__str__)
    print(P2.__repr__)
    P3 = Polytope(V=P1.V[0, None])
    print(P3)
    print(P3.__str__)
    print(P3.__repr__)


def test_constraints():
    P = Polytope(dim=2)
    x = cp.Variable(2)
    with pytest.raises(ValueError):
        P.containment_constraints(x)
