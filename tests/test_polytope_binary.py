# Copyright (C) 2020-2025 Mitsubishi Electric Research Laboratories (MERL)
# Copyright (c) 2019 Tor Aksel N. Heirung
#
# SPDX-License-Identifier: AGPL-3.0-or-later
# SPDX-License-Identifier: MIT

# Code purpose: Test the Polytope class methods for binary operations

import numpy as np
import pytest

from pycvxset import ConstrainedZonotope, Polytope
from pycvxset.common import check_matrices_are_equal_ignoring_row_order
from pycvxset.Polytope.operations_binary import inverse_affine_map_under_invertible_matrix, minus, plus


def test_addition_and_subtraction_by_a_vector():
    PV = Polytope(V=[[-1, 0], [1, 0], [0, 1]])
    PH = Polytope(A=[[-1, 0], [0, -1], [1, 1]], b=[-2, -3, 8])
    points = [
        (1, 1),
        [-1, 2],
        [[1.5], [-0.5]],
        np.array([-2, -0.1]),
        np.array([[-2], [-0.1]]),
    ]
    p_columns = [np.array(np.squeeze(p), dtype=float)[:, np.newaxis] for p in points]

    PV_plus_p_results = [PV + p for p in points]
    assert all(
        [
            check_matrices_are_equal_ignoring_row_order(PVpp.V, PV.V + p.T)
            for PVpp, p in zip(PV_plus_p_results, p_columns)
        ]
    )

    PV_plus_p_results = [p + PV for p in points]
    assert all(
        [
            check_matrices_are_equal_ignoring_row_order(PVpp.V, PV.V + p.T)
            for PVpp, p in zip(PV_plus_p_results, p_columns)
        ]
    )

    PV_minus_p_results = [PV - p for p in points]
    assert all(
        [
            check_matrices_are_equal_ignoring_row_order(PVmp.V, PV.V - p.T)
            for PVmp, p in zip(PV_minus_p_results, p_columns)
        ]
    )

    PV_minus_p_results = [p + (-PV) for p in points]
    assert all(
        [
            check_matrices_are_equal_ignoring_row_order(PVmp.V, p.T - PV.V)
            for PVmp, p in zip(PV_minus_p_results, p_columns)
        ]
    )

    PH_plus_p_results = [PH + p for p in points]
    assert all(
        [
            check_matrices_are_equal_ignoring_row_order(PHpp.V, PH.V + p.T)
            for PHpp, p in zip(PH_plus_p_results, p_columns)
        ]
    )

    PH_minus_p_results = [PH - p for p in points]
    assert all(
        [
            check_matrices_are_equal_ignoring_row_order(PHmp.V, PH.V - p.T)
            for PHmp, p in zip(PH_minus_p_results, p_columns)
        ]
    )

    with pytest.raises(TypeError):
        PV + "char"
    with pytest.raises(TypeError):
        "char" + PV
    translate = np.ones((2,))
    assert np.allclose((Polytope(dim=2) + translate).V, translate)
    PV + [[1, 1]]
    with pytest.raises(ValueError):
        PV + [[1, 1], [2, 3]]
    with pytest.raises(ValueError):
        PV + [1, 1, 2, 3]

    # Equality constrained polytope
    PH1 = Polytope(c=[1, 1, 0], h=1)
    PH1_3D_slice = PH1.slice(2, 0)
    PH1_3D_slice_translated = PH1_3D_slice - [1, 1, 0]
    assert PH1_3D_slice_translated.n_vertices == 4
    assert np.sum(np.isclose(abs(PH1_3D_slice_translated.V[:, :2]), 1)) == 8

    with pytest.raises(TypeError):
        "char" - PV
    with pytest.raises(TypeError):
        PV - "char"


def test_contains():
    P1 = Polytope(lb=[-1, -1], ub=[1, 1])
    P2 = Polytope(lb=[-0.5, -0.5], ub=[0.5, 0.5])
    P3 = Polytope(A=P1.A, b=P1.b)
    assert P1.contains(P2)
    assert P2 in P1
    assert not P2.contains(P1)
    assert P1 not in P2
    assert P1 == P3
    with pytest.raises(ValueError):
        P1.contains([1, 2, 3])
    with pytest.raises(ValueError):
        P1.contains("char")
    with pytest.raises(ValueError):
        P1.contains(Polytope(lb=-1, ub=1))
    assert all(P1.contains([[-1, 0.5], [1, 0.5]]))
    assert not any(P1.contains([[-2, 0.5], [2, 0.5]]))
    P4 = Polytope(dim=2)
    assert not any(P4.contains([[-1, 0.5], [1, 0.5]]))

    # A non-empty polytope always covers an empty polytope
    assert Polytope(V=np.eye(5)).contains(Polytope(dim=5))
    # Check <= and >=
    assert Polytope(V=np.eye(5)) >= Polytope(dim=5)
    assert Polytope(V=np.eye(5)) > Polytope(dim=5)
    assert Polytope(dim=5) <= Polytope(V=np.eye(5))
    assert Polytope(dim=5) < Polytope(V=np.eye(5))

    # V-Rep (and a single)
    assert (([[1, 2], [2, 1]] <= Polytope(c=[1, 2], h=0)) == [1, 0]).all()
    assert [1, 2] <= Polytope(c=[1, 2], h=0)

    # Equality constrained polytope
    PH1 = Polytope(c=[1, 1, 0], h=1)
    PH1_3D_slice = PH1.slice(2, 0)
    assert [1, 1, 1] in PH1
    assert [1, 1, 1] not in PH1_3D_slice
    assert ([[0, 0, 0], [1, 1, 0]] <= PH1_3D_slice).all()

    # A, b, Ae, be
    Pnew = Polytope(lb=[-1, 1], ub=[1, 1])
    assert not Pnew.is_full_dimensional
    assert Pnew.n_equalities == 1
    assert Pnew.n_halfspaces == 2
    assert Pnew == Pnew

    # Case where nH = 0 and nHe > 0 but bounded
    Pnew = Polytope(A=[1], b=1, Ae=[1], be=[1])
    assert Pnew == Pnew
    assert not Pnew.contains(Polytope(lb=-1, ub=1))

    # Empty polytope with single point
    assert [1, 1, 1] not in Polytope(dim=3)

    # Empty polytope with multiple points
    assert not Polytope(dim=3).contains(np.ones((5, 2))).any()

    # in operator takes only one point
    with pytest.raises(ValueError):
        [[0, 0], [2, 2]] in Polytope(c=[1, 0], h=1)

    # Use <= operator instead
    assert np.allclose([[0, 0], [2, 2]] <= Polytope(c=[1, 0], h=1), [1, 0])

    # Singleton
    P = Polytope(A=np.eye(2), b=[3, 3], Ae=np.eye(2), be=[2, 2])
    assert np.isclose(P.chebyshev_centering()[1], 0)
    assert P.contains([2, 2])
    assert not P.contains([1, 1])
    P = Polytope(V=[[2, 2]])
    assert np.isclose(P.chebyshev_centering()[1], 0)
    assert P.contains([2, 2])
    assert not P.contains([1, 1])

    # Testing from issue #2
    A = np.array([[1, 0], [0, 1], [-1, 0], [0, -1]])
    b = np.array([1, 1, 0, 0])
    polytope = Polytope(A=A, b=b)

    assert polytope.contains(np.array([0.5, 0.5]))  # True
    assert not polytope.contains(np.array([1.5, 1.5]))  # False
    assert not polytope.contains(np.array([1.5, 0.5]))  # False
    assert not polytope.contains(np.array([0.5, 1.5]))  # False
    assert (
        polytope.contains(np.array([[0.5, 0.5], [1.5, 1.5], [1.5, 0.5], [0.5, 1.5]])) == [True, False, False, False]
    ).all()


def test_intersection_and_redundant_inequalities():
    # Intersection calls redundant inequalities
    P1 = Polytope(lb=[0, 0], ub=[1, 1])
    # Sanity check
    assert P1.intersection(P1) == P1

    with pytest.raises(ValueError):
        P1.intersection("char")

    P2 = Polytope(lb=[0.25, 0.25], ub=[1.5, 1.5])
    P1_cap_P2_original_vertex = np.array([[0.25, 0.25], [0.25, 1], [1, 0.25], [1, 1]])
    P1_cap_P2 = P1.intersection(P2)
    assert check_matrices_are_equal_ignoring_row_order(P1_cap_P2.V, P1_cap_P2_original_vertex)

    P3 = Polytope()
    with pytest.raises(ValueError):
        P1.intersection(P3)
    P3 = Polytope(dim=2)
    assert P3.intersection(P1).is_empty
    assert P3.intersection(P1).dim == 2

    P4 = Polytope(lb=[1, 1], ub=[-1, -1])
    assert P4.is_empty
    P4.minimize_H_rep()
    assert P4.is_empty
    assert P4.dim == 2

    P5 = Polytope(lb=[-1, -1], ub=[1, 1])
    assert P5.intersection_with_halfspaces(A=[1, 1], b=[1]).H.shape == (5, P5.dim + 1)

    P6 = Polytope(lb=[-1, -1], ub=[1, 1])
    P6a = P6.intersection_with_halfspaces(A=[1, 1], b=[2])
    P6a.minimize_H_rep()
    assert P6a.H.shape == (4, P6a.dim + 1)

    with pytest.raises(ValueError):
        P6.intersection_with_halfspaces(A=[1, 1, 1], b=[2])

    P7 = Polytope(dim=2)
    P7a = P7.intersection_with_halfspaces(A=[1, 1], b=[2])
    assert P7a.is_empty
    P7a = P7.intersection_with_affine_set(Ae=[1, 1], be=[2])
    assert P7a.is_empty

    P = Polytope(c=[0, 0], h=1)
    assert P.intersection_with_affine_set([[0, 0]], [0]) == P
    assert P.intersection_with_affine_set([[1, 0]], [10]).is_empty
    assert np.allclose(P.intersection_with_affine_set(np.eye(2), [0.5, 0.25]).V, [0.5, 0.25])
    with pytest.raises(ValueError):
        P1.intersection_with_affine_set(Ae=[np.nan, 1], be=[1])
    with pytest.raises(ValueError):
        P1.intersection(ConstrainedZonotope(lb=[-1, -1], ub=[1, 1]))

    # Test intersection_under_inverse_affine_map
    P = Polytope(c=[0, 0], h=1)
    assert P.intersection_under_inverse_affine_map(Q=Polytope(c=[0, 0], h=1), R=np.eye(2)) == P
    P = Polytope(c=[0, 0], h=1).intersection_with_affine_set(Ae=[1, 0], be=[1])
    assert P.intersection_under_inverse_affine_map(Q=Polytope(c=[0, 0], h=1), R=np.eye(2)) == Polytope(
        lb=[1, -1], ub=[1, 1]
    )
    with pytest.raises(ValueError):
        P.intersection_under_inverse_affine_map(Q=ConstrainedZonotope(c=[0, 0], h=1), R=np.eye(2))
    # Invalid R
    with pytest.raises(ValueError):
        P.intersection_under_inverse_affine_map(Q=Polytope(c=[0, 0], h=1), R="char")
    # Incompatible Q and R by virtue of multiplication
    with pytest.raises(ValueError):
        P.intersection_under_inverse_affine_map(Q=Polytope(c=[0, 0], h=1), R=np.ones((3, 3)))
    # Dealing with unbounded Q and non-square R
    P = Polytope(c=[0, 0], h=1)
    P_unbounded_which_is_missed = Polytope(A=[[-1, 0, 0], [0, -1, 0], [1, 1, 0]], b=[0, 0, 1])
    assert P.intersection_under_inverse_affine_map(
        Q=P_unbounded_which_is_missed, R=np.array([[1, 0], [0, 1], [0, 0]])
    ) == Polytope(lb=[0, 0], ub=[1, 1]).intersection_with_halfspaces(A=[1, 1], b=1)
    with pytest.raises(ValueError):
        P_unbounded_which_is_missed.intersection_under_inverse_affine_map(Q=P, R=np.ones((2, 3)))


def test_addition_and_subtraction_by_a_polytope_and_negation():
    P1 = Polytope(lb=[-1, -1], ub=[1, 1])
    P2 = P1.copy()
    P1_plus_P2 = P1 + P2
    P1_plus_P2_correct_vertices = P1.V * 2
    check_matrices_are_equal_ignoring_row_order(P1_plus_P2.V, P1_plus_P2_correct_vertices)
    P1_minus_P2 = P1 - P2
    # Pontyagrin difference is the intersection of the polytope shifted by
    # the vertices
    P1_minus_P2_correct_vertices = np.array([[0, 0]])
    check_matrices_are_equal_ignoring_row_order(P1_minus_P2.V, P1_minus_P2_correct_vertices)

    P3 = Polytope(lb=[-0.5, -0.5], ub=[0.5, 0.5])
    P1_minus_P3 = P1 - P3
    assert P3.in_H_rep
    assert not P3.in_V_rep
    P3b = Polytope(V=P3.V)
    P1_minus_P3b = P1 - P3
    assert not P3b.in_H_rep
    assert P3b.in_V_rep
    assert P1_minus_P3 == P1_minus_P3b
    P3 = Polytope(dim=2)
    assert (P3 + P1) == P1
    assert plus(P3, P1) == P1
    assert (P1 + P3) == P1
    assert plus(P1, P3) == P1

    # Subtraction with P_from_lb_ub
    P3 = Polytope()
    with pytest.raises(ValueError):
        assert (P3 - P1) == -P1
    P3 = Polytope(dim=1)
    with pytest.raises(ValueError):
        assert (P3 - P1) == -P1
    P3 = Polytope(dim=2)
    assert (P1 - P3) == P1
    assert (P3 - P1).is_empty
    assert (P3 + (-P1)) == -P1

    # P4 (a 2D polytope in 3D space) - 3D full-dimensional polytope
    P4 = Polytope(lb=[-1, -1, -1], ub=[1, 1, 1]).intersection_with_affine_set(Ae=[0, 0, 1], be=[0])
    assert not P4.is_full_dimensional
    assert (P4 - Polytope(c=[0, 0, 0], h=1)).is_empty
    Q = P4 - P4
    Q.minimize_V_rep()
    assert not Q.is_empty
    assert Q.V.shape == (1, 3)
    assert np.allclose(Q.V, 0)

    with pytest.raises(TypeError):
        plus(P1, "char")
    with pytest.raises(TypeError):
        minus(P1, "char")
    with pytest.raises(ValueError):
        plus(P1, Polytope(lb=-1, ub=1))


def test_polytope_product_with_vector_m_times_set_set_times_m():
    P1 = Polytope(lb=[-1, -1], ub=[1, 1])
    M = np.diag([5, 5])
    P2 = M @ P1
    assert (5 * P1) == P2
    # Test inverse multiplication with anything
    with pytest.raises(TypeError):
        P1 * (1 / 5)
    assert (0 * P1) != Polytope(dim=2)
    assert np.all((0 * P1).V == np.zeros((1, 2)))

    P3 = P1.inverse_affine_map_under_invertible_matrix(M)
    assert ((1 / 5) * P1) == P3
    with pytest.raises(ValueError):
        [1, 2, 3] @ P2
    with pytest.raises(ValueError):
        [[1, 2, 3]] @ P2
    with pytest.raises(ValueError):
        [1, 2, 3] * P2
    with pytest.raises(ValueError):
        [[1, 2, 3]] * P2
    with pytest.raises(TypeError):
        P2 * P1
    with pytest.raises(TypeError):
        P1.affine_map(P2)
    with pytest.raises(TypeError):
        P1.affine_map("char")
    with pytest.raises(ValueError):
        P1.affine_map(np.zeros((3, 3, 2)))

    # Vector multiplication with 1D
    theta_vec = np.arange(0, 2 * np.pi, np.pi / 6)
    P4 = Polytope(V=np.vstack((np.cos(theta_vec), np.sin(theta_vec))).T)
    P5 = np.array([1, 0]) @ P4
    P5.minimize_V_rep()
    assert P5.dim == 1
    assert P5.n_vertices == 2
    assert np.isclose(np.max(P5.V), 1)
    assert np.isclose(np.min(P5.V), -1)

    # Multiplication with an empty polytope
    assert (Polytope(dim=5).affine_map(np.eye(5))).is_empty
    assert (
        Polytope(dim=5).affine_map(
            np.ones(
                5,
            )
        )
    ).is_empty
    with pytest.raises(ValueError):
        Polytope(dim=5).affine_map(np.eye(4))
    assert (Polytope(dim=5).affine_map(np.eye(5))).is_empty
    assert (
        Polytope(dim=5).affine_map(
            np.ones(
                5,
            )
        )
    ).is_empty
    with pytest.raises(ValueError):
        np.eye(4) * Polytope(V=np.eye(5))
    with pytest.raises(ValueError):
        np.eye(4) @ Polytope(V=np.eye(5))
    with pytest.raises(ValueError):
        np.ones(
            4,
        ) @ Polytope(V=np.eye(5))

    # M @ low-dimensional polytope
    P = Polytope(lb=[-1, -1], ub=[1, 1]).intersection_with_affine_set(Ae=[1, 0], be=1)
    Q = [1, 0] @ P
    Q.minimize_V_rep()
    assert Q.n_vertices == 1
    assert Q.dim == 1
    assert np.isclose(Q.V, 1)
    Q = 0.5 @ P
    Q.minimize_V_rep()
    assert Q.n_vertices == 2
    assert Q.dim == 2
    assert check_matrices_are_equal_ignoring_row_order(Q.V, [[0.5, -0.5], [0.5, 0.5]])

    # Inverse affine map with an empty polytope
    with pytest.raises(ValueError):
        assert Polytope(dim=5).inverse_affine_map_under_invertible_matrix(np.eye(5))
    with pytest.raises(ValueError):
        assert Polytope(dim=5).inverse_affine_map_under_invertible_matrix(
            np.ones(
                5,
            )
        )
    # High-dimensional M
    with pytest.raises(ValueError):
        Polytope(V=np.eye(5)).inverse_affine_map_under_invertible_matrix(
            np.array([[[1, 0, 0], [1, 2, 3]], [[1, 0, 0], [1, 2, 3]]])
        )
    # When M is 0
    with pytest.raises(TypeError):
        Polytope(V=np.eye(5)) * 0
    # Compute using inverse of M (vector)
    Q = Polytope(c=[0, 0], h=1)
    with pytest.raises(ValueError):
        Q @ np.array([[1, 0]]).T
    # Compute using inverse of M in V-Rep (scalar)
    R = Polytope(V=[[0, 0], [0, 1], [1, 0]])
    Ra = 0.5 * R
    assert np.isclose(R.volume(), Ra.volume() * 4)
    with pytest.raises(ValueError):
        Polytope(V=np.eye(5)).inverse_affine_map_under_invertible_matrix(
            np.ones(
                4,
            )
        )
    assert np.isclose(np.max(np.abs((0.5 * Polytope(V=np.eye(5))).V)), 0.5)

    assert (2 * Polytope(dim=5)).is_empty

    with pytest.raises(ValueError):
        inverse_affine_map_under_invertible_matrix(R, 2)

    with pytest.raises(TypeError):
        "char" * R

    with pytest.raises(ValueError):
        R @ np.zeros((2, 2))

    with pytest.raises(TypeError):
        R @ "char"

    with pytest.raises(TypeError):
        inverse_affine_map_under_invertible_matrix(R, "char")

    Q = Polytope(c=[0, 0, 0], h=1).intersection_with_affine_set(Ae=[0, 0, 1], be=1)
    Q_new = Q @ np.diag([2, 2, 2])
    assert np.allclose(Q_new.V[:, 2], 0.5)

    Rnew = R @ np.diag([2, 2])
    assert check_matrices_are_equal_ignoring_row_order(Rnew.V, [[0, 0], [0, 0.5], [0.5, 0]])


def test_project_closest_point_distance():
    P = Polytope(c=[0, 0, 0], h=0.5)
    P.project([1, 1, 1], p=1)
    P.project([1, 1, 1], p=2)
    assert np.allclose(P.closest_point([1, 1, 1]), P.project([1, 1, 1], p=2)[0])
    assert np.isclose(P.distance([1, 1, 1]), P.project([1, 1, 1], p=2)[1])
    P.project([1, 1, 1], p="inf")
    with pytest.raises(ValueError):
        P.project([1, 1, 1], p=3)
    # Check for CVXPY error
    P.cvxpy_args_socp = {"solver": "WRONG_SOLVER"}
    with pytest.raises(NotImplementedError):
        P.project([1, 1, 1], p=2)
    with pytest.raises(ValueError):
        P.project([[1, 1, 1, 0], [2, 3, 1, 0]], p=2)
    with pytest.raises(ValueError):
        P.project([[[1, 2, 1], [1, 3, 1]], [[1, 4, 1], [1, 5, 1]]], p=2)
    with pytest.raises(ValueError):
        Polytope(dim=3).project([1, 1, 1], p=2)


def test_slice():
    P = Polytope(c=[0, 0, 0], h=0.5)
    Q = P.slice(dims=2, constants=0.25)
    assert Q <= P
    assert Q.dim == 3
    Q_projection = Q.projection(2)
    assert Q_projection.dim == 2
    assert np.isclose(Q_projection.volume(), 1)
    print(Q.H)
    print(Q.He)
    print(P.intersection_with_affine_set(Ae=[0, 0, 1], be=0.25).H)
    print(P.intersection_with_affine_set(Ae=[0, 0, 1], be=0.25).He)
    print(Q == P.intersection_with_affine_set(Ae=[0, 0, 1], be=0.25))
    assert Q == P.intersection_with_affine_set(Ae=[0, 0, 1], be=0.25)
    assert P.slice(dims=[2], constants=[1]).is_empty
    # Dimension value out of bounds
    with pytest.raises(ValueError):
        Polytope(c=[0, 0, 0], h=0.5).slice(dims=3, constants=0.5)
    # Inconsistent dims and constants
    with pytest.raises(ValueError):
        Polytope(c=[0, 0, 0], h=0.5).slice(dims=2, constants=[0.5, 1])
    with pytest.raises(ValueError):
        Polytope(c=[0, 0, 0], h=0.5).slice(dims=[2, 1], constants=0.5)
    # Invalidself.dim
    with pytest.raises(ValueError):
        Polytope(c=[0, 0, 0], h=0.5).slice(dims=0.2, constants=0.5)


def test_slice_then_projection():
    P = Polytope(c=[0, 0, 0], h=0.5)
    Q = P.slice(dims=2, constants=0.25).projection(project_away_dims=2)
    Q_projection = P.slice_then_projection(dims=2, constants=0.25)
    assert Q.dim == 2
    assert Q == Q_projection


def test_chebyshev_centering_support_extreme():
    # Example based on
    # http://web.cvxr.com/cvx/examples/cvxbook/Ch04_cvx_opt_probs/html/chebyshev_center_2D.html
    A = np.array([[2, 1], [2, -1], [-1, 2], [-1, -2]])
    b = np.ones((4,))
    P = Polytope(A=A, b=b)
    center, radius = P.chebyshev_centering()
    assert P.contains(center)
    for support_direction in P.A:
        support_direction_unit = support_direction / np.linalg.norm(support_direction)
        assert np.abs(support_direction_unit @ center + radius - P.support(support_direction_unit)[0]) <= 1e-5

    # Check for CVXPY error
    P.cvxpy_args_lp = {"solver": "WRONG_SOLVER"}
    with pytest.raises(NotImplementedError):
        P.chebyshev_centering()

    P0 = Polytope()
    assert P0.chebyshev_centering() == (None, 0)
    with pytest.raises(ValueError):
        P0.support([1, 1])

    P1 = Polytope(lb=[-1, -1], ub=[1, 1])
    c, radius = P1.chebyshev_centering()
    assert np.allclose(c, np.array([0, 0]))
    assert np.allclose(radius, 1)
    support_direction = np.array([1, 1])
    assert np.allclose(P1.support(support_direction)[0], np.max(support_direction @ P1.V.T))

    P2 = Polytope(V=P1.V)
    c, radius = P2.chebyshev_centering()
    assert np.allclose(c, np.array([0, 0]))
    assert np.allclose(radius, 1)

    M = [[[1, 2], [2, 3]], [[1, 2], [2, 3]]]
    with pytest.raises(ValueError):
        P2.support(M)

    M = [[1, 2, 3], [2, 3, 4]]
    with pytest.raises(ValueError):
        P2.support(M)

    M = [[1, 2, 3]]
    with pytest.raises(ValueError):
        P2.support(M)

    M = [1, 2, 3]
    with pytest.raises(ValueError):
        P2.support(M)

    # Vertex support and polytope support matches
    P2 = Polytope(V=P1.V)
    assert np.isclose(P2.support([1, 1])[0], P1.support([1, 1])[0])
    assert np.all(np.isclose(P2.extreme([1, 1]), P1.extreme([1, 1])))

    # Check for CVXPY error
    P2.cvxpy_args_lp = {"solver": "WRONG_SOLVER"}
    with pytest.raises(NotImplementedError):
        P2.support([1, 1])

    # Case where nH = 0 and nHe > 0 but bounded
    Pnew = Polytope(A=[1], b=1, Ae=[1], be=[1])
    c, r = Pnew.chebyshev_centering()
    assert np.isclose(c, 1)
    assert np.isclose(r, 0)
    assert np.isclose(Pnew.support(1)[0], 1)
