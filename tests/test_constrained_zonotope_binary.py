# Copyright (C) 2020-2026 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

# Code purpose: Test the ConstrainedZonotope class for methods related to binary operations

import warnings

import numpy as np
import pytest

from pycvxset import ConstrainedZonotope, Ellipsoid, Polytope, spread_points_on_a_unit_sphere
from pycvxset.common.constants import PYCVXSET_ZERO, TESTING_STATEMENTS_INVOLVING_GUROBI


def assert_warns_or_raises_valueerror(callable_func):
    import warnings

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        try:
            callable_func()
        except ValueError:
            pass
        else:
            assert any(issubclass(warn.category, UserWarning) for warn in w), "Expected UserWarning not raised"


# Ignore solution inaccuracy error through out all tests in this module.
@pytest.fixture(autouse=True)
def suppress_cvxpy_warnings():
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Solution may be inaccurate.*", category=UserWarning, module="cvxpy")
        yield


def test_project_closest_point_distance():
    P = Polytope(c=[0, 0, 0], h=0.5)
    C = ConstrainedZonotope(polytope=P)
    point = [1, 1, 1]
    assert np.allclose(C.project(point, p=1)[0], P.project(point, p=1)[0])
    assert np.allclose(C.project(point, p=2)[0], P.project(point, p=2)[0])
    assert np.allclose(C.project(point, p="inf")[0], P.project(point, p="inf")[0])
    projection_result = C.project(point, p=2)
    assert np.allclose(C.closest_point([1, 1, 1]), projection_result[0])
    assert np.isclose(C.distance(point), projection_result[1])
    with pytest.raises(ValueError):
        C.project(point, p=3)
    # Check for CVXPY error
    C.cvxpy_args_socp = {"solver": "WRONG_SOLVER"}
    with pytest.raises(NotImplementedError):
        C.project(point, p=2)
    with pytest.raises(ValueError):
        C.project([[1, 1, 1, 0], [2, 3, 1, 0]], p=2)
    with pytest.raises(ValueError):
        C.project([[[1, 2, 1], [1, 3, 1]], [[1, 4, 1], [1, 5, 1]]], p=2)
    with pytest.raises(ValueError):
        ConstrainedZonotope(dim=3).project(point, p=2)


def test_support_extreme():
    points_on_unit_sphere = spread_points_on_a_unit_sphere(2, 5)[0]
    P = Polytope(V=points_on_unit_sphere)
    C = ConstrainedZonotope(polytope=P)
    direction_vectors = points_on_unit_sphere
    assert np.allclose(C.support(direction_vectors)[0], P.support(direction_vectors)[0])
    assert np.max(np.abs(C.extreme(direction_vectors) - P.extreme(direction_vectors))) <= PYCVXSET_ZERO

    # Check for CVXPY error
    C.cvxpy_args_lp = {"solver": "WRONG_SOLVER"}
    with pytest.raises(NotImplementedError):
        C.support([1, 1])


def test_plus_scalar_multiplication():
    # Test inverse multiplication with anything
    with pytest.raises(TypeError):
        ConstrainedZonotope(lb=[-1, -1], ub=[1, 1]) * (1 / 5)
    # Bad multiplier
    with pytest.raises(TypeError):
        "char" * ConstrainedZonotope(lb=[-1, -1], ub=[1, 1])
    # Bad addition
    with pytest.raises(TypeError):
        "char" + ConstrainedZonotope(lb=[-1, -1], ub=[1, 1])
    with pytest.raises(ValueError):
        [1, 1, 1] + ConstrainedZonotope(lb=[-1, -1], ub=[1, 1])
    with pytest.raises(ValueError):
        ConstrainedZonotope(lb=[-1, -1], ub=[1, 1]) + [1, 1, 1]
    with pytest.raises(ValueError):
        np.random.rand(2, 2) + ConstrainedZonotope(lb=[-1, -1], ub=[1, 1])
    with pytest.raises(ValueError):
        ConstrainedZonotope(lb=[-1, -1], ub=[1, 1]) + np.random.rand(2, 2)

    with pytest.raises(ValueError):
        ConstrainedZonotope(lb=[-1, -1], ub=[1, 1]) + ConstrainedZonotope(lb=[-1, -1, -1], ub=[1, 1, 1])

    # Test plus and multiplication of a zonotope with scalar and commutative property of plus
    Z1 = (0.5 * ConstrainedZonotope(lb=[-1, -1], ub=[1, 1])) + [0.5, 0.5]
    assert not Z1.is_empty
    assert Z1.is_zonotope
    assert np.allclose(Z1.G, 0.5 * np.eye(2))
    assert np.allclose(Z1.c, 0.5 * np.ones(2))
    Z2 = [0.5, 0.5] + (0.5 * ConstrainedZonotope(lb=[-1, -1], ub=[1, 1]))
    assert not Z2.is_empty
    assert Z2.is_zonotope
    assert np.allclose(Z2.G, 0.5 * np.eye(2))
    assert np.allclose(Z2.c, 0.5 * np.ones(2))

    # Test plus and multiplication of a constrained zonotope with scalar and commutative property of plus
    Z1 = (0.5 * ConstrainedZonotope(lb=[-1, -1], ub=[1, 1])) + [0.5, 0.5]
    P = Polytope(V=spread_points_on_a_unit_sphere(2, 7)[0])
    Z1 = (0.5 * ConstrainedZonotope(polytope=P)) + [0.5, 0.5]
    assert not Z1.is_empty
    assert not Z1.is_zonotope
    Z2 = [0.5, 0.5] + (0.5 * ConstrainedZonotope(polytope=P))
    assert not Z2.is_empty
    assert not Z2.is_zonotope
    if TESTING_STATEMENTS_INVOLVING_GUROBI == "full":
        assert_warns_or_raises_valueerror(lambda: Z1 == Z2)
    elif TESTING_STATEMENTS_INVOLVING_GUROBI == "minimal":
        assert_warns_or_raises_valueerror(lambda: Z1.contains(Z2, time_limit=0.1))
        assert_warns_or_raises_valueerror(lambda: Z2.contains(Z1, time_limit=0.1))

    # Test plus with polytope (and thereby with constrained zonotope)
    C3 = Z1 + P
    C4 = P + Z1
    assert not C3.is_empty
    assert not C4.is_empty
    assert not C3.is_zonotope
    assert not C4.is_zonotope
    if TESTING_STATEMENTS_INVOLVING_GUROBI == "full":
        assert_warns_or_raises_valueerror(lambda: C4 == C3)
    elif TESTING_STATEMENTS_INVOLVING_GUROBI == "minimal":
        assert_warns_or_raises_valueerror(lambda: C4.contains(C3, time_limit=0.1))
        assert_warns_or_raises_valueerror(lambda: C3.contains(C4, time_limit=0.1))

    assert ConstrainedZonotope(dim=2) + ConstrainedZonotope(dim=2) == ConstrainedZonotope(dim=2)
    C5 = ConstrainedZonotope(lb=[-1, -1], ub=[1, 1])
    C5_plus_empty = C5 + ConstrainedZonotope(dim=2)
    assert np.allclose(C5_plus_empty.G, C5.G)
    assert np.allclose(C5_plus_empty.c, C5.c)
    empty_plus_C5 = ConstrainedZonotope(dim=2) + C5
    assert np.allclose(empty_plus_C5.G, C5.G)
    assert np.allclose(empty_plus_C5.c, C5.c)


def test_matrix_times_set():
    # Bad multiplier
    with pytest.raises(ValueError):
        "char" @ ConstrainedZonotope(lb=[-1, -1], ub=[1, 1])
    with pytest.raises(ValueError):
        [1, 1, 1] @ ConstrainedZonotope(lb=[-1, -1], ub=[1, 1])
    with pytest.raises(ValueError):
        np.random.rand(2, 3) @ ConstrainedZonotope(lb=[-1, -1], ub=[1, 1])

    # Test zonotope multiplication with @
    Z = ConstrainedZonotope(lb=[-1, -1], ub=[1, 1])
    Z1 = (2 * np.eye(2)) @ Z
    Z2 = 2 * Z
    assert not Z.is_empty
    assert not Z.is_empty
    assert Z1.is_zonotope
    assert Z2.is_zonotope
    assert np.allclose(Z1.G, Z2.G)
    assert np.allclose(Z1.c, Z2.c)

    # Test constrained zonotope multiplication with @
    C = ConstrainedZonotope(polytope=Polytope(V=spread_points_on_a_unit_sphere(2, 7)[0]))
    C1 = (2 * np.eye(2)) @ C
    C2 = 2 * C
    assert not C1.is_empty
    assert not C2.is_empty
    assert not C1.is_zonotope
    assert not C2.is_zonotope
    if TESTING_STATEMENTS_INVOLVING_GUROBI == "full":
        with pytest.raises(ValueError):
            assert C1 == C2

    with pytest.raises(ValueError):
        np.ones((3, 2, 2)) @ C

    C4 = 0 * C
    assert not C4.is_empty
    assert C4.is_zonotope
    assert C4.G.size == 0
    assert np.allclose(C4.c, 0)

    empty_C = np.eye(4)[:, :2] @ ConstrainedZonotope(dim=2)
    assert empty_C.is_empty
    assert empty_C.dim == 4

    empty_C = np.eye(2) @ ConstrainedZonotope(dim=2)
    assert empty_C.is_empty
    assert empty_C.dim == 2

    empty_C = 0 @ ConstrainedZonotope(dim=2)
    assert empty_C.is_empty
    assert empty_C.dim == 2


def test_set_times_matrix():
    # Test zonotope multiplication with @
    Z = ConstrainedZonotope(lb=[-1, -1], ub=[1, 1])
    Z1 = Z @ (2 * np.eye(2))
    Z2 = 0.5 * Z
    assert not Z.is_empty
    assert not Z.is_empty
    assert Z1.is_zonotope
    assert Z2.is_zonotope
    assert np.allclose(Z1.G, Z2.G)
    assert np.allclose(Z1.c, Z2.c)

    # Test constrained zonotope multiplication with @
    C = ConstrainedZonotope(polytope=Polytope(V=spread_points_on_a_unit_sphere(2, 7)[0]))
    C1 = C @ (2 * np.eye(2))
    C2 = 0.5 * C
    assert not C1.is_empty
    assert not C2.is_empty
    assert not C1.is_zonotope
    assert not C2.is_zonotope
    if TESTING_STATEMENTS_INVOLVING_GUROBI == "full":
        with pytest.raises(ValueError):
            assert C1 == C2

    # Bad multiplier
    with pytest.raises(ValueError):
        ConstrainedZonotope(lb=[-1, -1], ub=[1, 1]) @ "char"
    with pytest.raises(ValueError):
        ConstrainedZonotope(lb=[-1, -1], ub=[1, 1]) @ [1, 1, 1]
    with pytest.raises(ValueError):
        ConstrainedZonotope(lb=[-1, -1], ub=[1, 1]) @ np.random.rand(2, 3)
    with pytest.raises(ValueError):
        ConstrainedZonotope(lb=[-1, -1], ub=[1, 1]) @ np.zeros((2, 2))


def test_rsub_neg():
    C = ConstrainedZonotope(lb=[-1, -1], ub=[1, 1])
    C_without_using_rsub = [0.5, 0.5] + (-C)
    assert C_without_using_rsub.is_zonotope
    with pytest.raises(TypeError):
        [0.5, 0.5] - C
    with pytest.raises(TypeError):
        "char" - C


def test_contains():
    # Define two polytopes, one inside the other, and check if the corresponding constrained zonotopes contain each
    # other
    P1_in_V_rep = Polytope(V=spread_points_on_a_unit_sphere(2, 5)[0])
    C1 = ConstrainedZonotope(polytope=P1_in_V_rep)
    with pytest.raises(ValueError):
        [[0, 0], [2, 2]] in ConstrainedZonotope(c=[0, 0], G=np.eye(2))
    assert [0, 0] in C1
    assert [10, 10] not in C1
    assert [0, 0] not in ConstrainedZonotope(dim=2)
    assert [0, 0] in ConstrainedZonotope(c=[0, 0], G=None)
    assert [1, 1] not in ConstrainedZonotope(c=[0, 0], G=None)
    assert not ConstrainedZonotope(dim=2).contains(np.random.rand(5, 2)).any()
    with pytest.raises(ValueError):
        C1.contains([10, 10, 10])
    assert ConstrainedZonotope(dim=2) <= C1
    assert not C1 <= ConstrainedZonotope(dim=2)
    with pytest.raises(ValueError):
        ConstrainedZonotope() <= C1
    with pytest.raises(ValueError):
        ConstrainedZonotope(lb=[-1, -1, -1], ub=[1, 1, 1]) <= C1

    P1_in_H_rep = Polytope(A=P1_in_V_rep.A, b=P1_in_V_rep.b)
    # Check Polytope == Polytope (V-Rep and H-Rep)
    assert P1_in_H_rep == P1_in_V_rep
    # Check Polytope == ConstrainedZonotope (via V-Rep)
    assert P1_in_V_rep == C1
    # Strict containment
    P2 = 0.5 * P1_in_H_rep
    C2 = ConstrainedZonotope(polytope=P2)
    assert P2 in P1_in_H_rep
    # Polytope.contains(CZ) via <=
    assert C2 <= P1_in_H_rep
    # Polytope.contains(CZ) via >=
    assert P1_in_H_rep >= C2
    # Polytope.contains(CZ) via <=
    assert C2 <= P1_in_V_rep
    # Polytope.contains(CZ) via >=
    assert P1_in_V_rep >= C2
    if TESTING_STATEMENTS_INVOLVING_GUROBI == "full":
        # CZ.contains(Polytope) via >=
        assert C1 >= P2
        # CZ.contains(Polytope) via <=
        assert P2 <= C1
        assert P2.slice(dims=[1], constants=[0.5]) <= C1
        # Check Polytope == ConstrainedZonotope (via H-Rep)
        assert P1_in_H_rep == C1
        Z1 = ConstrainedZonotope(lb=[-1, -1], ub=[1, 1])
        Z2 = 0.1 * Z1
        # Test in, neq
        # CZ.contains(CZ) via in
        assert C2 in C1
        # Zonotope.contains(CZ) and CZ.contains(Zonotope) via !=
        assert Z2 != C2
        # Zonotope.contains(CZ) via >=
        assert Z1 >= C2
        # Zonotope.contains(CZ) via <=
        assert Z2 <= Z1
    elif TESTING_STATEMENTS_INVOLVING_GUROBI == "minimal":
        assert C2.contains(P2)
        Z1 = ConstrainedZonotope(lb=[-1, -1], ub=[1, 1])
        assert Z1 >= C2
        assert Z1 >= P2
        assert C2 >= 0.2 * Z1
        assert P2 >= 0.2 * Z1

    # Sanity check
    C1 = ConstrainedZonotope(lb=[-1, -1], ub=[1, 1])
    C2 = ConstrainedZonotope(polytope=Polytope(lb=[-1, -1], ub=[1, 1]))
    assert C1.G.shape == (2, 2)
    assert C2.G.shape == (2, 6)
    if TESTING_STATEMENTS_INVOLVING_GUROBI == "full":
        assert C1 == C2

    # Testing from issue #2
    A = np.array([[1, 0], [0, 1], [-1, 0], [0, -1]])
    b = np.array([1, 1, 0, 0])
    CZ = ConstrainedZonotope(polytope=Polytope(A=A, b=b))

    assert CZ.contains(np.array([0.5, 0.5]))
    assert not CZ.contains(np.array([1.5, 1.5]))
    assert not CZ.contains(np.array([1.5, 0.5]))
    assert not CZ.contains(np.array([0.5, 1.5]))
    assert (
        CZ.contains(np.array([[0.5, 0.5], [1.5, 1.5], [1.5, 0.5], [0.5, 1.5]])) == np.array([True, False, False, False])
    ).all()

    # Definitely not contained
    P1_in_H_rep = Polytope(A=P1_in_V_rep.A, b=P1_in_V_rep.b)
    C1 = ConstrainedZonotope(polytope=P1_in_H_rep)
    P2 = P1_in_H_rep + [100, 100]
    C2 = ConstrainedZonotope(polytope=P2)
    assert not C1 == C2

    # __ge__ testing (and a single)
    with pytest.raises(TypeError):
        assert (([[1, 2], [2, 1]] <= ConstrainedZonotope(c=[1, 2], h=0)) == [1, 0]).all()
    with pytest.raises(TypeError):
        assert ((ConstrainedZonotope(c=[1, 2], h=0) >= [[1, 2], [2, 1]]) == [1, 0]).all()
    ConstrainedZonotope(c=[1, 2], h=0).contains([[1, 2], [2, 1]])
    assert [1, 2] <= ConstrainedZonotope(c=[1, 2], h=0)

    with pytest.raises(TypeError):
        assert "char" <= ConstrainedZonotope(c=[1, 0], h=1)
    with pytest.raises(TypeError):
        assert ConstrainedZonotope(c=[1, 0], h=1) >= "char"
    with pytest.raises(ValueError):
        ConstrainedZonotope(c=[1, 0], h=1).contains(Ellipsoid(c=[1, 0], r=1))


def test_intersection():
    # Test intersection of two zonotopes
    C1 = ConstrainedZonotope(lb=[-1, -1], ub=[1, 1])
    C2 = ConstrainedZonotope(lb=[0, 0], ub=[2, 2])
    C1_cap_C2 = C1.intersection(C2)
    lb, ub = C1_cap_C2.minimum_volume_circumscribing_rectangle()
    assert C1.is_zonotope
    assert C2.is_zonotope
    assert np.allclose(lb, [0, 0])
    assert np.allclose(ub, [1, 1])
    assert not C1_cap_C2.is_empty
    if TESTING_STATEMENTS_INVOLVING_GUROBI == "full":
        assert C1_cap_C2 == Polytope(lb=lb, ub=ub)

    # Test intersection of two constrained zonotopes (and intersection with halfspaces)
    C1 = ConstrainedZonotope(lb=[-1, -1], ub=[1, 1]).intersection_with_halfspaces(A=[1, 1], b=1)
    P1 = Polytope(lb=[-1, -1], ub=[1, 1]).intersection_with_halfspaces(A=[1, 1], b=1)
    C2 = ConstrainedZonotope(lb=[0, 0], ub=[2, 2]).intersection_with_halfspaces(A=[-1, -1], b=-0.5)
    P2 = Polytope(lb=[0, 0], ub=[2, 2]).intersection_with_halfspaces(A=[-1, -1], b=-0.5)
    assert not C1.is_zonotope
    assert not C2.is_zonotope
    P1_cap_P2 = P1.intersection(P2)
    C1_cap_C2 = C1.intersection(C2)
    assert not P1_cap_P2.is_empty
    assert not C1_cap_C2.is_empty
    lb_C, ub_C = C1_cap_C2.minimum_volume_circumscribing_rectangle()
    lb_P, ub_P = P1_cap_P2.minimum_volume_circumscribing_rectangle()
    assert np.allclose(lb_C, lb_P)
    assert np.allclose(ub_C, ub_P)
    if TESTING_STATEMENTS_INVOLVING_GUROBI == "full":
        assert C1_cap_C2 == P1_cap_P2

    # no halfspace
    C1 = ConstrainedZonotope(lb=[-1, -1], ub=[1, 1]).intersection_with_halfspaces(A=None, b=None)
    P1 = Polytope(lb=[-1, -1], ub=[1, 1]).intersection_with_halfspaces(A=None, b=None)
    assert C1.is_zonotope
    lb_C, ub_C = C1.minimum_volume_circumscribing_rectangle()
    lb_P, ub_P = P1.minimum_volume_circumscribing_rectangle()
    assert np.allclose(lb_C, lb_P)
    assert np.allclose(ub_C, ub_P)
    if TESTING_STATEMENTS_INVOLVING_GUROBI == "full":
        assert C1_cap_C2 == P1_cap_P2
    with pytest.raises(ValueError):
        ConstrainedZonotope(lb=[-1, -1], ub=[1, 1]).intersection_with_halfspaces(A=None, b=1)
    with pytest.raises(ValueError):
        ConstrainedZonotope(lb=[-1, -1], ub=[1, 1]).intersection_with_halfspaces(A=[1, 1], b=None)

    # ConstrainedZonotope intersected with a non-overlapping halfspace => Empty
    C3 = ConstrainedZonotope(lb=[0, 0], ub=[2, 2]).intersection_with_halfspaces(A=[-1, -1], b=-5)
    assert C3.is_empty
    with pytest.raises(ValueError):
        ConstrainedZonotope(lb=[0, 0], ub=[2, 2]).intersection_with_halfspaces(A=[-1, -1, 0], b=-5)
    assert ConstrainedZonotope(dim=2).intersection_with_halfspaces(A=[-1, -1], b=-5).is_empty

    # Test intersection of two constrained zonotopes (and intersection with affine_sets)
    C1 = ConstrainedZonotope(lb=[-1, -1], ub=[1, 1]).intersection_with_affine_set(Ae=[1, 1], be=1)
    P1 = Polytope(lb=[-1, -1], ub=[1, 1]).intersection_with_affine_set(Ae=[1, 1], be=1)
    C2 = ConstrainedZonotope(lb=[0, 0], ub=[2, 2]).intersection_with_affine_set(Ae=[1, 1], be=1)
    P2 = Polytope(lb=[0, 0], ub=[2, 2]).intersection_with_affine_set(Ae=[1, 1], be=1)
    assert not C1.is_zonotope
    assert not C2.is_zonotope
    P1_cap_P2 = P1.intersection(P2)
    P1_cap_P2.minimize_H_rep()
    C1_cap_C2 = C1.intersection(C2)
    assert not P1_cap_P2.is_empty
    assert not C1_cap_C2.is_empty
    lb_C, ub_C = C1_cap_C2.minimum_volume_circumscribing_rectangle()
    lb_P, ub_P = P1_cap_P2.minimum_volume_circumscribing_rectangle()
    assert np.allclose(lb_C, lb_P)
    assert np.allclose(ub_C, ub_P)
    if TESTING_STATEMENTS_INVOLVING_GUROBI == "full":
        with pytest.raises(ValueError):
            assert C1_cap_C2 == P1_cap_P2

    # Test intersection of a constrained zonotope and a zonotope
    C1 = ConstrainedZonotope(lb=[-1, -1], ub=[1, 1]).intersection_with_halfspaces(A=[1, 1], b=1)
    P1 = Polytope(lb=[-1, -1], ub=[1, 1]).intersection_with_halfspaces(A=[1, 1], b=1)
    C2 = ConstrainedZonotope(lb=[0, 0], ub=[2, 2])
    P2 = Polytope(lb=[0, 0], ub=[2, 2])
    P2_cap_P1 = P2.intersection(P1)
    C1_cap_C2 = C1.intersection(C2)
    C2_cap_C1 = C2.intersection(C1)
    assert not C1_cap_C2.is_empty
    assert not C2_cap_C1.is_empty
    if TESTING_STATEMENTS_INVOLVING_GUROBI == "full":
        assert C1_cap_C2 == C2_cap_C1
        assert P2_cap_P1 == C2_cap_C1

    # Test intersection of a constrained zonotope and a polytope (in V-Rep)
    C1 = ConstrainedZonotope(lb=[-1, -1], ub=[1, 1])
    P2_V = Polytope(V=[[0, 0], [0, 2], [2, 0], [2, 2]])
    C1_cap_P2_V = C1.intersection(P2_V)
    lb_C1_P2, ub_C1_P2 = C1_cap_P2_V.minimum_volume_circumscribing_rectangle()
    assert np.allclose(lb_C1_P2, [0, 0])
    assert np.allclose(ub_C1_P2, [1, 1])
    assert not C1_cap_P2_V.is_empty
    if TESTING_STATEMENTS_INVOLVING_GUROBI == "full":
        assert C1_cap_P2_V == Polytope(lb=lb_C1_P2, ub=ub_C1_P2)

    # Test intersection of a constrained zonotope and a polytope (no He)
    C1 = ConstrainedZonotope(lb=[-1, -1], ub=[1, 1])
    P2 = Polytope(lb=[0, 0], ub=[2, 2])
    C1_cap_P2 = C1.intersection(P2)
    lb_C1_P2, ub_C1_P2 = C1_cap_P2.minimum_volume_circumscribing_rectangle()
    assert np.allclose(lb_C1_P2, [0, 0])
    assert np.allclose(ub_C1_P2, [1, 1])
    assert not C1_cap_P2.is_empty
    assert not P2_cap_P1.is_empty
    if TESTING_STATEMENTS_INVOLVING_GUROBI == "full":
        assert C1_cap_P2_V == C1_cap_P2

    # Test intersection of a constrained zonotope and a polytope (with He)
    C1 = ConstrainedZonotope(lb=[-1, -1], ub=[1, 1])
    P2 = Polytope(lb=[0, 0], ub=[2, 2]).intersection_with_affine_set(Ae=[1, -1], be=[0])
    C1_cap_P2 = C1.intersection(P2)
    lb_C1_P2, ub_C1_P2 = C1_cap_P2.minimum_volume_circumscribing_rectangle()
    assert np.allclose(lb_C1_P2, [0, 0])
    assert np.allclose(ub_C1_P2, [1, 1])
    assert not C1_cap_P2.is_empty
    assert C1_cap_P2 == Polytope(V=[[0, 0], [1, 1]])
    if TESTING_STATEMENTS_INVOLVING_GUROBI == "full":
        assert C1_cap_P2 != Polytope(lb=lb_C1_P2, ub=ub_C1_P2)

    with pytest.raises(TypeError):
        C1.intersection("char")

    with pytest.raises(ValueError):
        C1.intersection_with_affine_set(Ae=[1, 1, 1], be=1)

    # no Affine set
    C2 = C1.intersection_with_affine_set(Ae=[], be=[])
    assert np.allclose(C2.G, C1.G)
    assert np.allclose(C2.c, C1.c)
    C2 = C1.intersection_with_affine_set(Ae=None, be=None)
    assert np.allclose(C2.G, C1.G)
    assert np.allclose(C2.c, C1.c)
    with pytest.raises(ValueError):
        C1.intersection_with_affine_set(Ae=None, be=[0.5, 0.5])
    with pytest.raises(ValueError):
        C1.intersection_with_affine_set(Ae=np.eye(2), be=None)

    # Affine set is infeasible
    C2 = C1.intersection_with_affine_set(Ae=[[1, 0], [1, 0]], be=[2, 3])
    assert C2.is_empty

    # Affine set does not intersect with C1
    C2 = C1.intersection_with_affine_set(Ae=np.eye(2), be=[2, 3])
    assert C2.is_empty

    # Affine set is a point
    C2 = C1.intersection_with_affine_set(Ae=np.eye(2), be=[0.5, 0.5])
    assert np.allclose(C2.c, [0.5, 0.5])
    assert C2.G.size == 0

    # Constrained zonotope intersected with affine set
    C1 = (
        ConstrainedZonotope(lb=[-1, -1], ub=[1, 1])
        .intersection_with_halfspaces(A=[1, 1], b=1)
        .intersection_with_affine_set(Ae=[1, 0], be=-1)
    )
    P1 = (
        Polytope(lb=[-1, -1], ub=[1, 1])
        .intersection_with_halfspaces(A=[1, 1], b=1)
        .intersection_with_affine_set(Ae=[1, 0], be=-1)
    )
    assert np.isclose(P1.extreme([0, 1]), C1.extreme([0, 1])).all()
    assert np.isclose(P1.extreme([0, -1]), C1.extreme([0, -1])).all()

    # Invalid intersections under inverse_affine_map
    with pytest.raises(ValueError):
        C1.intersection_under_inverse_affine_map("char", np.eye(2))
    with pytest.raises(ValueError):
        C1.intersection_under_inverse_affine_map(C2, "char")
    with pytest.raises(ValueError):
        C1.intersection_under_inverse_affine_map(C2, np.zeros((2, 3)))
    with pytest.raises(ValueError):
        C1.intersection_under_inverse_affine_map(C2, np.array([np.zeros((2, 2)), np.ones((2, 2))]))

    # Empty intersections under inverse_affine_map and standard intersection
    assert ConstrainedZonotope(dim=2).intersection_under_inverse_affine_map(C1, np.eye(2)).is_empty
    assert C1.intersection_under_inverse_affine_map(ConstrainedZonotope(dim=2), np.eye(2)).is_empty
    assert ConstrainedZonotope(dim=2).intersection(C1).is_empty
    assert C1.intersection(ConstrainedZonotope(dim=2)).is_empty


def test_minus():
    # Test subtraction of two zonotopes
    C1 = ConstrainedZonotope(lb=[-1, -1], ub=[1, 1])
    C2 = ConstrainedZonotope(lb=[-0.5, -0.5], ub=[0.5, 0.5])
    P1 = Polytope(lb=[-1, -1], ub=[1, 1])
    P2 = Polytope(lb=[-0.5, -0.5], ub=[0.5, 0.5])
    P1_minus_P2 = P1 - P2
    with pytest.warns(UserWarning, match="This function*"):
        C1_minus_C2 = C1 - C2
    C1_minus_C2 = C1.minus(C2, enable_warning=False)
    assert C1_minus_C2.is_zonotope
    assert not P1_minus_P2.is_empty
    assert not C1_minus_C2.is_empty
    assert C1_minus_C2 <= P1_minus_P2

    # Test subtraction of constrained zonotope and zonotope
    C1 = ConstrainedZonotope(lb=[-1, -1], ub=[1, 1]).intersection_with_halfspaces(A=[1, 1], b=0.75)
    C2 = ConstrainedZonotope(lb=[-0.5, -0.5], ub=[0.5, 0.5])
    P1 = Polytope(lb=[-1, -1], ub=[1, 1])
    P2 = Polytope(lb=[-0.5, -0.5], ub=[0.5, 0.5])
    P1_minus_P2 = P1 - P2
    with pytest.warns(UserWarning, match="This function*"):
        C1_minus_C2 = C1 - C2
    assert not C1_minus_C2.is_zonotope
    assert not P1_minus_P2.is_empty
    assert not C1_minus_C2.is_empty
    assert C1_minus_C2 <= P1_minus_P2

    # Test subtraction of zonotope minus constrained zonotope
    C1 = ConstrainedZonotope(lb=[-1, -1], ub=[1, 1]).intersection_with_halfspaces(A=[1, 1], b=0)
    C2 = ConstrainedZonotope(lb=[-0.5, -0.5], ub=[0.5, 0.5])
    with pytest.raises(TypeError):
        C2 - C1

    # Test subtraction of constrained zonotope minus ellipsoid
    C1 = ConstrainedZonotope(lb=[-1, -1], ub=[1, 1]).intersection_with_halfspaces(A=[1, 1], b=0)
    P1 = Polytope(lb=[-1, -1], ub=[1, 1]).intersection_with_halfspaces(A=[1, 1], b=0)
    ell = Ellipsoid(c=[0, 0], r=0.1)
    P1_minus_ell = P1 - ell
    with pytest.warns(UserWarning, match="This function*"):
        C1_minus_ell = C1 - ell
    C1_minus_ell = C1.minus(ell, enable_warning=False)
    assert not C1_minus_ell.is_empty
    assert not P1_minus_ell.is_empty
    assert C1_minus_ell <= P1_minus_ell

    # Test subtraction of constrained zonotope minus unit l1-norm ball
    C1 = ConstrainedZonotope(lb=[-1, -1], ub=[1, 1]).intersection_with_halfspaces(A=[1, 1], b=1.95)
    P1 = Polytope(lb=[-1, -1], ub=[1, 1]).intersection_with_halfspaces(A=[1, 1], b=1.95)
    P2 = Polytope(V=np.vstack((np.eye(2), -np.eye(2))))
    C1_minus_P2_inner = C1.approximate_pontryagin_difference(1, np.eye(2), np.zeros((2,)), enable_warning=False)
    P1_minus_P2 = P1 - P2
    assert not P1_minus_P2.is_empty
    assert not C1_minus_P2_inner.is_empty
    assert C1_minus_P2_inner <= P1_minus_P2
    C1 = ConstrainedZonotope(lb=[-1, -1], ub=[1, 1]).intersection_with_halfspaces(A=[1, 1], b=0)
    C1_minus_P2_inner = C1.approximate_pontryagin_difference(1, np.eye(2), np.zeros((2,)), enable_warning=False)
    assert C1_minus_P2_inner.is_empty

    # Test subtraction of constrained zonotope minus point
    C1 = ConstrainedZonotope(lb=[-1, -1], ub=[1, 1]).intersection_with_halfspaces(A=[1, 1], b=0)
    Q = np.ones((2,))
    C1_minus_Q = C1 - Q
    assert not C1_minus_Q.is_empty
    assert np.allclose(C1_minus_Q.G, C1.G)
    assert np.allclose(C1_minus_Q.c, C1.c - Q)
    assert np.allclose(C1_minus_Q.Ae, C1.Ae)
    assert np.allclose(C1_minus_Q.be, C1.be)
    C1 = ConstrainedZonotope(lb=[-1, -1], ub=[1, 1]).intersection_with_halfspaces(A=[1, 1], b=0)
    Q = ConstrainedZonotope(G=None, c=np.ones((2,)))
    C1_minus_Q = C1 - Q
    assert not C1_minus_Q.is_empty
    assert np.allclose(C1_minus_Q.G, C1.G)
    assert np.allclose(C1_minus_Q.c, C1.c - Q.c)
    assert np.allclose(C1_minus_Q.Ae, C1.Ae)
    assert np.allclose(C1_minus_Q.be, C1.be)
    C1 = ConstrainedZonotope(lb=[-1, -1], ub=[1, 1]).intersection_with_halfspaces(A=[1, 1], b=0)
    Q = Ellipsoid(c=np.ones((2,)))
    C1_minus_Q = C1 - Q
    assert not C1_minus_Q.is_empty
    assert np.allclose(C1_minus_Q.G, C1.G)
    assert np.allclose(C1_minus_Q.c, C1.c - Q.c)
    assert np.allclose(C1_minus_Q.Ae, C1.Ae)
    assert np.allclose(C1_minus_Q.be, C1.be)
    C1 = ConstrainedZonotope(lb=[-1, -1], ub=[1, 1]).intersection_with_halfspaces(A=[1, 1], b=0)
    Q = Ellipsoid(c=np.ones((2,)), r=0)
    C1_minus_Q = C1 - Q
    assert not C1_minus_Q.is_empty
    assert np.allclose(C1_minus_Q.G, C1.G)
    assert np.allclose(C1_minus_Q.c, C1.c - Q.c)
    assert np.allclose(C1_minus_Q.Ae, C1.Ae)
    assert np.allclose(C1_minus_Q.be, C1.be)

    # CZ minus polytope
    with pytest.raises(TypeError):
        C1 - Polytope(c=[2, 2], h=1)

    # CZ minus char invalid input
    with pytest.raises(TypeError):
        C1 - "char"

    # Invalid input: norm_type
    with pytest.raises(ValueError):
        C1.approximate_pontryagin_difference(10, C2.G, C2.c)
    # Invalid input: method is invalid
    with pytest.raises(ValueError):
        C1.approximate_pontryagin_difference(1, C2.G, C2.c, method="invalid_method")
    # Invalid input: 2D_CZ - 3D_Z
    with pytest.raises(ValueError):
        with pytest.warns(UserWarning, match="This function*"):
            C1 - ConstrainedZonotope(lb=[-1, -1, -1], ub=[1, 1, 1])

    # Many redundancies
    C1 = ConstrainedZonotope(
        G=np.hstack((np.eye(2), np.zeros((2, 1)))), c=np.ones((2,)), Ae=np.ones((4, 3)), be=np.ones((4,))
    )
    with pytest.raises(ValueError):
        with pytest.warns(UserWarning, match="This function*"):
            C1 - Ellipsoid(c=[0, 0], r=2)
    with pytest.raises(ValueError):
        C1.approximate_pontryagin_difference(2, G_S=np.eye(2), c_S=[0, 0])
    C1.remove_redundancies()
    with pytest.warns(UserWarning, match="This function*"):
        C1 - Ellipsoid(c=[0, 0], r=2)

    C1 = ConstrainedZonotope(dim=5)
    with pytest.warns(UserWarning, match="This function*"):
        C1_minus_ell = C1 - Ellipsoid(c=np.zeros((5,)), r=1)
    assert C1_minus_ell.is_empty
    C1 = ConstrainedZonotope(lb=np.zeros((5,)), ub=np.ones((5,)))
    C1_minus_empty = C1 - ConstrainedZonotope(dim=5)
    assert C1_minus_empty.is_zonotope
    assert not C1_minus_empty.is_empty
    assert np.allclose(C1_minus_empty.G, C1.G)
    assert np.allclose(C1_minus_empty.c, C1.c)


def test_remove_redundant_inequalities():

    # Single point
    Z = ConstrainedZonotope(G=np.eye(2), Ae=np.eye(2), c=np.zeros((2,)), be=np.ones((2,)))
    Z.remove_redundancies()
    assert not Z.is_empty
    assert Z.dim == 2
    assert Z.latent_dim == 0
    assert Z.is_zonotope
    assert not Z.is_empty
    assert Z.Ae.shape == (0, 0)
    assert Z.be.size == 0
    assert Z.He.shape == (0, Z.latent_dim + 1)
    assert Z.G.size == 0
    assert np.allclose(Z.c, 1)
    Z = ConstrainedZonotope(lb=3 * np.ones((2,)), ub=3 * np.ones((2,)))
    Z.remove_redundancies()
    assert not Z.is_empty
    assert Z.dim == 2
    assert Z.latent_dim == 0
    assert Z.is_zonotope
    assert not Z.is_empty
    assert Z.Ae.shape == (0, 0)
    assert Z.be.size == 0
    assert Z.He.shape == (0, Z.latent_dim + 1)
    assert Z.G.size == 0
    assert np.allclose(Z.c, 3)
    Z = ConstrainedZonotope(G=None, c=[3, 3])
    Z.remove_redundancies()
    assert not Z.is_empty
    assert Z.dim == 2
    assert Z.latent_dim == 0
    assert Z.is_zonotope
    assert not Z.is_empty
    assert Z.Ae.shape == (0, 0)
    assert Z.be.size == 0
    assert Z.He.shape == (0, Z.latent_dim + 1)
    assert Z.G.size == 0
    assert np.allclose(Z.c, 3)

    # Define infeasible (Ae, be)
    Z = ConstrainedZonotope(G=np.eye(2), Ae=[[1, 0], [1, 0]], c=np.zeros((2,)), be=[1, 2])
    Z.remove_redundancies()
    assert Z.dim == 2
    assert Z.latent_dim == 0
    assert Z.is_zonotope
    assert Z.is_empty
    assert Z.Ae.shape == (0, 0)
    assert Z.be.size == 0
    assert Z.He.shape == (0, Z.latent_dim + 1)
    assert Z.G.size == 0
    assert Z.c is None
