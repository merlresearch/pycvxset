# Copyright (C) 2020-2026 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

# Code purpose: Test the ConstrainedZonotope class for constructor and initialization

import cvxpy as cp
import numpy as np
import pytest

from pycvxset import ConstrainedZonotope, Ellipsoid, Polytope
from pycvxset.common.constants import DEFAULT_CVXPY_ARGS_LP, TESTING_STATEMENTS_INVOLVING_GUROBI


def test_constrained_zonotope_init():
    # Zonotope from bounds
    lb = [-1, -1]
    ub = [1, 1]
    Z = ConstrainedZonotope(lb=lb, ub=ub)
    assert np.allclose(Z.G, np.eye(2))
    assert np.allclose(Z.c, np.zeros((2,)))
    assert Z.dim == 2
    assert Z.latent_dim == 2
    assert Z.Ae.shape == (0, 2)
    assert Z.be.shape == (0,)
    assert not Z.is_empty
    assert Z.is_bounded
    assert Z.is_zonotope
    assert Z.He.shape == (0, Z.latent_dim + 1)

    Z_alt = ConstrainedZonotope(c=[0, 0], h=1)
    assert np.allclose(Z.G, Z_alt.G)
    assert np.allclose(Z.c, Z_alt.c)
    assert Z_alt.is_zonotope
    assert Z_alt.is_bounded
    assert not Z_alt.is_empty

    Z_alt = ConstrainedZonotope(c=[0, 0], h=[1, 1])
    assert np.allclose(Z.G, Z_alt.G)
    assert np.allclose(Z.c, Z_alt.c)
    assert Z_alt.is_zonotope
    assert Z_alt.is_bounded
    assert not Z_alt.is_empty

    # Empty constrained zonotope
    Z = ConstrainedZonotope(dim=2)
    assert Z.is_empty
    assert Z.is_bounded
    assert Z.is_zonotope
    assert Z.dim == 2
    assert Z.latent_dim == 0
    assert Z.Ae.shape == (0, 0)
    assert Z.be.size == 0
    assert Z.He.shape == (0, Z.latent_dim + 1)
    assert Z.G.size == 0
    assert Z.c is None
    Z = ConstrainedZonotope(G=np.eye(2), c=None)
    assert Z.is_empty
    assert Z.is_bounded
    assert Z.is_zonotope
    assert Z.dim == 2
    assert Z.latent_dim == 0
    assert Z.Ae.shape == (0, 0)
    assert Z.be.size == 0
    assert Z.He.shape == (0, Z.latent_dim + 1)
    assert Z.G.size == 0
    assert Z.c is None
    Z = ConstrainedZonotope()
    assert Z.is_empty
    assert Z.dim == 0
    assert Z.latent_dim == 0
    assert Z.is_zonotope
    assert Z.is_empty
    assert Z.Ae.shape == (0, 0)
    assert Z.be.size == 0
    assert Z.He.shape == (0, Z.latent_dim + 1)
    assert Z.G.size == 0
    assert Z.c is None
    Z = ConstrainedZonotope(lb=[1, 1], ub=[2, -1])
    assert Z.is_empty
    assert Z.is_bounded
    assert Z.is_zonotope
    assert Z.dim == 2
    assert Z.latent_dim == 0
    assert Z.Ae.shape == (0, 0)
    assert Z.be.size == 0
    assert Z.He.shape == (0, Z.latent_dim + 1)
    assert Z.G.size == 0
    assert Z.c is None

    for Z in [
        ConstrainedZonotope(G=np.eye(2), c=np.zeros((2,)), Ae=None, be=None),
        ConstrainedZonotope(G=np.eye(2), c=np.zeros((2,)), Ae=None, be=[]),
        ConstrainedZonotope(G=np.eye(2), c=np.zeros((2,)), Ae=[], be=None),
        ConstrainedZonotope(G=np.eye(2), c=np.zeros((2,)), Ae=[], be=[]),
    ]:
        assert not Z.is_empty
        assert Z.is_bounded
        assert Z.is_zonotope
        assert Z.dim == 2
        assert Z.latent_dim == 2
        assert Z.Ae.shape == (0, 2)
        assert Z.be.size == 0
        assert Z.G.size == 4
        assert np.allclose(Z.c, np.zeros((2,)))

    # Constrained zonotope
    Z = ConstrainedZonotope(G=np.eye(2), c=np.zeros((2,)), Ae=[1, 0], be=[2])
    assert Z.is_empty
    assert Z.is_bounded
    assert not Z.is_zonotope
    assert Z.dim == 2
    assert Z.latent_dim == 2
    assert Z.is_empty
    assert Z.Ae.shape == (1, 2)
    assert Z.be.size == 1
    assert np.allclose(Z.He, np.array((1, 0, 2)))
    assert Z.G.size == 4
    assert np.allclose(Z.c, np.zeros((2,)))

    # Zonotope even though Ae, be are defined
    Z = ConstrainedZonotope(G=np.eye(2), c=[[1, 1]], Ae=[0, 0], be=[0])
    assert np.allclose(Z.G, np.eye(2))
    assert np.allclose(Z.c, np.ones((2,)))
    assert Z.dim == 2
    assert Z.latent_dim == 2
    assert Z.Ae.shape == (0, 2)
    assert Z.be.shape == (0,)
    assert Z.He.shape == (0, Z.latent_dim + 1)
    assert not Z.is_empty
    assert Z.is_bounded
    assert Z.is_zonotope

    with pytest.raises(ValueError):
        ConstrainedZonotope(G=None, c=None)
    C = ConstrainedZonotope(G=np.eye(2), c=None)
    assert C.is_empty
    assert C.dim == 2
    assert ConstrainedZonotope(G=None, c=[1, 1]).is_singleton


def test_constrained_zonotope_from_bad_inputs():
    # Mismatched dimensions in G and c
    with pytest.raises(ValueError):
        ConstrainedZonotope(G=np.eye(2), c=np.zeros((3,)))
    # G is not 2D matrix
    with pytest.raises(ValueError):
        ConstrainedZonotope(G=[[[1, 2], [1, 2]], [[1, 2], [1, 2]]], c=np.zeros((3,)))
    # G is not float matrix
    ConstrainedZonotope(G=[0, 1], c=0)
    with pytest.raises(ValueError):
        ConstrainedZonotope(G=[0, "char"], c=0)
    # c is not convertible into float vector
    ConstrainedZonotope(G=np.eye(2), c=[0, 1])
    with pytest.raises(ValueError):
        ConstrainedZonotope(G=np.eye(2), c=[0, "char"])
    # G or c has nan
    with pytest.raises(ValueError):
        ConstrainedZonotope(G=np.eye(2), c=[0, np.nan])
    with pytest.raises(ValueError):
        ConstrainedZonotope(G=[0, np.nan], c=1)
    # c is None
    Z = ConstrainedZonotope(G=[0, 1], c=None)
    assert Z.is_empty
    assert Z.dim == 1
    Z = ConstrainedZonotope(G=np.eye(3), c=None)
    assert Z.is_empty
    assert Z.dim == 3
    # c is not 1D array
    ConstrainedZonotope(G=np.eye(2), c=[[0, 1]])
    with pytest.raises(ValueError):
        ConstrainedZonotope(G=np.eye(2), c=np.eye(2))
    # Invalid combinations
    with pytest.raises(ValueError):
        ConstrainedZonotope(G=np.eye(2), Ae=np.eye(2), be=np.zeros((2,)))
    with pytest.raises(ValueError):
        ConstrainedZonotope(G=np.eye(2), Ae=np.eye(2))
    with pytest.raises(ValueError):
        ConstrainedZonotope(Ae=np.eye(2), be=np.eye(2))
    with pytest.raises(ValueError):
        ConstrainedZonotope(G=np.eye(2), Ae=np.eye(2), c=np.zeros((2,)))
    with pytest.raises(ValueError):
        ConstrainedZonotope(dim=2, G=np.eye(2))
    with pytest.raises(ValueError):
        ConstrainedZonotope(Ae=2, lb=[-1, -1], ub=[1, 1])
    with pytest.raises(ValueError):
        ConstrainedZonotope(c=[3, 3])
    with pytest.raises(ValueError):
        ConstrainedZonotope(G=[3, 3])
    with pytest.raises(ValueError):
        ConstrainedZonotope(G=None, c=None)
    # Ae and G have different columns!
    with pytest.raises(ValueError):
        ConstrainedZonotope(G=np.eye(2), c=[[0, 1]], Ae=[1], be=[1])
    # Bad lb, ub
    with pytest.raises(ValueError):
        ConstrainedZonotope(lb=[-1, -1], ub=[1, 1, 1])
    with pytest.raises(ValueError):
        ConstrainedZonotope(lb=[-1, -1, -1], ub=[1, 1])
    with pytest.raises(ValueError):
        ConstrainedZonotope(lb=[-1, "char"], ub=[1, 1])
    with pytest.raises(ValueError):
        ConstrainedZonotope(lb=[-1, -1], ub=[1, "char"])
    with pytest.raises(ValueError):
        ConstrainedZonotope(lb=[[-1, 2], [-1, 2]], ub=[1, 1])
    with pytest.raises(ValueError):
        ConstrainedZonotope(lb=[-1, -1], ub=[[-1, 2], [-1, 2]])
    with pytest.raises(ValueError):
        ConstrainedZonotope(lb=[-1, -1], polytope=Polytope(dim=2))
    with pytest.raises(NotImplementedError):
        Z = ConstrainedZonotope(G=np.eye(2), c=np.zeros((2,)), Ae=[1, 0], be=[2])
        Z.cvxpy_args_lp = {"solver": "WRONG_SOLVER"}
        assert Z.is_empty

    # Create zonotope from c and h
    with pytest.raises(ValueError):
        ConstrainedZonotope(c=((0, 0), (0, 0)), h=1)

    with pytest.raises(ValueError):
        ConstrainedZonotope(c=(0, 0, 0), h=(1, 1, 1, 1))

    with pytest.raises(ValueError):
        ConstrainedZonotope(c=(0, 0, 0), h=((1, 1, 0), (1, 1, 0)))

    with pytest.raises(ValueError):
        ConstrainedZonotope(c=((0, 0), (0, 0)), h=1, lb=1)

    with pytest.raises(ValueError):
        ConstrainedZonotope(c="char", h=1)

    with pytest.raises(ValueError):
        ConstrainedZonotope(c=[0, 0], h="char")

    with pytest.raises(ValueError):
        ConstrainedZonotope(G=None, c=[0, 0], Ae=[1, 1], be=[1])


def test_constrained_zonotope_from_polytope():
    lb = [-1, -1]
    ub = [1, 1]

    # From H-Rep and V-Rep
    P = Polytope(lb=lb, ub=ub)
    Z = ConstrainedZonotope(polytope=P)
    Z_V = ConstrainedZonotope(polytope=Polytope(V=[[-1, -1], [-1, 1], [1, -1], [1, 1]]))

    # From H-Rep
    assert Z.dim == 2
    assert Z.latent_dim == P.n_halfspaces + P.dim
    assert Z.G.shape == (2, P.n_halfspaces + P.dim)
    assert Z.c.shape == (2,)
    assert Z.Ae.shape == (P.n_equalities + P.n_halfspaces, P.n_halfspaces + P.dim)
    assert Z.be.shape == (P.n_equalities + P.n_halfspaces,)
    assert not Z.is_empty

    # From V-Rep
    assert Z_V.dim == 2
    assert Z_V.latent_dim == P.n_vertices
    assert Z_V.G.shape == (2, P.n_vertices)
    assert Z_V.c.shape == (2,)
    assert Z_V.Ae.shape == (1, P.n_vertices)
    assert Z_V.be.shape == (1,)
    assert not Z_V.is_empty

    # Unbounded polytope
    unbounded_P = Polytope(A=np.eye(2), b=np.ones((2,)))
    with pytest.raises(ValueError):
        ConstrainedZonotope(polytope=unbounded_P)

    # Sliced polytope
    sliced_P = P.intersection_with_affine_set(Ae=[1, 0], be=1)
    with pytest.warns(UserWarning, match="Removed some rows in (Ae, be)*"):
        CZ = ConstrainedZonotope(polytope=sliced_P)
    assert CZ.dim == 2
    assert not CZ.is_empty

    # Polytope in V-Rep
    polytope_in_v_rep = Polytope(V=[[1, 0], [0, 0], [0, 1]])
    CZ = ConstrainedZonotope(polytope=polytope_in_v_rep)
    assert CZ.dim == 2
    assert CZ.latent_dim == polytope_in_v_rep.n_vertices
    assert CZ.G.shape == (2, polytope_in_v_rep.n_vertices)
    assert CZ.c.shape == (2,)
    assert CZ.latent_dim == polytope_in_v_rep.n_vertices
    assert CZ.Ae.shape == (1, polytope_in_v_rep.n_vertices)
    assert CZ.be.shape == (1,)
    assert not CZ.is_empty
    if TESTING_STATEMENTS_INVOLVING_GUROBI == "full":
        CZ == polytope_in_v_rep

    # Sliced polytope in V-Rep
    polytope_in_v_rep = Polytope(V=[[1, 0, 2], [0, 0, 2], [0, 1, 2]])
    CZ = ConstrainedZonotope(polytope=polytope_in_v_rep)
    assert CZ.dim == 3
    assert CZ.latent_dim == polytope_in_v_rep.n_vertices
    assert CZ.G.shape == (3, polytope_in_v_rep.n_vertices)
    assert CZ.c.shape == (3,)
    assert CZ.latent_dim == polytope_in_v_rep.n_vertices
    assert CZ.Ae.shape == (1, polytope_in_v_rep.n_vertices)
    assert CZ.be.shape == (1,)
    assert not CZ.is_empty
    if TESTING_STATEMENTS_INVOLVING_GUROBI == "full":
        CZ == polytope_in_v_rep

    # Single point
    Z1 = ConstrainedZonotope(polytope=Polytope(V=[[1]]))
    assert Z1.is_full_dimensional
    Z1 = ConstrainedZonotope(c=[[1]], G=None)
    assert Z1.is_full_dimensional
    Z1 = ConstrainedZonotope(c=[[1]], h=0)
    assert Z1.is_full_dimensional
    Z1 = ConstrainedZonotope(lb=[1], ub=[1])
    assert Z1.is_full_dimensional
    Z2 = ConstrainedZonotope(polytope=Polytope(V=[[1, 1]]))
    assert not Z2.is_full_dimensional
    Z2 = ConstrainedZonotope(c=[[1, 1]], G=None)
    assert not Z2.is_full_dimensional
    Z2 = ConstrainedZonotope(c=[[1, 1]], h=0)
    assert not Z2.is_full_dimensional
    Z2 = ConstrainedZonotope(lb=[[1, 1]], ub=[[1, 1]])
    assert not Z2.is_full_dimensional

    with pytest.raises(ValueError):
        ConstrainedZonotope(polytope=None)
    with pytest.raises(ValueError):
        ConstrainedZonotope(polytope=Ellipsoid(c=[0, 0], r=1))


def test_print():
    Z0 = ConstrainedZonotope(polytope=Polytope(dim=2))
    assert str(Z0) == "Constrained Zonotope (empty) in R^2"
    assert repr(Z0) == "Constrained Zonotope (empty) in R^2"
    Z1 = ConstrainedZonotope(polytope=Polytope(V=[[1, 1]]))
    assert str(Z1) == "Constrained Zonotope in R^2"
    assert repr(Z1) == "Constrained Zonotope in R^2\n\tthat is a zonotope representing a single point"
    Z1a = ConstrainedZonotope(c=[[1, 1]], G=None)
    assert str(Z1a) == "Constrained Zonotope in R^2"
    assert repr(Z1a) == "Constrained Zonotope in R^2\n\tthat is a zonotope representing a single point"
    Z2 = ConstrainedZonotope(ub=[1, 1], lb=[-1, -1])
    assert str(Z2) == "Constrained Zonotope in R^2"
    assert repr(Z2) == "Constrained Zonotope in R^2\n\tthat is a zonotope with latent dimension 2"
    Z2a = ConstrainedZonotope(G=np.eye(2), c=[0, 0])
    assert str(Z2a) == "Constrained Zonotope in R^2"
    assert repr(Z2a) == "Constrained Zonotope in R^2\n\tthat is a zonotope with latent dimension 2"
    C1 = Z2.intersection_with_affine_set(Ae=[1, 1], be=[1])
    assert str(C1) == "Constrained Zonotope in R^2"
    assert repr(C1) == "Constrained Zonotope in R^2\n\twith latent dimension 2 and 1 equality constraint"
    with pytest.warns(UserWarning, match="Removed some rows in "):
        C2 = ConstrainedZonotope(polytope=Polytope(ub=[1, 1], lb=[1, -1]))
    assert str(C2) == "Constrained Zonotope in R^2"
    assert repr(C2) == "Constrained Zonotope in R^2\n\twith latent dimension 3 and 2 equality constraints"


def test_neg():
    C1 = ConstrainedZonotope(lb=[-1, -1], ub=[1, 1])
    C2 = -C1
    assert np.allclose(C1.G, -C2.G)
    assert np.allclose(C1.c, -C2.c)
    assert np.allclose(C1.Ae, C2.Ae)
    assert np.allclose(C1.be, C2.be)
    assert not any([C1.is_empty, C2.is_empty])
    assert all([C1.is_zonotope, C2.is_zonotope])
    assert all([C1.is_bounded, C2.is_bounded])

    C1 = [0.5, 0.5] + ConstrainedZonotope(lb=[-1, -1], ub=[1, 1])
    C2 = -C1
    assert np.allclose(C1.G, -C2.G)
    assert np.allclose(C1.c, -C2.c)
    assert np.allclose(C1.Ae, C2.Ae)
    assert np.allclose(C1.be, C2.be)
    assert not any([C1.is_empty, C2.is_empty])
    assert all([C1.is_zonotope, C2.is_zonotope])
    assert all([C1.is_bounded, C2.is_bounded])


def test_copy():
    C1 = ConstrainedZonotope(lb=[-1, -1], ub=[1, 1])
    C1.copy()
    C2 = ConstrainedZonotope(polytope=Polytope(lb=[-1, -1], ub=[1, 1]))
    C2.copy()
    if TESTING_STATEMENTS_INVOLVING_GUROBI == "full":
        assert C2 == C1


def test_pow_and_cartesian_product():
    C1 = ConstrainedZonotope(lb=[-1, -1], ub=[1, 1])
    C2 = C1**2
    assert C2.dim == 4
    if TESTING_STATEMENTS_INVOLVING_GUROBI == "full":
        assert C2 == ConstrainedZonotope(lb=[-1, -1, -1, -1], ub=[1, 1, 1, 1])
    C3 = C1.cartesian_product(C1)
    assert C2.dim == 4
    if TESTING_STATEMENTS_INVOLVING_GUROBI == "full":
        assert C3 == C2
    P1 = Polytope(lb=[-1, -1], ub=[1, 1])
    C4 = C1.cartesian_product(P1)
    assert C4.dim == 4
    if TESTING_STATEMENTS_INVOLVING_GUROBI == "full":
        assert C4 == C2
    with pytest.raises(ValueError):
        C1.cartesian_product(Ellipsoid(c=[1, 1], r=1))
    C5 = ConstrainedZonotope(dim=5)
    C5_pow = C5**3
    assert C5_pow.dim == 15
    assert C5_pow.is_empty


def test_is_full_dimensional_is_empty_is_singleton():
    # Empty CZ > Not full-dimensional
    C = ConstrainedZonotope(G=np.hstack((np.eye(2), np.zeros((2, 1)))), c=np.ones((2,)), Ae=[1, 1, 1], be=[3.1])
    assert C.is_empty
    assert not C.is_full_dimensional
    assert not C.is_singleton
    # Non-empty CZ and full-dimensional
    C = ConstrainedZonotope(G=np.hstack((np.eye(2), np.zeros((2, 1)))), c=np.ones((2,)), Ae=[1, 1, 1], be=[0])
    assert C.is_full_dimensional
    assert not C.is_empty
    assert not C.is_singleton
    # Intersected with an affine set > Non-empty and non-full dimensional
    C_new = C.intersection_with_affine_set(Ae=[1, 0], be=0)
    assert not C_new.is_empty
    assert not C_new.is_full_dimensional
    assert not C_new.is_singleton
    # Zonotope > Full-dimensional and non-empty
    Z = ConstrainedZonotope(G=np.hstack((np.eye(2), np.zeros((2, 1)))), c=np.ones((2,)))
    assert Z.is_full_dimensional
    assert not Z.is_empty
    assert not Z.is_singleton
    # Intersected with an affine set > Non-empty and non-full dimensional
    Z_new = Z.intersection_with_affine_set(Ae=[1, 0], be=0)
    assert not Z_new.is_empty
    assert not Z_new.is_full_dimensional
    assert not Z.is_singleton
    # Z_new defined directly to check full-dimensional despite remove_redundancies
    CZ = ConstrainedZonotope(G=np.eye(2), c=np.ones((2,)), Ae=np.eye(2), be=[0, 0])
    assert not CZ.is_full_dimensional
    assert CZ.is_singleton
    # Singleton Zonotope in 2D > Non-empty and non-full-dimensional
    Z = ConstrainedZonotope(G=None, c=[0, 0])
    assert not Z.is_full_dimensional
    assert not Z.is_empty
    assert Z.is_singleton
    # Singleton Zonotope in 1D > Non-empty and Full-dimensional
    Z = ConstrainedZonotope(G=None, c=[0])
    assert Z.is_full_dimensional
    assert not Z.is_empty
    assert Z.is_singleton
    # Singleton due Ae, be
    Z3 = ConstrainedZonotope(c=[[1, 1]], G=np.eye(2), Ae=np.eye(2), be=[0.5, 0.5])
    assert not Z3.is_empty
    assert not Z3.is_full_dimensional
    assert Z3.is_singleton
    assert [[1, 1]] not in Z3
    assert [[1.5, 1.5]] in Z3
    Z4 = ConstrainedZonotope(c=[[1]], G=[1, 1], Ae=np.eye(2), be=[0.5, 0.5])
    assert not Z4.is_empty
    assert Z4.is_full_dimensional  # 1-D point in 1D is full-dimensional
    assert Z4.is_singleton
    assert [[1]] not in Z4
    assert [[2]] in Z4
    Z5 = ConstrainedZonotope(c=[[1]], G=[1, 1], Ae=np.eye(2), be=[1.5, 1.5])
    assert Z5.is_empty
    assert not Z5.is_full_dimensional
    assert not Z5.is_singleton

    # A CZ with unnecessary latent variable 4 (Effect of remove_redundancies) > Full-dimensional and non-empty
    C = ConstrainedZonotope(G=np.hstack((np.eye(3), np.zeros((3, 1)))), c=np.zeros((3,)), Ae=[0, 0, 0, 1], be=[0])
    C_new = [0, 1, 0] @ C
    assert not C.is_zonotope  # Technically it is a zonotope, Ae, be is present, so we say it is not a zonotope
    assert not C_new.is_zonotope  # Technically it is a zonotope, Ae, be is present, so we say it is not a zonotope
    assert C.is_full_dimensional
    assert C_new.is_full_dimensional
    assert not C_new.is_singleton
    # After intersection, the constrained zonotope remains full-dimensional and becomes a singleton
    C_new_2 = C_new.intersection_with_affine_set(Ae=[1], be=[0])
    assert C_new_2.latent_dim == 0
    assert C_new_2.is_full_dimensional
    assert C_new_2.is_singleton

    # After affine transformation, the constrained zonotope remains full-dimensional
    C_new = [[0, 1, 0], [1, 0, 0]] @ C
    assert C_new.is_full_dimensional
    assert C_new.latent_dim == 3  # latent_dim has 1, 2 and 4 columns left, due to remove_redundancies
    # After intersection, the constrained zonotope is no longer full-dimensional
    C_new_2 = C_new.intersection_with_affine_set(Ae=[1, 0], be=[0])
    assert not C_new_2.is_full_dimensional

    C = ConstrainedZonotope(dim=3)
    C.remove_redundancies()
    assert C.is_empty


def test_containment_constraints():
    C = ConstrainedZonotope(lb=[-1, -1], ub=[1, 1])
    opt_x_value = C.extreme([1, 1])
    for test_shape in [2, (2,), (2, 1), (1, 2), (1, 1, 2), (1, 2, 1)]:
        x = cp.Variable(test_shape).flatten(order="F")
        # Warn only if 3D variables
        check_warning = False
        try:
            check_warning = len(test_shape) > 2
        except TypeError:
            pass
        prob = cp.Problem(cp.Maximize(cp.sum(x)), C.containment_constraints(x)[0])
        if check_warning:
            with pytest.warns(UserWarning):
                prob.solve(**DEFAULT_CVXPY_ARGS_LP)
        else:
            prob.solve(**DEFAULT_CVXPY_ARGS_LP)
        assert np.isclose(x.value, opt_x_value).all()
        prob = cp.Problem(cp.Maximize(cp.sum(x)), C.containment_constraints(x.flatten(order="F"))[0])
        if check_warning:
            with pytest.warns(UserWarning):
                prob.solve(**DEFAULT_CVXPY_ARGS_LP)
        else:
            prob.solve(**DEFAULT_CVXPY_ARGS_LP)
        assert np.isclose(x.value, opt_x_value).all()

    # Empty constrained zonotope
    C = ConstrainedZonotope(dim=2)
    x = cp.Variable(2)
    with pytest.raises(ValueError):
        C.containment_constraints(x)
