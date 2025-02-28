# Copyright (C) 2020-2025 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

# Code purpose: Test the ConstrainedZonotope class for constructor and initialization

import numpy as np
import pytest

from pycvxset import ConstrainedZonotope, Polytope
from pycvxset.common.constants import TESTING_CONTAINMENT_STATEMENTS_INVOLVING_GUROBI


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
    assert Z.is_empty
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
    assert Z.is_empty
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
    assert Z.is_empty
    assert Z.Ae.shape == (0, 0)
    assert Z.be.size == 0
    assert Z.He.shape == (0, Z.latent_dim + 1)
    assert Z.G.size == 0
    assert Z.c is None

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

    # Sanity check
    assert np.allclose(Z.G, Z_V.G)
    assert np.allclose(Z.c, Z_V.c)

    # From H-Rep
    assert Z.dim == 2
    assert Z.latent_dim == 6
    assert Z.Ae.shape == (P.n_halfspaces, 6)
    assert Z.be.shape == (P.n_halfspaces,)
    assert not Z.is_empty

    # From V-Rep
    assert Z_V.dim == 2
    assert Z_V.latent_dim == 6
    assert Z_V.Ae.shape == (P.n_halfspaces, 6)
    assert Z_V.be.shape == (P.n_halfspaces,)
    assert not Z_V.is_empty

    # Unbounded polytope
    unbounded_P = Polytope(A=np.eye(2), b=np.ones((2,)))
    with pytest.raises(ValueError):
        ConstrainedZonotope(polytope=unbounded_P)

    # Sliced polytope
    sliced_P = P.intersection_with_affine_set(Ae=[1, 0], be=1)
    with pytest.warns(UserWarning, match="Removed some rows in (Ae, be)*"):
        Z = ConstrainedZonotope(polytope=sliced_P)
    assert Z.dim == 2
    assert Z.latent_dim == 6
    assert Z.Ae.shape == (3, 6)
    assert Z.be.shape == (3,)
    assert not Z.is_empty


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
    assert repr(C2) == "Constrained Zonotope in R^2\n\twith latent dimension 4 and 2 equality constraints"


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
    if TESTING_CONTAINMENT_STATEMENTS_INVOLVING_GUROBI:
        assert C2 == C1


def test_pow():
    C1 = ConstrainedZonotope(lb=[-1, -1], ub=[1, 1])
    C2 = C1**2
    assert C2.dim == 4
    if TESTING_CONTAINMENT_STATEMENTS_INVOLVING_GUROBI:
        assert C2 == ConstrainedZonotope(lb=[-1, -1, -1, -1], ub=[1, 1, 1, 1])


def test_is_full_dimensional():
    C = ConstrainedZonotope(G=np.hstack((np.eye(2), np.zeros((2, 1)))), c=np.ones((2,)), Ae=[1, 1, 1], be=[3.1])
    assert not C.is_full_dimensional
    Z = ConstrainedZonotope(G=np.hstack((np.eye(2), np.zeros((2, 1)))), c=np.ones((2,)))
    assert Z.is_full_dimensional
    C = ConstrainedZonotope(G=np.hstack((np.eye(2), np.zeros((2, 1)))), c=np.ones((2,)), Ae=[1, 1, 1], be=[2.1])
    with pytest.raises(NotImplementedError):
        C.is_full_dimensional
    C = ConstrainedZonotope(G=None, c=[0, 0])
    assert not C.is_full_dimensional
