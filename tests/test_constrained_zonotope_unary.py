# Copyright (C) 2020-2025 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

# Code purpose: Test the ConstrainedZonotope class for unary operations

import numpy as np
import pytest

from pycvxset import ConstrainedZonotope, Polytope, approximate_volume_from_grid
from pycvxset.common.constants import TESTING_STATEMENTS_INVOLVING_GUROBI


def test_approximate_volume_from_grid_and_minimum_volume_circumscribing_rectangle():
    # Empty ConstrainedZonotope
    C = ConstrainedZonotope(dim=2)
    assert np.isclose(approximate_volume_from_grid(C, area_grid_step_size=0.5), 0)
    with pytest.raises(ValueError):
        C.minimum_volume_circumscribing_rectangle()

    # Single-point ConstrainedZonotope
    C = ConstrainedZonotope(c=[0, 0], G=None)
    assert np.isclose(approximate_volume_from_grid(C, area_grid_step_size=0.5), 0)

    # 3D ConstrainedZonotope
    C = ConstrainedZonotope(lb=[-1, -1, -1], ub=[1, 1, 1])
    with pytest.raises(ValueError):
        approximate_volume_from_grid(C, area_grid_step_size=0.5)

    # 2D ConstrainedZonotope
    lb = [-1, -1]
    ub = [1, 1]
    C = ConstrainedZonotope(lb=lb, ub=ub)
    assert np.isclose(approximate_volume_from_grid(C, area_grid_step_size=0.5), 4)
    lb_rect, ub_rect = C.minimum_volume_circumscribing_rectangle()
    assert np.allclose(lb, lb_rect)
    assert np.allclose(ub, ub_rect)
    with pytest.raises(ValueError):
        approximate_volume_from_grid(C, area_grid_step_size=-0.5)
    with pytest.raises(ValueError):
        approximate_volume_from_grid(C, area_grid_step_size=[0.5, 0.1])


def test_interior_point():
    lb = [-1, -1]
    ub = [1, 1]
    C = ConstrainedZonotope(lb=lb, ub=ub)
    assert np.allclose(C.interior_point(), 0)
    C = ConstrainedZonotope(lb=lb, ub=ub).intersection_with_affine_set(Ae=[1, 0], be=1)
    assert C.interior_point() <= C


def test_projection():
    lb = [-1, -2]
    ub = [1, 2]
    C = ConstrainedZonotope(lb=lb, ub=ub)
    C_projected = C.projection([1])
    assert C_projected.dim == 1
    assert np.allclose(C_projected.support([[1], [-1]])[0], 1)
    C_projected = C.projection([0])
    assert C_projected.dim == 1
    assert np.allclose(C_projected.support([[1], [-1]])[0], 2)


def test_slice():
    lb = [-1, -2]
    ub = [1, 2]
    C = ConstrainedZonotope(lb=lb, ub=ub)
    C_sliced = C.slice([1], [0.75])
    assert C_sliced.dim == 2
    assert np.allclose(C_sliced.support([[1, 0], [-1, 0]])[1][:, 1], 0.75)
    C_sliced = C.slice([1], [0.75])
    assert C_sliced.dim == 2
    assert np.allclose(C_sliced.support([[1, 0], [-1, 0]])[1][:, 1], 0.75)


def test_slice_then_projection():
    P = ConstrainedZonotope(c=[0, 0, 0], h=0.5)
    Q = P.slice(dims=2, constants=0.25).projection(project_away_dims=2)
    Q_projection = P.slice_then_projection(dims=2, constants=0.25)
    assert Q.dim == 2
    if TESTING_STATEMENTS_INVOLVING_GUROBI == "full":
        assert Q == Q_projection


def test_chebyshev_center_and_redundant_inequalities():
    lb = [-1, -2]
    ub = [1, 2]
    C = ConstrainedZonotope(lb=lb, ub=ub)
    P = Polytope(lb=lb, ub=ub)
    with pytest.warns(UserWarning, match="This function returns a sub-optimal*"):
        c_C, r_C = C.chebyshev_centering()
    c_P, r_P = P.chebyshev_centering()
    assert np.isclose(r_P, r_C)
    assert np.allclose(c_P, c_C)

    G, c = [[1, 0, 0], [0, 1, 0]], [0, 0]
    Ae, be = [-1, -1, 1.25], -0.75
    C1 = ConstrainedZonotope(G=G, c=c, Ae=Ae, be=be)
    P1 = Polytope(lb=[-1, -1], ub=[1, 1]).intersection_with_halfspaces([-1, -1], 0.5)
    with pytest.warns(UserWarning, match="This function returns a sub-optimal*"):
        _, r_C = C1.chebyshev_centering()
    _, r_P = P1.chebyshev_centering()
    assert r_P > r_C

    # Clearly empty set
    C = ConstrainedZonotope(lb=lb, ub=ub).intersection_with_halfspaces([1, 1], -10)
    with pytest.raises(ValueError):
        _, r_C = C.chebyshev_centering()

    # Not trivially an empty set
    C = ConstrainedZonotope(G=np.hstack((np.eye(2), np.zeros((2, 1)))), c=np.ones((2,)), Ae=[1, 1, 1], be=[3.1])
    with pytest.raises(ValueError):
        _, r_C = C.chebyshev_centering()

    # Low-dimensional set
    C = ConstrainedZonotope(lb=lb, ub=ub).intersection_with_affine_set([1, 1], 0)
    with pytest.raises(ValueError):
        _, r_C = C.chebyshev_centering()

    # Many redundancies
    C1 = ConstrainedZonotope(
        G=np.hstack((np.eye(2), np.zeros((2, 1)))), c=np.ones((2,)), Ae=np.ones((4, 3)), be=np.ones((4,))
    )
    C2 = ConstrainedZonotope(G=np.hstack((np.eye(2), np.zeros((2, 1)))), c=np.ones((2,)), Ae=[1, 1, 1], be=1)
    with pytest.warns(UserWarning, match="This function returns a sub-optimal*"):
        with pytest.raises(ValueError):
            C1.chebyshev_centering()
        C1.remove_redundancies()
        c_C1, r_C1 = C1.chebyshev_centering()
        c_C2, r_C2 = C2.chebyshev_centering()
    assert np.allclose(c_C1, c_C2)
    assert np.isclose(r_C1, r_C2)

    # Solver error
    C = ConstrainedZonotope(lb=lb, ub=ub)
    C.cvxpy_args_lp = {"solver": "WRONG_SOLVER"}
    with pytest.raises(NotImplementedError):
        with pytest.warns(UserWarning, match="This function returns a sub-optimal*"):
            C.chebyshev_centering()


def test_maximum_volume_inscribed_ellipsoids():
    lb = [-1, -2]
    ub = [1, 2]
    C = ConstrainedZonotope(lb=lb, ub=ub)
    P = Polytope(lb=lb, ub=ub)
    with pytest.warns(UserWarning, match="This function returns a sub-optimal*"):
        c_C, Q_C, G_C = C.maximum_volume_inscribing_ellipsoid()
    c_P, Q_P, G_P = P.maximum_volume_inscribing_ellipsoid()
    assert np.allclose(c_P, c_C)
    assert np.allclose(Q_P, Q_C)
    assert np.allclose(G_P, G_C)

    G, c = [[1, 0, 0], [0, 1, 0]], [0, 0]
    Ae, be = [-1, -1, 1.25], -0.75
    C1 = ConstrainedZonotope(G=G, c=c, Ae=Ae, be=be)
    P1 = Polytope(lb=[-1, -1], ub=[1, 1]).intersection_with_halfspaces([-1, -1], 0.5)
    with pytest.warns(UserWarning, match="This function returns a sub-optimal*"):
        c_C, Q_C, G_C = C1.maximum_volume_inscribing_ellipsoid()
    c_P, Q_P, G_P = P1.maximum_volume_inscribing_ellipsoid()
    assert np.prod(np.diag(G_P)) > np.prod(np.diag(G_C))

    # Clearly empty set
    C = ConstrainedZonotope(lb=lb, ub=ub).intersection_with_halfspaces([1, 1], -10)
    with pytest.raises(ValueError):
        C.maximum_volume_inscribing_ellipsoid()

    # Not trivially an empty set
    C = ConstrainedZonotope(G=np.hstack((np.eye(2), np.zeros((2, 1)))), c=np.ones((2,)), Ae=[1, 1, 1], be=[3.1])
    with pytest.raises(ValueError):
        C.maximum_volume_inscribing_ellipsoid()

    # Low-dimensional set
    C = ConstrainedZonotope(lb=lb, ub=ub).intersection_with_affine_set([1, 1], 0)
    with pytest.raises(ValueError):
        C.maximum_volume_inscribing_ellipsoid()

    # Many redundancies
    C1 = ConstrainedZonotope(
        G=np.hstack((np.eye(2), np.zeros((2, 1)))), c=np.ones((2,)), Ae=np.ones((4, 3)), be=np.ones((4,))
    )
    C2 = ConstrainedZonotope(G=np.hstack((np.eye(2), np.zeros((2, 1)))), c=np.ones((2,)), Ae=[1, 1, 1], be=1)
    with pytest.warns(UserWarning, match="This function returns a sub-optimal*"):
        with pytest.raises(ValueError):
            C1.maximum_volume_inscribing_ellipsoid()
        C1.remove_redundancies()
        c_C1, Q_C1, G_C1 = C1.maximum_volume_inscribing_ellipsoid()
        c_C2, Q_C2, G_C2 = C2.maximum_volume_inscribing_ellipsoid()
    assert np.allclose(c_C1, c_C2)
    assert np.allclose(Q_C1, Q_C2)
    assert np.allclose(G_C1, G_C2)

    # Solver error
    C = ConstrainedZonotope(lb=lb, ub=ub)
    C.cvxpy_args_lp = {"solver": "WRONG_SOLVER"}
    with pytest.raises(NotImplementedError):
        with pytest.warns(UserWarning, match="This function returns a sub-optimal*"):
            C.maximum_volume_inscribing_ellipsoid()
