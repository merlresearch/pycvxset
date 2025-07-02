# Copyright (C) 2020-2025 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

# Code purpose: Test the Polytope class methods for unary operations

import numpy as np
import pytest

from pycvxset import Ellipsoid, Polytope, approximate_volume_from_grid
from pycvxset.common import check_matrices_are_equal_ignoring_row_order
from pycvxset.common.constants import TESTING_SHOW_PLOTS

TESTING_SHOW_PLOTS = False or TESTING_SHOW_PLOTS


def test_scale_and_normalize():
    # Create a polytope from a vertex list, scale, and check resulting vertices
    V = np.array([[1.9, 0.2], [0, -2], [-0.3, 0.17], [3, 4.01]])
    PV = Polytope(V=V)
    PV.normalize()
    PV_sliced = PV.intersection_with_affine_set(Ae=[1, 0], be=0.5)
    PV_sliced.normalize()

    factorV = 1.3
    factorV * PV
    with pytest.raises(TypeError):
        PV * factorV

    # Create a polytope from inequalities, scale, and check resulting (A, b)
    A = np.array([[-1, -2], [2, 0.9], [-1.3, 3]])
    b = np.array([-0.6, 3.1, 4])  # b as a 1D array is also ok (best)
    PH = Polytope(A=A, b=b)
    factorH = -0.8
    factorH * PH
    with pytest.raises(TypeError):
        PH * factorH

    # Test negation
    lb = [-1, -1, -1]
    ub = [1, 1, 1]
    P1 = Polytope(lb=lb, ub=ub)
    P2 = -P1
    assert (-P2) == P1


def test_chebyshev_centering():
    # The test is in test_polytope_binary
    pass


def test_interior_point():
    P1 = Polytope(lb=[-1, -1], ub=[1, 1])
    assert np.allclose(
        P1.interior_point(point_type="centroid"),
        P1.interior_point(point_type="chebyshev"),
    )
    P2 = Polytope(V=P1.V)
    assert np.allclose(
        P2.interior_point(point_type="centroid"),
        P2.interior_point(point_type="chebyshev"),
    )
    A = np.array([[2, 1], [2, -1], [-1, 2], [-1, -2]])
    b = np.ones((4,))
    P3 = Polytope(A=A, b=b)
    assert not np.allclose(
        P3.interior_point(point_type="centroid"),
        P3.interior_point(point_type="chebyshev"),
    )
    with pytest.raises(NotImplementedError):
        P3.interior_point(point_type="random")

    P4 = Polytope(dim=3)
    with pytest.raises(ValueError):
        P4.interior_point(point_type="centroid")
    with pytest.raises(ValueError):
        P4.interior_point(point_type="chebyshev")

    P7 = Polytope(A=[[1, 0], [0, 1]], b=[1, 1])  # Unbounded polytope
    with pytest.raises(ValueError):
        P7.interior_point(point_type="chebyshev")
    with pytest.raises(ValueError):
        P7.interior_point(point_type="centroid")


def test_projection_and_copy():
    lb = [-1, -1, -1]
    ub = [1, 1, 1]
    P1 = Polytope(lb=lb, ub=ub)
    P1_projection = P1.projection(2)
    P1_projection.minimize_V_rep()
    true_vertices = Polytope(lb=lb[:2], ub=ub[:2]).V
    assert check_matrices_are_equal_ignoring_row_order(P1_projection.V, true_vertices)

    # Projection of a cylinder using HRep only
    P2 = Polytope(A=[[-1, 0, 0], [0, -1, 0], [1, 1, 0], [0, 0, 1], [0, 0, -1]], b=[0, 0, 1, 1, 1])
    P2_projection = P2.projection(2)
    assert not P2_projection.in_H_rep
    assert P2_projection.in_V_rep
    P2_projection.minimize_V_rep()
    true_vertices = np.array([[1, 0], [0, 1], [0, 0]])
    assert check_matrices_are_equal_ignoring_row_order(P2_projection.V, true_vertices)

    # Create copies
    P1.copy()
    Polytope(V=P1.V).copy()
    Polytope(dim=5).copy()

    # Project l1-norm ball
    P = Polytope(V=np.vstack((np.eye(3), -np.eye(3))))
    P_projection = P.projection(2)
    P_projection.minimize_V_rep()
    assert np.isclose(np.max(np.max(P_projection.V)), 1)
    assert np.isclose(np.min(np.min(P_projection.V)), -1)
    assert np.all(np.isclose(np.sum(np.abs(P_projection.V), axis=1), np.ones((4,))))

    with pytest.raises(ValueError):
        P.projection(3)

    with pytest.raises(ValueError):
        P.projection(-1)

    with pytest.raises(ValueError):
        P.projection([0, 1, 2])

    assert P.projection([]) == P


def test_minimum_volume_ellipsoid():
    # Full-dimensional case
    lb = [-1, -1]
    ub = [1, 1]
    P_full_dim = Polytope(lb=lb, ub=ub)
    covering_ellipsoid_center, covering_ellipsoid_shape, covering_ellipsoid_shape_sqrt = (
        P_full_dim.minimum_volume_circumscribing_ellipsoid()
    )
    assert np.allclose(covering_ellipsoid_shape, 2 * np.eye(2))
    assert np.allclose(covering_ellipsoid_shape_sqrt, np.sqrt(2) * np.eye(2))
    assert np.allclose(covering_ellipsoid_center, np.zeros((2,)))

    if TESTING_SHOW_PLOTS:
        import matplotlib.pyplot as plt

        plt.figure()
        plt.gca().set_aspect("equal")
        P_full_dim.plot(plt.gca(), patch_args={"alpha": 0.2})
        E = Ellipsoid(c=covering_ellipsoid_center, Q=covering_ellipsoid_shape)
        E.plot(method="outer", ax=plt.gca(), patch_args={"alpha": 0.2, "facecolor": "r"})
        plt.xlim([-2, 2])
        plt.ylim([-2, 2])
        plt.show()

    # Low-dimensional case
    lb = [-1, -1, -1]
    ub = [1, 1, 1]
    P_low_dim = Polytope(lb=lb, ub=ub).intersection_with_affine_set(Ae=[1, 1, 1], be=[0])
    covering_ellipsoid_center, covering_ellipsoid_shape, _ = P_low_dim.minimum_volume_circumscribing_ellipsoid()

    # Check for CVXPY error
    P_full_dim.cvxpy_args_socp = {"solver": "WRONG_SOLVER"}
    P_low_dim.cvxpy_args_sdp = {"solver": "WRONG_SOLVER"}
    with pytest.raises(NotImplementedError):
        P_full_dim.minimum_volume_circumscribing_ellipsoid()
    with pytest.raises(NotImplementedError):
        P_low_dim.minimum_volume_circumscribing_ellipsoid()

    # Empty polytope
    with pytest.raises(ValueError):
        Polytope().minimum_volume_circumscribing_ellipsoid()


def test_maximum_volume_ellipsoid():
    # Full-dimensional case
    lb = [-1, -1]
    ub = [1, 1]
    P_full_dim = Polytope(lb=lb, ub=ub)
    ellipsoid_center, ellipsoid_shape, ellipsoid_sqrt_shape = P_full_dim.maximum_volume_inscribing_ellipsoid()
    assert np.allclose(ellipsoid_shape, np.eye(2))
    assert np.allclose(ellipsoid_sqrt_shape, np.eye(2))
    assert np.allclose(ellipsoid_center, np.zeros((2,)))

    if TESTING_SHOW_PLOTS:
        import matplotlib.pyplot as plt

        plt.figure()
        plt.gca().set_aspect("equal")
        P_full_dim.plot(plt.gca(), patch_args={"alpha": 0.2})
        E = Ellipsoid(c=ellipsoid_center, Q=ellipsoid_shape)
        E.plot(method="inner", ax=plt.gca(), patch_args={"alpha": 0.2, "facecolor": "r"})
        plt.xlim([-2, 2])
        plt.ylim([-2, 2])
        plt.show()

    # Low-dimensional case
    lb = [-1, -1, -1]
    ub = [1, 1, 1]
    P_low_dim = Polytope(lb=lb, ub=ub).intersection_with_affine_set(Ae=[1, 1, 1], be=[0])
    with pytest.warns(UserWarning):
        P_low_dim.maximum_volume_inscribing_ellipsoid()
    P_low_dim = Polytope(lb=[-1, -1], ub=[1, 1]).intersection_with_halfspaces(A=[[1, 0], [-1, 0]], b=[1, -1])
    with pytest.warns(UserWarning):
        P_low_dim.maximum_volume_inscribing_ellipsoid()

    # Check for CVXPY error
    P_full_dim.cvxpy_args_socp = {"solver": "WRONG_SOLVER"}
    P_low_dim.cvxpy_args_sdp = {"solver": "WRONG_SOLVER"}
    with pytest.raises(NotImplementedError):
        P_full_dim.maximum_volume_inscribing_ellipsoid()
    with pytest.raises(NotImplementedError):
        P_low_dim.maximum_volume_inscribing_ellipsoid()

    # Empty polytope
    with pytest.raises(ValueError):
        Polytope().maximum_volume_inscribing_ellipsoid()

    # Unbounded polytope
    P = Polytope(A=[[1, 0], [0, 1]], b=[1, 1])
    with pytest.raises(ValueError):
        P.maximum_volume_inscribing_ellipsoid()
    P = Polytope(A=[[1, 0], [0, 1]], b=[1, 1]).intersection_with_affine_set(Ae=[1, 0], be=[1])
    with pytest.raises(ValueError):
        P.maximum_volume_inscribing_ellipsoid()

    P = Polytope(A=np.vstack((np.eye(2), -np.eye(2))), b=[2, 2, -3, -3])
    assert P.is_empty
    with pytest.raises(ValueError):
        P.maximum_volume_inscribing_ellipsoid()


def test_volume():
    P = Polytope(lb=[-1, -1], ub=[1, 1])
    assert np.isclose(P.volume(), 4)
    P = Polytope(lb=[-1, -1], ub=[1, 1]).intersection_with_affine_set(Ae=[1, 0], be=1)
    with pytest.raises(ValueError):
        P.volume()
    P = Polytope()
    assert P.volume() == 0


def test_minimum_volume_rectangle_and_deflate_rectangle():
    # Full-dimensional case: H-Rep case
    lb = [-1, -1]
    ub = [1, 1]
    P_full_dim = Polytope(lb=lb, ub=ub)
    lb_min_vol, ub_min_vol = P_full_dim.minimum_volume_circumscribing_rectangle()
    assert np.allclose(lb_min_vol, lb)
    assert np.allclose(ub_min_vol, ub)

    # Full-dimensional case: V-Rep case
    P_simplex = Polytope(V=[[1, 0], [0, 1], [0, 0]])
    lb_min_vol, ub_min_vol = P_simplex.minimum_volume_circumscribing_rectangle()
    assert np.allclose(lb_min_vol, [0, 0])
    assert np.allclose(ub_min_vol, [1, 1])

    # Empty polytope
    with pytest.raises(ValueError):
        Polytope().minimum_volume_circumscribing_rectangle()

    # deflate_rectangle
    P_full_dim_bounding_box = Polytope.deflate_rectangle(P_full_dim)
    assert P_full_dim == P_full_dim_bounding_box


def test_approximate_volume_from_grid_and_minimum_volume_circumscribing_rectangle():
    # Empty Polytope
    C = Polytope(dim=2)
    assert np.isclose(approximate_volume_from_grid(C, area_grid_step_size=0.5), 0)
    with pytest.raises(ValueError):
        C.minimum_volume_circumscribing_rectangle()

    # 3D Polytope
    C = Polytope(lb=[-1, -1, -1], ub=[1, 1, 1])
    with pytest.raises(ValueError):
        approximate_volume_from_grid(C, area_grid_step_size=0.5)

    # 2D Polytope
    lb = [-1, -1]
    ub = [1, 1]
    C = Polytope(lb=lb, ub=ub)
    assert np.isclose(approximate_volume_from_grid(C, area_grid_step_size=0.5), 4)
    lb_rect, ub_rect = C.minimum_volume_circumscribing_rectangle()
    assert np.allclose(lb, lb_rect)
    assert np.allclose(ub, ub_rect)
    with pytest.raises(ValueError):
        approximate_volume_from_grid(C, area_grid_step_size=-0.5)
    with pytest.raises(ValueError):
        approximate_volume_from_grid(C, area_grid_step_size=[0.5, 0.1])
