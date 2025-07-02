# Copyright (C) 2020-2025 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

# Code purpose: Test the Ellipsoid class

import math
import unittest

import cvxpy as cp
import numpy as np
import pytest

from pycvxset import PYCVXSET_ZERO, ConstrainedZonotope, Ellipsoid, Polytope, spread_points_on_a_unit_sphere
from pycvxset.common.constants import DEFAULT_CVXPY_ARGS_SOCP, TEST_3D_PLOTTING, TESTING_SHOW_PLOTS

TESTING_SHOW_PLOTS = False or TESTING_SHOW_PLOTS


class TestEllipsoid(unittest.TestCase):
    def test___init__(self):
        E = Ellipsoid(c=np.zeros((2,)), Q=np.eye(2))
        assert E.is_full_dimensional
        assert not E.is_empty
        E = Ellipsoid(c=np.zeros((2,)), G=np.ones((2, 2)))
        assert not E.is_full_dimensional
        assert not E.is_empty
        E = Ellipsoid(c=np.zeros((2,)), G=np.eye(2))
        assert E.is_full_dimensional
        assert not E.is_empty
        # Full-dimensional ellipsoid unnecessarily complicated
        E1 = Ellipsoid(c=np.zeros((2,)), G=np.hstack((np.eye(2), np.zeros((2, 1)))))
        assert E1.is_full_dimensional
        assert not E1.is_empty
        assert np.isclose(E1.G, E.G).all()
        assert np.isclose(E1.c, E.c).all()
        assert np.isclose(E1.Q, E.Q).all()

        # Swap dimensions after affine transformation
        E = Ellipsoid(c=np.zeros((2,)), G=[[0, 1], [1, 0]])
        assert E.dim == 2
        assert E.is_full_dimensional
        assert not E.is_singleton

        # Additional degenerate ellipsoid tests
        E = Ellipsoid(c=np.zeros((2,)), G=np.ones((2, 3)))
        assert E.dim == 2
        assert not E.is_full_dimensional

        E = Ellipsoid(c=np.zeros((3,)), G=np.ones((3, 2)))
        assert E.dim == 3
        assert not E.is_full_dimensional

        E = Ellipsoid(c=np.zeros((3,)), G=np.zeros((3, 2)))
        assert E.dim == 3
        assert not E.is_full_dimensional

        with pytest.raises(ValueError):
            Ellipsoid(c=np.zeros((2,)), Q=[[1, 0], [0, 0]])

        E1 = Ellipsoid(c=np.zeros((2,)))
        assert E1.dim == 2
        assert E1.is_singleton

        E2 = Ellipsoid(c=np.zeros((2,)), r=0)
        E3 = Ellipsoid(c=np.zeros((2,)), G=PYCVXSET_ZERO / 10 * np.eye(2))
        E4 = Ellipsoid(c=np.zeros((2,)), Q=((PYCVXSET_ZERO / 10) ** 2) * np.eye(2))
        assert not E2.is_full_dimensional
        assert not E3.is_full_dimensional
        assert not E4.is_full_dimensional
        assert E2.is_singleton
        assert E3.is_singleton
        assert E4.is_singleton

        assert E1 == E2
        assert E2 == E3
        assert E3 == E4

        E2 = Ellipsoid(c=np.zeros((2,)), r=2)
        E3 = Ellipsoid(c=np.zeros((2,)), G=2 * np.eye(2))
        E4 = Ellipsoid(c=np.zeros((2,)), Q=4 * np.eye(2))
        assert E2.is_full_dimensional
        assert E3.is_full_dimensional
        assert E4.is_full_dimensional
        assert not E3.is_singleton
        assert not E4.is_singleton

        assert E1 != E2  # Singleton vs not singleton
        assert E2 == E3
        assert E3 == E4

        E3 = Ellipsoid(c=np.zeros((2,)), G=np.diag([1, 0]))
        E4 = Ellipsoid(c=np.zeros((2,)), Q=np.diag([1, 1])).intersection_with_affine_set(Ae=[0, 1], be=0)
        assert E3.polytopic_inner_approximation() == E4.polytopic_inner_approximation()
        assert not E3.is_full_dimensional
        assert not E4.is_full_dimensional
        assert not E3.is_singleton
        assert not E4.is_singleton

        # Can not check for containment between non-full-dimensional ellipsoids
        with pytest.raises(ValueError):
            E3 == E4

        with pytest.raises(ValueError):
            Ellipsoid(c=np.zeros((3,)), r=2) != E3  # Dimension mismatch

        # Q must be positive definite
        with pytest.raises(ValueError):
            Ellipsoid(c=np.zeros((2,)), Q=[[1, 0], [0, -1]])

        with pytest.raises(ValueError):
            Ellipsoid(c=np.zeros((2,)), Q=[[1, 0], [0, 0]])

        with pytest.raises(ValueError):
            Ellipsoid(c=np.zeros((2,)), Q=[[1, 1], [0, 1]])

        # G must have self.dim rows
        with pytest.raises(ValueError):
            Ellipsoid(c=np.zeros((3,)), G=[[1, 0], [0, 0]])

        # Can't do 2D center
        with pytest.raises(ValueError):
            Ellipsoid(c=np.eye(2), Q=np.eye(2))

        with pytest.raises(ValueError):
            Ellipsoid(c=np.zeros((2,)), Q=np.zeros((2,)))

        with pytest.raises(ValueError):
            Ellipsoid(c=np.zeros((2,)), Q=np.zeros((2, 3)))

        with pytest.raises(ValueError):
            Ellipsoid(c=np.zeros((2,)), Q=np.eye(3))

        with pytest.raises(ValueError):
            Ellipsoid(c=np.zeros((2,)), Q=np.eye(2), G=np.eye(2))

        with pytest.raises(ValueError):
            Ellipsoid(c=np.zeros((2,)), A=np.eye(2))

        with pytest.raises(ValueError):
            Ellipsoid(c="char", r=1)

        with pytest.raises(ValueError):
            Ellipsoid(m=[1, 0])

    def test_plotting(self):
        a = 1
        b = 3
        n_vectors = 102
        Q = [[a**2, 0], [0, b**2]]
        ell = Ellipsoid(c=np.zeros((2,)), Q=Q)
        ax, _, _ = ell.plot(method="inner", n_vertices=n_vectors, center_args={"s": 50})
        ell.plot(
            method="outer",
            n_halfspaces=8,
            ax=ax,
            patch_args={"alpha": 0, "edgecolor": "k"},
            center_args={"s": 50, "color": "blue"},
        )
        dir_vectors = spread_points_on_a_unit_sphere(ell.dim, n_vectors)[0]
        scaled_dir_vectors = np.tile(np.array([[a, b]]), (n_vectors, 1)) * dir_vectors
        ax.plot(scaled_dir_vectors[:, 0], scaled_dir_vectors[:, 1])
        ax.set_xlim([-10, 10])
        ax.set_ylim([-10, 10])

        if TEST_3D_PLOTTING:
            ell = Ellipsoid(c=(0, 0, 0), r=3)
            ell.plot(method="inner")
            ell.plot(method="outer")

    def test_sphere_and_support_function(self):
        radius = 5
        sphere = Ellipsoid(c=(0, 0), r=radius)
        for support_direction in ((0, 1), (0, -1), (1, 0), (-1, 0)):
            assert sphere.support(support_direction)[0] == radius

        with pytest.raises(ValueError):
            Ellipsoid(c=(0, 0), r=-1)

        low_dim_sphere = Ellipsoid(c=(1, 0), G=[[1, 1], [1, 1]])
        assert np.isclose(low_dim_sphere.support([1, 1])[0], 1 + 2 * np.sqrt(2))
        # Orthogonal to the direction
        assert np.isclose(low_dim_sphere.support([1, -1])[0], 1)

        singleton = Ellipsoid(c=(1, 0))
        assert np.isclose(singleton.support([1, -1])[1], [1, 0]).all()

    def test_add(self):
        sphere_1 = Ellipsoid(c=(1, 0), r=3)
        sphere_2 = Ellipsoid(c=(0, 0), r=3)
        polytope = Polytope(lb=[-2, -2], ub=[2, 2])

        # All of these actions will cause error due to lack of exact representation
        with pytest.raises(ValueError):
            sphere_1 + sphere_2
        with pytest.raises(ValueError):
            sphere_1 + polytope
        with pytest.raises(ValueError):
            sphere_1 + ConstrainedZonotope(lb=[-1, -1], ub=[1, 1])
        with pytest.raises(TypeError):
            polytope + sphere_1
        with pytest.raises(TypeError):
            "str" + sphere_1
        with pytest.raises(ValueError):
            [[[1, 1], [1, 1]], [[1, 1], [1, 1]]] + sphere_1
        with pytest.raises(ValueError):
            [1, 1, 1] + sphere_1
        with pytest.raises(ValueError):
            [[[1, 1, 1]]] + sphere_1

        # Check addition
        sphere_3 = sphere_1 + (1, 1)
        assert np.max(np.abs(sphere_3.c - sphere_1.c)) == 1
        assert np.min(np.abs(sphere_3.c - sphere_1.c)) == 1

        # Commutativity
        sphere_3 = np.array([1, 1]) + sphere_1
        assert np.max(np.abs(sphere_3.c - sphere_1.c)) == 1
        assert np.min(np.abs(sphere_3.c - sphere_1.c)) == 1

        # Minus
        sphere_3 = np.array([1, 1]) + (-sphere_1)
        assert np.max(np.abs(sphere_3.c + sphere_1.c)) == 1
        assert np.min(np.abs(sphere_3.c + sphere_1.c)) == 1

    def test_mul_and_rmul(self):
        # Major semi-axis lengths of 1 and 2
        E1 = Ellipsoid(c=(0, 0), Q=np.diag((1, 4)))
        # Define E2 as rotation of E1
        rotation_angle = np.pi / 2
        R = np.array(
            [
                [np.cos(rotation_angle), np.sin(rotation_angle)],
                [-np.sin(rotation_angle), np.cos(rotation_angle)],
            ]
        )
        E2_orig = Ellipsoid(c=(0, 0), Q=R @ E1.Q @ R.T)
        E2 = R @ E1
        assert np.all(np.isclose(E2_orig.c, E2.c))
        assert np.all(np.all(np.isclose(E2_orig.Q, E2.Q)))
        E2a = E1.affine_map(R)
        assert np.all(np.isclose(E2a.c, E2.c))
        assert np.all(np.all(np.isclose(E2a.Q, E2.Q)))
        # Ok
        R * E1

        E3 = E2 @ R
        assert np.all(np.isclose(E3.c, E1.c))
        assert np.all(np.isclose(E3.Q, E1.Q))
        E4 = E2.inverse_affine_map_under_invertible_matrix(R)
        assert np.all(np.isclose(E3.c, E4.c))
        assert np.all(np.isclose(E3.Q, E4.Q))
        with pytest.raises(TypeError):
            E2 @ np.zeros((2, 2))

        with pytest.raises(TypeError):
            E2 * 0

        with pytest.raises(TypeError):
            E2 * "char"

        with pytest.raises(TypeError):
            E2 @ "char"

        with pytest.raises(TypeError):
            "char" * E2

        with pytest.raises(TypeError):
            "char" @ E2

        with pytest.raises(TypeError):
            E2 * np.zeros((2, 2))

        with pytest.raises(TypeError):
            E1 * R

        np.array([np.eye(2)]) * E1

        with pytest.raises(ValueError):
            np.eye(3) * E1

        with pytest.raises(TypeError):
            E1 * np.array([np.eye(2)])

        with pytest.raises(TypeError):
            E1 * np.eye(3)

        with pytest.raises(ValueError):
            np.array([np.eye(2)]) @ E1
        # Allowed!
        assert (2 @ E1).dim == 2
        E1D = np.array([1, 2]) @ E1
        assert E1D.dim == 1
        # Scalar multiplication with an ellipsoid with zero
        zero_ellipsoid = 0 @ E1
        assert np.isclose(zero_ellipsoid.c, np.zeros((E1.dim,))).all()
        assert np.isclose(zero_ellipsoid.Q, np.zeros((E1.dim, E1.dim))).all()
        zero_ellipsoid = 0 * E1
        assert np.isclose(zero_ellipsoid.c, np.zeros((E1.dim,))).all()
        assert np.isclose(zero_ellipsoid.Q, np.zeros((E1.dim, E1.dim))).all()
        # Not full row rank
        E1D = np.array([[1, 2], [1, 2]]) @ E1
        assert not E1D.is_full_dimensional

        with pytest.raises(TypeError):
            E1 @ 2

        with pytest.raises(TypeError):
            E1.affine_map(Polytope(dim=2))

        with pytest.raises(TypeError):
            E1.inverse_affine_map_under_invertible_matrix(Polytope(dim=2))

    def test_sub(self):
        E = Ellipsoid(c=[0, 0], r=1)
        assert np.allclose((E - [1, 1]).c, [-1, -1])
        assert np.allclose((E - [1, 1]).Q, np.eye(2))
        with pytest.raises(TypeError):
            E - E
        with pytest.raises(TypeError):
            E - 2
        with pytest.raises(TypeError):
            E - Polytope(dim=2)

    def test_shrink_over_inflate(self):
        lb = [-1, -1]
        ub = [1, 1]
        P = Polytope(lb=lb, ub=ub)
        E = Ellipsoid.deflate(P)
        assert np.allclose(E.Q, 2 * np.eye(2))
        assert np.allclose(E.c, np.zeros((2,)))
        if TESTING_SHOW_PLOTS:
            import matplotlib.pyplot as plt

            plt.figure()
            plt.gca().set_aspect("equal")
            P.plot(plt.gca(), patch_args={"alpha": 0.2})
            E.plot(method="inner", ax=plt.gca(), patch_args={"facecolor": "r", "alpha": 0.2})
            plt.xlim([-2, 2])
            plt.ylim([-2, 2])
            plt.show()
        E2 = Ellipsoid.inflate(P)
        assert np.allclose(E2.Q, np.eye(2))
        assert np.allclose(E2.c, np.zeros((2,)))
        if TESTING_SHOW_PLOTS:
            import matplotlib.pyplot as plt

            plt.figure()
            plt.gca().set_aspect("equal")
            P.plot(plt.gca(), patch_args={"alpha": 0.2})
            E2.plot(method="inner", ax=plt.gca(), patch_args={"facecolor": "r", "alpha": 0.2})
            plt.xlim([-2, 2])
            plt.ylim([-2, 2])
            plt.show()

        E3 = Ellipsoid.inflate_ball(P)
        assert np.allclose(E3.Q, np.eye(2))
        assert np.allclose(E3.c, np.zeros((2,)))
        if TESTING_SHOW_PLOTS:
            import matplotlib.pyplot as plt

            plt.figure()
            plt.gca().set_aspect("equal")
            P.plot(plt.gca(), patch_args={"alpha": 0.2})
            E3.plot(method="inner", ax=plt.gca(), patch_args={"facecolor": "r", "alpha": 0.2})
            plt.xlim([-2, 2])
            plt.ylim([-2, 2])
            plt.show()

    def test_polytopic_outer_approximation(self):
        E = Ellipsoid(c=np.zeros((2,)), Q=np.eye(2))
        P = E.polytopic_outer_approximation(n_halfspaces=8)
        assert P.dim == 2
        assert np.allclose(P.b, np.ones((8,)))
        dir_vectors, _, _ = spread_points_on_a_unit_sphere(2, 6, verbose=False)
        P = E.polytopic_outer_approximation(direction_vectors=dir_vectors)
        assert (P.n_halfspaces, P.dim) == dir_vectors.shape
        assert np.allclose(P.b, np.ones((6,)))
        E.polytopic_outer_approximation(direction_vectors=dir_vectors, n_halfspaces=4)
        assert (P.n_halfspaces, P.dim) == dir_vectors.shape
        assert np.allclose(P.b, np.ones((6,)))

        with pytest.raises(ValueError):
            E.polytopic_outer_approximation(n_halfspaces=-1)

        E = Ellipsoid(c=np.zeros((1,)), Q=[[1]])
        assert E.polytopic_outer_approximation(n_halfspaces=2) == Polytope(c=0, h=1)

        # Low-dimensional ellipsoid
        E = Ellipsoid(c=np.zeros((2,)), G=np.ones((2, 2)))
        P = E.polytopic_outer_approximation()
        assert P.H.shape == (84, 3)
        assert P.He.shape == (1, 3)
        P.minimize_H_rep()
        assert P.H.shape == (2, 3)
        assert P.He.shape == (1, 3)

    def test_polytopic_inner_approximation(self):
        E = Ellipsoid(c=np.zeros((2,)), Q=np.eye(2))
        P = E.polytopic_inner_approximation(n_vertices=8)
        assert P.dim == 2
        assert np.allclose(np.linalg.norm(P.V, ord=2, axis=1), np.ones((8,)))
        dir_vectors, _, _ = spread_points_on_a_unit_sphere(2, 6, verbose=False)
        P = E.polytopic_inner_approximation(direction_vectors=dir_vectors)
        assert (P.n_vertices, P.dim) == dir_vectors.shape
        # Increase rtol to 1e-4 since we are rounding of digits
        assert np.allclose(np.linalg.norm(P.V, ord=2, axis=1), np.ones((6,)), rtol=1e-4)
        E.polytopic_inner_approximation(direction_vectors=dir_vectors, n_vertices=4)
        assert (P.n_vertices, P.dim) == dir_vectors.shape

        with pytest.raises(ValueError):
            E.polytopic_inner_approximation(n_vertices=-1)

        E = Ellipsoid(c=np.zeros((1,)), Q=[[1]])
        assert E.polytopic_inner_approximation(n_vertices=2) == Polytope(c=0, h=1)

        # Low-dimensional ellipsoid
        E = Ellipsoid(c=np.zeros((2,)), G=np.ones((2, 2)))
        P = E.polytopic_inner_approximation()
        assert P.V.shape == (84, 2)
        P.minimize_V_rep()
        assert P.V.shape == (2, 2)

    def test_project_closest_point_distance(self):
        E = Ellipsoid(c=[0, 0, 0], r=2)
        assert np.isclose(E.distance([3, 0, 0]), 1)
        assert np.allclose(E.closest_point([3, 0, 0]), [2, 0, 0])
        assert np.isclose(E.project([3, 0, 0], p=1)[1], 1)
        assert np.isclose(E.project([3, 0, 0], p="inf")[1], 1)

        E = Ellipsoid(c=[0, 1, 0])
        assert np.isclose(E.project([1, 1, 1])[0], E.c).all()

        with pytest.raises(ValueError):
            E.project([3, 0, 0], p="WHAT")

        E.cvxpy_args_socp = {"solver": "WRONG_SOLVER"}
        with pytest.raises(NotImplementedError):
            E.project([3, 0, 0])

        E.cvxpy_args_lp = {"solver": "WRONG_SOLVER"}
        with pytest.raises(NotImplementedError):
            E.project([3, 0, 0], p=1)

    def test_contains(self):
        E = Ellipsoid(c=[0, 0, 0], r=2)
        assert [1, 1, 1] in E
        assert E.contains([1, 1, 1]).all()
        assert E.contains([[1, 1, 1], [1, -1, 1]]).all()
        assert not E.contains([1, 10, 1]).all()
        with pytest.raises(ValueError):
            E.contains([1, 1, 1, 1])
        with pytest.raises(ValueError):
            E.contains([[1, 2], [2, 1], [2, 3]])
        with pytest.raises(ValueError):
            E.contains([[[1, 2], [2, 1]], [[2, 3], [5, 3]]])

        E.cvxpy_args_socp = {"solver": "WRONG_SOLVER"}
        with pytest.raises(NotImplementedError):
            E.project([3, 0, 0])

        E = Ellipsoid(c=[0, 0], r=2)
        P = E.polytopic_outer_approximation(n_halfspaces=8)
        assert P >= E
        assert E <= P
        P = E.polytopic_inner_approximation(n_vertices=8)
        assert E >= P
        assert P <= E
        E_larger = Ellipsoid(c=[0, 0], r=3)
        assert E_larger.contains(E)
        assert E_larger >= E
        assert not E_larger <= E
        assert E == E

        E = Ellipsoid(c=[0, 0], r=2)
        with pytest.raises(ValueError):
            E >= ConstrainedZonotope(lb=[-1, -1], ub=[1, 1])
        with pytest.raises(ValueError):
            E <= ConstrainedZonotope(lb=[-1, -1], ub=[1, 1])

        E.cvxpy_args_sdp = {"solver": "WRONG_SOLVER"}
        with pytest.raises(NotImplementedError):
            E == E

        E = Ellipsoid(c=[[1, 1]])
        assert E.is_singleton
        assert E.contains([[1, 1], [1, 1]]).all()
        assert not E.contains([[1, 1], [2, 1]]).all()
        assert not E.contains([2, 2])
        assert E.contains([1, 1])

    def test_copy(self):
        E = Ellipsoid(c=[0, 0], r=2)
        E.copy()

    def test_print(self):
        E = Ellipsoid(c=[0, 0], r=2)
        print(E)
        print(E.__repr__)
        print(E.__str__)
        print(E.__doc__)
        E = Ellipsoid(c=[0, 0], G=[[1, 0], [0, 0]])
        print(E)
        print(E.__repr__)
        print(E.__str__)
        print(E.__doc__)
        E = Ellipsoid(c=[0, 0])
        print(E)
        print(E.__repr__)
        print(E.__str__)
        print(E.__doc__)

    def test_volume(self):
        r = 2
        E = Ellipsoid(c=[0, 0], r=r)
        assert np.isclose(E.volume(), np.pi * (r**2))
        E = Ellipsoid(c=[0, 0, 0], r=r)
        assert np.isclose(E.volume(), 4 / 3 * np.pi * (r**3))
        E = Ellipsoid(c=[0, 0, 0], G=np.zeros((3, 3)))
        assert np.isclose(E.volume(), 0)

    def test_projection(self):
        a = 1
        b = 3
        Q = [[a**2, 0], [0, b**2]]
        ell = Ellipsoid(c=np.ones((2,)), Q=Q)
        assert ell.projection(1) == Ellipsoid(c=ell.c[0], r=a)

    def test_circumscribing_inscribing_ellipsoid_rectangle_chebyshev(self):
        a = 1
        b = 3
        Q = [[a**2, 0], [0, b**2]]
        ell = Ellipsoid(c=np.ones((2,)), Q=Q)
        c, Q, G = ell.minimum_volume_circumscribing_ellipsoid()
        assert np.allclose(c, ell.c)
        assert np.allclose(Q, ell.Q)
        assert np.allclose(G, ell.G)
        c, Q, G = ell.maximum_volume_inscribing_ellipsoid()
        assert np.allclose(c, ell.c)
        assert np.allclose(Q, ell.Q)
        assert np.allclose(G, ell.G)
        c, r = ell.chebyshev_centering()
        assert np.allclose(c, ell.c)
        assert np.isclose(r, min(a, b))
        c, r = ell.minimum_volume_circumscribing_ball()
        assert np.allclose(c, ell.c)
        assert np.isclose(r, max(a, b))
        lb, ub = ell.minimum_volume_circumscribing_rectangle()
        assert np.allclose(lb, ell.c - np.array([a, b]))
        assert np.allclose(ub, ell.c + np.array([a, b]))

        # Low-dimensional ellipsoid
        E = Ellipsoid(c=[0, 0, 0], G=np.zeros((3, 3)))
        c, r = E.chebyshev_centering()
        assert np.isclose(r, 0.0)

    def test_affine_hull_and_quadratic_form(self):
        E = Ellipsoid(c=[0, 0], r=1)
        Qmat_E = E.quadratic_form_as_a_symmetric_matrix()
        Ae = np.array([1, 1])
        be = 0.9
        Eslice = E.intersection_with_affine_set(Ae=Ae, be=be)
        Qmat_Eslice = Eslice.quadratic_form_as_a_symmetric_matrix()
        Eslice_Ae, Eslice_be = Eslice.affine_hull()
        scaling = be / Eslice_be
        assert np.isclose(scaling * Eslice_Ae, Ae).all()
        assert E.affine_hull()[0] is None
        assert E.affine_hull()[1] is None

        if TESTING_SHOW_PLOTS:
            import itertools as itert

            import matplotlib.pyplot as plt

            fig, (ax1, ax2) = plt.subplots(ncols=2, sharex=True)
            x_grid = np.arange(start=-1.2, stop=1.201, step=0.1)
            for x, y in itert.product(x_grid, repeat=2):
                if [x, y, 1] @ Qmat_Eslice @ [x, y, 1] <= 0 and abs(Eslice_Ae @ [x, y] - Eslice_be) <= PYCVXSET_ZERO:
                    ax1.scatter(x, y, s=10, color="r")
                elif [x, y, 1] @ Qmat_E @ [x, y, 1] <= 0:
                    ax1.scatter(x, y, s=10, color="b")
                elif [x, y, 1] @ Qmat_Eslice @ [x, y, 1] <= 0:
                    ax1.scatter(x, y, s=10, color="g")
                else:
                    ax1.scatter(x, y, s=10, color="k")
            ax1.set_aspect("equal")
            ax1.set_title("Grid-based containment check")
            E.plot(ax=ax2)
            Eslice.plot(ax=ax2, patch_args={"color": "r"})
            ax2.set_aspect("equal")
            ax2.set_title("Plotting via pycvxset")
            plt.show()

    def test_affine_set_intersection(self):
        E = Ellipsoid(c=[0, 0], r=2)
        Eslice = E.intersection_with_affine_set(Ae=[1, 1], be=1.5)
        assert [1, 1] not in Eslice
        assert Eslice <= E

        P = Eslice.polytopic_inner_approximation()
        P.minimize_V_rep()
        assert Eslice.contains(P.V).all()
        assert E.contains(P.V).all()

        if TESTING_SHOW_PLOTS:
            import matplotlib.pyplot as plt

            ax, _, _ = E.plot()
            Eslice.plot(method="outer", ax=ax, patch_args={"edgecolor": "r"})
            ax.set_aspect("equal")
            plt.show()

        E = Ellipsoid(c=[0, 0, 0], r=1) + [1, 0, 1]
        Ae = [1, 1, 1]

        # Intersection is with whole R^n
        assert E.intersection_with_affine_set(Ae=[], be=[]) == E

        with pytest.raises(ValueError):
            E.intersection_with_affine_set(Ae=Ae, be=Ae @ E.c + np.sqrt(3) + 0.01)

        # Intersection is a point
        intersection_point = np.array([1, 1, 1]) / np.sqrt(3) + E.c
        Eslice = E.intersection_with_affine_set(Ae=Ae, be=Ae @ E.c + np.sqrt(3))
        assert Eslice.is_singleton
        assert intersection_point in Eslice

        # Intersection of the affine set is a point inside E
        intersection_point = np.array([0.1, 0, 0.1]) + E.c
        Eslice = E.intersection_with_affine_set(Ae=np.eye(3), be=intersection_point)
        assert Eslice.is_singleton
        assert intersection_point in Eslice

        # Intersection of the affine set is a point outside E
        intersection_point = np.array([1, 1, 1]) + E.c
        with pytest.raises(ValueError):
            E.intersection_with_affine_set(Ae=np.eye(3), be=intersection_point)

        # Intersection with an empty set
        with pytest.raises(ValueError):
            E.intersection_with_affine_set(Ae=np.vstack((np.eye(3), [1, 1, 1])), be=[0, 0, 0, 1])

        # Intersection with a different affine set
        with pytest.raises(ValueError):
            E.intersection_with_affine_set(Ae=np.eye(4), be=[0, 0, 0, 1])

        # Intersection is a 2D circle
        Eslice = E.intersection_with_affine_set(Ae=Ae, be=Ae @ E.c + np.sqrt(3) / 2)
        assert [1, 1, 1] not in Eslice
        assert Eslice <= E

        P = Eslice.polytopic_inner_approximation(n_vertices=30)
        P.minimize_V_rep()
        assert Eslice.contains(P.V).all()
        assert E.contains(P.V).all()

        if TESTING_SHOW_PLOTS:
            import matplotlib.pyplot as plt

            ax, _, _ = E.plot()
            ax.scatter(P.V[:, 0], P.V[:, 1], P.V[:, 2])
            ax.set_aspect("equal")
            plt.show()

    def test_slice_then_projection(self):
        P = Ellipsoid(c=[0, 0, 0], r=0.5)
        Q = P.slice(dims=2, constants=0.25).projection(project_away_dims=2)
        Q_projection = P.slice_then_projection(dims=2, constants=0.25)
        assert Q.dim == 2
        assert Q == Q_projection
        assert Q == Ellipsoid(c=[0, 0], r=math.sqrt(0.5**2 - 0.25**2))

    def test_containment_constraints(self):
        E = Ellipsoid(c=[0, 0], r=2)
        opt_x_value = E.extreme([1, 1])
        for test_shape in [2, (2,), (2, 1), (1, 2), (1, 1, 2), (1, 2, 1)]:
            x = cp.Variable(test_shape)
            # Warn only if 3D variables
            check_warning = False
            try:
                check_warning = len(test_shape) > 2
            except TypeError:
                pass
            prob = cp.Problem(objective=cp.Maximize(cp.sum(x)), constraints=E.containment_constraints(x)[0])
            if check_warning:
                with pytest.warns(UserWarning):
                    prob.solve(**DEFAULT_CVXPY_ARGS_SOCP)
            else:
                prob.solve(**DEFAULT_CVXPY_ARGS_SOCP)
            assert np.isclose(x.value, opt_x_value).all()
            prob = cp.Problem(
                objective=cp.Maximize(cp.sum(x)), constraints=E.containment_constraints(x.flatten(order="F"))[0]
            )
            if check_warning:
                with pytest.warns(UserWarning):
                    prob.solve(**DEFAULT_CVXPY_ARGS_SOCP)
            else:
                prob.solve(**DEFAULT_CVXPY_ARGS_SOCP)
            assert np.isclose(x.value, opt_x_value).all()

        # Skip checking empty ellipsoid
