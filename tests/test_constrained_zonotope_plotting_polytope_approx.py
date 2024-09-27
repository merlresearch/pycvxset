# Copyright (C) 2020-2024 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

# Code purpose: Test the ConstrainedZonotope class for plotting and polytope-related operations

import matplotlib.pyplot as plt
import pytest

from pycvxset import ConstrainedZonotope, Polytope, spread_points_on_a_unit_sphere
from pycvxset.common.constants import TESTING_SHOW_PLOTS

TESTING_SHOW_PLOTS = False or TESTING_SHOW_PLOTS


def test_approximate_empty_sets():
    C = ConstrainedZonotope(dim=2)
    C_inner_as_P = C.polytopic_inner_approximation()
    assert C_inner_as_P.is_empty
    assert C_inner_as_P.dim == 2
    C_outer_as_P = C.polytopic_outer_approximation()
    assert C_outer_as_P.is_empty
    assert C_outer_as_P.dim == 2


def test_plot_2D():
    FEW_POINTS = 4
    MANY_POINTS = 10
    P = Polytope(V=spread_points_on_a_unit_sphere(2, 7)[0])
    C = ConstrainedZonotope(polytope=P)
    ax, _, _ = C.plot(
        method="outer", n_halfspaces=FEW_POINTS, patch_args={"alpha": 0.3, "label": "Outer-approximation"}
    )
    P.plot(ax=ax, patch_args={"facecolor": "lightpink", "alpha": 0.3, "label": "True polytope"})
    C.plot(
        method="inner",
        ax=ax,
        n_vertices=FEW_POINTS,
        patch_args={"facecolor": None, "edgecolor": "k", "label": "Inner-approximation"},
    )
    ax.set_title("2D example with few vertices")
    ax.legend()

    ax, _, _ = C.plot(
        method="outer", n_halfspaces=MANY_POINTS, patch_args={"alpha": 0.3, "label": "Outer-approximation"}
    )
    P.plot(ax=ax, patch_args={"facecolor": "lightpink", "alpha": 0.3, "label": "True polytope"})
    C.plot(
        method="inner",
        ax=ax,
        n_vertices=MANY_POINTS,
        patch_args={"facecolor": None, "edgecolor": "k", "label": "Inner-approximation"},
    )
    ax.set_title("2D example with many vertices")
    ax.legend()

    if TESTING_SHOW_PLOTS:
        plt.show()
    else:
        plt.close()


def test_plot_3D():
    FEW_POINTS = 6
    MANY_POINTS = 12
    P = Polytope(V=spread_points_on_a_unit_sphere(3, 6)[0])
    C = ConstrainedZonotope(polytope=P)
    patch_args_outer = {"edgecolor": "blue", "facecolor": None, "label": "Outer-approximation"}
    patch_args_inner = {"edgecolor": "k", "facecolor": None, "label": "Inner-approximation"}
    ax, _, _ = C.plot(method="outer", n_halfspaces=FEW_POINTS, patch_args=patch_args_outer)
    P.plot(ax=ax, patch_args={"edgecolor": "lightpink", "facecolor": None, "label": "True polytope"})
    C.plot(method="inner", ax=ax, n_vertices=FEW_POINTS, patch_args=patch_args_inner)
    ax.set_title("3D example with few vertices")
    ax.legend()

    with pytest.warns(UserWarning, match="Invalid combination of"):
        ax, _, _ = C.plot(
            method="outer",
            n_halfspaces=MANY_POINTS,
            patch_args={"edgecolor": "blue", "facecolor": None, "label": "Outer-approximation"},
        )
    P.plot(ax=ax, patch_args={"edgecolor": "lightpink", "facecolor": None, "label": "True polytope"})
    with pytest.warns(UserWarning, match="Invalid combination of"):
        C.plot(
            method="inner",
            ax=ax,
            n_vertices=MANY_POINTS,
            patch_args={"facecolor": None, "edgecolor": "k", "label": "Inner-approximation"},
        )
    ax.set_title("3D example with many vertices")
    ax.legend()

    if TESTING_SHOW_PLOTS:
        plt.show()
    else:
        plt.close()


def test_plotting_with_specified_dir_vectors():
    P = Polytope(V=spread_points_on_a_unit_sphere(3, 6)[0])
    C = ConstrainedZonotope(polytope=P)
    with pytest.warns(UserWarning, match="Invalid combination of"):
        dir_vectors = spread_points_on_a_unit_sphere(3, 15)[0]
    C.plot(method="outer", direction_vectors=dir_vectors)
    C.plot(method="inner", direction_vectors=dir_vectors)
    plt.close()


if __name__ == "__main__":
    test_plot_3D()
