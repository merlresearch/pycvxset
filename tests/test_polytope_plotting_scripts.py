# Copyright (C) 2020-2025 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

# Code purpose: Test the Polytope class methods for plotting

import matplotlib.pyplot as plt
import pytest

from pycvxset import Polytope
from pycvxset.common.constants import TESTING_SHOW_PLOTS

TESTING_SHOW_PLOTS = False or TESTING_SHOW_PLOTS


def test_print_and_plot():
    # Plotting
    with pytest.warns(UserWarning, match="Can not plot an empty polytope!"):
        Polytope(dim=2).plot()
    plt.close()
    with pytest.warns(UserWarning, match="Can not plot an empty polytope!"):
        Polytope(dim=2).plot2d()
    plt.close()
    with pytest.warns(UserWarning, match="Can not plot an empty polytope!"):
        Polytope(dim=3).plot()
    plt.close()
    with pytest.warns(UserWarning, match="Can not plot an empty polytope!"):
        Polytope(dim=3).plot3d()
    plt.close()
    P = Polytope(A=[[1, 1], [-1, -1]], b=[1, 1])
    # Issues two warnings!
    with pytest.warns(UserWarning, match="bounded"):
        try:
            P.plot()
        except NotImplementedError as err:
            print(err)
        P.cvxpy_args_lp = {"solver": "OSQP"}
        P.plot()
    plt.close()
    with pytest.warns(UserWarning, match="Can not plot an unbounded polytope!"):
        P.plot2d()
    plt.close()
    P3D = Polytope(A=[[1, 0, 0], [-1, 0, 0], [0, -1, 0], [0, 0, -1]], b=[1, 0, 0, 0])
    # Issues two warnings!
    with pytest.warns(UserWarning, match="bounded"):
        P3D.plot()
    plt.close()
    with pytest.warns(UserWarning, match="Can not plot an unbounded polytope!"):
        P3D.plot3d()
    plt.close()

    # Incorrect dimension errors
    with pytest.raises(ValueError):
        Polytope(c=[0, 0], h=1).plot3d()
    with pytest.raises(ValueError):
        Polytope(c=[0, 0, 0], h=1).plot2d()

    # Labeling
    Polytope(c=[0, 0, 0], h=1).plot(patch_args={"label": "Test"})
    plt.close()
    Polytope(c=[0, 0, 0], h=1).plot(vertex_args={"label": "Test"})
    plt.close()
    Polytope(c=[0, 0], h=1).plot(vertex_args={"label": "Test"})
    plt.close()
    Polytope(c=[0, 0], h=1).plot(patch_args={"label": "Test"})
    plt.close()
    Polytope(c=[0, 0], h=1).plot(patch_args={"label": "Test", "fill": False})
    plt.close()
    Polytope(c=[0, 0], h=1).plot(patch_args={"label": "Test", "facecolor": None})
    plt.close()

    # Testing sanitize patch_args
    # 2D
    P = Polytope(c=[0, 0], h=1)
    # No color
    P.plot(patch_args={"fill": False})
    P.plot(patch_args={"facecolor": None})
    P.plot(patch_args={"fill": False, "facecolor": None})
    # With color
    P.plot(patch_args={"fill": True})
    P.plot(patch_args={"facecolor": "r"})
    P.plot(patch_args={"fill": True, "facecolor": "r"})
    plt.close()
    # Raise error for incorrect patch_args
    with pytest.raises(ValueError):
        P.plot(patch_args={"fill": False, "facecolor": "r"})
    with pytest.raises(ValueError):
        P.plot(patch_args={"fill": True, "facecolor": None})

    # 3D
    P = Polytope(c=[0, 0, 0], h=1)
    # No color
    P.plot(patch_args={"fill": False})
    P.plot(patch_args={"facecolor": None})
    P.plot(patch_args={"fill": False, "facecolor": None})
    # With color
    P.plot(patch_args={"fill": True})
    P.plot(patch_args={"facecolor": "r"})
    P.plot(patch_args={"fill": True, "facecolor": "r"})
    plt.close()
    P.plot(patch_args={"label": "Plot without facecolor", "facecolor": None})
    P.plot(patch_args={"label": "Plot without facecolor", "facecolor": "blue"})
    P.plot(patch_args={"label": "Plot with facecolor"})
    P.plot(patch_args={"label": "Plot with facecolor"}, vertex_args={"visible": True, "marker": "x"})
    plt.close()
    # Raise error for incorrect patch_args
    with pytest.raises(ValueError):
        P.plot(patch_args={"fill": False, "facecolor": "r"})
    with pytest.raises(ValueError):
        P.plot(patch_args={"fill": True, "facecolor": None})
    with pytest.raises(ValueError):
        P.plot(patch_args={"alpha": 0.5, "label": "Can not plot without facecolor and alpha set", "facecolor": None})
    # Can't plot 4D
    with pytest.raises(NotImplementedError):
        Polytope(c=[0, 0, 0, 0], h=1).plot()
    # Can't turn off autoscale for 3D
    with pytest.raises(NotImplementedError):
        Polytope(c=[0, 0, 0], h=1).plot(autoscale_enable=False)

    if TESTING_SHOW_PLOTS:
        P.plot()
        P3 = Polytope(lb=-1, ub=1)
        P3.plot()
        P3.plot(edgecolor="k")
        P3.plot(edgecolor="k")
        P4 = Polytope(V=[[-2, 1], [3, 2], [4, 5]])
        P4.plot()
        P5 = Polytope(V=[[-2, 3, 4], [3, 3, 4], [4, 3, 4]])
        P5.plot()
    else:
        P4 = Polytope(V=[[-2, 1], [3, 2], [4, 5]])
        ax, h4, _ = P4.plot()
        _, h4_2, h4_2v = (P4 + [1, 1]).plot(ax=ax, vertex_args={"color": "g"}, patch_args={"facecolor": "m"})
        ax.legend([h4, h4_2, h4_2v], ["P", "shifted P", "shifted P.v"])
        plt.close()
        # Low-dimensional plot
        Polytope(V=[[-2, 1, 1], [3, 1, 2], [5, 1, 5]]).plot()
        plt.close()
        P5 = Polytope(V=[[-2, 1, 1], [3, 10, 2], [5, 1, 5], [1, 1, 4]])
        assert P5.is_full_dimensional
        ax, h5, _ = P5.plot()
        _, h5_2, h5_2v = (P5 + [1, 1, 1]).plot(ax=ax, vertex_args={"color": "g"}, patch_args={"facecolor": "m"})
        ax.legend([h5, h5_2, h5_2v], ["P", "shifted P", "shifted P.v"])
        plt.close()
