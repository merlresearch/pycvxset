# Copyright (C) 2020-2024 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

# Code purpose: Python script to generate pycvxset logo.

from argparse import ArgumentParser

import matplotlib.pyplot as plt

from pycvxset import Ellipsoid, Polytope


def generate_logo(use_plot_show, save_plot):
    V = [[-1, 0.5], [-1, 1], [1, 1], [1, -1], [0.5, -1]]
    P = Polytope(V=V)
    bounding_box = Polytope.deflate_rectangle(P)
    maximum_volume_inscribing_ellipsoid = Ellipsoid.inflate(P)
    chebyshev_ball_of_maximum_volume_inscribing_ellipsoid = Ellipsoid.inflate_ball(maximum_volume_inscribing_ellipsoid)
    plt.figure(figsize=(2, 2), dpi=320)
    ax = plt.gca()
    bounding_box.plot(ax=ax, patch_args={"facecolor": "pink", "linewidth": 3, "linestyle": ":"})
    P.plot(ax=ax)
    maximum_volume_inscribing_ellipsoid.plot(ax=ax, patch_args={"facecolor": "gold", "alpha": 0.6})
    maximum_volume_inscribing_ellipsoid.plot(ax=ax, patch_args={"facecolor": None})
    chebyshev_ball_of_maximum_volume_inscribing_ellipsoid.plot(
        ax=ax, center_args={"color": "k"}, patch_args={"facecolor": "gray"}
    )
    plt.tick_params(
        which="both",  # both major and minor ticks are affected
        bottom=False,  # ticks along the bottom edge are off
        top=False,  # ticks along the top edge are off
        left=False,  # ticks along the left edge are off
        right=False,  # ticks along the right edge are off
        labelleft=False,  # labels along the left edge are off
        labelbottom=False,  # labels along the bottom edge are off
    )
    plt.subplots_adjust(left=0.02, bottom=0.02, right=0.98, top=0.98)
    plt.tight_layout
    if save_plot:
        plt.savefig("docs/source/_static/pycvxset_logo.png", dpi=320)
        print("Logo saved!")
    if use_plot_show:
        plt.show()
    else:
        plt.close()


if __name__ == "__main__":
    parser = ArgumentParser(
        prog="pycvxset_logo",
        description="A python script to generate pycvxset logo",
    )
    parser.add_argument("--do_not_use_plot_show", action="store_false")
    parser.add_argument("--save_plot", action="store_true")
    args = parser.parse_args()
    generate_logo(args.do_not_use_plot_show, args.save_plot)
