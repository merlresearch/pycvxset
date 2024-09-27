# Copyright (C) 2020-2024 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

# Code purpose: A diagnostic python script to check if pycvxset is installed correctly.

from argparse import ArgumentParser

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation

from pycvxset import Ellipsoid, Polytope


def run_demo(use_plot_show, save_plot):
    # Polytope definition
    P = Polytope(c=[0, 0, 0], h=[1, 0.5, 0.1])
    P = P.intersection_with_halfspaces([1, -0.5, 0], 0.25)

    # Transformation details
    rotate_angle = np.pi / 4
    R = Rotation.from_rotvec(rotate_angle * np.array([0, 0, 1])).as_matrix()
    shift_vec = [5, 4, 3]

    # Plot the 3D plot
    fig = plt.figure(figsize=(8, 4))
    ax_3D = fig.add_subplot(1, 2, 1, projection="3d")
    P.plot(ax=ax_3D, patch_args={"facecolor": "lightblue", "label": "Original polytope"})
    P_after_affine_transformation = R @ P + shift_vec
    P_after_affine_transformation.plot(
        ax=ax_3D, patch_args={"facecolor": "lightgreen", "label": "Transformed polytope"}
    )
    ax_3D.legend(loc="best")
    ax_3D.set_title(
        f"Rotate a polytope about z-axis by {np.rad2deg(rotate_angle)}" + r"$^\circ$" + f"\nand shift it by {shift_vec}"
    )
    ax_3D.set_xlabel("x")
    ax_3D.set_ylabel("y")
    ax_3D.set_zlabel("z")

    transformed_P_projected_to_XY = P_after_affine_transformation.projection(project_away_dim=2)
    minimum_volume_circumscribing_ellipsoid = Ellipsoid.deflate(transformed_P_projected_to_XY)
    maximum_volume_inscribing_ellipsoid = Ellipsoid.inflate(transformed_P_projected_to_XY)

    # Plot the 2D plot
    ax_2d = fig.add_subplot(1, 2, 2)
    transformed_P_projected_to_XY.plot(
        ax=ax_2d, patch_args={"facecolor": "lightgreen", "label": "Transformed polytope"}
    )
    minimum_volume_circumscribing_ellipsoid.plot(
        method="outer", ax=ax_2d, patch_args={"fill": False, "label": "Min. vol. circumscribing ellipsoid"}
    )
    maximum_volume_inscribing_ellipsoid.plot(
        method="inner", ax=ax_2d, patch_args={"facecolor": "lightpink", "label": "Max. vol. inscribing ellipsoid"}
    )
    ax_2d.set_title("Projection of transformed polytope on to XY\n and ellipsoidal approximations")
    ax_2d.set_xlabel("x")
    ax_2d.set_ylabel("y")
    ax_2d.grid()
    ax_2d.set_xlim([3.5, 6.5])
    ax_2d.set_ylim([2.85, 5.85])
    ax_2d.legend(loc="best")
    plt.subplots_adjust(left=0.026, bottom=0.15, right=0.95, top=0.88, wspace=0.41, hspace=0.2)
    if save_plot:
        plt.savefig("docs/source/_static/pycvxset_diag.png", dpi=300)
        print("Plot saved!")
    if use_plot_show:
        plt.show()
    else:
        plt.close()


if __name__ == "__main__":
    parser = ArgumentParser(
        prog="pycvxset_diag",
        description="A python script that performs simple Polytope manipulations to make sure "
        "pycvxset is installed correctly",
    )
    parser.add_argument("--do_not_use_plot_show", action="store_false")
    parser.add_argument("--save_plot", action="store_true")
    args = parser.parse_args()
    run_demo(args.do_not_use_plot_show, args.save_plot)
