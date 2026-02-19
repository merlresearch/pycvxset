# Copyright (C) 2020-2026 Mitsubishi Electric Research Laboratories (MERL)
# Copyright (c) 2019 Tor Aksel N. Heirung
#
# SPDX-License-Identifier: AGPL-3.0-or-later
# SPDX-License-Identifier: MIT

# Code purpose:  Define the plotting methods for the Polytope class
# Coverage: This file has 5 missing statements + 4 excluded statements + 1 partial branches.

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any, Optional, cast

if TYPE_CHECKING:
    from pycvxset.Polytope import Polytope

import cdd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.patches import Polygon
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection
from mpl_toolkits.mplot3d.axes3d import Axes3D
from scipy.spatial.transform import Rotation  # pyright: ignore[reportUnknownVariableType]

from pycvxset.common import prune_and_round_vertices
from pycvxset.common.constants import (
    DEFAULT_PATCH_ARGS_2D,
    DEFAULT_PATCH_ARGS_3D,
    DEFAULT_VERTEX_ARGS,
    PLOTTING_DECIMAL_PRECISION_CDD,
)
from pycvxset.Polytope.vertex_halfspace_enumeration import (
    get_cdd_polyhedron_from_Ab_Aebe,
    get_cdd_polyhedron_from_V,
    set_attributes_minimal_Ab_Aebe_from_cdd_polyhedron,
    set_attributes_V_from_cdd,
)


def plot(
    self: "Polytope",
    ax: Optional[Axes | Axes3D | None] = None,
    patch_args: Optional[dict[str, Any]] = None,
    vertex_args: Optional[dict[str, Any]] = None,
    autoscale_enable: bool = True,
    decimal_precision: int = PLOTTING_DECIMAL_PRECISION_CDD,
    enable_warning: bool = True,
) -> tuple[Any, Any, Any]:
    """
    Plot a 2D or 3D polytope.

    Args:
        ax (Axes | Axes3D | None, optional): Axis on which the patch is to be plotted
        patch_args (dict, optional): Arguments to pass for plotting faces and edges. See [Matplotlib-patch]_ for options
            for patch_args for 2D and [Matplotlib-Line3DCollection]_ for options for patch_args for 3D. Defaults to
            None, in which case we set edgecolor to black, and facecolor to skyblue.
        vertex_args (dict, optional): Arguments to pass for plotting vertices. See [Matplotlib-scatter]_ for
            options for vertex_args for 2D and [Matplotlib-Axes3D.scatter]_ for options for vertex_args for 3D. Defaults
            to None, in which case we skip plotting the vertices.
        autoscale_enable (bool, optional): When set to True, matplotlib adjusts axes to view full polytope. Defaults
            to True.
        decimal_precision (int, optional): When plotting a 3D polytope that is in V-Rep and not in H-Rep, we round
            vertex to the specified precision to avoid numerical issues. Defaults to PLOTTING_DECIMAL_PRECISION_CDD
            specified in pycvxset.common.constants.
        enable_warning (bool, optional): Enables the UserWarning. May be turned off if expected. Defaults to True.

    Raises:
        ValueError: When an incorrect axes is provided.
        NotImplementedError: When a polytope.dim >= 4 or == 1 | autoscale_enable is set to False for 3D plotting.
        UserWarning: When an empty polytope or an unbounded polytope is provided.
        UserWarning: In 3D plotting, when all faces have less than 3 vertices

    Returns:
        (axes, handle, handle): Tuple with axes containing the polygon, handle for plotting patch, handle for plotting
        vertices

    Notes:
        - plot is a wrapper for :meth:`plot2d` and :meth:`plot3d` functions. See
          their documentation for more details.
        - *Vertex-halfspace enumeration for 2D polytopes*: If a 2D polytope is in H-Rep, vertex enumeration is performed
          to plot the polytope in 2D. No vertex enumeration is performed for a 2D polytope in V-Rep. This function calls
          minimize_V_rep to simplify plotting.
        - *Vertex-halfspace enumeration for 3D polytopes*: The 3D polytope needs to have both V-Rep and H-Rep.
          Consequently, at least a halfspace/vertex enumeration is performed for a 3D polytope in single representation.
        - *Rounding vertices*: This function calls :meth:`pycvxset.common.prune_and_round_vertices` when a 3D polytope
          in V-Rep is given to be plotted.
        - *Handle returned for the patches*: For 2D plot, axis handle of the single patch is returned. For 3D
          plotting, multiple patches are present, and plot returns the first patch's axis handle. In this case, the
          order of patches are determined by the use of `pycddlib.copy_input_incidence
          <https://pycddlib.readthedocs.io/en/latest/cdd.html#cdd.copy_input_incidence>`_ from the from the specified
          (A, b). Also, labeling the first patch is done using Poly3DCollection.
        - *Disabling face colors*: We can plot the polytope frames without filling it by setting
          `patch_args['facecolor'] = None` or `patch_args['fill'] = False`. If visual comparison of 3D polytopes is
          desired, it may be better to plot just the polytope frames instead by setting patch_args['facecolor'] = None.

    Warning:
        This function may produce erroneous-looking plots when visually comparing multiple 3D sets. For more info on
        matplotlib limitations, see
        https://matplotlib.org/stable/api/toolkits/mplot3d/faq.html#my-3d-plot-doesn-t-look-right-at-certain-viewing-angles.
        If visual comparison is desired, it may be better to plot just the polytope frame instead by setting
        patch_args['facecolor'] = None.
    """

    # Start plotting
    if self.dim == 2:
        return plot2d(
            self,
            ax=ax,
            patch_args=patch_args,
            vertex_args=vertex_args,
            autoscale_enable=autoscale_enable,
            enable_warning=enable_warning,
        )
    elif self.dim == 3:
        return plot3d(
            self,
            ax=ax,
            patch_args=patch_args,
            vertex_args=vertex_args,
            autoscale_enable=autoscale_enable,
            decimal_precision=decimal_precision,
            enable_warning=enable_warning,
        )
    else:
        raise NotImplementedError("Plot is not implemented for n >= 4 or n <= 1.")


def plot2d(
    self: "Polytope",
    ax: Optional[Axes | Axes3D | None] = None,
    patch_args: Optional[dict[str, Any]] = None,
    vertex_args: Optional[dict[str, Any]] = None,
    autoscale_enable: bool = True,
    enable_warning: bool = True,
) -> tuple[Any, Any, Any]:
    """
    Plot a 2D polytope using matplotlib's add_patch(Polygon()).

    Args:
        ax (Axes | Axes3D | None, optional): Axis on which the patch is to be plotted
        patch_args (dict, optional): Arguments to pass for plotting faces and edges. See [Matplotlib-patch]_
            for options for patch_args. Defaults to None, in which case we set edgecolor to black, and facecolor to
            skyblue.
        vertex_args (dict, optional): Arguments to pass for plotting vertices. See [Matplotlib-scatter]_ for
            options for vertex_args. Defaults to None, in which case we skip plotting the vertices.
        autoscale_enable (bool, optional): When set to True (default), matplotlib adjusts axes to view full polytope.
        enable_warning (bool, optional): Enables the UserWarning. May be turned off if expected. Defaults to True.

    Raises:
        ValueError: When a 2D polytope is provided

    Returns:
        (axes, handle, handle): Tuple with axes containing the polygon, handle for plotting patch, handle for plotting
        vertices

    Notes:
        This function requires polytope in V-Rep, and performs a vertex enumeration if the polytope is H-Rep.

        We sort the vertices in counter-clockwise direction with respect to the centroid, and then plot it using
        matplotlib's Polygon. We can plot just the polytope frames without filling it by setting
        `patch_args['facecolor'] = None` or `patch_args['fill'] = False`.
    """
    if self.dim != 2:
        raise ValueError("Expected a 2D polytope")
    elif self.is_empty:
        # Can't plot an empty polytope
        if enable_warning:
            warnings.warn("Can not plot an empty polytope!", UserWarning)
        return plt.gca(), None, None
    elif not self.is_bounded:
        # Can't plot an unbounded polytope
        if enable_warning:
            warnings.warn("Can not plot an unbounded polytope!", UserWarning)
        return plt.gca(), None, None

    if not ax:
        plt.figure()
        ax = plt.gca()

    patch_args, vertex_args = sanitize_patch_args_and_vertex_args(patch_args, vertex_args, 2)

    # Ensures that redundant vertices are removed, and if no V-rep, we will call it as well
    self.minimize_V_rep()

    # Order the vertices
    c = self.interior_point(point_type="centroid")
    order = np.argsort(np.arctan2(self.V[:, 1] - c[1], self.V[:, 0] - c[0]))
    # Plot the patch
    sorted_V = self.V[order, :]
    h_patch = ax.add_patch(Polygon(sorted_V, closed=True, **patch_args))
    # Plot vertices
    h_vert = ax.scatter(sorted_V[:, 0], sorted_V[:, 1], **vertex_args)
    # Set up autoscaling
    ax.autoscale(enable=autoscale_enable)
    return ax, h_patch, h_vert  # handle(s) to the patch(es)


def plot3d(
    self: "Polytope",
    ax: Optional[Axes | Axes3D | None] = None,
    patch_args: Optional[dict[str, Any]] = None,
    vertex_args: Optional[dict[str, Any]] = None,
    autoscale_enable: bool = True,
    decimal_precision: int = PLOTTING_DECIMAL_PRECISION_CDD,
    enable_warning: bool = True,
    prune_V: bool = True,
) -> tuple[Any, Any, Any]:
    """Plot a 3D polytope using matplotlib's Line3DCollection.

    Args:
        ax (Axes | Axes3D | None, optional): Axes to plot. Defaults to None, in which case a new axes is created. The
            function assumes that the provided axes was defined with projection='3d'.
        patch_args (dict, optional): Arguments to pass for plotting faces and edges. See [Matplotlib-Line3DCollection]_
            for options for patch_args. Defaults to None, in which case we set edgecolor to black, and facecolor to
            skyblue.
        vertex_args (dict, optional): Arguments to pass for plotting vertices. See [Matplotlib-Axes3D.scatter]_ for
            options for vertex_args. Defaults to None, in which case we skip plotting the vertices.
        autoscale_enable (bool, optional): When set to True (default), matplotlib adjusts axes to view full polytope.
        decimal_precision (int, optional): When plotting a 3D polytope that is in V-Rep and not in H-Rep, we round
            vertex to the specified precision to avoid numerical issues. Defaults to PLOTTING_DECIMAL_PRECISION_CDD
            specified in pycvxset.common.constants.
        enable_warning (bool, optional): Enables the UserWarning. May be turned off if expected. Defaults to True.
        prune_V (bool, optional): When True, prune vertices before plotting. Defaults to True.

    Raises:
        ValueError: When either a non-3D polytope is provided.
        UserWarning: When all faces have less than 3 vertices
        NotImplementedError: autoscale_enable needs to be currently enabled for 3D plotting

    Returns:
        (axes, handle, handle): Tuple with axes containing the polygon, handle for plotting patch, handle for plotting
        vertices

    Notes:
        - This function requires polytope in H-Rep and V-Rep, and uses CDD to compute incidences, and vertices.
          Consequently, at least one halfspace/vertex enumeration is performed.
        - Since 3D plotting involves multiple patches, the first patch's axis handle is returned. In this case, the
          order of patches are determined by the use of `pycddlib.copy_input_incidence
          <https://pycddlib.readthedocs.io/en/latest/cdd.html#cdd.copy_input_incidence>`_ from the specified (A, b).
        - This function calls :meth:`pycvxset.common.prune_and_round_vertices` when a 3D polytope
          in V-Rep is given to be plotted.
        - When label is passed in patch_args, the label is only applied to the first patch. Subsequent patches will not
          have a label. Moreover, the first patch is plotted using Poly3DCollection, while the subsequent patches are
          plotted using Line3DCollection in this case. Otherwise, when no label is provided, all patches are plotted
          using Line3DCollection.
        - We iterate over each halfspace, rotate the halfspace about its centroid so that halfspace is now parallel to
          XY plane, and then sort the vertices based on their XY coordinates in counter-clockwise direction with respect
          to the centroid, and then plot it using matplotlib's Line3DCollection.

    Warning:
        This function may produce erroneous-looking plots when visually comparing multiple 3D sets. For more info on
        matplotlib limitations, see
        https://matplotlib.org/stable/api/toolkits/mplot3d/faq.html#my-3d-plot-doesn-t-look-right-at-certain-viewing-angles.
        If visual comparison is desired, it may be better to plot just the polytope frame instead by setting
        patch_args['facecolor'] = None.
    """

    if self.dim != 3:
        raise ValueError("Expected a 3D polytope")
    elif self.is_empty:
        # Can't plot an empty polytope
        if enable_warning:
            warnings.warn("Can not plot an empty polytope!", UserWarning)
        return plt.gca(), None, None
    elif not self.is_bounded:
        # Can't plot an unbounded polytope
        if enable_warning:
            warnings.warn("Can not plot an unbounded polytope!", UserWarning)
        return plt.gca(), None, None
    elif not autoscale_enable:
        raise NotImplementedError("autoscale_enable needs to be currently enabled for 3D plotting")

    # Create an axes if not yet defined
    if not ax:
        plt.figure()
        ax = plt.axes(projection="3d")

    patch_args, vertex_args = sanitize_patch_args_and_vertex_args(patch_args, vertex_args, 3)

    # If in V-Rep and not self.in_H_rep, then first filter the vertices and then perform facet enumeration
    if self.in_V_rep and not self.in_H_rep:
        V = prune_and_round_vertices(self.V, decimal_precision=decimal_precision)
        cdd_polytope = get_cdd_polyhedron_from_V(V, prune_V=prune_V)
        try:
            set_attributes_minimal_Ab_Aebe_from_cdd_polyhedron(self, cdd_polytope)
        except (ValueError, IndexError) as err:
            raise ValueError("Ran into issues during facet enumeration!") from err

    # Use CDD to compute list_incidences and copy_input_incidence makes sense when generated from H-Rep
    if self.n_equalities > 0:
        cdd_polytope = get_cdd_polyhedron_from_Ab_Aebe(self.A, self.b, self.Ae, self.be)
    else:
        cdd_polytope = get_cdd_polyhedron_from_Ab_Aebe(self.A, self.b)
    cdd_polytope_A = -np.array(cdd.copy_inequalities(cdd_polytope).array)[:, 1:]
    list_incidences = [list(x) for x in cdd.copy_input_incidence(cdd_polytope)]
    # Overwrite all vertices as a n_vertices times n matrix
    try:
        tV_cdd_matrix = cdd.copy_generators(cdd_polytope)
        set_attributes_V_from_cdd(self, tV_cdd_matrix)
    except ValueError as err:
        raise ValueError("Ran into issues during vertex enumeration!") from err
    cdd_polytope_vertices = self.V

    # Plot each face
    h_first_patch = Line3DCollection([], **patch_args)  # pyright: ignore[reportArgumentType]
    is_first_patch = True
    h_vert = None
    for face_vertex_indices, halfspace_direction in zip(list_incidences, cdd_polytope_A):
        if len(face_vertex_indices) >= 3:
            # Define relative vectors
            poly_face_V = cdd_polytope_vertices[face_vertex_indices, :]
            poly_face_V_centroid = np.mean(poly_face_V, axis=0)
            relative_vectors = poly_face_V - poly_face_V_centroid
            # Rotate the relative vectors so that they all lie in XY plane
            unit_halfspace_direction = halfspace_direction / np.linalg.norm(halfspace_direction)
            rot_axis_vector = np.cross(unit_halfspace_direction, [0, 0, 1])
            rot_angle = np.arccos(np.dot(unit_halfspace_direction, [0, 0, 1]))
            R_matrix = cast(np.ndarray, Rotation.from_rotvec(rot_angle * rot_axis_vector).as_matrix())
            rotated_relative_vectors = (R_matrix @ relative_vectors.T).T
            # Compute angle subtended by relative vectors (after rotation) to a mean vec (centroid)
            mean_rotated_relative_vec = np.mean(rotated_relative_vectors, axis=0)
            angle_between_relative_vectors = [
                np.arctan2(
                    rotated_relative_vec[1] - mean_rotated_relative_vec[1],
                    rotated_relative_vec[0] - mean_rotated_relative_vec[0],
                )
                for rotated_relative_vec in rotated_relative_vectors
            ]
            # Obtain the order of angles
            order = np.argsort(angle_between_relative_vectors)
            poly_face_V = np.vstack((poly_face_V[order, :], poly_face_V[order[0], :]))
            # Plot faces
            poly_face_vertices = [[tuple(v) for v in poly_face_V]]
            if "label" in patch_args:
                # Use Poly3DCollection to get a patch in the legend
                if patch_args["facecolor"] is None:
                    # However, facecolor=None doesn't work if unfilled polytope is desired, so set alpha=0
                    patch_args["alpha"] = 0
                    h = Poly3DCollection(poly_face_vertices, **patch_args)  # pyright: ignore[reportArgumentType]
                    # Remove label from future patches/lines and store h with label handle for future reference
                    patch_args.pop("alpha")
                else:
                    h = Poly3DCollection(poly_face_vertices, **patch_args)  # pyright: ignore[reportArgumentType]
                patch_args.pop("label")
            else:
                h = Line3DCollection(poly_face_vertices, **patch_args)  # pyright: ignore[reportArgumentType]
            if is_first_patch:
                h_first_patch = h
                is_first_patch = False
            ax.add_collection3d(h)  # pyright: ignore[reportAttributeAccessIssue]
            # Plot vertices
            h_vert = ax.scatter(poly_face_V[:, 0], poly_face_V[:, 1], poly_face_V[:, 2], **vertex_args)
    if h_vert is None:
        warnings.warn("No plot generated since all faces had at most 2 vertices!")
    return ax, h_first_patch, h_vert


def sanitize_patch_args_and_vertex_args(
    patch_args: Optional[dict[str, Any]], vertex_args: Optional[dict[str, Any]], dimension: int
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Sanitize patch_args and vertex_args

    Args:
        patch_args (dict): Arguments to pass for plotting faces and edges.
        vertex_args (dict): Arguments to pass for plotting vertices.
        dimension (int): Dimension of the polytope.
    """
    if dimension == 2:
        # Main goal: facecolor=None in 2D plotting does not produce frame plot | In that event, set fill to False
        if patch_args is None:
            patch_args = {}
        if "fill" in patch_args and not patch_args["fill"]:
            if "facecolor" in patch_args:
                if patch_args["facecolor"] is not None:
                    raise ValueError("Can not have facecolor is not None and fill=False together!")
                patch_args.pop("facecolor")
        if "facecolor" in patch_args and patch_args["facecolor"] is None:
            patch_args.pop("facecolor")
            if "fill" in patch_args and patch_args["fill"]:
                raise ValueError("Can not have facecolor is None and fill=True together!")
            patch_args["fill"] = False
        patch_args = dict(DEFAULT_PATCH_ARGS_2D, **patch_args)  # Override default values
    else:  # dimension is 3
        # Main goal: fill is not supported in 3D plotting | labeling requires Poly3DCollection |
        # Unfilled plot Poly3DCollection requires alpha=0 while Line3DCollection requires facecolor=None
        # "facecolor"=None is equivalent to fill=False
        if patch_args is None:
            # By default, prefer line plots in 3D
            patch_args = {"facecolor": None}
        if "fill" in patch_args:
            if not patch_args["fill"]:
                if "facecolor" in patch_args and patch_args["facecolor"] is not None:
                    raise ValueError("Can not have facecolor is not None and fill=False together!")
                else:
                    patch_args["facecolor"] = None
            else:
                if "facecolor" in patch_args and patch_args["facecolor"] is None:
                    raise ValueError("Can not have facecolor is None and fill=True together!")
            patch_args.pop("fill")
        patch_args = dict(DEFAULT_PATCH_ARGS_3D, **patch_args)  # Set other keys with default values
        if "label" in patch_args and "alpha" in patch_args and patch_args["facecolor"] is None:
            # We will use alpha = 0 so the user can not expect any other alpha to work!
            raise ValueError("Can not set 'facecolor' to None (or unspecified) AND set 'alpha' if labeling is desired!")
    if vertex_args:
        if "visible" not in vertex_args:
            vertex_args["visible"] = True
        vertex_args = dict(DEFAULT_VERTEX_ARGS, **vertex_args)  # Override default values
    else:
        vertex_args = DEFAULT_VERTEX_ARGS
    return patch_args, vertex_args
