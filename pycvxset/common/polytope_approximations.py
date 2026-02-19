# Copyright (C) 2020-2026 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

# Code purpose:  Define the methods to compute polytopic approximations of sets

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Sequence, cast

import numpy as np

from pycvxset.common import is_ellipsoid, is_polytope, make_aspect_ratio_equal
from pycvxset.Polytope import Polytope

if TYPE_CHECKING:
    from pycvxset.ConstrainedZonotope import ConstrainedZonotope
    from pycvxset.Ellipsoid import Ellipsoid


def polytopic_outer_approximation(
    cvx_set: ConstrainedZonotope | Ellipsoid | Polytope,
    direction_vectors: Optional[Sequence[Sequence[float]] | np.ndarray] = None,
    n_halfspaces: Optional[int] = None,
    verbose: bool = False,
    enable_warning: bool = True,
) -> Polytope:
    r"""Compute a polytopic outer-approximation of a given set via ray shooting.

    Args:
        cvx_set (ConstrainedZonotope | Ellipsoid | Polytope): Set to be approximated,
        direction_vectors (Sequence[Sequence[float]] | np.ndarray, optional): Directions to use when performing ray
            shooting. Matrix (N times self.dim) for some :math:`N \geq 1`. Defaults to None.
        n_halfspaces (int, optional): Number of halfspaces to be used for the inner-approximation. n_vertices is
            overridden whenever direction_vectors are provided.  Defaults to None.
        verbose (bool, optional): If true, :meth:`pycvxset.common.spread_points_on_a_unit_sphere` is passed with
            verbose. Defaults to False.
        enable_warning (bool, optional): Enables the UserWarning. May be turned off if expected. Defaults to True.

    Returns:
        Polytope: Polytopic outer-approximation in H-Rep of a given set with n_halfspaces no smaller than user-provided
        n_vertices.

    Notes:
        We compute the polytope using :meth:`support` evaluated along the direction vectors computed by
        :meth:`pycvxset.common.spread_points_on_a_unit_sphere`. When direction_vectors is None and n_halfspaces is None,
        we select :math:`\text{n\_halfspaces} = 2 \text{self.dim} + 2^\text{self.dim}
        \text{SPOAUS\_DIRECTIONS\_PER\_QUADRANT}` (as defined in :py:mod:`pycvxset.common.constants`).  [BV04]_
    """
    # Doing inline import to reduce import-time overhead
    from pycvxset.common import spread_points_on_a_unit_sphere

    if cvx_set.is_empty:
        return Polytope(dim=cvx_set.dim)
    else:
        if direction_vectors is None:
            direction_vectors = spread_points_on_a_unit_sphere(
                cvx_set.dim, n_halfspaces, verbose=verbose, enable_warning=enable_warning
            )[0]
        if is_ellipsoid(cvx_set) or is_polytope(cvx_set):
            ray_shooting_center = cvx_set.interior_point()
        else:
            ray_shooting_center = cast("ConstrainedZonotope", cvx_set).interior_point(enable_warning=enable_warning)
        shifted_set = cvx_set - ray_shooting_center
        P = Polytope(A=direction_vectors, b=shifted_set.support(direction_vectors)[0])
        if is_ellipsoid(shifted_set) and not shifted_set.is_full_dimensional:
            # We can do better than just outer-approximation via support, since we know the affine hull
            Ae, be = shifted_set.affine_hull()
            P = P.intersection_with_affine_set(Ae, be)
        return P + ray_shooting_center


def polytopic_inner_approximation(
    cvx_set: ConstrainedZonotope | Ellipsoid | Polytope,
    direction_vectors: Optional[Sequence[Sequence[float]] | np.ndarray] = None,
    n_vertices: Optional[int] = None,
    verbose: bool = False,
    enable_warning: bool = True,
) -> Polytope:
    r"""Compute a polytopic inner-approximation of a given set via ray shooting.

    Args:
        cvx_set (ConstrainedZonotope | Ellipsoid | Polytope): Set to be approximated,
        direction_vectors (Sequence[Sequence[float]] | np.ndarray, optional): Directions to use when performing ray
            shooting. Matrix (N times self.dim) for some :math:`N \geq 1`. Defaults to None.
        n_vertices (int, optional): Number of vertices to be used for the inner-approximation. n_vertices is
            overridden whenever direction_vectors are provided.  Defaults to None.
        verbose (bool, optional): If true, :meth:`pycvxset.common.spread_points_on_a_unit_sphere` is passed with
            verbose. Defaults to False.
        enable_warning (bool, optional): Enables the UserWarning. May be turned off if expected. Defaults to True.

    Returns:
        Polytope: Polytopic inner-approximation in V-Rep of a given set with n_vertices no smaller than user-provided
        n_vertices.

    Notes:
        We compute the polytope using :meth:`extreme` evaluated along the direction vectors computed by
        :meth:`pycvxset.common.spread_points_on_a_unit_sphere`. When direction_vectors is None and n_vertices is None,
        we select :math:`\text{n\_vertices} = 2 \text{self.dim} + 2^\text{self.dim}
        \text{SPOAUS\_DIRECTIONS\_PER\_QUADRANT}` (as defined in :meth:`pycvxset.common.constants`). [BV04]_
        The function also uses :meth:`pycvxset.common.make_aspect_ratio_equal` to account for possibly non-symmetric
        sets.
    """
    # Doing inline import to reduce import-time overhead
    from pycvxset.common import spread_points_on_a_unit_sphere

    if cvx_set.is_empty:
        return Polytope(dim=cvx_set.dim)
    else:
        if direction_vectors is None:
            direction_vectors = spread_points_on_a_unit_sphere(
                cvx_set.dim, n_vertices, verbose=verbose, enable_warning=enable_warning
            )[0]

        # Shift the set and then do ray-shooting
        if is_ellipsoid(cvx_set) or is_polytope(cvx_set):
            ray_shooting_center = cvx_set.interior_point()
        else:
            ray_shooting_center = cast("ConstrainedZonotope", cvx_set).interior_point(enable_warning=enable_warning)
        if cvx_set.is_full_dimensional:
            (
                shifted_set_with_equal_aspect_ratio,
                shift_to_under_the_affine_transform,
                scaling_matrix_to_undo_the_affine_transform,
            ) = make_aspect_ratio_equal(cvx_set - ray_shooting_center)
            polytope_shifted_set_with_equal_aspect_ratio = Polytope(
                V=shifted_set_with_equal_aspect_ratio.extreme(direction_vectors)
            )
            return (
                (scaling_matrix_to_undo_the_affine_transform @ polytope_shifted_set_with_equal_aspect_ratio)
                + shift_to_under_the_affine_transform
                + ray_shooting_center
            )
        else:
            shifted_set = cvx_set - ray_shooting_center
            return Polytope(V=shifted_set.extreme(direction_vectors)) + ray_shooting_center
