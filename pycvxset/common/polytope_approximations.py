# Copyright (C) 2020-2025 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

# Code purpose:  Define the methods to compute polytopic approximations of sets

from pycvxset.common import is_ellipsoid, spread_points_on_a_unit_sphere
from pycvxset.Polytope import Polytope


def polytopic_outer_approximation(self, direction_vectors=None, n_halfspaces=None, verbose=False):
    r"""Compute a polytopic outer-approximation of a given set via ray shooting.

    Args:
        direction_vectors (array_like, optional): Directions to use when performing ray shooting. Matrix (N times
            self.dim) for some :math:`N \geq 1`. Defaults to None.
        n_halfspaces (int, optional): Number of halfspaces to be used for the inner-approximation. n_vertices is
            overridden whenever direction_vectors are provided.  Defaults to None.
        verbose (bool, optional): If true, :meth:`pycvxset.common.spread_points_on_a_unit_sphere` is passed with
            verbose. Defaults to False.

    Returns:
        Polytope: Polytopic outer-approximation in H-Rep of a given set with n_halfspaces no smaller than user-provided
        n_vertices.

    Notes:
        We compute the polytope using :meth:`support` evaluated along the direction vectors computed by
        :meth:`pycvxset.common.spread_points_on_a_unit_sphere`. When direction_vectors is None and n_halfspaces is None,
        we select :math:`\text{n\_halfspaces} = 2 \text{self.dim} + 2^\text{self.dim}
        \text{SPOAUS\_DIRECTIONS\_PER\_QUADRANT}` (as defined in :py:mod:`pycvxset.common.constants`).  [BV04]_
    """
    if self.is_empty:
        return Polytope(dim=self.dim)
    else:
        if direction_vectors is None:
            direction_vectors = spread_points_on_a_unit_sphere(self.dim, n_halfspaces, verbose=verbose)[0]
        ray_shooting_center = self.interior_point()
        shifted_set = self - ray_shooting_center
        P = Polytope(A=direction_vectors, b=shifted_set.support(direction_vectors)[0])
        if is_ellipsoid(shifted_set) and not shifted_set.is_full_dimensional:
            # We can do better than just outer-approximation via support, since we know the affine hull
            Ae, be = shifted_set.affine_hull()
            P = P.intersection_with_affine_set(Ae, be)
        return P + ray_shooting_center


def polytopic_inner_approximation(self, direction_vectors=None, n_vertices=None, verbose=False):
    r"""Compute a polytopic inner-approximation of a given set via ray shooting.

    Args:
        direction_vectors (array_like, optional): Directions to use when performing ray shooting. Matrix (N times
            self.dim) for some :math:`N \geq 1`. Defaults to None.
        n_vertices (int, optional): Number of vertices to be used for the inner-approximation. n_vertices is
            overridden whenever direction_vectors are provided.  Defaults to None.
        verbose (bool, optional): If true, :meth:`pycvxset.common.spread_points_on_a_unit_sphere` is passed with
            verbose. Defaults to False.

    Returns:
        Polytope: Polytopic inner-approximation in V-Rep of a given set with n_vertices no smaller than user-provided
        n_vertices.

    Notes:
        We compute the polytope using :meth:`extreme` evaluated along the direction vectors computed by
        :meth:`pycvxset.common.spread_points_on_a_unit_sphere`. When direction_vectors is None and n_vertices is None,
        we select :math:`\text{n\_vertices} = 2 \text{self.dim} + 2^\text{self.dim}
        \text{SPOAUS\_DIRECTIONS\_PER\_QUADRANT}` (as defined in :meth:`pycvxset.common.constants`). [BV04]_
    """
    if self.is_empty:
        return Polytope(dim=self.dim)
    else:
        if direction_vectors is None:
            direction_vectors = spread_points_on_a_unit_sphere(self.dim, n_vertices, verbose=verbose)[0]

        # Shift the set and then do ray-shooting
        ray_shooting_center = self.interior_point()
        shifted_set = self - ray_shooting_center
        return Polytope(V=shifted_set.extreme(direction_vectors)) + ray_shooting_center
