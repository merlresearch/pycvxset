# Copyright (C) 2020-2026 Mitsubishi Electric Research Laboratories (MERL)
# Copyright (c) 2019 Tor Aksel N. Heirung
#
# SPDX-License-Identifier: AGPL-3.0-or-later
# SPDX-License-Identifier: MIT

# Code purpose:  Define the methods for vertex-halfspace enumeration for the Polytope class
# Coverage: This file has 11 missing statements + 2 excluded statements + 4 partial branches.

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional, Sequence, cast

import cdd  # pycddlib -- for vertex enumeration from H-representation
import numpy as np
from scipy.spatial import ConvexHull, HalfspaceIntersection

from pycvxset.common import prune_and_round_vertices, sanitize_Ab
from pycvxset.common.constants import (
    DEFAULT_QHULL_OPTIONS,
    PLOTTING_DECIMAL_PRECISION_CDD,
    PREFER_QHULL_OVER_CDD,
    PYCVXSET_ZERO,
)

if TYPE_CHECKING:
    from pycvxset.Polytope import Polytope


def get_cdd_polyhedron_from_V(
    V: Sequence[Sequence[float]] | np.ndarray,
    prune_V: bool = False,
) -> cdd.Polyhedron:
    """Get CDD polyhedron in generator form from given V

    Args:
        V (Sequence[Sequence[float]] | np.ndarray): self.n_vertices times n matrix
        prune_V (bool): Whether to prune and round vertices. See PLOTTING_DECIMAL_PRECISION_CDD in constants.py for the
            default decimals places retained.

    Returns:
        cdd.Polyhedron: CDD Polyhedron
    """
    V_arr: np.ndarray = np.asarray(V, dtype=float)
    if prune_V:
        V_arr = prune_and_round_vertices(V_arr, decimal_precision=PLOTTING_DECIMAL_PRECISION_CDD)
    n_vertices = V_arr.shape[0]
    # t is 1 to indicate that all are vertices
    tV_list = np.hstack((np.ones((n_vertices, 1)), V_arr)).tolist()
    tV_cdd = cdd.matrix_from_array(tV_list, rep_type=cdd.RepType.GENERATOR)
    try:
        return cdd.polyhedron_from_matrix(tV_cdd)
    except RuntimeError as err:
        raise ValueError(
            "Computation of CDD polyhedron failed due to numerical inconsistency in vertex list! Try setting prune_V "
            "to True to prune and round the vertices before computing the CDD polyhedron."
        ) from err


def get_cdd_polyhedron_from_Ab_Aebe(
    A: Optional[Sequence[Sequence[float]] | np.ndarray | None] = None,
    b: Optional[Sequence[float] | np.ndarray] = None,
    Ae: Optional[Sequence[Sequence[float]] | np.ndarray] = None,
    be: Optional[Sequence[float] | np.ndarray] = None,
) -> cdd.Polyhedron:
    """Get CDD polyhedron in inequality form from given (A, b, Ae, be)

    Args:
        A (Sequence[Sequence[float]] | np.ndarray | None, optional): Inequality coefficient vectors (H-Rep). The
            vectors are stacked vertically.
        b (Sequence[float] | np.ndarray | None, optional): Inequality constants (H-Rep). The constants are expected to
            be in a 1D numpy array.
        Ae (Sequence[Sequence[float]] | np.ndarray | None, optional): Equality coefficient vectors (H-Rep). The vectors
            are stacked vertically.  Defaults to None.
        be (Sequence[float] | np.ndarray | None, optional): Equality constants (H-Rep). The constants are expected to be
            in a 1D numpy array. Defaults to None.

    Returns:
        cdd.Polyhedron: CDD Polyhedron
    """
    A, b = sanitize_Ab(A, b)
    if A is None or b is None:
        H_He_cdd = None
    else:
        A_arr: np.ndarray = np.asarray(A, dtype=float)
        b_arr: np.ndarray = np.asarray(b, dtype=float)
        b_mA = np.hstack((np.array([b_arr]).T, -A_arr))
        H_He_cdd = cdd.matrix_from_array(b_mA, rep_type=cdd.RepType.INEQUALITY)
    if (Ae is None and be is not None) or (Ae is not None and be is None):
        raise ValueError("Expected (Ae, be) to be either both provided or both None.")
    if Ae is not None and np.asarray(Ae).size > 0:
        # Add all equalities to obtain H_He_cdd
        Ae, be = sanitize_Ab(Ae, be)
        Ae_arr: np.ndarray = np.asarray(Ae, dtype=float)
        be_arr: np.ndarray = np.asarray(be, dtype=float)
        be_mAe = np.hstack((np.array([be_arr]).T, -Ae_arr))
        He_cdd = cdd.matrix_from_array(be_mAe, lin_set=set(range(len(be_arr))), rep_type=cdd.RepType.INEQUALITY)
        if H_He_cdd is None:
            H_He_cdd = He_cdd
        else:
            cdd.matrix_append_to(H_He_cdd, He_cdd)
    H_He_cdd = cast(cdd.Matrix, H_He_cdd)
    return cdd.polyhedron_from_matrix(H_He_cdd)


def determine_H_rep(
    self: Polytope,
    prefer_qhull_over_cdd: bool = PREFER_QHULL_OVER_CDD,
    prune_V: bool = False,
) -> None:
    """Determine the halfspace representation from a given vertex representation of the polytope.

    Args:
        prefer_qhull_over_cdd (bool, optional): When True, determine_H_rep uses qhull when possible. Otherwise, we use
            cdd. Default is True.
        prune_V (bool, optional): When True, prune vertices before halfspace enumeration. Defaults to False.

    Raises:
        ValueError: When H-rep computation fails OR Polytope is not bounded!

    Notes:
        - When the set is empty, we define an empty polytope.
        - Otherwise, we use cdd for the halfspace enumeration.
        - The computed vertex representation need not be minimal.
        - We do not check for bounded polytope since we have V-Rep.
    """
    if self.in_H_rep:
        # Do nothing! Already have a h-rep!
        pass
    elif self.is_empty:
        self._set_polytope_to_empty(self.dim)
    else:
        try:
            old_info = (self.is_full_dimensional, self.is_empty, self.is_bounded)
            if prefer_qhull_over_cdd and self.is_full_dimensional and self.n_vertices >= self.dim + 1:
                # qhull handles this case. Requires full_dimensional and at least self.dim + 1 vertices
                hull = ConvexHull(self.V, qhull_options=DEFAULT_QHULL_OPTIONS)
                # hull.equations has shape (m, d+1): [n0, n1, ..., nd-1, offset]
                # Interior satisfies n·x + offset <= 0  =>  A = n, b = -offset
                halfspaces = cast(np.ndarray, hull.equations)
                A, b = halfspaces[:, :-1], -halfspaces[:, -1]
                self._set_attributes_from_Ab_Aebe(A, b, erase_V_rep=False)
            else:
                # cdd handles this case.
                cdd_polyhedron = get_cdd_polyhedron_from_V(self.V, prune_V=prune_V)
                set_attributes_minimal_Ab_Aebe_from_cdd_polyhedron(self, cdd_polyhedron)
        except ValueError as err:
            # Error can come from halfspace enumeration of an unbounded set
            raise ValueError("Computation of H-rep failed!") from err
        raise_error_if_property_changed(self, old_info)


def determine_V_rep(self: Polytope, prefer_qhull_over_cdd: bool = PREFER_QHULL_OVER_CDD) -> None:
    r"""Determine the vertex representation from a given halfspace representation of the polytope.

    Args:
        prefer_qhull_over_cdd (bool, optional): When True, determine_V_rep uses qhull when possible. Otherwise, we use
            cdd. Default is True.

    Raises:
        ValueError: Vertex enumeration yielded rays! Possibly due to numerical issues or unbounded polytope!
        ValueError: Polytope is not bounded!

    Notes:
        We use cdd for the vertex enumeration. cdd uses the halfspace representation :math:`[b, -A]` where :math:`b - Ax
        \geq 0  \Leftrightarrow Ax \leq b`.

        For a polyhedron described as

        .. math ::
            \mathcal{P} = \text{conv}(v_1, ..., v_n) + \text{nonneg}(r_1, ..., r_s),

        the V-representation matrix in cdd is [t V] where t is the column vector with n ones followed by s zeroes, and V
        is the stacked matrix of n vertex row vectors on top of s ray row vectors. `pycvxset` uses only bounded
        polyhedron, so we should never observe rays.
    """
    # Vertex enumeration from halfspace representation using cdd.
    if self.in_V_rep:
        # Do nothing! Already have a V_Rep!
        pass
    elif self.is_empty:
        self._set_polytope_to_empty(self.dim)
    else:
        try:
            old_info = (self.is_full_dimensional, self.is_empty, self.is_bounded)
            if not self.is_bounded:
                raise ValueError("Polytope is not bounded!")
            elif prefer_qhull_over_cdd and self.is_full_dimensional:
                # qhull handles this case. self.n_equalities is zero since full-dimensional.
                halfspaces = np.hstack([self.A, -self.b[:, None]])
                x0 = self.chebyshev_centering()[0]
                halfspace_intersection = HalfspaceIntersection(halfspaces, x0, qhull_options=DEFAULT_QHULL_OPTIONS)
                V = np.array(halfspace_intersection.intersections)
                self._set_attributes_from_V(V, erase_H_rep=False)
            else:
                # cdd handles this case.
                cdd_polyhedron = get_cdd_polyhedron_from_Ab_Aebe(self.A, self.b, self.Ae, self.be)
                tV_cdd_matrix = cdd.copy_generators(cdd_polyhedron)  # Perform vertex enumeration
                set_attributes_V_from_cdd(self, tV_cdd_matrix)
        except ValueError as err:
            # Error can come from vertex enumeration of an unbounded set
            raise ValueError("Computation of V-rep failed!") from err
        raise_error_if_property_changed(self, old_info)


def minimize_H_rep(self: Polytope) -> None:
    r"""Remove any redundant inequalities from the halfspace representation of the polytope using cdd.

    Raises:
        ValueError: When minimal H-Rep computation fails OR polytope is not bounded!
    """
    if self.is_empty:
        self._set_polytope_to_empty(self.A.shape[1])
    else:
        try:
            old_info = (self.is_full_dimensional, self.is_empty, self.is_bounded)
            if not self.is_bounded:
                raise ValueError("Polytope is not bounded!")
            cdd_polyhedron = get_cdd_polyhedron_from_Ab_Aebe(self.A, self.b, self.Ae, self.be)
            set_attributes_minimal_Ab_Aebe_from_cdd_polyhedron(self, cdd_polyhedron)
        except ValueError as err:
            # Error can come from halfspace enumeration of an unbounded set
            raise ValueError("Computation of minimal H-rep failed!") from err
        raise_error_if_property_changed(self, old_info)


def minimize_V_rep(
    self: Polytope,
    prefer_qhull_over_cdd: bool = PREFER_QHULL_OVER_CDD,
    prune_V: bool = False,
) -> None:
    """Remove any redundant vertices from the vertex representation of the polytope.

    Args:
        prefer_qhull_over_cdd (bool, optional): When True, minimize_V_rep uses qhull when possible. Otherwise, we use
            cdd. Default is True.
        prune_V (bool, optional): When True, prune vertices before minimizing. Defaults to False.

    Raises:
        ValueError: When minimal V-Rep computation fails!


    Notes:
        Use cdd or qhull for the reduction of vertices.
    """
    if self.is_empty:
        self._set_polytope_to_empty(self.dim)
    else:
        old_info = (self.is_full_dimensional, self.is_empty, self.is_bounded)
        if self.n_vertices == 1:
            pass
        elif self.dim == 1:
            V_minimal = np.vstack((np.min(self.V, keepdims=True), np.max(self.V, keepdims=True)))
            if np.diff(V_minimal, axis=0) <= PYCVXSET_ZERO:
                # Extrema are same. So pick only the top row.
                self._set_attributes_from_V(V_minimal[:1, :], erase_H_rep=False)
            else:
                self._set_attributes_from_V(V_minimal, erase_H_rep=False)
        elif prefer_qhull_over_cdd and self.is_full_dimensional:
            # qhull handles this case. It calls determine_V_rep if necessary via self.V getter
            # Indices of the unique vertices forming the convex hull:
            i_V_minimal = cast(np.ndarray, ConvexHull(self.V).vertices)
            V_minimal = self.V[i_V_minimal, :]
            self._set_attributes_from_V(V_minimal, erase_H_rep=False)
        else:
            # cdd handles this case. It calls determine_V_rep if necessary via self.V getter
            try:
                cdd_polyhedron = get_cdd_polyhedron_from_V(self.V, prune_V=prune_V)
                tV_cdd_matrix = cdd.copy_generators(cdd_polyhedron)
                cdd.matrix_canonicalize(tV_cdd_matrix)  # Minimize redundant vertices
                set_attributes_V_from_cdd(self, tV_cdd_matrix)  # Set attributes from V
            except ValueError as err:
                # Error can come from vertex enumeration of an unbounded set
                raise ValueError("Computation of minimal V-rep failed!") from err
        raise_error_if_property_changed(self, old_info)


def raise_error_if_property_changed(self: Polytope, old_info: Sequence[Any]) -> None:
    """Raise error if property changed compared to the provided (old) property information

    Args
        old_info (tuple): 3-dimensional tuple of the form (self.is_full_dimensional, self.is_empty, self.is_bounded)

    Raises:
        ValueError: If any of the property changed.
    """
    new_info = (self.is_full_dimensional, self.is_empty, self.is_bounded)
    error_message = ""
    for old, new, text in zip(old_info, new_info, ["Full-dimensionality: ", "Empty: ", "Bounded: "]):
        if old != new:
            error_message += f"{text:s} {str(old):s} -> {str(new):s}"
    if error_message != "":
        error_message += "Unexpected change in polytope properties! This may be due to numerical issues.\n"
        raise ValueError(error_message)


def set_attributes_minimal_Ab_Aebe_from_cdd_polyhedron(self: Polytope, cdd_polyhedron: cdd.Polyhedron) -> None:
    """Get (A, b, Ae, be) from CDD polyhedron in inequality form

    Args:
        cdd.Polyhedron: CDD Polyhedron

    Returns:
        (np.ndarray, np.ndarray, np.ndarray, np.ndarray): (A, b, Ae, be). (Ae, be) is (None, None) if no equality
        constraints
    """
    H_cdd_matrix = cdd.copy_inequalities(cdd_polyhedron)
    cdd.matrix_canonicalize(H_cdd_matrix)  # Identify linear equalities if any
    H_cdd_values = np.array(H_cdd_matrix.array)
    Ae, be = None, None
    if H_cdd_values.size == 0:
        raise ValueError("Did not expect facet list to be empty after minimization!")
    elif len(H_cdd_matrix.lin_set):
        # Extract equalities
        He_cdd = np.array([v for index, v in enumerate(H_cdd_values) if index in H_cdd_matrix.lin_set])
        be, Ae = He_cdd[:, 0], -He_cdd[:, 1:]
        if len(H_cdd_matrix.lin_set) < H_cdd_values.shape[0]:
            # Extract inequalities
            H_cdd = np.array([v for index, v in enumerate(H_cdd_values) if index not in H_cdd_matrix.lin_set])
            b, A = H_cdd[:, 0], -H_cdd[:, 1:]
        else:
            A = np.empty((0, Ae.shape[1]))
            b = np.empty((0,))
    else:
        # Extract inequalities
        H_cdd = np.array(H_cdd_matrix.array)
        b, A = H_cdd[:, 0], -H_cdd[:, 1:]
    self._set_attributes_from_Ab_Aebe(A, b, Ae=Ae, be=be, erase_V_rep=False)


def set_attributes_V_from_cdd(self: Polytope, tV_cdd_matrix: cdd.Matrix) -> None:
    tV = np.array(tV_cdd_matrix.array)
    if (tV[:, 0] == 0).any():
        raise ValueError("Vertex enumeration yielded rays! Possibly due to numerical issues or unbounded polytope!")
    else:
        V = tV[:, 1:]
        self._set_attributes_from_V(V, erase_H_rep=False)


def valid_rows_with_not_all_zeros_in_A_and_no_inf_in_b(
    A: Sequence[Sequence[float]] | np.ndarray, b: Sequence[float] | np.ndarray
) -> np.ndarray:
    """
    valid_rows_with_not_all_zeros_in_A_and_no_inf_in_b collects row indices of (A, b) such that for b is not infinity
    and the row of A is not a zero vector.

    Args:
        A (numpy.ndarray): 2D numpy array
        b (numpy.ndarray): 1D numpy array

    Returns:
        bool numpy.ndarray: Valid rows in the H-representation after removing invalid rows
    """
    valid_rows_of_b = b < np.inf
    valid_rows_of_A = (np.abs(A) > PYCVXSET_ZERO).any(axis=1)
    valid_rows = np.bitwise_and(valid_rows_of_A, valid_rows_of_b)
    return valid_rows
