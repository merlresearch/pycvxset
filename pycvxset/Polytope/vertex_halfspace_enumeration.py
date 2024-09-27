# Copyright (C) 2020-2024 Mitsubishi Electric Research Laboratories (MERL)
# Copyright (c) 2019 Tor Aksel N. Heirung
#
# SPDX-License-Identifier: AGPL-3.0-or-later
# SPDX-License-Identifier: MIT

# Code purpose:  Define the methods for vertex-halfspace enumeration for the Polytope class
# Coverage: This file has 9 untested statements + 1 partial to handle unexpected errors from pycddlib

import cdd  # pycddlib -- for vertex enumeration from H-representation
import numpy as np
from scipy.spatial import ConvexHull  # for finding A, b from V-representation

from pycvxset.common.constants import PYCVXSET_ZERO


def get_cdd_polyhedron_from_V(V):
    """Get CDD polyhedron in generator form from given V

    Args:
        V (array_like): self.n_vertices times n matrix

    Returns:
        cdd.Polyhedron: CDD Polyhedron
    """
    n_vertices = V.shape[0]
    tV_list = np.hstack((np.ones((n_vertices, 1)), V)).tolist()
    tV_cdd = cdd.Matrix(tV_list, number_type="float")
    tV_cdd.rep_type = cdd.RepType.GENERATOR  # specifies that this is V-rep
    try:
        return cdd.Polyhedron(tV_cdd)
    except RuntimeError as err:
        raise ValueError("Computation of CDD polyhedron failed due to numerical inconsistency in vertex list") from err


def get_cdd_polyhedron_from_Ab_Aebe(A, b, Ae=None, be=None):
    """Get CDD polyhedron in inequality form from given (A, b, Ae, be)

    Args:
        A (array_like): Inequality coefficient matrix A
        b (array_like): Inequality coefficient vector b
        Ae (array_like, optional): Equality coefficient matrix A. Defaults to None.
        be (array_like, optional): Equality coefficient vector b. Defaults to None.

    Returns:
        cdd.Polyhedron: CDD Polyhedron
    """
    b_mA = np.hstack((np.array([b]).T, -A))
    # Add all inequalities to H_He_cdd
    H_He_cdd = cdd.Matrix(b_mA, linear=False, number_type="float")
    H_He_cdd.rep_type = cdd.RepType.INEQUALITY  # specifies that this is H-rep
    if Ae is not None and Ae.size > 0:
        # Add all equalities to H_He_cdd
        be_mAe = np.hstack((np.array([be]).T, -Ae))
        H_He_cdd.extend(be_mAe, linear=True)
    return cdd.Polyhedron(H_He_cdd)


def determine_H_rep(self):
    """Determine the halfspace representation from a given vertex representation of the polytope.

    Raises:
        ValueError: When H-rep computation fails

    Notes:
        - When the set is empty, we define an empty polytope.
        - Otherwise, we use cdd for the halfspace enumeration.
        - The computed vertex representation need not be minimal.
    """
    if self.in_H_rep:
        # Do nothing! Already have a h-rep!
        pass
    elif self.is_empty:
        self._set_polytope_to_empty(self.dim)
    else:
        try:
            V_cddP = get_cdd_polyhedron_from_V(self.V)
            set_attributes_minimal_Ab_Aebe_from_cdd_polyhedron(self, V_cddP)
        except ValueError as err:
            # Error can come from halfspace enumeration of an unbounded set
            raise ValueError("Computation of H-rep failed!") from err


def determine_V_rep(self):  # also determines rays R (not implemented)
    r"""Determine the vertex representation from a given halfspace representation of the polytope.

    Raises:
        ValueError: Vertex enumeration yields rays, which indicates that the polytope is unbounded. Numerical issues may
            also be a culprit.

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
            cdd_polyhedron = get_cdd_polyhedron_from_Ab_Aebe(self.A, self.b, self.Ae, self.be)
            tV_cdd_matrix = cdd_polyhedron.get_generators()[:]  # Perform vertex enumeration
            set_attributes_V_from_cdd(self, tV_cdd_matrix)
        except ValueError as err:
            # Error can come from vertex enumeration of an unbounded set
            raise ValueError("Computation of V-rep failed!") from err


def minimize_H_rep(self):
    r"""Remove any redundant inequalities from the halfspace representation of the polytope using cdd.

    Raises:
        ValueError: When minimal H-Rep computation fails!
    """
    if self.is_empty:
        self._set_polytope_to_empty(self.A.shape[1])
    else:
        try:
            cdd_polyhedron = get_cdd_polyhedron_from_Ab_Aebe(self.A, self.b, self.Ae, self.be)
            set_attributes_minimal_Ab_Aebe_from_cdd_polyhedron(self, cdd_polyhedron)
        except ValueError as err:
            # Error can come from halfspace enumeration of an unbounded set
            raise ValueError("Computation of minimal H-rep failed!") from err


def minimize_V_rep(self, prefer_qhull_over_cdd=True):
    """Remove any redundant vertices from the vertex representation of the polytope.

    Args:
        prefer_qhull_over_cdd (bool, optional): When True, minimize_V_rep uses qhull. Otherwise, we use cdd.

    Raises:
        ValueError: When minimal V-Rep computation fails!

    Notes:
        Use cdd or qhull for the reduction of vertices. Requires the polytope to be bounded.
    """
    if not self.is_bounded:
        raise ValueError("Polytope is not bounded!")
    elif self.is_empty:
        self._set_polytope_to_empty(self.dim)
    else:
        if self.is_full_dimensional and self.dim >= 2 and prefer_qhull_over_cdd:
            # Indices of the unique vertices forming the convex hull:
            i_V_minimal = ConvexHull(self.V).vertices
            V_minimal = self.V[i_V_minimal, :]
            self._set_attributes_from_V(V_minimal, erase_H_rep=False)
        else:
            # cdd handles this case. It calls determine_V_rep if necessary via self.V getter
            try:
                cdd_polyhedron = get_cdd_polyhedron_from_V(self.V)
                tV_cdd_matrix = cdd_polyhedron.get_generators()
                tV_cdd_matrix.canonicalize()  # Minimize redundant vertices
                set_attributes_V_from_cdd(self, tV_cdd_matrix)  # Set attributes from V
            except ValueError as err:
                # Error can come from vertex enumeration of an unbounded set
                raise ValueError("Computation of minimal V-rep failed!") from err


def set_attributes_minimal_Ab_Aebe_from_cdd_polyhedron(self, cdd_polyhedron):
    """Get (A, b, Ae, be) from CDD polyhedron in inequality form

    Args:
        cdd.Polyhedron: CDD Polyhedron

    Returns:
        (array_like, array_like, array_like, array_like): (A, b, Ae, be). (Ae, be) is (None, None) if no equality
        constraints
    """
    H_cdd_matrix = cdd_polyhedron.get_inequalities()
    H_cdd_matrix.canonicalize()  # Identify linear equalities if any
    H_cdd_array = np.array(H_cdd_matrix[:])
    Ae, be = None, None
    if H_cdd_array.size == 0:
        raise ValueError("Did not expect facet list to be empty after minimization!")
    elif len(H_cdd_matrix.lin_set):
        # Extract equalities
        He_cdd = np.array([v for index, v in enumerate(H_cdd_array) if index in H_cdd_matrix.lin_set])
        be, Ae = He_cdd[:, 0], -He_cdd[:, 1:]
        if len(H_cdd_matrix.lin_set) < H_cdd_array.shape[0]:
            # Extract inequalities
            H_cdd = np.array([v for index, v in enumerate(H_cdd_array) if index not in H_cdd_matrix.lin_set])
            b, A = H_cdd[:, 0], -H_cdd[:, 1:]
        else:
            A = np.empty((0, Ae.shape[1]))
            b = np.empty((0,))
    else:
        # Extract inequalities
        H_cdd = np.array(H_cdd_matrix[:])
        b, A = H_cdd[:, 0], -H_cdd[:, 1:]
    self._set_attributes_from_Ab_Aebe(A, b, Ae=Ae, be=be, erase_V_rep=False)


def set_attributes_V_from_cdd(self, tV_cdd_matrix):
    tV = np.array(tV_cdd_matrix)
    if (tV[:, 0] == 0).any():
        raise ValueError("Vertex enumeration yielded rays! Possibly due to numerical issues or unbounded polytope!")
    else:
        V = tV[:, 1:]
        self._set_attributes_from_V(V, erase_H_rep=False)


def valid_rows_with_not_all_zeros_in_A_and_no_inf_in_b(A, b):
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
