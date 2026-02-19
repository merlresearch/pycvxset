# Copyright (C) 2020-2026 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

# Code purpose: Check if data members and methods in pycvxset are consistent across classes

import itertools as itert

import numpy as np

from pycvxset import ConstrainedZonotope, Ellipsoid, Polytope

CLASS_DICT = {
    "ellipsoid": Ellipsoid,
    "polytope": Polytope,
    "constrained_zonotope": ConstrainedZonotope,
}
DATA_MEMBERS_CLASS_SPECIFIC = {
    "polytope": set(
        [
            "A",
            "b",
            "H",
            "V",
            "in_H_rep",
            "in_V_rep",
            "n_vertices",
            "n_halfspaces",
        ]
    ),
    "constrained_zonotope": set(
        [
            "is_zonotope",
        ]
    ),
    "ellipsoid": set(
        [
            "Q",
        ]
    ),
}
EXPECTED_DATA_MEMBERS_1_BUT_NOT_IN_2 = {
    ("polytope", "ellipsoid"): set(
        [
            "Ae",
            "be",
            "He",
            "n_equalities",
        ]
    ),
    ("constrained_zonotope", "ellipsoid"): set(
        [
            "Ae",
            "be",
            "He",
            "n_equalities",
        ]
    ),
    ("ellipsoid", "polytope"): set(
        [
            "G",
            "c",
            "latent_dim",
        ]
    ),
    ("ellipsoid", "constrained_zonotope"): set(),
    ("polytope", "constrained_zonotope"): set(),
    ("constrained_zonotope", "polytope"): set(
        [
            "G",
            "c",
            "latent_dim",
        ]
    ),
}

METHODS_CLASS_SPECIFIC = {
    "polytope": set(
        [
            "_set_attributes_from_bounds",
            "_set_attributes_from_Ab_Aebe",
            "_update_emptiness_full_dimensionality_for_h_rep_polytope",
            "_update_boundedness_singleton_for_h_rep_polytope",
            # Polytope-specific decomposition for handling lower-dimensional polytopes
            "decompose_as_affine_transform_of_polytope_without_equalities",
            "normalize",
            "_set_attributes_from_V",
            "_set_polytope_to_empty",
            "plot3d",  # Polytope-specific plotting
            "plot2d",  # Polytope-specific plotting
            "determine_V_rep",  # Polytope-specific V-Rep <> H-Rep
            "determine_H_rep",  # Polytope-specific V-Rep <> H-Rep
            "minimize_V_rep",  # Polytope-specific V-Rep <> H-Rep
            "minimize_H_rep",  # Polytope-specific V-Rep <> H-Rep
        ]
    ),
    "constrained_zonotope": set(
        [
            "_get_Gc_Aebe_from_bounds",
            "_get_Gc_Aebe_for_empty_constrained_zonotope",
            "polytopic_inner_approximation",  # ConstrainedZonotope require plot approximation
            "polytopic_outer_approximation",  # ConstrainedZonotope require plot approximation
            "remove_redundancies",  # ConstraiendZonotope specific
        ]
    ),
    "ellipsoid": set(
        [
            "_set_radii_from_Q",
            "_set_ellipsoid_to_singleton",
            "polytopic_inner_approximation",  # Ellipsoids require plot approximation
            "polytopic_outer_approximation",  # Ellipsoids require plot approximation
            "quadratic_form_as_a_symmetric_matrix",  # Ellipsoid specific
        ]
    ),
}
EXPECTED_METHODS_1_BUT_NOT_IN_2 = {
    ("polytope", "ellipsoid"): set(
        [
            "__pow__",  # Ellipsoid self Cartesian product are not possible
            "__rsub__"  # REVISIT: Ellipsoid Pontryagin difference can not be exact
            "cartesian_product",  # Ellipsoid Cartesian product are not possible
            "minus",  # Ellipsoid Pontryagin difference can not be exact
            "intersection",  # Ellipsoid intersections are not closed
            "intersection_under_inverse_affine_map",  # Ellipsoid intersections are not closed
            "intersection_with_halfspaces",  # Ellipsoid intersections are not closed
            "deflate_rectangle",  # Ellipsoids can not have rectangles
        ]
    ),
    ("constrained_zonotope", "ellipsoid"): set(
        [
            "__pow__",  # Ellipsoid self Cartesian product are not possible
            "cartesian_product",  # Ellipsoid Cartesian product are not possible
            "__rsub__"  # REVISIT: Ellipsoid Pontryagin difference can not be exact
            "minus",  # Ellipsoid Pontryagin difference can not be exact
            "intersection",  # Ellipsoid intersections are not closed
            "intersection_under_inverse_affine_map",  # Ellipsoid intersections are not closed
            "intersection_with_halfspaces",  # Ellipsoid intersections are not closed
        ]
    ),
    ("ellipsoid", "polytope"): set(
        [
            "inflate_ball",  # Polytopes can not have balls
        ]
    ),
    ("ellipsoid", "constrained_zonotope"): set(
        [
            "inflate_ball",  # ConstrainedZonotope can not have balls
            "volume",  # ConstrainedZonotope volume computation is unknown
        ]
    ),
    ("polytope", "constrained_zonotope"): set(
        [
            "volume",  # ConstrainedZonotope volume computation is unknown
        ]
    ),
    ("constrained_zonotope", "polytope"): set(
        ["approximate_pontryagin_difference"]  # Polytope can compute Pontryagin difference exactly
    ),
}

data_members = {}
methods = {}
for class_name in CLASS_DICT:
    class_obj = CLASS_DICT[class_name]
    methods[class_name] = set([v for v in dir(class_obj) if callable(getattr(class_obj, v))])
    methods[class_name] -= METHODS_CLASS_SPECIFIC[class_name]
    data_members[class_name] = set([v for v in dir(class_obj) if not callable(getattr(class_obj, v))])
    data_members[class_name] -= DATA_MEMBERS_CLASS_SPECIFIC[class_name]

METHODS_TO_DEFINE = {key: [] for key in CLASS_DICT.keys()}
for name_1, name_2 in itert.permutations(CLASS_DICT.keys(), 2):
    in_1_but_not_in_2 = methods[name_1] - methods[name_2] - EXPECTED_METHODS_1_BUT_NOT_IN_2[(name_1, name_2)]
    METHODS_TO_DEFINE[name_2].extend(in_1_but_not_in_2)

DATA_MEMBERS_TO_DEFINE = {key: [] for key in CLASS_DICT.keys()}
for name_1, name_2 in itert.permutations(CLASS_DICT.keys(), 2):
    in_1_but_not_in_2 = (
        data_members[name_1] - data_members[name_2] - EXPECTED_DATA_MEMBERS_1_BUT_NOT_IN_2[(name_1, name_2)]
    )
    DATA_MEMBERS_TO_DEFINE[name_2].extend(in_1_but_not_in_2)

for key in CLASS_DICT.keys():
    print("Class :", key)
    print("Data member:", np.unique(DATA_MEMBERS_TO_DEFINE[key]))
    print("Method:", np.unique(METHODS_TO_DEFINE[key]), "\n")
