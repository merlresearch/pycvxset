# Copyright (C) 2020-2025 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

# Code purpose: Check if methods in pycvxset are consistent

from pycvxset import ConstrainedZonotope, Ellipsoid, Polytope

in_polytope_but_not_in_constrained_zonotope = [
    v
    for v in dir(Polytope)
    if (
        callable(getattr(Polytope, v))
        and not (v in dir(ConstrainedZonotope) and callable(getattr(ConstrainedZonotope, v)))
    )
]

in_constrained_zonotope_but_not_in_polytope = [
    v
    for v in dir(ConstrainedZonotope)
    if (callable(getattr(ConstrainedZonotope, v)) and not (v in dir(Polytope) and callable(getattr(Polytope, v))))
]

in_ellipsoid_but_not_in_polytope = [
    v
    for v in dir(Ellipsoid)
    if (callable(getattr(Ellipsoid, v)) and not (v in dir(Polytope) and callable(getattr(Polytope, v))))
]

print("in_polytope_but_not_in_constrained_zonotope\n", "\n".join(in_polytope_but_not_in_constrained_zonotope), end="\n")
print("\n\nin_constrained_zonotope_but_not_in_polytope\n", "\n".join(in_constrained_zonotope_but_not_in_polytope))
print("\n\nin_ellipsoid_but_not_in_polytope\n", "\n".join(in_ellipsoid_but_not_in_polytope))
