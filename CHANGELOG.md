<!--
Copyright (C) 2020-2025 Mitsubishi Electric Research Laboratories (MERL)

SPDX-License-Identifier: CC-BY-4.0
-->
# 1.1.0 /2025-06-30

- Update dependency to use pycddlib>=3.0.0
- General changes
  - Add citation.cff
  - Add a new tutorial going over the code snippets used in ACC 2025 paper
  - Add flatten to containment_constraints for easier interface with cvxpy
  - [Breaking change] projection method uses the keyword argument project_away_dims instead of project_away_dim
  - Align definitions for Ellipsoid and Zonotopes (via ConstrainedZonotope)
  - Add compute_irredundant_affine_set_using_cdd
- Add new methods/data members to ConstrainedZonotope
  - cartesian_product
  - is_singleton
  - is_full_dimensional
  - is_zonotope
  - slice_then_projection
- Add new methods to Ellipsoid
  - affine_hull
  - intersection_with_affine_set
  - quadratic_form_as_a_symmetric_matrix
  - slice_then_projection
- Changes to Polytope
  - Code changes to support breaking change in pycddlib update
  - slice_then_projection
- [Bugfix] Fix issue #2 in github


This release also includes other minor code refactoring and changes to tests for better code coverage.

# 1.0.2 / 2025-02-13

- Set up documentation website using Github Pages, which is released under CC-BY-4.0 license

# 1.0.1 / 2024-10-14

- Restrict pycddlib to 2.1.8
- Minor bugfixes

# 1.0.0 / 2024-09-27

Initial release
