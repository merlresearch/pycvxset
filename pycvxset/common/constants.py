# Copyright (C) 2020-2024 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

# Code purpose:  Specify the constants to be used with cvxpy when optimization problems as well as testing workflows

PYCVXSET_ZERO = 1e-6  # Zero threshold for numerical stability
PYCVXSET_ZERO_GUROBI = 1e-5  # Zero threshold for numerical stability when using GUROBI
# Zero threshold for vertex clustering when computing polytopic inner-approximation of constrained zonotopes
PYCVXSET_ZERO_CDD = 1e-5
PLOTTING_DECIMAL_PRECISION_CDD = 3

# Solvers used by default
DEFAULT_LP_SOLVER_STR = "CLARABEL"  # CLARABEL, MOSEK, CVXOPT, SCS, ECOS, GUROBI, OSQP
DEFAULT_SOCP_SOLVER_STR = "CLARABEL"  # CLARABEL, MOSEK, CVXOPT, SCS, ECOS, GUROBI
DEFAULT_SDP_SOLVER_STR = "SCS"  # CLARABEL, MOSEK, CVXOPT, SCS

# CVXPY args used by default (use "reoptimize": True when using GUROBI)
DEFAULT_CVXPY_ARGS_LP = {"solver": DEFAULT_LP_SOLVER_STR}
DEFAULT_CVXPY_ARGS_SOCP = {"solver": DEFAULT_SOCP_SOLVER_STR}
DEFAULT_CVXPY_ARGS_SDP = {"solver": DEFAULT_SDP_SOLVER_STR}

# Time limit for GUROBI when checking for containment of a constrained zonotope in an another constrained zonotope
TIME_LIMIT_FOR_GUROBI_NON_CONVEX = 60.0

# Constants for spread_points_on_a_unit_sphere (SPOAUS)
SPOAUS_SLACK_TOLERANCE = 1e-8
SPOAUS_COST_TOLERANCE = 1e-5
SPOAUS_INITIAL_TAU = 1.0
SPOAUS_SCALING_TAU = 1.1
SPOAUS_TAU_MAX = 1e4
SPOAUS_ITERATIONS_AT_TAU_MAX = 20
SPOAUS_MINIMUM_NORM_VALUE_SQR = 0.8**2
# For SPOAUS_DIRECTIONS_PER_QUADRANT=20, we have 2D = 84, 3D = 166, 4D = 328, 5D = 650
SPOAUS_DIRECTIONS_PER_QUADRANT = 20

# Testing workflow constants | You could also do "GUROBI" in cvxpy.installed_solvers()
TESTING_CONTAINMENT_STATEMENTS_INVOLVING_GUROBI = False
TESTING_SHOW_PLOTS = False
TEST_3D_PLOTTING = False

# Plotting constants for polytopes
DEFAULT_PATCH_ARGS_2D = {"edgecolor": "k", "facecolor": "skyblue"}
DEFAULT_PATCH_ARGS_3D = {"edgecolor": "k", "facecolor": None}
DEFAULT_VERTEX_ARGS = {"visible": False, "s": 30, "marker": "o", "color": "k"}
