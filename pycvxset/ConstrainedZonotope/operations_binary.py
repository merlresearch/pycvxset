# Copyright (C) 2020-2026 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

# Code purpose:  Define the methods involving another set or a point used with ConstrainedZonotope class
# Coverage: With TESTING_STATEMENTS_INVOLVING_GUROBI
# Coverage: "minimal" (default), this file has  3 missing statements + 6 excluded statements + 1 partial branches.
# Coverage: "full"             , this file has  2 missing statements + 6 excluded statements + 0 partial branches.
# Coverage: "no"               , this file has 10 missing statements + 6 excluded statements + 5 partial branches.

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any, Sequence, cast

import numpy as np

from pycvxset.common import (
    convex_set_contains_points,
    is_constrained_zonotope,
    is_ellipsoid,
    is_polytope,
    sanitize_Aebe,
    sanitize_and_identify_Aebe,
    sanitize_Gc,
)
from pycvxset.common.constants import (
    PYCVXSET_ZERO,
    PYCVXSET_ZERO_GUROBI,
    TIME_LIMIT_FOR_CONSTRAINED_ZONOTOPE_CONTAINMENT_GUROBI_IN_S,
)
from pycvxset.ConstrainedZonotope.operations_unary import get_matrix_least_norm_solution

if TYPE_CHECKING:
    from pycvxset.ConstrainedZonotope import ConstrainedZonotope
    from pycvxset.Ellipsoid import Ellipsoid
    from pycvxset.Polytope import Polytope

DOCSTRING_FOR_PROJECT = (
    "\n"
    + r"""
    Notes:
        Given a constrained zonotope :math:`\mathcal{P}` and a test point
        :math:`y\in\mathbb{R}^{\mathcal{P}.\text{dim}}`, this function solves a convex program with decision variables
        :math:`x\in\mathbb{R}^{\mathcal{P}.\text{dim}}` and :math:`\xi\in\mathbb{R}^{\mathcal{P}.\text{latent\_dim}}`,

        .. math::
            \text{minimize}    &\quad  \|x - y\|_p\\
            \text{subject to}  &\quad  x = G_P \xi + c_P\\
                               &\quad  A_P \xi = b_P\\
                               &\quad  \|\xi\|_\infty \leq 1
    """
)

DOCSTRING_FOR_PROJECTION = (
    "\n"
    + r"""
    Returns:
        ConstrainedZonotope: m-dimensional set obtained via projection.
"""
)


DOCSTRING_FOR_SLICE = (
    "\n"
    + r"""
    Returns:
        ConstrainedZonotope: Constrained zonotope that has been sliced at the specified dimensions.
    """
)

DOCSTRING_FOR_SLICE_THEN_PROJECTION = (
    "\n"
    + r"""
    Returns:
        ConstrainedZonotope: m-dimensional set obtained via projection after slicing.
    """
)

DOCSTRING_FOR_SUPPORT = (
    "\n"
    + r"""
    Notes:
        Given a constrained zonotope :math:`\mathcal{P}` and a support direction
        :math:`\eta\in\mathbb{R}^{\mathcal{P}.\text{dim}}`, this function solves a convex program with decision
        variables :math:`x\in\mathbb{R}^{\mathcal{P}.\text{dim}}` and
        :math:`\xi\in\mathbb{R}^{\mathcal{P}.\text{latent\_dim}}`,

        .. math ::
            \text{maximize}     &\quad \eta^\top x\\
            \text{subject to}   &\quad x = G_P \xi + c_P\\
                                &\quad A_P \xi = b_P\\
                                &\quad \|\xi\|_\infty \leq 1
    """
)


def affine_map(
    self: ConstrainedZonotope, M: int | float | Sequence[float] | Sequence[Sequence[float]] | np.ndarray
) -> ConstrainedZonotope:
    r"""Multiply a matrix or a scalar with a constrained zonotope

    Args:
        M (int | float | Sequence[float] | Sequence[Sequence[float]] | np.ndarray): Matrix (N times self.dim) or a
            scalar to be multiplied with a constrained zonotope

    Returns:
        ConstrainedZonotope: Constrained zonotope which is the product of M and self. Specifically, given a constrained
        zonotope :math:`\mathcal{P}`, and a matrix :math:`M\in\mathbb{R}^{m\times P.\text{dim}}` or scalar :math:`M`,
        then this function returns a constrained zonotope :math:`\mathcal{R}=\{Mx|x\in\mathcal{P}\}`.

    Notes:
        This function implements (11) of [SDGR16]_.
    """
    try:
        M_arr: np.ndarray = np.atleast_2d(M).astype(float)
    except (TypeError, ValueError) as err:
        raise ValueError(f"M must be convertible to a numpy 2D array of float! Got {type(M)}") from err
    if M_arr.ndim > 2:
        raise ValueError(f"M is must be convertible into a 2D numpy.ndarray. But got {M_arr.ndim:d}D array.")
    elif self.dim != 1 and M_arr.shape[0] == M_arr.shape[1] and M_arr.shape[0] == 1:
        # Scalar multiplication
        if self.c is None:
            # Do a lazy check of emptiness to handle scalar times empty
            return self.__class__(dim=self.dim)
        elif abs(M_arr[0][0]) <= PYCVXSET_ZERO:
            # m=0 multiplier => product is constrained zonotope that only has origin
            return self.__class__(c=np.zeros((self.dim,)), G=None)
        else:
            # This includes CZ that may be empty due to (Ae, be) but we just proceed with affine map
            M_arr = M_arr[0][0] * np.eye(self.dim)
    elif M_arr.shape[1] != self.dim:
        raise ValueError(f"Expected M upon promotion to 2D array to have {self.dim:d} columns. M: {M_arr.shape} matrix")
    elif self.c is None:
        # Check for emptiness lazily again to handle matrix times empty
        return self.__class__(dim=M_arr.shape[0])
    return self.__class__(G=M_arr @ self.G, c=M_arr @ self.c, Ae=self.Ae, be=self.be)


def approximate_pontryagin_difference(
    self: ConstrainedZonotope,
    norm_type: int | str,
    G_S: Sequence[Sequence[float]] | np.ndarray,
    c_S: Sequence[float] | np.ndarray,
    method: str = "inner-least-squares",
    enable_warning: bool = True,
) -> ConstrainedZonotope:
    r"""Approximate Pontryagin difference between a constrained zonotope and an affine transformation of a unit-norm
    ball.

    Specifically, we approximate the Pontryagin difference :math:`\mathcal{P}\ominus\mathcal{S}=\{x\in\mathbb{R}^n\ |\ x
    + \mathcal{S}\subseteq\mathcal{P}\}` between a constrained zonotope :math:`\mathcal{P}` and a subtrahend
    :math:`\mathcal{S}`. The subtrahend must be specifically of the form :math:`\mathcal{S}=\{G_S \xi + c_S |
    \|\xi\|_p\leq 1\}` for norm\_type :math:`p\in\{1,2,\infty\}`.

    * For p=1, the subtrahend is a convex hull of intervals specified by the columns of :math:`G_S` with each
      interval is symmetric about :math:`c_S`.
    * For p=2, the subtrahend is an ellipsoid is characterized by affine transformation :math:`G_S` of a unit
      Euclidean norm ball which is then shifted to :math:`c_S`. Here, :math:`G_S` is given by
      :attr:`pycvxset.Ellipsoid.Ellipsoid.G`. See :meth:`pycvxset.Ellipsoid.Ellipsoid` to obtain :math:`G_S` and
      :math:`c_S` for broader classes of ellipsoids.
    * For p='inf, the subtrahend is a zonotope characterized by (:math:`G_S`, :math:`c_S`).

    Args:
        norm_type (int | str): Norm type.
        G_S (Sequence[Sequence[float]] | numpy.ndarray): Affine transformation matrix (self.dim times N) for scaling the
            ball
        c_S (Sequence[float] | numpy.ndarray): Affine transformation vector (self.dim,) for translating the ball
        method (str, optional): Approximation method to use. Can be one of ['inner-least-squares']. Defaults to
            'inner-least-squares'.
        enable_warning (bool, optional): Enables the UserWarning. May be turned off if expected. Defaults to True.

    Raises:
        ValueError: Norm type is not in [1, 2, 'inf]
        ValueError: Mismatch in dimension (no. of columns of G_S, length of c_S, and self.dim should match)

    Returns:
        ConstrainedZonotope: Approximated Pontryagin difference between self and the affine transformation of the
        unit-norm ball.

    Notes:
        For method 'inner-least-squares', this function implements Table 1 of [VWD24]_.
    """
    if self.c is None:
        # When c is None, the Pontryagin difference is also empty.
        return self.__class__(dim=self.dim)
    if norm_type not in [1, 2, "inf"]:
        raise ValueError("Expected norm_type to be in [1, 2, 'inf']")
    G_S_arr, c_S_arr = sanitize_Gc(G_S, c_S)
    if G_S_arr.shape[0] != self.dim:
        raise ValueError(f"Expected subtrahend to have the dimension {self.dim:d}, but got {G_S_arr.shape[0]}!")
    if method == "inner-least-squares":
        matrix_least_norm_solution = get_matrix_least_norm_solution(
            self.G, self.Ae, "minus/approximate_pontryagin_difference", enable_warning=enable_warning
        )
        matrix_least_norm_solution = matrix_least_norm_solution @ G_S_arr

        if norm_type == 1:
            D_elements = 1 - np.linalg.norm(matrix_least_norm_solution, ord=np.inf, axis=1)
        elif norm_type == 2:
            D_elements = 1 - np.linalg.norm(matrix_least_norm_solution, ord=2, axis=1)
        else:  # norm_type is "inf":
            D_elements = 1 - np.linalg.norm(matrix_least_norm_solution, ord=1, axis=1)
        if min(D_elements) < 0:  # D_{ii} < 0 for some i
            return self.__class__(dim=self.dim)
        else:
            D = np.diag(D_elements)
            if self.n_equalities:
                return self.__class__(G=self.G @ D, c=self.c - c_S_arr, Ae=self.Ae @ D, be=self.be)
            else:
                return self.__class__(G=self.G @ D, c=self.c - c_S_arr)
    else:
        raise ValueError(f"Invalid method provided. Should be in ['inner-least-squares']. Got {method:s}!")


def cartesian_product(
    self: ConstrainedZonotope,
    sequence_Q: Polytope | ConstrainedZonotope | Sequence[Polytope | ConstrainedZonotope],
) -> ConstrainedZonotope:
    r"""Generate the Cartesian product of a set :math:`\mathcal{Q}` (or a list of `\mathcal{Q}`) with
    :math:`\mathcal{P}`.

    Args:
        sequence_Q (Polytope | ConstrainedZonotope | Sequence[Polytope | ConstrainedZonotope]): List of sets to take
            Cartesian product with

    Returns:
        ConstrainedZonotope: Cartesian product of self and all sets in sequence_Q
    """
    try:
        _ = (Q for Q in sequence_Q)  # pyright: ignore[reportGeneralTypeIssues, reportUnknownVariableType]
    except TypeError:
        # Not iterable. So make it iterable.
        sequence_Q = cast(tuple["Polytope | ConstrainedZonotope"], (sequence_Q,))
    sequence_Q = cast("Sequence[Polytope | ConstrainedZonotope]", sequence_Q)

    product_set = self.copy()
    for Q in sequence_Q:
        if is_ellipsoid(Q):
            raise ValueError(
                "Expected sequence_Q to be either an individual (OR a sequence of) ConstrainedZonotope or Polytope "
                f"object. Got {type(Q)}"
            )
        elif is_polytope(Q):
            Q = self.__class__(polytope=cast("Polytope", Q))
        Q_cz = cast("ConstrainedZonotope", Q)
        newobj_G = np.vstack(
            (
                np.hstack((product_set.G, np.zeros((product_set.dim, Q_cz.latent_dim)))),
                np.hstack((np.zeros((Q_cz.dim, product_set.latent_dim)), Q_cz.G)),
            )
        )
        newobj_c = np.hstack((product_set.c, Q_cz.c))
        newobj_Ae, newobj_be = get_first_two_blocks_for_Ae_be(product_set, Q_cz)
        product_set = self.__class__(G=newobj_G, c=newobj_c, Ae=newobj_Ae, be=newobj_be)
    return product_set


def contains(
    self: ConstrainedZonotope,
    Q: Sequence[float] | Sequence[Sequence[float]] | np.ndarray | Polytope | ConstrainedZonotope,
    verbose: bool = False,
    time_limit: float = TIME_LIMIT_FOR_CONSTRAINED_ZONOTOPE_CONTAINMENT_GUROBI_IN_S,
) -> bool | np.ndarray:
    r"""Check containment of a set :math:`\mathcal{Q}` (could be a polytope or a constrained zonotope), or a collection
    of points :math:`Q \in \mathbb{R}^{n_Q \times \mathcal{P}.\text{dim}}` in the given constrained zonotope.

    Args:
        Q (Sequence[float] | Sequence[Sequence[float]] | np.ndarray | Polytope | ConstrainedZonotope): Polytope object,
            ConstrainedZonotope object, or a collection of points to be tested for containment within the constrained
            zonotope. When providing a collection of points, Q is a matrix (N times self.dim) with each row is a point.
        verbose (bool, optional): Verbosity flag to provide cvxpy when solving the MIQP related to checking containment
            of a constrained zonotope within another constrained zonotope. Defaults to False.
        time_limit (float, optional): Time limit in seconds for GUROBI solver when solving the MIQP related to checking
            containment of a constrained zonotope within another constrained zonotope. Set time_limit to np.inf if no
            limit is desired. Defaults to 60s (see constants.py).

    Raises:
        ValueError: Dimension mismatch between Q and the constrained zonotope
        ValueError: Q is Ellipsoid
        ValueError: Unable to perform containment check between two constrained zonotopes (including time_limit issues)
        NotImplementedError: GUROBI is not installed when checking containment of two constrained zonotopes
        NotImplementedError: Failed to check containment between two constrained zonotopes, due to an unhandled status

    Returns:
        bool | numpy.ndarray([bool]): Boolean corresponding to :math:`\mathcal{Q}\subseteq\mathcal{P}` or
        :math:`\mathcal{Q}\in\mathcal{P}` .

    Notes:
        * We use :math:`\mathcal{P}` to denote the constrained zonotope characterized by self.
        * When Q is a constrained zonotope, a bool is returned which is True if and only if :math:`\mathcal{Q}\subseteq
          \mathcal{P}`. This function uses the non-convex programming capabilities of GUROBI (via CVXPY) to check
          containment between two constrained zonotopes.
        * When Q is a polytope, a bool is returned which is True if and only if :math:`\mathcal{Q}\subseteq
          \mathcal{P}`.

          * When Q is in V-Rep, the containment problem simplifies to checking if all vertices of Q are
            contained :math:`\mathcal{P}`.
          * When Q is in H-Rep, this function converts Q into a constrained zonotope, and then checks for containment.
        * When Q is a single n-dimensional point, a bool is returned which is True if and only if :math:`Q\in
          \mathcal{P}`. This function uses :meth:`distance` to check containment of a point in a constrained zonotope.
        * When Q is a collection of n-dimensional points (Q is a matrix with each row describing a point), an array
          of is returned where each element is True if and only if :math:`Q_i\in \mathcal{P}`.  This function uses
          :meth:`distance` to check containment of a point in a constrained zonotope.

    Warning:
        This function requires `gurobipy <https://pypi.org/project/gurobipy/>`_ when checking if the
        :class:`ConstrainedZonotope` object represented by `self` contains a :class:`ConstrainedZonotope` object `Q` or
        :class:`pycvxset.Polytope.Polytope` object `Q` in H-Rep.
    """
    import cvxpy as cp
    import gurobipy

    if is_ellipsoid(Q):
        raise ValueError("Checking containment of an Ellipsoid in a ConstrainedZonotope is not supported.")
    elif is_constrained_zonotope(Q) or is_polytope(Q):
        Q = cast("Polytope | ConstrainedZonotope", Q)
        if Q.dim != self.dim:
            raise ValueError(f"Expected Q to have dimension {self.dim:d}. Got Q with dimension {Q.dim:d}!")
        elif Q.is_empty:
            return True
        elif self.is_empty:
            return False
        elif is_polytope(Q):
            Q = cast("Polytope", Q)
            if Q.in_V_rep:
                return bool(np.all(self.contains(Q.V)))
            else:
                return self.contains(self.__class__(polytope=Q))
        else:
            Q = cast("ConstrainedZonotope", Q)
            Q_c = cast(np.ndarray, Q.c)  # Q is not empty, so c is not None
            if "GUROBI" not in cp.installed_solvers():
                raise NotImplementedError(
                    "Please set up CVXPY with GUROBI to check for containment of two constrained zonotopes"
                )
            Q_latent_var = cp.Variable((Q.latent_dim,))
            alpha_var = cp.Variable((self.dim,))
            dummy_var = cp.Variable(1, integer=True)  # Needed to tell cvxpy that this is a mixed-integer program
            const: list[cp.Constraint] = [cp.norm(Q_latent_var, p="inf") <= 1, dummy_var == 0]
            if Q.n_equalities:
                const += [Q.Ae @ Q_latent_var == Q.be]
            # Set up the bilinear objective
            alpha_var_Q_latent_var = cp.hstack((alpha_var, Q_latent_var))
            # We use -Q.G since we need to use +cp.quad_form
            negative_bilinear_coefficient_matrix = cp.vstack(
                (
                    cp.hstack((np.zeros((self.dim, self.dim)), -Q.G)),
                    np.zeros((Q.latent_dim, self.dim + Q.latent_dim)),
                )
            )
            if self.n_equalities:
                beta_var = cp.Variable((self.n_equalities,))
                const += [cp.norm(self.G.T @ alpha_var + self.Ae.T @ beta_var, p=1) <= 1]
                objective = (
                    1
                    - alpha_var @ (Q_c - self.c)
                    - beta_var @ self.be
                    + cp.quad_form(alpha_var_Q_latent_var, negative_bilinear_coefficient_matrix, assume_PSD=True)
                )
            else:
                const += [cp.norm(self.G.T @ alpha_var, p=1) <= 1]
                objective = (
                    1
                    - alpha_var @ (Q_c - self.c)
                    + cp.quad_form(alpha_var_Q_latent_var, negative_bilinear_coefficient_matrix, assume_PSD=True)
                )

            problem = cp.Problem(cp.Minimize(objective), const)
            try:
                problem.solve(solver="GUROBI", reoptimize=True, NonConvex=2, verbose=verbose, TimeLimit=time_limit)
                # From https://github.com/cvxpy/cvxpy/issues/1091, dispose the model explicitly
                cast(Any, problem.solver_stats.extra_stats).dispose()
            except (
                cp.error.SolverError,  # pyright: ignore[reportAttributeAccessIssue]
                gurobipy._exception.GurobiError,  # pyright: ignore[reportAttributeAccessIssue]
            ) as err:
                raise ValueError(
                    "Unable to perform containment check between two constrained zonotopes! CVXPY returned "
                    f"error: {str(err)}"
                ) from err
            if problem.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
                if objective.value >= -PYCVXSET_ZERO_GUROBI:
                    return True
                else:
                    return False
            elif problem.status in [cp.UNBOUNDED, cp.UNBOUNDED_INACCURATE]:
                return False
            elif problem.status == cp.USER_LIMIT:
                raise ValueError(
                    f"Exceeded specified time limit of {time_limit:.2f} seconds exceeded when checking containment of "
                    f"two constrained zonotopes. CVXPY status {problem.status:s}"
                )
            else:
                # Is not expected to happen!
                raise NotImplementedError(
                    "Failed to check containment between two constrained zonotopes, due to an unhandled "
                    f"status: {problem.status:s}"
                )
    else:
        # Q is a set of test_points
        return convex_set_contains_points(self, Q)


def intersection(self: ConstrainedZonotope, Q: ConstrainedZonotope | Polytope) -> ConstrainedZonotope:
    """Compute the intersection of constrained zonotope with another constrained zonotope or polytope.

    Args:
        Q (ConstrainedZonotope | Polytope): Set to intersect with

    Raises:
        TypeError: When Q is neither a ConstrainedZonotope object or a Polytope object

    Returns:
        ConstrainedZonotope: Intersection of self with Q

    Notes:
        * When Q is a constrained zonotope, this function uses :meth:`intersection_under_inverse_affine_map` with R set
          to identity matrix.
        * When Q is a polytope in H-Rep, this function uses  :meth:`intersection_with_halfspaces` and
          :meth:`intersection_with_affine_set`.
        * When Q is a polytope in V-Rep, this function converts Q into a constrained zonotope to avoid a halfspace
          enumeration.
    """
    if is_constrained_zonotope(Q):
        return self.intersection_under_inverse_affine_map(cast("ConstrainedZonotope", Q), np.eye(self.dim))
    elif is_polytope(Q):
        Q = cast("Polytope", Q)
        if Q.in_V_rep:
            return self.intersection(self.__class__(polytope=Q))
        else:
            constrained_zonotope_after_intersection_with_H = self.intersection_with_halfspaces(Q.A, Q.b)
            if Q.n_equalities > 0:
                return constrained_zonotope_after_intersection_with_H.intersection_with_affine_set(Q.Ae, Q.be)
            else:
                return constrained_zonotope_after_intersection_with_H
    else:
        raise TypeError(f"Unsupported operation between Constrained Zonotope and {type(Q)}")


def intersection_with_affine_set(
    self: ConstrainedZonotope, Ae: Sequence[Sequence[float]] | np.ndarray, be: Sequence[float] | np.ndarray
) -> ConstrainedZonotope:
    r"""Compute the intersection of a constrained zonotope with an affine set.

    Args:
        Ae (Sequence[Sequence[float]] | numpy.ndarray): Equality coefficient matrix (N times self.dim) that define the
            affine set :math:`\{x|A_ex = b_e\}`.
        be (Sequence[float] | numpy.ndarray): Equality constant vector (N,) that define the affine set :math:`\{x| jA_ex
            = b_e\}`.

    Raises:
        ValueError: When the number of columns in Ae is different from self.dim

    Returns:
        ConstrainedZonotope: The intersection of a constrained zonotope with the affine set.

    Notes:
        This function implements imposes the constraints :math:`\{A_ex = b_e\}` as constraints in the latent dimension
        of the constrained zonotope --- :math:`A_e (G \xi + c) = b_e` for every feasible :math:`\xi`.
    """
    Ae, be, Aebe_status, solution_to_Ae_x_eq_be = sanitize_and_identify_Aebe(Ae, be)
    if Aebe_status == "no_Ae_be":
        return self.copy()
    Ae_arr = cast(np.ndarray, Ae)  # Ae, be is not None since that is handled in no_Ae_be case
    be_arr = cast(np.ndarray, be)
    if self.dim != Ae_arr.shape[1]:
        raise ValueError(f"Expected Ae to have {self.dim:d} columns. Got Ae with shape {Ae_arr.shape}!")
    elif (
        self.c is None  # Do a lazy check of emptiness to handle empty set intersected with affine set
        or Aebe_status == "infeasible"
        or (Aebe_status == "single_point" and solution_to_Ae_x_eq_be is not None and solution_to_Ae_x_eq_be not in self)
    ):
        return self.__class__(dim=self.dim)
    elif Aebe_status == "single_point":
        return self.__class__(G=None, c=cast(np.ndarray, solution_to_Ae_x_eq_be))
    else:
        if self.n_equalities:
            new_Ae = np.vstack((self.Ae, Ae_arr @ self.G))
            new_be = np.hstack((self.be, be_arr - Ae_arr @ self.c))
        else:
            new_Ae = Ae_arr @ self.G
            new_be = be_arr - Ae_arr @ self.c
        return self.__class__(G=self.G, c=self.c, Ae=new_Ae, be=new_be)


def intersection_with_halfspaces(
    self: ConstrainedZonotope, A: Sequence[Sequence[float]] | np.ndarray, b: Sequence[float] | np.ndarray
) -> ConstrainedZonotope:
    r"""Compute the intersection of a constrained zonotope with a collection of halfspaces (polyhedron).

    Args:
        A (Sequence[Sequence[float]] | numpy.ndarray): Inequality coefficient matrix (N times self.dim) that define the
            polyhedron :math:`\{x|Ax \leq b\}`.
        b (Sequence[float] | numpy.ndarray): Inequality constant vector (N,) that define the polyhedron :math:`\{x| Ax
            \leq b\}`.

    Returns:
        ConstrainedZonotope: The intersection of a constrained zonotope with a collection of halfspaces.

    Notes:
        This function implements (10) of [RK22]_ for each halfspace. We skip redundant inequalities when encountered and
        we return empty set when intersection yields an empty set.
    """
    Q_A, Q_b = sanitize_Aebe(A, b)
    if Q_A is None and Q_b is None:
        return self.copy()
    elif cast(np.ndarray, Q_A).shape[1] != self.dim:
        raise ValueError(f"Expected A to have {self.dim:d} columns. Got A with shape {cast(np.ndarray, Q_A).shape}!")
    elif self.c is None:  # Do a lazy check of emptiness to handle empty set intersected with halfspaces
        return self.__class__(dim=self.dim)
    else:
        temp_set = self.copy()
        for Q_a_i, Q_b_i in zip(cast(np.ndarray, Q_A), cast(np.ndarray, Q_b)):
            if temp_set.support(Q_a_i)[0] <= Q_b_i:  # Skip the halfspace since it completely contains the set
                continue
            elif Q_b_i < -temp_set.support(-Q_a_i)[0]:  # The intersection is empty
                return self.__class__(dim=self.dim)
            else:
                newobj_G, newobj_c, newobj_Ae, newobj_be = temp_set.G, temp_set.c, temp_set.Ae, temp_set.be
                d_m = Q_b_i - Q_a_i @ newobj_c + np.linalg.norm(Q_a_i @ newobj_G, ord=1)
                new_row = np.hstack((Q_a_i @ newobj_G, d_m / 2))
                if newobj_Ae.size == 0:
                    newobj_Ae = np.array([new_row])
                else:
                    expanded_newobj_Ae = np.hstack((newobj_Ae, np.zeros((newobj_Ae.shape[0], 1))))
                    newobj_Ae = np.vstack((expanded_newobj_Ae, new_row))
                newobj_be = np.hstack((newobj_be, Q_b_i - Q_a_i @ newobj_c - d_m / 2))
                newobj_G = np.hstack((newobj_G, np.zeros((self.dim, 1))))
                temp_set = self.__class__(G=newobj_G, c=newobj_c, Ae=newobj_Ae, be=newobj_be)
        return temp_set


def intersection_under_inverse_affine_map(
    self: ConstrainedZonotope, Y: ConstrainedZonotope, R: Sequence[Sequence[float]] | np.ndarray
) -> ConstrainedZonotope:
    r"""Compute the intersection of constrained zonotope with another constrained zonotope under an inverse affine map

    Args:
        Y (ConstrainedZonotope): Set to intersect with
        R (Sequence[Sequence[float]] | numpy.ndarray): Matrix (Y.dim times self.dim) for the inverse-affine map.

    Raises:
        ValueError: When Y is not a ConstrainedZonotope
        ValueError: When R is not of correct dimension

    Returns:
        ConstrainedZonotope: Intersection of self with Y under an inverse affine map R. Specifically, given constrained
        zonotopes :math:`\mathcal{P}` (self) and :math:`\mathcal{Y}`, and a matrix :math:`R`, we compute the set
        :math:`\mathcal{P}\cap_R\mathcal{Y}=\{x \in \mathcal{P}| Rx \in \mathcal{Y}\}`. When :math:`R` is
        invertible, :math:`\mathcal{P}\cap_R\mathcal{Y}=(R^{-1}\mathcal{P})\cap\mathcal{Y}`.

    Notes:
        This function implements (13) of [SDGR16]_. This function does not require R to be invertible.
    """
    if not is_constrained_zonotope(Y):
        raise ValueError(f"Expected Y to be a constrained zonotope. Got {type(Y)}!")

    try:
        R_arr: np.ndarray = np.atleast_2d(R).astype(float)
    except (TypeError, ValueError) as err:
        raise ValueError("Expected R to convertible into a 2D numpy array of floats!") from err
    if R_arr.ndim != 2 or R_arr.shape != (Y.dim, self.dim):
        raise ValueError(f"Expected R to have the shape ({Y.dim:d}, {self.dim:d}). Got {R_arr.shape}!")
    elif self.c is None or Y.c is None:  # Do a lazy check of emptiness to handle empty set intersection
        return self.__class__(dim=self.dim)
    else:
        newobj_G = np.hstack((self.G, np.zeros((self.dim, Y.latent_dim))))
        newobj_c = self.c

        # Set up first two rows of A, b based on constrained zonotope or zonotope situations
        # Set up the last row
        intersection_specific_Ae = np.hstack((R_arr @ self.G, -Y.G))
        intersection_specific_be = Y.c - R_arr @ self.c
        newobj_Ae, newobj_be = get_first_two_blocks_for_Ae_be(self, Y)
        newobj_Ae = np.vstack((newobj_Ae, intersection_specific_Ae))
        newobj_be = np.hstack((newobj_be, intersection_specific_be))

        return self.__class__(G=newobj_G, c=newobj_c, Ae=newobj_Ae, be=newobj_be)


def inverse_affine_map_under_invertible_matrix(
    self: ConstrainedZonotope, M: Sequence[Sequence[float]] | np.ndarray
) -> ConstrainedZonotope:
    r"""Compute the inverse affine map of a constrained zonotope for a given matrix M.

    Args:
        M (Sequence[Sequence[float]] | numpy.ndarray): Square invertible matrix of dimension self.dim

    Raises:
        ValueError: M is not convertible into a 2D float array
        ValueError: M is not square
        ValueError: M is not invertible

    Returns:
        ConstrainedZonotope: Inverse affine map of self under M. Specifically, Given a constrained zonotope
        :math:`\mathcal{P}` and an invertible self.dim-dimensional matrix :math:`M\in\mathbb{R}^{\mathcal{P}.dim\times
        \mathcal{P}.\text{dim}}`, this function computes the constrained zonotope :math:`\mathcal{R}=M^{-1}\mathcal{P}`.

    Notes:
        This function is a wrapper for :meth:`affine_map`. We require M to be invertible in order to ensure that the
        resulting set is representable as a constrained zonotope.
    """
    try:
        M_arr: np.ndarray = np.atleast_2d(np.squeeze(M)).astype(float)
    except (TypeError, ValueError) as err:
        raise ValueError(f"Multiplication of ConstrainedZonotope with {type(M)} is not supported!") from err
    if M_arr.shape != (self.dim, self.dim):
        raise ValueError("Expected M to be a square matrix of shape ({self.dim:d},{self.dim:d}). Got {M_arr.shape}!")
    try:
        M_inv = np.linalg.inv(M_arr)
    except np.linalg.LinAlgError as err:
        raise ValueError("Expected M to be invertible!") from err
    return affine_map(self, M_inv)


def plus(
    self: ConstrainedZonotope, Q: Sequence[float] | np.ndarray | Polytope | ConstrainedZonotope
) -> ConstrainedZonotope:
    r"""Add a point or a set Q to a constrained zonotope (Minkowski sum).

    Args:
        Q (Sequence[float] | np.ndarray | Polytope | ConstrainedZonotope): The point or set to add

    Raises:
        TypeError: When Q is not one of the following --- convertible into a 1D numpy array or a constrained zonotope.
        ValueError: When Q has a dimension mismatch with self.
        UserWarning: When using with Q that is a set, warn that an inner-approximation is returned.

    Returns:
        ConstrainedZonotope: Minkowski sum of self and Q.

    Notes:
        Given a constrained zonotope :math:`\mathcal{P}` and a set :math:`Q`, this function computes the Minkowski sum
        of Q and the constrained zonotope, defined as :math:`\mathcal{R}=\{x + q|x\in\mathcal{P}, q\in\mathcal{Q}\}`. On
        the other hand, when Q is a point, this function computes the constrained zonotope :math:`\mathcal{R}=\{x +
        Q|x\in\mathcal{P}\}`.

        This function implements (12) of [SDGR16]_ when Q is a constrained zonotope. When Q is a Polytope, this function
        converts it into a constrained zonotope, and then uses (12) of [SDGR16]_.
    """
    if is_constrained_zonotope(Q):
        Q = cast("ConstrainedZonotope", Q)
        if Q.dim != self.dim:
            raise ValueError(f"Expected a constrained zonotope of dim. {self.dim:d}! Got Q with dim:{Q.dim:d}")
        elif self.c is None:  # Do a lazy check of emptiness to handle empty set plus another set
            return Q.copy()
        elif Q.c is None:  # Do a lazy check of emptiness to handle empty set plus another set
            return self.copy()
        else:
            newobj_G = np.hstack((self.G, Q.G))
            newobj_c = self.c + Q.c
            newobj_Ae, newobj_be = get_first_two_blocks_for_Ae_be(self, Q)
            return self.__class__(G=newobj_G, c=newobj_c, Ae=newobj_Ae, be=newobj_be)
    elif is_polytope(Q):
        return self.plus(self.__class__(polytope=cast("Polytope", Q)))
    else:
        try:
            Q = np.atleast_1d(np.squeeze(Q)).astype(float)
        except (TypeError, ValueError) as err:
            raise TypeError(f"Unsupported operation: {type(Q)} + ConstrainedZonotope") from err
        Q_arr = cast(np.ndarray, Q)
        if Q_arr.ndim != 1 or Q_arr.size != self.dim:
            raise ValueError(f"Q must be a numpy 1D array of length {self.dim:d}! Got Q with shape:{Q_arr.shape}")
        if self.c is None:  # Do a lazy check of emptiness to handle empty set plus another set
            return self.__class__(G=None, c=Q_arr)
        else:
            return self.__class__(G=self.G, c=self.c + Q_arr, Ae=self.Ae, be=self.be)


def minus(
    self: ConstrainedZonotope,
    Q: "Sequence[float] | np.ndarray | Ellipsoid | ConstrainedZonotope",
    enable_warning: bool = True,
) -> ConstrainedZonotope:
    r"""Implement - operation: Pontryagin difference with a constrained zonotope minuend

    When Q is an ellipsoid or a zonotope, minus returns an inner-approximation of the set corresponding to the
    Pontryagin difference of a constrained zonotope and Q. When Q is a point or a singleton set, an exact set
    corresponding to the translation by -Q or -Q.c is returned.

    Args:
        Q (Sequence[float] | numpy.ndarray | Ellipsoid | ConstrainedZonotope): Point/set to use as subtrahend in the
            Pontryagin difference.
        enable_warning (bool, optional): Enables the UserWarning. May be turned off if expected. Defaults to True.

    Raises:
        TypeError: When Q is not one of the following --- convertible into a 1D numpy array, or an ellipsoid, or a
            zonotope.
        ValueError: When Q has a dimension mismatch with self.
        UserWarning: When using with Q that is a set, warn that an inner-approximation is returned.

    Returns:
        ConstrainedZonotope: Pontryagin difference of self and Q

    Notes:
        This function uses :meth:`approximate_pontryagin_difference` when Q is a set and uses :meth:`plus` when Q is a
        point.
    """
    if is_ellipsoid(Q):
        Q = cast("Ellipsoid", Q)
        if Q.is_singleton:
            return self.plus(-Q.c)
        elif enable_warning:
            warnings.warn(
                "This function returns an inner-approximation of the Pontryagin difference with an ellipsoid.",
                UserWarning,
            )
        G_S, c_S = Q.G, Q.c
        p = 2
    elif is_constrained_zonotope(Q):
        Q = cast("ConstrainedZonotope", Q)
        if Q.n_equalities:
            raise TypeError("Unsupported operation ConstrainedZonotope - (ConstrainedZonotope with n_equalities>0)!")
        elif Q.c is None:
            # Pontryagin difference is the same as the original set since the subtrahend is empty.
            return self.copy()
        elif Q.is_singleton:
            return self.plus(-Q.c)
        elif enable_warning:
            warnings.warn(
                "This function returns an inner-approximation of the Pontryagin difference with a zonotope.",
                UserWarning,
            )
        G_S, c_S = Q.G, Q.c
        p = "inf"
    else:
        try:
            Q = np.atleast_1d(Q).astype(float)
        except (TypeError, ValueError) as err:
            raise TypeError(f"Unsupported operation ConstrainedZonotope - {type(Q)}!") from err
        return self.plus(-cast(np.ndarray, Q))
    return self.approximate_pontryagin_difference(
        p, G_S, c_S, method="inner-least-squares", enable_warning=enable_warning
    )


##################
# Helper functions
##################
def get_first_two_blocks_for_Ae_be(self: ConstrainedZonotope, Y: ConstrainedZonotope) -> tuple[np.ndarray, np.ndarray]:
    if self.n_equalities and Y.n_equalities:
        newobj_Ae = np.block(
            [
                [self.Ae, np.zeros((self.n_equalities, Y.latent_dim))],
                [np.zeros((Y.n_equalities, self.latent_dim)), Y.Ae],
            ]
        )
        newobj_be = np.hstack((self.be, Y.be))
    elif self.n_equalities == 0 and Y.n_equalities == 0:
        new_latent_dim = self.latent_dim + Y.latent_dim
        newobj_Ae = np.empty((0, new_latent_dim))
        newobj_be = np.empty((0,))
    elif self.n_equalities and Y.n_equalities == 0:
        newobj_Ae = np.hstack((self.Ae, np.zeros((self.n_equalities, Y.latent_dim))))
        newobj_be = self.be
    else:
        newobj_Ae = np.hstack((np.zeros((Y.n_equalities, self.latent_dim)), Y.Ae))
        newobj_be = Y.be
    return newobj_Ae, newobj_be
