# Copyright (C) 2020-2025 Mitsubishi Electric Research Laboratories (MERL)
# Copyright (c) 2019 Tor Aksel N. Heirung
#
# SPDX-License-Identifier: AGPL-3.0-or-later
# SPDX-License-Identifier: MIT

from setuptools import setup

with open("README.md", "r") as fh:
    long_description = fh.read()

# numpy>=1.14 for rcond=None correct defaults from https://stackoverflow.com/a/44678023
# scipy>=1.3.0 for the reflective-simplex option in linprog()
# matplotlib>=3.8 for https://github.com/matplotlib/matplotlib/pull/25565
# cvxpy>=1.5.3 for gurobipy error fix https://github.com/cvxpy/cvxpy/issues/2493
INSTALL_REQUIRES = [
    "numpy>=1.14",
    "scipy>=1.3.0",
    "pycddlib>=3.0.0",
    "matplotlib>=3.8",
    "cvxpy>=1.5.3",
    "gurobipy>=11.0.0",
]
TESTS_REQUIRES = ["pytest", "coverage", "lark"]
DOCS_REQUIRES = ["sphinx", "notebook", "ipykernel", "nbconvert", "ipympl", "myst-parser", "nbqa", "black"]

setup(
    name="pycvxset",
    version="1.1.0",
    description=("A Python package for manipulation and visualization of convex sets.")[0],
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/merlresearch/pycvxset",
    author="Abraham P. Vinod",
    author_email="vinod@merl.com, abraham.p.vinod@ieee.org",
    license="AGPL-3.0-or-later",
    packages=["pycvxset"],
    install_requires=INSTALL_REQUIRES,
    extras_require={
        "with_tests": TESTS_REQUIRES,
        "with_docs_and_tests": [*TESTS_REQUIRES, *DOCS_REQUIRES],
    },
    python_requires=">=3.9",  # for matplotlib>=3.8
    zip_safe=False,
)
