<!--
Copyright (C) 2020-2024 Mitsubishi Electric Research Laboratories (MERL)
Copyright (c) 2019 Tor Aksel N. Heirung

SPDX-License-Identifier: AGPL-3.0-or-later
SPDX-License-Identifier: MIT
-->

# pycvxset: Convex sets in Python

- [pycvxset: Convex sets in Python](#pycvxset-convex-sets-in-python)
  - [What can pycvxset do for me?](#what-can-pycvxset-do-for-me)
  - [Quick start](#quick-start)
    - [Requirements](#requirements)
    - [Installation](#installation)
      - [Sanity check](#sanity-check)
    - [Optional: Testing](#optional-testing)
    - [Optional: Documentation and testing](#optional-documentation-and-testing)
  - [Getting help](#getting-help)
  - [Contributing](#contributing)
  - [License](#license)
  - [Acknowledgements](#acknowledgements)
  - [Contact](#contact)

## What can pycvxset do for me?

pycvxset is a Python package for manipulation and visualization of convex sets.

Currently, pycvxset supports the following three representations:
1. **polytopes**,
2. **ellipsoids**, and
3. **constrained zonotopes** (which are equivalent to polytopes).

Some of the operations enabled by pycvxset include:
* construct sets from a variety of representations and transform between these representations,
* perform various operations on these sets including (but not limited to):
  * plot in 2D and 3D,
  * affine and inverse-affine transformation,
  * Minkowski sum,
  * Pontryagin difference,
  * intersection,
  * checking for containment,
  * projection,
  * slicing, and
  * support function evaluation.

> See the Jupyter notebooks in [examples](./examples) folder for more details on how pycvxset can be used in set-based control and perform reachability analysis.


## Quick start

### Requirements

pycvxset supports Python 3.9+ on Ubuntu, Windows, and MacOS. As described in [setup.py](./setup.py), pycvxset has the following core dependencies:
1. [numpy](https://numpy.org/)
2. [scipy](https://scipy.org/)
3. [cvxpy](https://www.cvxpy.org/)
4. [matplotlib](https://matplotlib.org/)
5. [pycddlib](https://pycddlib.readthedocs.io/en/latest/)
6. [gurobipy](https://pypi.org/project/gurobipy/): This dependency is **optional**. Almost all functionalities of pycvxset are available without [Gurobi](https://www.gurobi.com/). However, pycvxset uses [Gurobi](https://www.gurobi.com/) (through [cvxpy](https://www.cvxpy.org/)) to perform *some* containment and equality checks involving constrained zonotopes. See [License](#license) section for more details.

### Installation

Refer to [.github/workflows](.github/workflows) for exact steps to install pycvxset for different OSes. These steps are summarized below:
1. OS-dependent pre-installation steps:
   * **Ubuntu**: Install [gmp](https://gmplib.org/).
     ```
     $ sudo apt-get install libgmp-dev
     ```
   * **MacOS**: Install [pycddlib](https://github.com/mcmtroffaes/pycddlib/) manually in order to link with [gmp](https://gmplib.org/).
     ```
     % brew install gmp
     % python3 -m pip install --upgrade pip
     % git clone https://github.com/mcmtroffaes/pycddlib.git
     % cd pycddlib
     % git checkout master
     % git submodule update --init
     % ./cddlib-makefile-gmp.sh
     % env "CFLAGS=-I$(brew --prefix)/include -L$(brew --prefix)/lib" python3 -m pip install .
     % cd ..
     ```
     These steps are adapted from [pycddlib/build.yml](https://github.com/mcmtroffaes/pycddlib/blob/master/.github/workflows/build.yml).
   * **Windows**: No special steps required since pip takes care of it. If plotting fails, you can set matplotlib backend via an environment variable `set MPLBACKEND=Agg`.
     See [https://matplotlib.org/3.5.1/users/explain/backends.html#selecting-a-backend](https://matplotlib.org/3.5.1/users/explain/backends.html#selecting-a-backend) for more details.
2. Clone the pycvxset repository into a desired folder `PYCVXSET_DIR`.
3. Run `pip install -e .` in the folder `PYCVXSET_DIR`.

#### Sanity check
Check your installation by running  `python3 examples/pycvxset_diag.py` in the folder `PYCVXSET_DIR`. The script should generate a figure with two subplots, each generated using pycvxset.
  - *Left subplot*: A 3D plot of two polytopes (one at the origin, and the other translated and rotated). You can interact with this plot using your mouse.
  - *Right subplot*: A 2D plot of the projection of the polytope, and its corresponding minimum volume circumscribing and maximum volume inscribing ellipsoids.

![](./docs/source/_static/pycvxset_diag.png)
![](_static/pycvxset_diag.png)

### Optional: Testing

1. Use `pip install -e ".[with_tests]"` to install the additional dependencies.
2. Run `$ ./scripts/run_tests_and_update_docs.sh` to view testing results on the command window.
3. Open [./docs/source/_static/codecoverage/overall/index.html](./docs/source/_static/codecoverage/overall/index.html) in your browser for coverage results.

### Optional: Documentation and testing

1. Use `pip install -e ".[with_docs_and_tests]"` to install the additional dependencies.
2. Run `$./scripts/run_all.sh`. This will take about 5 minutes.
     * **Faster but incomplete option:** Run only `$./scripts/run_sphinx_docs.sh` to build the API documentation without rendering the tutorial notebooks or performing coverage.
3. In your browser, view
      * API documentation at [./docs/build/index.html](./docs/build/index.html).
      * View the Jupyter notebooks at [./docs/build/tutorials/tutorials.html](./docs/build/tutorials/tutorials.html).
      * View code coverage from testing at [./docs/build/_static/codecoverage/overall/index.html](./docs/build/tutorials/tutorials.html).

## Getting help

Let us say that you have read the available [Documentation](https://pycvxset.readthedocs.io/en/latest/) and searched the
[Discussion](https://github.com/merlresearch/pycvxset/discussions) and the
[Issue](https://github.com/merlresearch/pycvxset/issues) webpages, but still need help.

**Please start a [Discussion](https://github.com/merlresearch/pycvxset/discussions)** for...

- Getting help in setting up pycvxset to work on your computer.
- Advice on how to use pycvxset for your set manipulations problem.
- Suggestions on documentation of the code.
- Collecting ideas for potential new features/enhancements from the community.

**Please open an [Issue](https://github.com/merlresearch/pycvxset/issues)** if you believe that fixing your problem will involve a change in the pycvxset source code. For example...

- Bug reports.
- New feature/enhancement proposals with details.

**How to submit a good bug report?** When opening an [Issue](https://github.com/merlresearch/pycvxset/issues), please consider providing:

- High-level description of the behavior you would expect and the actual behavior.
- Which version of pycvxset are you using? The bleeding edge or the version number.
- OS (Windows, Linux, Mac, or something else)
- Can you reliably reproduce the issue?
- Details of the bug (a stack trace can be useful).
- A reduced test case that reproduces the issue for our development team.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for our policy on contributions.

## License

pycvxset is released under `AGPL-3.0-or-later` license, as found in the [LICENSE.md](LICENSE.md) file.

All files:
```
Copyright (c) 2020-2024 Mitsubishi Electric Research Laboratories (MERL).

SPDX-License-Identifier: AGPL-3.0-or-later
```
except the following files:
1. .gitignore
1. README.md
1. pycvxset/Polytope/\_\_init\_\_.py
1. pycvxset/Polytope/operations_binary.py
1. pycvxset/Polytope/operations_unary.py
1. pycvxset/Polytope/plotting_scripts.py
1. pycvxset/Polytope/vertex_halfspace_enumeration.py
1. pycvxset/\_\_init\_\_.py
1. pycvxset/common/\_\_init\_\_.py
1. setup.py
1. tests/test\_polytope\_binary.py
1. tests/test\_polytope\_init.py
1. tests/test\_polytope\_vertex_facet_enum.py

which have the copyright
```
Copyright (C) 2020-2024 Mitsubishi Electric Research Laboratories (MERL)
Copyright (c) 2019 Tor Aksel N. Heirung

SPDX-License-Identifier: AGPL-3.0-or-later
SPDX-License-Identifier: MIT
```

The method `contains` in [pycvxset/ConstrainedZonotope/operations_binary.py](./pycvxset/ConstrainedZonotope/operations_binary.py) uses [gurobipy](https://pypi.org/project/gurobipy/) (through [cvxpy](cvxpy.org)) and requires acceptance of appropriate license terms.

## Acknowledgements

The development of pycvxset started from commit
[ebd85404](https://github.com/heirung/pytope/commit/ebd85404ba235e8223fca1f6ba8817decccc4797) of
[pytope](https://github.com/heirung/pytope.git).
```
Copyright (c) 2019 Tor Aksel N. Heirung

SPDX-License-Identifier: MIT
```

pycvxset extends [pytope](https://github.com/heirung/pytope.git) in several new directions, including:
- Plotting for 3D polytopes using [matplotlib](https://matplotlib.org/stable/gallery/mplot3d/index.html),
- Interface with [cvxpy](https://www.cvxpy.org/) for access to a wider range of solvers,
- Introduce new methods for *Polytope* class including circumscribing and inscribing ellipsoids, volume computation, and containment checks,
- Minimize the use of vertex-halfspace enumeration using convex optimization,
- Include extensive documentation and example Jupyter notebooks,
- Implement exhaustive testing with nearly 100% coverage, and
- Support for *Ellipsoid* and *ConstrainedZonotope* set representations.

## Contact

For questions or bugs, contact Abraham P. Vinod (Email: [vinod@mail.com](mailto:vinod@mail.com) or [abraham.p.vinod@ieee.org](mailto:abraham.p.vinod@ieee.org)).
