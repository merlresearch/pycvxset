# Copyright (C) 2020-2025 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

# Code purpose: Run all unit tests and update docs.
# Note: This script must be run at the root of the repository


echo "Running pycvxset_diag silently"
python examples/pycvxset_diag.py --do_not_use_plot_show

# -W error:UserWarning will error at any uncaught warnings
coverage run --rcfile=tests/.coveragerc --source=pycvxset -m pytest tests -W error::UserWarning
coverage report --skip-covered --rcfile=tests/.coveragerc
coverage html --rcfile=tests/.coveragerc

# Move the results of the coverage to docs
mkdir -p docs/source/_static/codecoverage
rm -rf docs/source/_static/codecoverage/overall
mv htmlcov docs/source/_static/codecoverage/overall
