# Copyright (C) 2020-2026 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

# Code purpose: Run all unit tests and update docs.
# Note: This script must be run at the root of the repository


echo "Running pycvxset_diag silently"
python examples/pycvxset_diag.py --do_not_use_plot_show

echo "Remove tmp directory if it exists"
rm -r tmp

echo "Running code coverage"
# -W error:UserWarning will error at any uncaught warnings
coverage run --source=pycvxset -m pytest tests -W error::UserWarning
coverage report --skip-covered
coverage html

# Move the results of the coverage to docs
echo "Move coverage results"
mkdir -p docs/source/_static/codecoverage
rm -rf docs/source/_static/codecoverage/overall
mv htmlcov docs/source/_static/codecoverage/overall
echo "Updated code coverage report at docs/source/_static/codecoverage/overall/index.html"

echo "pyright run started!"
pyright
echo "pyright run complete!"
