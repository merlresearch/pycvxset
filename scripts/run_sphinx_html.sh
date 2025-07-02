# Copyright (C) 2020-2025 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

# Code purpose: Run sphinx-based auto-documentation.
# Note: This script must be run at the root of the repository

# Update docs
cd docs
make clean
if [ -z "${1}" ]; then
  echo "Performing a local html build"
else
  echo "Checking out git worktree"
  git worktree add -f build/html gh-pages
fi
rm -r source/api_summary
sphinx-build -b html source build
make html
cd ..

echo "The coverage and tutorials files have been overwritten."
python3 examples/pycvxset_diag.py --save_plot --do_not_use_plot_show
echo "Ran pycvxset_diag with save plots"

if [ -z "${1}" ]; then
  echo "Skipped deployment! Use ./scripts/run_sphinx_html.sh deploy to deploy the website!"
  echo "Run python -m http.server --directory ./docs/build/ to view the website at localhost:8000"
else
  rm -r docs/build/html/_static/codecoverage/overall
  echo "Remove code coverage for documentation update"

  cd docs/build/html
  git add -A
  git commit -m "Documentation update"
  git push origin gh-pages
fi
