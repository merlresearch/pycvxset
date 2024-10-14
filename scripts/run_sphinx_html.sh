# Copyright (C) 2020-2024 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

# Code purpose: Run sphinx-based auto-documentation.
# Note: This script must be run at the root of the repository


# Update docs
(
  cd docs || exit
  make clean
  rm -r source/api_summary
  sphinx-build -b html source build
  make html
)
echo "The coverage and tutorials files have been overwritten."
python3 examples/pycvxset_diag.py --save_plot --do_not_use_plot_show
echo "Ran pycvxset_diag with save plots"
