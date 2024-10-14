# Copyright (C) 2020-2024 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

# Code purpose: Run sphinx-based auto-documentation to generate MANUAL.pdf
# Note: This script must be run at the root of the repository


# Update docs
(
  cd docs || exit
  # Remove all untracked and ignored files and directories
  make clean
  rm -r source/api_summary

  # Build the tex file
  sphinx-build -b latex source build
  cd build

  ##### Make hard-coded eits
  ## sphinxmanual.cls
  # Make author information right justified
  sed -i '67s/.*/        \\begin{tabular}[t]{r}/' sphinxmanual.cls
  ## pycvxset.tex
  # Comment out the README TOC
  sed -i '87,150 s/^/%/' pycvxset.tex
  # Comment out the sanity check png included for README
  sed -i '340 s/^/%/' pycvxset.tex

  #### After edits to latex, now build
  latexmk pycvxset.tex
)
cp docs/build/pycvxset.pdf MANUAL.pdf
