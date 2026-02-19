# Copyright (C) 2020-2026 Mitsubishi Electric Research Laboratories (MERL)
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

  ##### Make hard-coded edits

  ## sphinxmanual.cls
  # Make author information right justified
  sed -i '67s/.*/        \\begin{tabular}[t]{r}/' sphinxmanual.cls

  ## pycvxset.tex
  # Comment out the README TOC (Entire two-staged itemize)
  sed -i '91,148 s/^/%/' pycvxset.tex
  # Replace {docs/source/_static/pycvxset_diag}.png with pycvxset_diag.png
  cp ../source/_static/pycvxset_diag.png .
  sed -i 's/\\sphinxincludegraphics{{docs\/source\/_static\/pycvxset_diag}.png}//g' pycvxset.tex

  #### After edits to latex, now build
  latexmk -pdf pycvxset.tex
)
cp docs/build/pycvxset.pdf MANUAL.pdf
