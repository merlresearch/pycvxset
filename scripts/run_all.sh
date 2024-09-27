# Copyright (C) 2020-2024 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

# Code purpose: Run all scripts to prepare for version release.
# Note: This script must be run at the root of the repository


# Remove all untracked and ignored files and directories
echo "Cleaning up untracked and ignored files for fresh build..."
git clean -fdX

# Run example notebooks
./scripts/run_example_notebooks_and_update_docs.sh

# Run tests
./scripts/run_tests_and_update_docs.sh

# Run coverage from notebooks
./scripts/run_coverage_on_example_notebooks.sh

# Update docs
./scripts/run_sphinx_docs.sh

if [ -z "${1}" ]; then
    echo "Skipped cleaning after! Use ./scripts/run_all.sh clean_after to clean all files after scripts!"
else
    # Remove all untracked and ignored files and directories
    echo "Cleaning up untracked and ignored files before release..."
    git clean -fdX
fi
