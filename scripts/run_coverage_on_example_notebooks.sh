# Copyright (C) 2020-2026 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

# Code purpose: Run coverage on example notebooks.
# Note: This script must be run at the root of the repository

# List of notebooks to process
relevant_notebooks=(examples/ConstrainedZonotope.ipynb, examples/Polytope.ipynb)

echo "Remove tmp directory if it exists"
rm -r examples/tmp

for notebook in ${relevant_notebooks[@]}; do
    # Get the filename without the extension
    filename=$(basename -- "$notebook")
    filename="${filename%.*}"

    # Execute the notebook and export it to html
    jupyter nbconvert --to script examples/${filename}.ipynb
    echo "Converted the notebook ${filename}.ipynb into script ${filename}.py!"

    # Remove magic line
    sed -i '/get_ipython/d' examples/${filename}.py

    # Run coverage
    coverage run --source=pycvxset/${filename} examples/${filename}.py > /dev/null
    echo "Completed the coverage check on the script ${filename}.ipy!"

    # Remove the python file
    rm examples/${filename}.py

    # Generate coverage
    coverage report --skip-covered
    coverage html

    # Move the results of the coverage to docs
    mkdir -p docs/source/_static/codecoverage
    rm -rf docs/source/_static/codecoverage/${filename}_notebook
    mv htmlcov docs/source/_static/codecoverage/${filename}_notebook
done
