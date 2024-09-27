# Copyright (C) 2020-2024 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later

# Code purpose: Run example notebooks, update docs, and add copyright to the generated HTML pages.
# Note: This script must be run at the root of the repository

# Start time
start=`date +%s`

# REUSE-IgnoreStart
COPYRIGHT_TEXT="<!--\n\
Copyright (C) 2020-2024 Mitsubishi Electric Research Laboratories (MERL)\n\
SPDX-License-Identifier: AGPL-3.0-or-later\n\
-->"
# REUSE-IgnoreEnd

# Escape special characters for sed
ESCAPED_COPYRIGHT_TEXT=$(printf '%s\n' "$COPYRIGHT_TEXT" | sed 's/[\/&]/\\&/g')

# List of notebooks to process
for notebook in examples/*.ipynb; do
    # Get the filename without the extension
    filename=$(basename -- "$notebook")
    filename="${filename%.*}"

    # Black the notebook
    # Chose 75 characters as the line length to match the width of the HTML page
    nbqa black examples/${filename}.ipynb --line-length=75
    echo "Black the notebook ${filename}.ipynb!"

    # Execute the notebook and export it to html
    jupyter nbconvert --execute --to html examples/${filename}.ipynb
    echo "Converted the notebook ${filename}.ipynb into HTML ${filename}.html!"

    # Clear the cells in the notebook
    jupyter nbconvert --clear-output --inplace examples/${filename}.ipynb
    echo "Cleaned the notebook ${filename}.ipynb for git!"

    # Move the tutorial
    mv examples/${filename}.html docs/source/_static/${filename}.html
    echo "Moved the HTML ${filename}.html into docs!"

    # Insert the copyright text at the top of the HTML file
    sed -i "1i $ESCAPED_COPYRIGHT_TEXT" docs/source/_static/${filename}.html
    echo "Added the copyright info to HTML ${filename}.html!"
done

# End time
end=`date +%s`
runtime=$((end-start))
echo "Example notebooks to HTML generation took $runtime seconds!"
