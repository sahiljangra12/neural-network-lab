#!/bin/sh
# GitHub Pages serves this repo from main:/docs. Copy the site there before each commit.
cd "$(dirname "$0")/.." || exit 1
rm -rf docs && mkdir docs
cp -r index.html learn.html advanced.html css js img report docs/
touch docs/.nojekyll
echo "docs/ rebuilt: $(find docs -type f | wc -l) files"
