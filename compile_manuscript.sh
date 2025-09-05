#!/bin/bash

# Compile LaTeX manuscript with bibliography
# This script runs the necessary commands to properly compile the manuscript
# with citations and bibliography

echo "Compiling LaTeX manuscript with bibliography..."

# First compilation
echo "Step 1: First LaTeX compilation..."
pdflatex manuscript.tex

# Generate bibliography
echo "Step 2: Generating bibliography..."
bibtex manuscript

# Second compilation (to include bibliography references)
echo "Step 3: Second LaTeX compilation..."
pdflatex manuscript.tex

# Third compilation (to resolve all cross-references)
echo "Step 4: Third LaTeX compilation..."
pdflatex manuscript.tex

echo "Compilation complete! Check manuscript.pdf for the final output."

# Clean up auxiliary files (optional)
echo "Cleaning up auxiliary files..."
rm -f manuscript.aux manuscript.bbl manuscript.blg manuscript.log manuscript.out manuscript.toc

echo "Done!"
