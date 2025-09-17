# LaTeX Manuscript Compilation Guide

## Issue: Citations Not Appearing

The citations in your manuscript are properly formatted but require the correct LaTeX compilation process to appear correctly.

## Required LaTeX Packages

Make sure you have the following packages installed:
- `texlive-full` or `texlive-latex-extra`
- `texlive-bibtex-extra`
- `pdflatex`
- `bibtex`

## Installation (Ubuntu/Debian)

```bash
sudo apt-get update
sudo apt-get install texlive-full
```

## Compilation Process

The citations require a multi-step compilation process:

### Method 1: Using the provided script

```bash
./compile_manuscript.sh
```

### Method 2: Manual compilation

```bash
# Step 1: First compilation
pdflatex manuscript.tex

# Step 2: Generate bibliography
bibtex manuscript

# Step 3: Second compilation (includes bibliography)
pdflatex manuscript.tex

# Step 4: Third compilation (resolves cross-references)
pdflatex manuscript.tex
```

## Expected Citation Format

After proper compilation, your citations should appear as:

- **In-text citations**: (Mandelbrot and Van Ness, 1968; Beran, 1994)
- **Bibliography**: Full reference list at the end of the document

## Troubleshooting

### If citations still don't appear:

1. **Check bibliography file**: Ensure `references.bib` is in the same directory
2. **Check citation keys**: Verify all `\citep{}` commands match keys in `references.bib`
3. **Check bibliography style**: The manuscript uses `plainnat` style
4. **Check for errors**: Look for error messages during compilation

### Common Issues:

- **Missing bibliography**: Run `bibtex manuscript` after first compilation
- **Missing references**: Check that all cited keys exist in `references.bib`
- **Style issues**: Ensure `natbib` package is loaded (it is in your manuscript)

## Alternative: Online LaTeX Compilers

If you don't want to install LaTeX locally, you can use online compilers:

1. **Overleaf** (https://www.overleaf.com)
   - Upload `manuscript.tex` and `references.bib`
   - Compile online with full bibliography support

2. **ShareLaTeX** (now part of Overleaf)
   - Similar functionality to Overleaf

## Verification

After compilation, check that:
- Citations appear as (Author, Year) format
- Bibliography section is included at the end
- All cited references appear in the bibliography
- No "undefined citation" warnings

## File Structure

Your manuscript should have:
```
manuscript.tex          # Main document
references.bib          # Bibliography database
manuscript.pdf          # Compiled output (after compilation)
```

The citations are correctly formatted in your manuscript - they just need proper LaTeX compilation to render correctly!
