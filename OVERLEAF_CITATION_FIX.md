# Overleaf Citation Fix Guide

## Issues Fixed in manuscript.tex

### 1. Package Order
- Moved `natbib` before `hyperref` to avoid conflicts
- Added `\bibliographystyle{plainnat}` in the preamble
- Removed duplicate bibliography style declaration

### 2. Package Conflicts
- Added proper `\hypersetup` configuration
- Added `filecontents` and `etoolbox` packages for Overleaf compatibility

### 3. Bibliography Setup
- Single `\bibliography{references}` command at the end
- Proper `plainnat` style declaration

## Steps to Fix Citations in Overleaf

### Step 1: Upload Files
1. Upload `manuscript.tex` to your Overleaf project
2. Upload `references.bib` to the same project
3. Make sure both files are in the root directory

### Step 2: Compile
1. Click "Compile" in Overleaf
2. If citations don't appear, try "Recompile from scratch"
3. Check the "Logs and output files" for any errors

### Step 3: Check for Common Issues

#### Issue: "Undefined citation" warnings
- **Solution**: Ensure all citation keys in `\citep{}` match exactly with keys in `references.bib`
- **Check**: No typos in citation keys

#### Issue: Bibliography not appearing
- **Solution**: Make sure `references.bib` is uploaded and in the same directory
- **Check**: File name is exactly `references.bib` (case-sensitive)

#### Issue: Citations appear as [?]
- **Solution**: This usually means the bibliography wasn't processed
- **Fix**: Try "Recompile from scratch" or check for compilation errors

### Step 4: Alternative Bibliography Styles

If `plainnat` doesn't work, try these alternatives:

```latex
% Option 1: Plain style
\bibliographystyle{plain}

% Option 2: Alpha style  
\bibliographystyle{alpha}

% Option 3: Abbrv style
\bibliographystyle{abbrv}
```

### Step 5: Test with Minimal Example

If issues persist, test with this minimal example:

```latex
\documentclass{article}
\usepackage{natbib}
\bibliographystyle{plainnat}

\begin{document}
This is a test citation \citep{mandelbrot1968}.

\bibliography{references}
\end{document}
```

## Expected Output

After successful compilation, citations should appear as:
- **In-text**: (Mandelbrot and Van Ness, 1968)
- **Bibliography**: Full reference list at the end

## Troubleshooting Checklist

- [ ] `references.bib` file uploaded to Overleaf
- [ ] File names match exactly (case-sensitive)
- [ ] No compilation errors in the log
- [ ] Bibliography style declared before `\begin{document}`
- [ ] `\bibliography{references}` command at the end
- [ ] All citation keys exist in `references.bib`

## Contact Overleaf Support

If issues persist:
1. Check Overleaf's documentation on bibliographies
2. Use Overleaf's support chat
3. Share the compilation log for specific error messages

The manuscript.tex file has been updated with the correct package configuration for Overleaf!
