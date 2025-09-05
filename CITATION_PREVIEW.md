# Citation Preview - How Your Citations Should Appear

## Current Citation Format in manuscript.tex

Your manuscript uses the correct LaTeX citation format:

```latex
\citep{mandelbrot1968, beran1994}
\citep{cont2001}
\citep{willinger1995}
\citep{ivanov1999}
\citep{pelletier2001}
```

## How They Should Appear After Compilation

### In the Text:
- Long-range dependence (LRD), characterized by the Hurst parameter H, is a fundamental property of time series that quantifies the persistence of temporal correlations over extended time scales **(Mandelbrot and Van Ness, 1968; Beran, 1994)**.

- This phenomenon is ubiquitous across scientific domains, from financial markets **(Cont, 2001)** and network traffic analysis **(Willinger et al., 1995)** to physiological signals **(Ivanov et al., 1999)** and climate data **(Pelletier and Turcotte, 2001)**.

### In the Bibliography Section:
```
References

Abry, P. and Veitch, D. (2000). Wavelet analysis of long-range-dependent traffic. 
IEEE Transactions on Information Theory, 44(1):2–15.

Alessio, E., Carbone, A., Castelli, G., and Frappietro, V. (2002). Second-order 
moving average and scaling of stochastic time series. The European Physical 
Journal B-Condensed Matter and Complex Systems, 27(2):197–200.

Beran, J. (1994). Statistics for long-memory processes, volume 61. CRC press.

Cont, R. (2001). Empirical properties of asset returns: stylized facts and 
statistical issues. Quantitative Finance, 1(2):223–236.

Ivanov, P. C., Amaral, L. A. N., Goldberger, A. L., Havlin, S., Rosenblum, M. G., 
Struzik, Z. R., and Stanley, H. E. (1999). Multifractality in human heartbeat 
dynamics. Nature, 399(6735):461–465.

Mandelbrot, B. B. and Van Ness, J. W. (1968). Fractional brownian motions, 
fractional noises and applications. SIAM Review, 10(4):422–437.

Pelletier, J. D. and Turcotte, D. L. (2001). Long-range persistence in climate 
and the 0.5 power. Geophysical Research Letters, 28(3):423–426.

Willinger, W., Taqqu, M. S., Sherman, R., and Wilson, D. V. (1995). Self-similarity 
through high-variability: statistical analysis of ethernet lan traffic at the 
source level. ACM SIGCOMM Computer Communication Review, 25(4):100–113.
```

## Citation Style Details

Your manuscript uses the `natbib` package with `plainnat` bibliography style, which produces:

- **In-text citations**: (Author, Year) format
- **Multiple authors**: (Author1 and Author2, Year) or (Author1 et al., Year)
- **Multiple citations**: (Author1, Year1; Author2, Year2)
- **Bibliography**: Author, Year format with full details

## Why Citations Might Not Appear

1. **Incomplete compilation**: LaTeX requires multiple passes to resolve citations
2. **Missing bibtex step**: Bibliography must be generated with `bibtex`
3. **Missing LaTeX installation**: Need full LaTeX distribution
4. **File path issues**: Ensure `references.bib` is in the same directory

## Quick Fix

If you're using an online LaTeX compiler (like Overleaf):
1. Upload both `manuscript.tex` and `references.bib`
2. Compile the document
3. Citations should appear automatically

The citation format in your manuscript is correct - it just needs proper LaTeX compilation to render!
