# lrdbenchmark Demonstration Notebooks

The tutorial curriculum that originally shipped as Jupyter notebooks is now stored as Markdown in this directory. Each notebook mirrors the narrative tutorials published in `docs/tutorials/` and walks through the full workflow: synthetic data generation, estimator evaluation, benchmarking, robustness diagnostics, and leaderboard creation.

## Contents

The Markdown notebooks live in `markdown/` and follow a progressive structure:

1. **01_data_generation_and_visualisation.md** – generating and inspecting stochastic LRD processes (FBM, FGN, ARFIMA, MRW, α-stable).
2. **02_estimation_and_validation.md** – exercising the twenty estimators (13 classical, 3 ML, 4 neural) with statistical validation.
3. **03_custom_models_and_estimators.md** – extending lrdbenchmark with bespoke data models and estimators.
4. **04_comprehensive_benchmarking.md** – running the benchmark driver, contamination panels, and diagnostics.
5. **05_leaderboard_generation.md** – building stratified leaderboards and publication-ready summaries.

Each notebook is accompanied by the image assets it produces (saved to `*_files/`) so that documentation renders deterministically.

## Opening the notebooks interactively

The Markdown format keeps diffs tidy, yet you can still work interactively by converting back to `.ipynb`:

```bash
pip install jupytext jupyter matplotlib seaborn

# Example: convert the first notebook back to ipynb
jupytext --to notebook notebooks/markdown/01_data_generation_and_visualisation.md

# Launch Jupyter in the markdown directory
jupyter notebook notebooks/markdown/
```

Editors such as VS Code, JupyterLab, and Obsidian can also render the Markdown files directly.

## What the notebooks demonstrate

- **Data generation** – five stochastic families with configurable Hurst parameters and validation plots.
- **Estimation & validation** – twenty estimators with confidence intervals, bootstrap diagnostics, and bias analysis.
- **Extensibility** – guidance for subclassing the data-model and estimator APIs.
- **Benchmarking** – classical/ML/neural benchmarks, contamination stress tests, stratified reporting, significance testing.
- **Leaderboards** – composite scoring, robustness panels, and export utilities (CSV/JSON/LaTeX/Markdown).

Outputs (figures, CSV tables, LaTeX snippets) are written to `outputs/` just as in the original notebooks, ensuring reproducibility between the Markdown and HTML documentation views.

## Tips

- Use the `quick` runtime profile during interactive experimentation for faster turnaround (`export LRDBENCHMARK_RUNTIME_PROFILE=quick`).
- For long sessions or GPU work, enable the desired acceleration extras when installing lrdbenchmark.
- Cross-reference the documentation tutorials if you prefer a narrative that embeds the same code snippets.

---

Made with care for the time-series analysis community. If you enhance a notebook or port it to another format, please consider opening a pull request. 
