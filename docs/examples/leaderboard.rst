.. _leaderboard_significance:

Leaderboard Significance Analysis
=================================

This example explains how the leaderboard workflow now embeds
non-parametric significance testing in line with Brigato et al.'s call
for statistically defensible comparisons.

Workflow Summary
----------------

* Run the benchmark sweeps with :class:`lrdbenchmark.analysis.benchmark.ComprehensiveBenchmark`.
* Collect leaderboard tables from the generated CSV or directly inside the
  ``notebooks/markdown/05_leaderboard_generation.md`` notebook (or the
  companion tutorial :doc:`/tutorials/tutorial_05_leaderboards`).
* Inspect the ``significance_analysis`` field returned with each benchmarking
  summary to obtain omnibus and post-hoc test statistics.
* Adjust the protocol in ``config/benchmark_protocol.yaml`` to standardise
  preprocessing, scale-selection, and estimator overrides across runs.

Code Snippet
------------

.. code-block:: python

   from lrdbenchmark.analysis.benchmark import ComprehensiveBenchmark
   import pandas as pd

   benchmark = ComprehensiveBenchmark()
   results = benchmark.run_comprehensive_benchmark(
       data_length=1000,
       benchmark_type="comprehensive",
       save_results=False,
   )

   significance = results.get("significance_analysis", {})
   if significance.get("status") == "ok":
       friedman = significance["friedman"]
       print(
           f"Friedman χ²={friedman['statistic']:.4f} "
           f"(p={friedman['p_value']:.4f}) across "
           f"{friedman['n_data_models']} data models and "
           f"{friedman['n_estimators']} estimators"
       )

       mean_ranks = (
           pd.DataFrame(
               list(significance["mean_ranks"].items()),
               columns=["Estimator", "Mean Rank"],
           ).sort_values("Mean Rank")
       )
       print(mean_ranks.to_string(index=False))

       pairwise = pd.DataFrame(
           [
               {
                   "Estimator A": res["pair"][0],
                   "Estimator B": res["pair"][1],
                   "Holm p-value": res.get("holm_p_value"),
                   "Significant": res.get("significant"),
               }
               for res in significance["post_hoc"]
               if res.get("p_value") is not None
           ]
       )
       if not pairwise.empty:
           print("\nHolm-corrected Wilcoxon comparisons:")
           print(pairwise.sort_values("Holm p-value").to_string(index=False))
   else:
       print(significance.get("reason", "No significance analysis available."))

Output Interpretation
---------------------

* **Friedman χ², p-value** – ombuds the null hypothesis that all estimators
  perform equally across the assessed data models.
* **Mean Rank Table** – lower mean ranks indicate better aggregate performance.
* **Holm-corrected Wilcoxon** – highlights pairwise differences that remain
  statistically significant after controlling the family-wise error rate.
* **Coverage / CI width** – the benchmark automatically reports empirical
  coverage rates and mean 95% interval widths so calibration quality can be
  assessed alongside raw error.
* **Stratified metrics** – the ``stratified_metrics`` payload provides error,
  coverage, and confidence-width summaries across H bands, tail classes, data
  lengths, and contamination regimes to prevent regime averaging.
* **Robustness panels** – advanced benchmark runs embed scaling influence
  diagnostics and stress tests for missingness, regime shifts, seasonal drift,
  and burst noise so protocol choices are stress-tested alongside leaderboard
  scores.

Additional Stratified Reporting
-------------------------------

The JSON artefact saved by :class:`~lrdbenchmark.analysis.benchmark.ComprehensiveBenchmark`
now includes a ``stratified_metrics`` section. To produce a publishable summary:

.. code-block:: python

   from lrdbenchmark.analytics.dashboard import AnalyticsDashboard

   dashboard = AnalyticsDashboard()
   print(dashboard.generate_stratified_report("benchmark_results/comprehensive_benchmark_latest.json"))

The dashboard can also render dedicated figures showcasing scaling slopes and
robustness stress responses captured during advanced benchmarks:

.. code-block:: python

   dashboard.create_advanced_diagnostics_visuals(
       "benchmark_results/advanced_benchmark_latest.json",
       output_dir="benchmark_results/figures",
   )

Best Practices
--------------

* Ensure each estimator succeeds on all benchmarked data models; otherwise,
  the significance module drops incomplete rows to maintain valid paired
  tests.
* Record the provenance bundle saved alongside the JSON benchmark artefact
  so that statistical claims remain reproducible.
* Present the mean-rank and Holm-adjusted tables alongside error metrics in
  manuscripts to prevent "champion" narratives that rely on marginal,
  non-significant improvements.

