Uncertainty Quantification API
==============================

Confidence interval estimation and coverage probability analysis.

UncertaintyQuantifier
---------------------

.. autoclass:: lrdbenchmark.analysis.uncertainty.UncertaintyQuantifier
   :members:
   :undoc-members:

Supported Methods
~~~~~~~~~~~~~~~~~

* **Block Bootstrap**: Moving-block bootstrap for dependent time series
* **Wavelet Bootstrap**: Wavelet-domain resampling preserving scale-wise energy
* **Parametric Monte Carlo**: Simulation from known data model
* **Studentized Bootstrap**: Bias-corrected intervals with t-distribution CIs

CoverageAnalyzer
----------------

Monte Carlo estimation of confidence interval coverage probabilities.

.. autoclass:: lrdbenchmark.analysis.uncertainty.CoverageAnalyzer
   :members:
   :undoc-members:

CoverageResult
--------------

.. autoclass:: lrdbenchmark.analysis.uncertainty.CoverageResult
   :members:
   :undoc-members:

Example Usage
~~~~~~~~~~~~~

.. code-block:: python

   from lrdbenchmark.analysis.uncertainty import UncertaintyQuantifier, CoverageAnalyzer
   
   # Compute confidence intervals
   uq = UncertaintyQuantifier(confidence_level=0.95)
   intervals = uq.compute_intervals(
       estimator=dfa_estimator,
       data=signal,
       base_result=estimation_result,
       true_value=0.7
   )
   
   print(f"Block bootstrap CI: {intervals['block_bootstrap']['confidence_interval']}")
   print(f"Studentized CI: {intervals['studentized_bootstrap']['confidence_interval']}")
   
   # Analyze coverage probability
   analyzer = CoverageAnalyzer(n_trials=200)
   coverage = analyzer.analyze_estimator_coverage(
       estimator_cls=DFAEstimator,
       data_model_cls=FBMModel,
       true_H=0.7,
       length=1000
   )
   
   for method, result in coverage.items():
       print(f"{method}: {result.empirical_coverage:.1%} coverage")

Utility Functions
-----------------

.. autofunction:: lrdbenchmark.analysis.uncertainty.run_comprehensive_coverage_analysis
