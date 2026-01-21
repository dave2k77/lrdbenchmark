Diagnostics API
===============

Power-law diagnostics, scale sensitivity analysis, and structural break detection.

PowerLawDiagnostics
-------------------

.. autoclass:: lrdbenchmark.analysis.diagnostics.PowerLawDiagnostics
   :members:
   :undoc-members:

ScaleWindowSensitivityAnalyser
------------------------------

.. autoclass:: lrdbenchmark.analysis.diagnostics.ScaleWindowSensitivityAnalyser
   :members:
   :undoc-members:

StructuralBreakDetector
-----------------------

Detect stationarity violations that invalidate classical estimator assumptions.

.. autoclass:: lrdbenchmark.analysis.diagnostics.StructuralBreakDetector
   :members:
   :undoc-members:

Example Usage
~~~~~~~~~~~~~

.. code-block:: python

   from lrdbenchmark.analysis.diagnostics import StructuralBreakDetector
   
   detector = StructuralBreakDetector(significance_level=0.05)
   
   # Run all tests
   result = detector.detect_all(data)
   
   if result['any_break_detected']:
       print("⚠️ Stationarity violated!")
       for warning in result['warnings']:
           print(f"  - {warning}")
   
   # Individual tests
   cusum_result = detector.cusum_test(data)
   chow_result = detector.chow_test(data, break_index=500)
   icss_result = detector.icss_algorithm(data)

Utility Functions
-----------------

.. autofunction:: lrdbenchmark.analysis.diagnostics.run_comprehensive_diagnostics
