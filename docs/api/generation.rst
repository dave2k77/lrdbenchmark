Generation Module API
=====================

This module provides data generators for testing LRD estimators under various conditions.

Time Series Generator
---------------------

.. autoclass:: lrdbenchmark.generation.TimeSeriesGenerator
   :members:
   :undoc-members:
   :show-inheritance:

Nonstationarity Generators
--------------------------

Generators for time-varying Hurst parameter testing.

RegimeSwitchingProcess
~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: lrdbenchmark.generation.RegimeSwitchingProcess
   :members:
   :undoc-members:

ContinuousDriftProcess
~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: lrdbenchmark.generation.ContinuousDriftProcess
   :members:
   :undoc-members:

StructuralBreakProcess
~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: lrdbenchmark.generation.StructuralBreakProcess
   :members:
   :undoc-members:

EnsembleTimeAverageProcess
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: lrdbenchmark.generation.EnsembleTimeAverageProcess
   :members:
   :undoc-members:

Critical Regime Models
----------------------

Physics-motivated generators for critical and nonequilibrium regimes.

OrnsteinUhlenbeckProcess
~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: lrdbenchmark.generation.OrnsteinUhlenbeckProcess
   :members:
   :undoc-members:

SubordinatedProcess
~~~~~~~~~~~~~~~~~~~

.. autoclass:: lrdbenchmark.generation.SubordinatedProcess
   :members:
   :undoc-members:

FractionalLevyMotion
~~~~~~~~~~~~~~~~~~~~

.. autoclass:: lrdbenchmark.generation.FractionalLevyMotion
   :members:
   :undoc-members:

SOCAvalancheModel
~~~~~~~~~~~~~~~~~

.. autoclass:: lrdbenchmark.generation.SOCAvalancheModel
   :members:
   :undoc-members:

Surrogate Data Generators
-------------------------

Generators for hypothesis testing of LRD and nonlinearity.

IAFFTSurrogate
~~~~~~~~~~~~~~

.. autoclass:: lrdbenchmark.generation.IAFFTSurrogate
   :members:
   :undoc-members:

PhaseRandomizedSurrogate
~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: lrdbenchmark.generation.PhaseRandomizedSurrogate
   :members:
   :undoc-members:

ARSurrogate
~~~~~~~~~~~

.. autoclass:: lrdbenchmark.generation.ARSurrogate
   :members:
   :undoc-members:

Factory Functions
-----------------

.. autofunction:: lrdbenchmark.generation.create_nonstationary_process

.. autofunction:: lrdbenchmark.generation.create_critical_regime_process

.. autofunction:: lrdbenchmark.generation.create_surrogate_generator
