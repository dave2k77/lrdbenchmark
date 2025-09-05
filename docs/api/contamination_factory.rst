Contamination Factory API
=========================

The Contamination Factory provides a comprehensive system for generating realistic confounding profiles and applying them to pure signals for testing estimator robustness under various real-world conditions.

Contamination Factory
---------------------

.. autoclass:: lrdbenchmark.models.contamination.contamination_factory.ContaminationFactory
   :members:
   :undoc-members:
   :show-inheritance:

Confounding Scenarios
---------------------

The contamination factory supports various domain-specific confounding scenarios:

.. autoclass:: lrdbenchmark.models.contamination.contamination_factory.ConfoundingScenario
   :members:
   :undoc-members:
   :show-inheritance:

Confounding Profiles
--------------------

.. autoclass:: lrdbenchmark.models.contamination.contamination_factory.ConfoundingProfile
   :members:
   :undoc-members:
   :show-inheritance:

Usage Examples
--------------

Basic Contamination Application
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from lrdbenchmark.models.contamination.contamination_factory import ContaminationFactory, ConfoundingScenario
   from lrdbenchmark.models.data_models.fbm.fbm_model import FBMModel
   import numpy as np
   
   # Create contamination factory
   factory = ContaminationFactory()
   
   # Generate pure data
   model = FBMModel(H=0.7, sigma=1.0)
   pure_data = model.generate(1000, seed=42)
   
   # Apply different contamination scenarios
   scenarios = [
       ConfoundingScenario.PURE,
       ConfoundingScenario.GAUSSIAN_NOISE,
       ConfoundingScenario.TREND,
       ConfoundingScenario.OUTLIERS
   ]
   
   for scenario in scenarios:
       contaminated_data, description = factory.apply_confounding(
           pure_data, scenario, intensity=0.2
       )
       print(f"{scenario.value}: {description}")

EEG Contamination Scenarios
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from lrdbenchmark.models.contamination.contamination_factory import ContaminationFactory, ConfoundingScenario
   from lrdbenchmark.models.data_models.fbm.fbm_model import FBMModel
   import numpy as np
   
   # Create contamination factory
   factory = ContaminationFactory()
   
   # Generate pure data
   model = FBMModel(H=0.7, sigma=1.0)
   pure_data = model.generate(1000, seed=42)
   
   # EEG-specific contamination scenarios
   eeg_scenarios = [
       ConfoundingScenario.EEG_OCULAR_ARTIFACTS,
       ConfoundingScenario.EEG_MUSCLE_ARTIFACTS,
       ConfoundingScenario.EEG_CARDIAC_ARTIFACTS,
       ConfoundingScenario.EEG_ELECTRODE_POPPING,
       ConfoundingScenario.EEG_ELECTRODE_DRIFT,
       ConfoundingScenario.EEG_60HZ_NOISE,
       ConfoundingScenario.EEG_SWEAT_ARTIFACTS,
       ConfoundingScenario.EEG_MOVEMENT_ARTIFACTS
   ]
   
   print("EEG Contamination Testing:")
   for scenario in eeg_scenarios:
       contaminated_data, description = factory.apply_confounding(
           pure_data, scenario, intensity=0.3
       )
       print(f"{scenario.value}: {description}")

Financial Contamination Scenarios
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from lrdbenchmark.models.contamination.contamination_factory import ContaminationFactory, ConfoundingScenario
   from lrdbenchmark.models.data_models.fbm.fbm_model import FBMModel
   import numpy as np
   
   # Create contamination factory
   factory = ContaminationFactory()
   
   # Generate pure data
   model = FBMModel(H=0.7, sigma=1.0)
   pure_data = model.generate(1000, seed=42)
   
   # Financial-specific contamination scenarios
   financial_scenarios = [
       ConfoundingScenario.FINANCIAL_CRASH,
       ConfoundingScenario.VOLATILITY_CLUSTERING,
       ConfoundingScenario.MARKET_MICROSTRUCTURE
   ]
   
   print("Financial Contamination Testing:")
   for scenario in financial_scenarios:
       contaminated_data, description = factory.apply_confounding(
           pure_data, scenario, intensity=0.4
       )
       print(f"{scenario.value}: {description}")

Physiological Contamination Scenarios
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from lrdbenchmark.models.contamination.contamination_factory import ContaminationFactory, ConfoundingScenario
   from lrdbenchmark.models.data_models.fbm.fbm_model import FBMModel
   import numpy as np
   
   # Create contamination factory
   factory = ContaminationFactory()
   
   # Generate pure data
   model = FBMModel(H=0.7, sigma=1.0)
   pure_data = model.generate(1000, seed=42)
   
   # Physiological-specific contamination scenarios
   physiological_scenarios = [
       ConfoundingScenario.PHYSIOLOGICAL_DRIFT,
       ConfoundingScenario.SENSOR_ARTIFACTS,
       ConfoundingScenario.MEASUREMENT_ERRORS
   ]
   
   print("Physiological Contamination Testing:")
   for scenario in physiological_scenarios:
       contaminated_data, description = factory.apply_confounding(
           pure_data, scenario, intensity=0.25
       )
       print(f"{scenario.value}: {description}")

Environmental Contamination Scenarios
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from lrdbenchmark.models.contamination.contamination_factory import ContaminationFactory, ConfoundingScenario
   from lrdbenchmark.models.data_models.fbm.fbm_model import FBMModel
   import numpy as np
   
   # Create contamination factory
   factory = ContaminationFactory()
   
   # Generate pure data
   model = FBMModel(H=0.7, sigma=1.0)
   pure_data = model.generate(1000, seed=42)
   
   # Environmental-specific contamination scenarios
   environmental_scenarios = [
       ConfoundingScenario.ENVIRONMENTAL_SEASONAL,
       ConfoundingScenario.NETWORK_BURSTS,
       ConfoundingScenario.INDUSTRIAL_CALIBRATION
   ]
   
   print("Environmental Contamination Testing:")
   for scenario in environmental_scenarios:
       contaminated_data, description = factory.apply_confounding(
           pure_data, scenario, intensity=0.3
       )
       print(f"{scenario.value}: {description}")

Custom Contamination Profiles
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from lrdbenchmark.models.contamination.contamination_factory import ContaminationFactory, ConfoundingScenario, ConfoundingProfile
   from lrdbenchmark.models.data_models.fbm.fbm_model import FBMModel
   import numpy as np
   
   # Create contamination factory
   factory = ContaminationFactory()
   
   # Generate pure data
   model = FBMModel(H=0.7, sigma=1.0)
   pure_data = model.generate(1000, seed=42)
   
   # Create custom contamination profile
   custom_profile = ConfoundingProfile(
       scenario=ConfoundingScenario.GAUSSIAN_NOISE,
       intensity=0.5,
       parameters={
           'noise_std': 0.3,
           'noise_correlation': 0.1
       },
       description="Custom high-intensity correlated noise"
   )
   
   # Apply custom contamination
   contaminated_data, description = factory.apply_confounding(
       pure_data, custom_profile
   )
   print(f"Custom contamination: {description}")

Intensity Variation Testing
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from lrdbenchmark.models.contamination.contamination_factory import ContaminationFactory, ConfoundingScenario
   from lrdbenchmark.models.data_models.fbm.fbm_model import FBMModel
   import numpy as np
   
   # Create contamination factory
   factory = ContaminationFactory()
   
   # Generate pure data
   model = FBMModel(H=0.7, sigma=1.0)
   pure_data = model.generate(1000, seed=42)
   
   # Test different intensity levels
   intensities = [0.1, 0.2, 0.3, 0.4, 0.5]
   scenario = ConfoundingScenario.EEG_OCULAR_ARTIFACTS
   
   print("Intensity Variation Testing:")
   for intensity in intensities:
       contaminated_data, description = factory.apply_confounding(
           pure_data, scenario, intensity=intensity
       )
       print(f"Intensity {intensity}: {description}")

Contamination Analysis
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from lrdbenchmark.models.contamination.contamination_factory import ContaminationFactory, ConfoundingScenario
   from lrdbenchmark.models.data_models.fbm.fbm_model import FBMModel
   import numpy as np
   
   # Create contamination factory
   factory = ContaminationFactory()
   
   # Generate pure data
   model = FBMModel(H=0.7, sigma=1.0)
   pure_data = model.generate(1000, seed=42)
   
   # Analyze contamination effects
   scenarios = [
       ConfoundingScenario.PURE,
       ConfoundingScenario.GAUSSIAN_NOISE,
       ConfoundingScenario.EEG_OCULAR_ARTIFACTS,
       ConfoundingScenario.FINANCIAL_CRASH
   ]
   
   print("Contamination Analysis:")
   for scenario in scenarios:
       contaminated_data, description = factory.apply_confounding(
           pure_data, scenario, intensity=0.3
       )
       
       # Calculate contamination metrics
       contamination_level = np.std(contaminated_data - pure_data) / np.std(pure_data)
       correlation = np.corrcoef(pure_data, contaminated_data)[0, 1]
       
       print(f"{scenario.value}:")
       print(f"  Description: {description}")
       print(f"  Contamination level: {contamination_level:.3f}")
       print(f"  Correlation with pure: {correlation:.3f}")

Best Practices
--------------

1. **Intensity Selection**: Use appropriate intensity levels (0.1-0.5) for realistic testing
2. **Scenario Selection**: Choose scenarios relevant to your application domain
3. **Multiple Scenarios**: Test robustness across different contamination types
4. **Intensity Variation**: Test across different intensity levels
5. **Analysis**: Always analyze the effects of contamination on your data
6. **Validation**: Compare contaminated results with pure data baselines

.. note::
   The contamination factory provides realistic confounding profiles based on
   real-world scenarios. This makes it ideal for testing estimator robustness
   in practical applications.

.. warning::
   High intensity contamination (> 0.5) may significantly distort the underlying
   signal characteristics and should be used with caution.
