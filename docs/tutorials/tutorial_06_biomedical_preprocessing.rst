Biomedical Preprocessing Tutorial
=================================

This tutorial demonstrates the new domain-specific preprocessing utilities for
EEG and HRV/ECG signals together with the deterministic real-world validation
workflow.

Loading the Surrogate Datasets
------------------------------

.. code-block:: python

    from lrdbenchmark.real_world_validation import RealWorldDataValidator
    from lrdbenchmark.random_manager import initialise_global_rng

    initialise_global_rng(4242)
    validator = RealWorldDataValidator(results_dir="results/biomedical-demo", seed=None)

    eeg_record = validator.datasets["physiological_eeg"]
    hrv_record = validator.datasets["physiological_hrv"]

Preparing the Signals
---------------------

.. code-block:: python

    from lrdbenchmark.robustness.adaptive_preprocessor import AdaptiveDataPreprocessor

    preprocessor = AdaptiveDataPreprocessor()

    eeg_clean, eeg_meta = preprocessor.preprocess(
        eeg_record.values,
        domain="eeg",
        sampling_rate_hz=256,
    )

    hrv_clean, hrv_meta = preprocessor.preprocess(
        hrv_record.values,
        domain="hrv",
        sampling_rate_hz=4,
    )

    print(eeg_meta["domain_preprocessing"])
    # {'domain': 'eeg', 'sampling_rate_hz': 256, 'bandpass_hz': (1.0, 45.0), 'notch_hz': 50.0}

Benchmarking the Cleaned Series
-------------------------------

.. code-block:: python

    results = validator.run(persist=False)
    physiological_estimates = [
        ds for ds in results["datasets"] if ds["domain"] == "physiological"
    ]
    for ds in physiological_estimates:
        print(ds["name"], ds["estimates"]["DFA"]["hurst_parameter"])

The run artefacts include provenance metadata with the RNG seeds required to
reproduce the generated surrogate datasets and preprocessing configuration.

