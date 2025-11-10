Domain-Specific Preprocessing
============================

The :mod:`lrdbenchmark.domain.preprocessing` module adds lightweight helpers for
handling biomedical signals directly within the benchmarking workflow.  The
``DomainPreprocessor`` class exposes an automatic switcher that applies the
appropriate cleaning steps for EEG and ECG/HRV series while recording the
choices in the metadata returned by :class:`~lrdbenchmark.robustness.adaptive_preprocessor.AdaptiveDataPreprocessor`.

Sampling-Rate Guidance
----------------------

The helper provides curated sampling-rate recommendations that can be queried
at runtime::

    >>> from lrdbenchmark.domain.preprocessing import DomainPreprocessor
    >>> DomainPreprocessor().sampling_guidance()
    {'eeg': {'recommended_range_hz': (128, 512),
             'comment': 'Use ≥256 Hz when analysing beta activity or higher.'},
     'ecg': {'recommended_range_hz': (100, 500),
             'comment': 'For HRV, 250 Hz provides robust R-peak localisation.'},
     'hrv': {'recommended_range_hz': (4, 16),
             'comment': 'Resampled RR-intervals at 4 Hz are standard for HRV metrics.'}}

EEG Pipeline
------------

``DomainPreprocessor`` applies a 1–45 Hz band-pass filter, a mains notch
filter (50 Hz or 60 Hz depending on the sampling rate), and returns the
processed signal together with the configuration::

    >>> preprocessor = DomainPreprocessor()
    >>> cleaned, meta = preprocessor.preprocess(eeg_signal, domain="eeg", sampling_rate_hz=256)
    >>> meta["bandpass_hz"]
    (1.0, 45.0)

ECG/HRV Pipeline
----------------

For ECG and HRV channels a high-pass filter removes baseline wander before a
40 Hz low-pass and mains notch filter are applied.  The metadata records the
choices for downstream provenance.

Tutorial Walkthrough
--------------------

The new ``docs/tutorials/tutorial_06_biomedical_preprocessing.rst`` tutorial
provides a step-by-step walk-through covering:

* loading the surrogate EEG/HRV datasets from :mod:`lrdbenchmark.real_world_validation`,
* applying the domain-specific preprocessor, and
* benchmarking classical estimators on the cleaned series.

