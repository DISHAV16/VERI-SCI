"""Trusted, read-only evaluation harness for VERI-SCI.

Nothing in this package is ever importable or readable by agent-generated
code: the AST guard in :mod:`sandbox.ast_guard` rejects any reference to
``trusted_eval``.
"""

from .data_loader import (
    BAND_HZ,
    CLASS_NAMES,
    RUNS,
    SUBJECTS,
    DataUnavailableError,
    EEGDataset,
    LOSOSplit,
    load_dataset,
    load_eegbci,
    loso_splits,
    make_synthetic_dataset,
)

__all__ = [
    "BAND_HZ",
    "CLASS_NAMES",
    "RUNS",
    "SUBJECTS",
    "DataUnavailableError",
    "EEGDataset",
    "LOSOSplit",
    "load_dataset",
    "load_eegbci",
    "loso_splits",
    "make_synthetic_dataset",
]
