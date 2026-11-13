"""EEGBCI data loading and read-only Leave-One-Subject-Out (LOSO) splits.

The loader downloads the PhysioNet EEG Motor Movement/Imagery dataset through
MNE-Python (subjects 1-5, runs 4, 8, 12: left- vs right-fist motor imagery),
applies an 8-30 Hz band-pass filter, epochs the imagery period, and decimates
to 80 Hz.  Every array handed out by this module is flagged non-writeable so
that downstream code cannot mutate the evaluation data in place.

A synthetic fallback generator is provided for offline smoke-testing only.
Datasets carry a ``source`` tag (``"eegbci"`` or ``"synthetic"``) that the
benchmark prints in its results table so synthetic numbers are never mistaken
for real EEG results.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional, Sequence, Tuple

import numpy as np

LOGGER = logging.getLogger(__name__)

SUBJECTS: Tuple[int, ...] = (1, 2, 3, 4, 5)
RUNS: Tuple[int, ...] = (4, 8, 12)
BAND_HZ: Tuple[float, float] = (8.0, 30.0)
EPOCH_TMIN_S: float = 0.0
EPOCH_TMAX_S: float = 4.0
DECIMATION: int = 2
CLASS_NAMES: Tuple[str, str] = ("left_hand", "right_hand")
ERD_RANGE: Tuple[float, float] = (0.8, 0.97)
INTERFERENCE_SCALE: float = 2.0


class DataUnavailableError(RuntimeError):
    """Raised when the EEGBCI dataset cannot be downloaded or parsed."""


def _freeze(arr: np.ndarray) -> np.ndarray:
    """Return a contiguous copy of ``arr`` flagged read-only."""
    out = np.ascontiguousarray(arr).copy()
    out.setflags(write=False)
    return out


@dataclass(frozen=True)
class EEGDataset:
    """Immutable pooled EEG epochs.

    Attributes:
        X: Float32 array of shape ``(n_trials, n_channels, n_times)``.
        y: Int64 labels in ``{0, 1}`` (0 = left hand, 1 = right hand).
        subjects: Int64 subject id for every trial.
        sfreq: Sampling frequency in Hz after decimation.
        source: ``"eegbci"`` for real data, ``"synthetic"`` for the fallback.
    """

    X: np.ndarray
    y: np.ndarray
    subjects: np.ndarray
    sfreq: float
    source: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "X", _freeze(np.asarray(self.X, dtype=np.float32)))
        object.__setattr__(self, "y", _freeze(np.asarray(self.y, dtype=np.int64)))
        object.__setattr__(self, "subjects", _freeze(np.asarray(self.subjects, dtype=np.int64)))
        if not (len(self.X) == len(self.y) == len(self.subjects)):
            raise ValueError("X, y and subjects must have the same length")

    @property
    def n_channels(self) -> int:
        """Number of EEG channels."""
        return int(self.X.shape[1])

    @property
    def n_times(self) -> int:
        """Number of time samples per epoch."""
        return int(self.X.shape[2])

    @property
    def subject_ids(self) -> Tuple[int, ...]:
        """Sorted unique subject ids."""
        return tuple(int(s) for s in np.unique(self.subjects))

    def fingerprint(self) -> str:
        """SHA-256 digest of the data, used to prove the harness data was not altered."""
        digest = hashlib.sha256()
        for arr in (self.X, self.y, self.subjects):
            digest.update(arr.tobytes())
        return digest.hexdigest()


@dataclass(frozen=True)
class LOSOSplit:
    """One read-only Leave-One-Subject-Out fold."""

    test_subject: int
    train_subjects: Tuple[int, ...]
    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray

    def __post_init__(self) -> None:
        for name in ("X_train", "y_train", "X_test", "y_test"):
            object.__setattr__(self, name, _freeze(getattr(self, name)))
        if self.test_subject in self.train_subjects:
            raise ValueError("LOSO violation: test subject present in the training set")


def loso_splits(
    dataset: EEGDataset, subjects: Optional[Sequence[int]] = None
) -> Iterator[LOSOSplit]:
    """Yield one read-only LOSO fold per held-out subject.

    Args:
        dataset: Pooled dataset.
        subjects: Subjects to hold out; defaults to every subject in the dataset.

    Yields:
        :class:`LOSOSplit` whose training data never contains the test subject.
    """
    held_out = tuple(subjects) if subjects is not None else dataset.subject_ids
    for sid in held_out:
        test_mask = dataset.subjects == sid
        if not test_mask.any():
            raise ValueError(f"subject {sid} not present in dataset")
        train_mask = ~test_mask
        train_subjects = tuple(int(s) for s in np.unique(dataset.subjects[train_mask]))
        yield LOSOSplit(
            test_subject=int(sid),
            train_subjects=train_subjects,
            X_train=dataset.X[train_mask],
            y_train=dataset.y[train_mask],
            X_test=dataset.X[test_mask],
            y_test=dataset.y[test_mask],
        )


def load_eegbci(
    data_dir: Optional[Path] = None,
    subjects: Sequence[int] = SUBJECTS,
    runs: Sequence[int] = RUNS,
    band: Tuple[float, float] = BAND_HZ,
) -> EEGDataset:
    """Download and preprocess the EEGBCI motor-imagery dataset.

    Args:
        data_dir: MNE data cache directory (``None`` uses MNE's default).
        subjects: PhysioNet subject numbers.
        runs: Motor-imagery runs (4, 8, 12 = left vs right fist imagery).
        band: Band-pass edges in Hz.

    Returns:
        Pooled :class:`EEGDataset` with ``source="eegbci"``.

    Raises:
        DataUnavailableError: If the download or parsing fails.
    """
    try:
        import mne
        from mne.datasets import eegbci
        from mne.io import concatenate_raws, read_raw_edf
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise DataUnavailableError("mne is required: pip install mne") from exc

    mne.set_log_level("WARNING")
    xs, ys, ss = [], [], []
    sfreq_out = 0.0
    for sid in subjects:
        try:
            kwargs = {"path": str(data_dir)} if data_dir else {}
            paths = eegbci.load_data(sid, list(runs), **kwargs)
            raw = concatenate_raws([read_raw_edf(p, preload=True, verbose="WARNING") for p in paths])
            if hasattr(eegbci, "standardize"):
                eegbci.standardize(raw)
            raw.filter(band[0], band[1], fir_design="firwin", skip_by_annotation="edge")
            events, event_id = mne.events_from_annotations(raw, event_id=dict(T1=1, T2=2))
            epochs = mne.Epochs(
                raw, events, event_id, tmin=EPOCH_TMIN_S, tmax=EPOCH_TMAX_S,
                baseline=None, picks="eeg", preload=True, verbose="WARNING",
            )
        except Exception as exc:
            raise DataUnavailableError(
                f"Could not load EEGBCI subject {sid} (runs {list(runs)}): {exc}"
            ) from exc
        fs = float(epochs.info["sfreq"])
        n_keep = int(round((EPOCH_TMAX_S - EPOCH_TMIN_S) * fs))
        data = epochs.get_data()[:, :, :n_keep] * 1e6  # volts -> microvolts
        data = data[:, :, ::DECIMATION]
        sfreq_out = fs / DECIMATION
        labels = epochs.events[:, 2] - 1  # T1 -> 0 (left), T2 -> 1 (right)
        xs.append(data.astype(np.float32))
        ys.append(labels.astype(np.int64))
        ss.append(np.full(len(labels), sid, dtype=np.int64))
        LOGGER.info("subject %d: %d trials, %d channels", sid, len(labels), data.shape[1])

    if len({x.shape[1:] for x in xs}) != 1:
        raise DataUnavailableError("subjects have inconsistent channel/time shapes")
    return EEGDataset(
        X=np.concatenate(xs), y=np.concatenate(ys), subjects=np.concatenate(ss),
        sfreq=sfreq_out, source="eegbci",
    )


def make_synthetic_dataset(
    n_subjects: int = 5,
    trials_per_subject: int = 45,
    n_channels: int = 64,
    n_times: int = 320,
    sfreq: float = 80.0,
    seed: int = 0,
) -> EEGDataset:
    """Generate band-limited synthetic motor-imagery-like EEG (offline smoke tests only).

    Imagery of the left (right) hand attenuates mu-band power over the
    right (left) channel group, blurred by a subject-specific mixing matrix so
    that cross-subject generalisation is non-trivial.  **Never report results
    from this generator as real EEG performance.**
    """
    rng = np.random.default_rng(seed)
    freqs = np.fft.rfftfreq(n_times, d=1.0 / sfreq)
    band_mask = (freqs >= BAND_HZ[0]) & (freqs <= BAND_HZ[1])
    half = n_channels // 2
    xs, ys, ss = [], [], []
    for subj in range(1, n_subjects + 1):
        mixing = np.eye(n_channels) + 0.35 * rng.normal(size=(n_channels, n_channels)) / np.sqrt(n_channels)
        gain = rng.uniform(0.8, 1.4)
        labels = rng.permutation(np.arange(trials_per_subject) % 2)
        trials = np.empty((trials_per_subject, n_channels, n_times), dtype=np.float32)
        for k, label in enumerate(labels):
            noise = rng.normal(size=(n_channels, n_times))
            sig = np.fft.irfft(np.fft.rfft(noise, axis=1) * band_mask, n=n_times, axis=1)
            amp = np.ones(n_channels)
            erd = rng.uniform(ERD_RANGE[0], ERD_RANGE[1])
            if label == 0:
                amp[half:] = erd
            else:
                amp[:half] = erd
            interference = np.fft.irfft(
                np.fft.rfft(rng.normal(size=(n_channels, n_times)), axis=1) * band_mask, n=n_times, axis=1)
            trials[k] = gain * (mixing @ (sig * amp[:, None] + INTERFERENCE_SCALE * interference))
        xs.append(trials)
        ys.append(labels.astype(np.int64))
        ss.append(np.full(trials_per_subject, subj, dtype=np.int64))
    return EEGDataset(
        X=np.concatenate(xs), y=np.concatenate(ys), subjects=np.concatenate(ss),
        sfreq=sfreq, source="synthetic",
    )


def load_dataset(
    data_dir: Optional[Path] = None, synthetic: bool = False, allow_synthetic_fallback: bool = False
) -> EEGDataset:
    """Load real EEGBCI data, optionally falling back to the synthetic generator.

    Args:
        data_dir: MNE cache directory.
        synthetic: Force the synthetic generator.
        allow_synthetic_fallback: Use synthetic data if the real download fails.

    Raises:
        DataUnavailableError: Real data unavailable and fallback not allowed.
    """
    if synthetic:
        LOGGER.warning("Using SYNTHETIC data; results are not real EEG results.")
        return make_synthetic_dataset()
    try:
        return load_eegbci(data_dir=data_dir)
    except DataUnavailableError:
        if not allow_synthetic_fallback:
            raise
        LOGGER.warning("EEGBCI unavailable; falling back to SYNTHETIC data.")
        return make_synthetic_dataset()
