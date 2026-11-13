"""Deliberately invalid experiments ("traps") used to test the statistical gate.

Each trap really executes a fast log-variance + logistic-regression pipeline on
the supplied dataset and then records the provenance the trusted harness would
record for such a run.  The five traps are:

1. Leaky random window split   - windows from the test subject are in training.
2. Single-seed cherry-pick     - best of N seeds reported as if it were one run.
3. Label-shuffled noise model  - trained on permuted labels, i.e. pure noise.
4. Test-set scaler leakage     - the scaler is fitted on train *and* test data.
5. Unseeded non-deterministic  - no seeds, so replays cannot reproduce.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler

from auditor.stats_gate import (
    FIT_LEAKAGE,
    INSUFFICIENT_SEEDS,
    NOT_ABOVE_CHANCE,
    SPLIT_LEAKAGE,
    UNSEEDED_RUN,
    ExperimentRecord,
    FoldManifest,
    build_loso_manifest,
)

from .data_loader import EEGDataset

TRAP_LEAKY_SPLIT = "leaky_random_window_split"
TRAP_CHERRY_PICK = "single_seed_cherry_pick"
TRAP_SHUFFLED = "label_shuffled_noise_model"
TRAP_SCALER_LEAK = "test_set_scaler_leakage"
TRAP_UNSEEDED = "unseeded_nondeterministic_run"


@dataclass(frozen=True)
class Trap:
    """A deliberately invalid experiment and the flag the gate must raise."""

    name: str
    description: str
    expected_flag: str
    record: ExperimentRecord


def log_variance_features(X: np.ndarray) -> np.ndarray:
    """Per-channel log band power (variance of the band-passed signal)."""
    return np.log(np.var(X, axis=2) + 1e-8)


def _windows(X: np.ndarray, n_windows: int = 3) -> Tuple[np.ndarray, np.ndarray]:
    """Cut each epoch into overlapping half-length windows (50 % overlap)."""
    length = X.shape[2] // 2
    step = (X.shape[2] - length) // max(1, n_windows - 1)
    wins, owner = [], []
    for w in range(n_windows):
        wins.append(X[:, :, w * step : w * step + length])
        owner.append(np.arange(len(X)))
    return np.concatenate(wins), np.concatenate(owner)


def _fit_score(
    feat_tr: np.ndarray, y_tr: np.ndarray, feat_te: np.ndarray, y_te: np.ndarray,
    seed: Optional[int], *, scaler_sees_test: bool = False, shuffle_labels: bool = False,
) -> float:
    """Bootstrap-train a logistic regression and return macro-F1 on the test features."""
    rng = np.random.default_rng(seed)  # seed=None -> OS entropy (non-deterministic)
    idx = rng.integers(0, len(feat_tr), len(feat_tr))
    ytr = y_tr[idx]
    if shuffle_labels:
        ytr = rng.permutation(ytr)
    fit_data = np.vstack([feat_tr[idx], feat_te]) if scaler_sees_test else feat_tr[idx]
    scaler = StandardScaler().fit(fit_data)
    clf = LogisticRegression(C=0.1, max_iter=500).fit(scaler.transform(feat_tr[idx]), ytr)
    return float(f1_score(y_te, clf.predict(scaler.transform(feat_te)), average="macro", zero_division=0))


def _loso_matrix(
    ds: EEGDataset, seeds: Sequence[Optional[int]], **kwargs: bool
) -> Tuple[np.ndarray, List[Tuple[float, float]]]:
    """Run a LOSO experiment per seed; also return one replay pair of (seed0, subject0)."""
    feats = log_variance_features(ds.X)
    ids = ds.subject_ids
    mat = np.zeros((len(seeds), len(ids)))
    for i, seed in enumerate(seeds):
        for j, sid in enumerate(ids):
            te = ds.subjects == sid
            mat[i, j] = _fit_score(feats[~te], ds.y[~te], feats[te], ds.y[te], seed, **kwargs)
    te0 = ds.subjects == ids[0]
    replay = _fit_score(feats[~te0], ds.y[~te0], feats[te0], ds.y[te0], seeds[0], **kwargs)
    return mat, [(float(mat[0, 0]), replay)]


def _trap_leaky_split(ds: EEGDataset, seeds: Sequence[int]) -> Trap:
    wins, owner = _windows(ds.X)
    feats = log_variance_features(wins)
    y, subj = ds.y[owner], ds.subjects[owner]
    ids = ds.subject_ids

    def run(seed: int) -> np.ndarray:
        rng = np.random.default_rng(seed)
        test = rng.random(len(y)) < 0.2
        scaler = StandardScaler().fit(feats[~test])
        clf = LogisticRegression(C=0.1, max_iter=500).fit(scaler.transform(feats[~test]), y[~test])
        pred = clf.predict(scaler.transform(feats[test]))
        return np.array([
            f1_score(y[test][subj[test] == s], pred[subj[test] == s], average="macro", zero_division=0)
            for s in ids
        ])

    mat = np.stack([run(s) for s in seeds])
    manifests = [FoldManifest(s, tuple(ids), tuple(ids), "random_window") for s in ids]
    rec = ExperimentRecord(
        TRAP_LEAKY_SPLIT, list(ids), list(seeds), mat, manifests, len(seeds),
        [(float(mat[0, 0]), float(run(seeds[0])[0]))],
    )
    return Trap(TRAP_LEAKY_SPLIT,
                "Overlapping windows split at random, so test-subject windows leak into training.",
                SPLIT_LEAKAGE, rec)


def _trap_cherry_pick(ds: EEGDataset, pool: int) -> Trap:
    pool_seeds = list(range(100, 100 + pool))
    mat, _ = _loso_matrix(ds, pool_seeds)
    best = int(np.argmax(mat.mean(axis=1)))
    feats = log_variance_features(ds.X)
    ids = ds.subject_ids
    te0 = ds.subjects == ids[0]
    replay = _fit_score(feats[~te0], ds.y[~te0], feats[te0], ds.y[te0], pool_seeds[best])
    rec = ExperimentRecord(
        TRAP_CHERRY_PICK, list(ids), [pool_seeds[best]], mat[best : best + 1],
        build_loso_manifest(ids), pool, [(float(mat[best, 0]), replay)],
        reported_mean_f1=float(mat[best].mean()),
    )
    return Trap(TRAP_CHERRY_PICK,
                f"Best of {pool} seeds reported as a single run.", INSUFFICIENT_SEEDS, rec)


def _trap_shuffled(ds: EEGDataset, seeds: Sequence[int]) -> Trap:
    mat, replay = _loso_matrix(ds, list(seeds), shuffle_labels=True)
    rec = ExperimentRecord(
        TRAP_SHUFFLED, list(ds.subject_ids), list(seeds), mat,
        build_loso_manifest(ds.subject_ids), len(seeds), replay,
    )
    return Trap(TRAP_SHUFFLED, "Model trained on permuted labels (pure noise).", NOT_ABOVE_CHANCE, rec)


def _trap_scaler_leak(ds: EEGDataset, seeds: Sequence[int]) -> Trap:
    mat, replay = _loso_matrix(ds, list(seeds), scaler_sees_test=True)
    rec = ExperimentRecord(
        TRAP_SCALER_LEAK, list(ds.subject_ids), list(seeds), mat,
        build_loso_manifest(ds.subject_ids, fit_includes_test=True), len(seeds), replay,
    )
    return Trap(TRAP_SCALER_LEAK, "StandardScaler fitted on train and test features.", FIT_LEAKAGE, rec)


def _trap_unseeded(ds: EEGDataset, n_runs: int) -> Trap:
    mat, replay = _loso_matrix(ds, [None] * n_runs)
    rec = ExperimentRecord(
        TRAP_UNSEEDED, list(ds.subject_ids), [None] * n_runs, mat,
        build_loso_manifest(ds.subject_ids), n_runs, replay,
    )
    return Trap(TRAP_UNSEEDED, "No random seeds; replay cannot reproduce results.", UNSEEDED_RUN, rec)


def generate_traps(
    dataset: EEGDataset, seeds: Sequence[int] = (0, 1, 2), cherry_pick_pool: int = 10
) -> List[Trap]:
    """Execute the five invalid experiments and return their records.

    Args:
        dataset: Pooled dataset (real or synthetic).
        seeds: Seeds used by the traps that are otherwise well-formed.
        cherry_pick_pool: Number of seeds tried for the cherry-pick trap.

    Returns:
        The five :class:`Trap` objects in a fixed order.
    """
    return [
        _trap_leaky_split(dataset, seeds),
        _trap_cherry_pick(dataset, cherry_pick_pool),
        _trap_shuffled(dataset, seeds),
        _trap_scaler_leak(dataset, seeds),
        _trap_unseeded(dataset, len(seeds)),
    ]
