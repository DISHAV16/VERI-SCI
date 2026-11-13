"""Independent statistical gate: no discovery is promoted unless it passes here.

The gate consumes :class:`ExperimentRecord` objects produced by the *trusted*
harness (never by the agent) and applies two layers of checks:

1. **Provenance checks** (hard failures): split protocol must be LOSO with
   disjoint subjects, no fitted statistic may touch the test subject, at least
   ``min_seeds`` explicit seeds, seeds reported must equal seeds executed
   (anti cherry-picking), and a replay of one run must reproduce bit-for-bit.
2. **Statistical checks**: subject-level paired t-test (primary), Wilcoxon
   signed-rank (reported as a robustness check), Cohen's d_z >= 0.5, a 95 %
   t-interval for the mean paired difference that must exclude zero,
   Holm-Bonferroni correction across the family of hypotheses, an above-chance
   floor, and a requirement that the improvement exceeds the seed-level noise.

Note on the Wilcoxon test: with n = 5 subjects the smallest attainable
two-sided exact p-value is 2/2^5 = 0.0625, so it can never reach alpha = 0.05.
It is therefore reported but not used as a gating criterion.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import stats
from statsmodels.stats.multitest import multipletests

SPLIT_LEAKAGE = "SPLIT_LEAKAGE"
FIT_LEAKAGE = "FIT_LEAKAGE"
UNSEEDED_RUN = "UNSEEDED_RUN"
INSUFFICIENT_SEEDS = "INSUFFICIENT_SEEDS"
SEED_CHERRY_PICK = "SEED_CHERRY_PICK"
REPORT_MISMATCH = "REPORT_MISMATCH"
NONDETERMINISTIC = "NONDETERMINISTIC"
NO_REPLAY_CHECK = "NO_REPLAY_CHECK"
MALFORMED = "MALFORMED"
NOT_ABOVE_CHANCE = "NOT_ABOVE_CHANCE"
NOT_SIGNIFICANT = "NOT_SIGNIFICANT"
SMALL_EFFECT = "SMALL_EFFECT"
CI_INCLUDES_ZERO = "CI_INCLUDES_ZERO"
SEED_NOISE_DOMINATES = "SEED_NOISE_DOMINATES"

FLAG_DESCRIPTIONS: Dict[str, str] = {
    SPLIT_LEAKAGE: "Evaluation was not strict Leave-One-Subject-Out: test-subject data was available for training.",
    FIT_LEAKAGE: "A fitted statistic (e.g. scaler) used data from the held-out test subject.",
    UNSEEDED_RUN: "At least one run used no explicit random seed.",
    INSUFFICIENT_SEEDS: "Fewer distinct explicit seeds than the required minimum.",
    SEED_CHERRY_PICK: "Fewer seeds were reported than were executed (possible seed cherry-picking).",
    REPORT_MISMATCH: "Reported mean F1 exceeds the mean of the recorded per-run F1 scores.",
    NONDETERMINISTIC: "Replaying an identical run did not reproduce the original F1 score.",
    NO_REPLAY_CHECK: "No replay evidence was supplied to prove determinism.",
    MALFORMED: "Record is malformed (shape mismatch, non-finite values or inconsistent subjects).",
    NOT_ABOVE_CHANCE: "Mean LOSO F1 is not above the chance-level floor.",
    NOT_SIGNIFICANT: "Improvement over baseline is not significant after Holm-Bonferroni correction.",
    SMALL_EFFECT: "Paired Cohen's d_z is below the minimum effect size.",
    CI_INCLUDES_ZERO: "The 95% CI of the mean paired improvement includes zero.",
    SEED_NOISE_DOMINATES: "Improvement does not exceed the seed-level noise floor.",
}


@dataclass(frozen=True)
class GateConfig:
    """Thresholds for the statistical gate."""

    alpha: float = 0.05
    min_effect_size: float = 0.5
    min_seeds: int = 3
    chance_f1: float = 0.5
    chance_margin: float = 0.02
    replay_atol: float = 1e-6
    ci_level: float = 0.95
    max_seed_noise_ratio: float = 1.0


@dataclass(frozen=True)
class FoldManifest:
    """Provenance of one evaluation fold, recorded by the trusted harness.

    Attributes:
        test_subject: Held-out subject id.
        train_subjects: Subjects whose data was available for training.
        fit_subjects: Subjects whose data influenced any fitted statistic.
        protocol: ``"LOSO"`` or e.g. ``"random_window"``.
    """

    test_subject: int
    train_subjects: Tuple[int, ...]
    fit_subjects: Tuple[int, ...]
    protocol: str = "LOSO"


@dataclass
class ExperimentRecord:
    """Everything the gate needs to audit one candidate or baseline.

    Attributes:
        name: Identifier.
        subject_ids: Subject ids matching the columns of ``f1``.
        seeds: Seed per row of ``f1`` (``None`` = unseeded).
        f1: Macro-F1 matrix of shape ``(n_seeds, n_subjects)``.
        folds: Per-subject provenance manifests.
        n_runs_logged: Number of seed-level runs the harness executed.
        replay_pairs: ``(original_f1, replay_f1)`` for replayed runs.
        reported_mean_f1: Mean F1 the experimenter claims (if any).
        metadata: Free-form extra information.
    """

    name: str
    subject_ids: List[int]
    seeds: List[Optional[int]]
    f1: np.ndarray
    folds: List[FoldManifest]
    n_runs_logged: int
    replay_pairs: List[Tuple[float, float]] = field(default_factory=list)
    reported_mean_f1: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.f1 = np.atleast_2d(np.asarray(self.f1, dtype=float))

    def mean_f1(self) -> float:
        """Grand mean macro-F1 over seeds and subjects."""
        return float(np.mean(self.f1))

    def subject_means(self) -> Dict[int, float]:
        """Seed-averaged F1 per subject."""
        return {int(s): float(self.f1[:, j].mean()) for j, s in enumerate(self.subject_ids)}


@dataclass
class AuditReport:
    """Verdict of the gate for one candidate."""

    name: str
    verdict: str
    flags: List[str]
    reasons: List[str]
    stats: Dict[str, Any]
    p_holm: float
    family_size: int

    @property
    def verified(self) -> bool:
        """True iff the candidate passed every check."""
        return self.verdict == "VERIFIED"

    def to_dict(self) -> Dict[str, Any]:
        """JSON-serialisable representation."""
        return to_jsonable(
            {
                "name": self.name, "verdict": self.verdict, "flags": self.flags,
                "reasons": self.reasons, "stats": self.stats,
                "p_holm": self.p_holm, "family_size": self.family_size,
            }
        )


def to_jsonable(obj: Any) -> Any:
    """Recursively convert numpy types / NaN / inf into strict-JSON-safe values."""
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return to_jsonable(obj.tolist())
    if isinstance(obj, (np.floating, float)):
        val = float(obj)
        return val if math.isfinite(val) else None
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


def build_loso_manifest(subject_ids: Sequence[int], fit_includes_test: bool = False) -> List[FoldManifest]:
    """Build honest LOSO manifests (optionally simulating fit-on-test leakage)."""
    ids = [int(s) for s in subject_ids]
    folds = []
    for test in ids:
        train = tuple(s for s in ids if s != test)
        fit = train + (test,) if fit_includes_test else train
        folds.append(FoldManifest(test_subject=test, train_subjects=train, fit_subjects=fit))
    return folds


def variance_decomposition(f1: np.ndarray) -> Dict[str, float]:
    """Separate seed-level from subject-level variance.

    Args:
        f1: Matrix ``(n_seeds, n_subjects)``.

    Returns:
        Mean within-subject seed variance, between-subject variance of the
        seed-averaged scores, the seed std and their ratio (NaN if undefined).
    """
    arr = np.atleast_2d(np.asarray(f1, dtype=float))
    n_seeds, n_subj = arr.shape
    within = float(np.mean(np.var(arr, axis=0, ddof=1))) if n_seeds > 1 else float("nan")
    between = float(np.var(arr.mean(axis=0), ddof=1)) if n_subj > 1 else float("nan")
    ratio = within / between if math.isfinite(within) and math.isfinite(between) and between > 0 else float("nan")
    return {
        "seed_variance_within_subject": within,
        "subject_variance_between": between,
        "seed_std": math.sqrt(within) if math.isfinite(within) else float("nan"),
        "seed_to_subject_variance_ratio": ratio,
    }


def holm_bonferroni(p_values: Sequence[float], alpha: float = 0.05) -> Tuple[np.ndarray, np.ndarray]:
    """Holm-Bonferroni step-down correction.

    Returns:
        ``(reject, adjusted_p)`` arrays in the input order.
    """
    p = np.asarray(p_values, dtype=float)
    if p.size == 0:
        return np.array([], dtype=bool), np.array([], dtype=float)
    reject, adjusted, _, _ = multipletests(p, alpha=alpha, method="holm")
    return reject, adjusted


def paired_subject_stats(
    candidate: ExperimentRecord, baseline: ExperimentRecord, cfg: Optional[GateConfig] = None
) -> Dict[str, Any]:
    """Subject-level paired comparison of a candidate against the baseline.

    Seeds are averaged *within* each subject first, so the unit of analysis is
    the subject (n = number of LOSO folds), not the seed.
    """
    cfg = cfg or GateConfig()
    ids = sorted(set(candidate.subject_ids) & set(baseline.subject_ids))
    if len(ids) < 2:
        raise ValueError("need at least two shared subjects for a paired test")
    cand_m, base_m = candidate.subject_means(), baseline.subject_means()
    c = np.array([cand_m[i] for i in ids])
    b = np.array([base_m[i] for i in ids])
    diff = c - b
    n = len(diff)
    mean_d = float(diff.mean())
    sd = float(diff.std(ddof=1))
    if sd < 1e-12:
        t_stat = float("nan")
        p_t = 1.0 if abs(mean_d) < 1e-12 else 0.0
        dz = 0.0 if abs(mean_d) < 1e-12 else math.copysign(1e6, mean_d)
        ci = (mean_d, mean_d)
    else:
        t_stat, p_t = stats.ttest_rel(c, b)
        t_stat, p_t = float(t_stat), float(p_t)
        dz = mean_d / sd
        half = float(stats.t.ppf(0.5 + cfg.ci_level / 2.0, n - 1)) * sd / math.sqrt(n)
        ci = (mean_d - half, mean_d + half)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            p_w = float(stats.wilcoxon(diff, alternative="two-sided").pvalue)
        if not math.isfinite(p_w):
            p_w = 1.0
    except ValueError:
        p_w = 1.0
    cand_var = variance_decomposition(candidate.f1)
    base_var = variance_decomposition(baseline.f1)
    noise_terms = [v for v in (cand_var["seed_variance_within_subject"], base_var["seed_variance_within_subject"])
                   if math.isfinite(v)]
    return {
        "n_subjects": n,
        "mean_f1_candidate": float(c.mean()),
        "mean_f1_baseline": float(b.mean()),
        "mean_diff": mean_d,
        "t_stat": t_stat,
        "p_ttest": p_t,
        "p_wilcoxon": p_w,
        "cohens_dz": dz,
        "ci_low": float(ci[0]),
        "ci_high": float(ci[1]),
        "ci_level": cfg.ci_level,
        "seed_noise_sd": math.sqrt(sum(noise_terms)) if len(noise_terms) == 2 else float("nan"),
        "candidate_variance": cand_var,
        "baseline_variance": base_var,
    }


def check_provenance(
    record: ExperimentRecord, baseline: Optional[ExperimentRecord] = None, cfg: Optional[GateConfig] = None
) -> List[str]:
    """Hard provenance checks; returns the list of raised flags."""
    cfg = cfg or GateConfig()
    flags: List[str] = []
    n_seeds, n_subj = record.f1.shape
    if (n_seeds != len(record.seeds) or n_subj != len(record.subject_ids)
            or not np.all(np.isfinite(record.f1)) or not record.folds):
        return [MALFORMED]
    if baseline is not None and set(record.subject_ids) != set(baseline.subject_ids):
        flags.append(MALFORMED)
    if {f.test_subject for f in record.folds} != set(record.subject_ids):
        flags.append(MALFORMED)

    if any(f.protocol != "LOSO" or f.test_subject in f.train_subjects for f in record.folds):
        flags.append(SPLIT_LEAKAGE)
    if any(f.test_subject in f.fit_subjects or not set(f.fit_subjects) <= set(f.train_subjects)
           for f in record.folds):
        flags.append(FIT_LEAKAGE)

    if any(s is None for s in record.seeds):
        flags.append(UNSEEDED_RUN)
    explicit = {s for s in record.seeds if s is not None}
    if len(explicit) < cfg.min_seeds:
        flags.append(INSUFFICIENT_SEEDS)
    if record.n_runs_logged > len(record.seeds):
        flags.append(SEED_CHERRY_PICK)
    if record.reported_mean_f1 is not None and record.reported_mean_f1 > record.mean_f1() + 1e-6:
        flags.append(REPORT_MISMATCH)

    if not record.replay_pairs:
        flags.append(NO_REPLAY_CHECK)
    elif any(abs(a - b) > cfg.replay_atol for a, b in record.replay_pairs):
        flags.append(NONDETERMINISTIC)
    return flags


def _verdict(
    record: ExperimentRecord,
    hard_flags: List[str],
    st: Optional[Dict[str, Any]],
    p_holm: float,
    family_size: int,
    cfg: GateConfig,
) -> AuditReport:
    flags = list(hard_flags)
    if st is not None and MALFORMED not in flags:
        if st["mean_diff"] <= 0 or p_holm >= cfg.alpha:
            flags.append(NOT_SIGNIFICANT)
        if st["cohens_dz"] < cfg.min_effect_size:
            flags.append(SMALL_EFFECT)
        if st["ci_low"] <= 0:
            flags.append(CI_INCLUDES_ZERO)
        if st["mean_f1_candidate"] <= cfg.chance_f1 + cfg.chance_margin:
            flags.append(NOT_ABOVE_CHANCE)
        noise = st["seed_noise_sd"]
        if math.isfinite(noise) and st["mean_diff"] <= cfg.max_seed_noise_ratio * noise:
            flags.append(SEED_NOISE_DOMINATES)
    elif st is None and MALFORMED not in flags:
        flags.append(MALFORMED)
    verdict = "VERIFIED" if not flags else "REJECTED"
    return AuditReport(
        name=record.name,
        verdict=verdict,
        flags=flags,
        reasons=[FLAG_DESCRIPTIONS[f] for f in flags],
        stats=st or {},
        p_holm=float(p_holm),
        family_size=family_size,
    )


def audit_family(
    candidates: Sequence[ExperimentRecord],
    baseline: ExperimentRecord,
    cfg: Optional[GateConfig] = None,
    n_untested: int = 0,
) -> List[AuditReport]:
    """Audit a family of candidates with Holm-Bonferroni correction.

    Args:
        candidates: Records to audit against ``baseline``.
        baseline: Reference record evaluated by the same trusted harness.
        cfg: Gate thresholds.
        n_untested: Extra hypotheses that were attempted but produced no
            record (e.g. crashed candidates); they enter the Holm family with
            p = 1 so the family-wise error rate stays honest.

    Returns:
        One :class:`AuditReport` per candidate, in input order.
    """
    cfg = cfg or GateConfig()
    analyses: List[Tuple[List[str], Optional[Dict[str, Any]]]] = []
    for rec in candidates:
        hard = check_provenance(rec, baseline, cfg)
        st: Optional[Dict[str, Any]] = None
        if MALFORMED not in hard:
            try:
                st = paired_subject_stats(rec, baseline, cfg)
            except (ValueError, KeyError):
                hard.append(MALFORMED)
        analyses.append((hard, st))
    raw = [st["p_ttest"] if st else 1.0 for _, st in analyses] + [1.0] * n_untested
    _, adjusted = holm_bonferroni(raw, cfg.alpha)
    family = len(raw)
    return [
        _verdict(rec, hard, st, adjusted[i], family, cfg)
        for i, (rec, (hard, st)) in enumerate(zip(candidates, analyses))
    ]


def audit_candidate(
    candidate: ExperimentRecord,
    baseline: ExperimentRecord,
    family_size: int = 1,
    cfg: Optional[GateConfig] = None,
) -> AuditReport:
    """Audit one candidate inside a known family of ``family_size`` hypotheses.

    Without the other p-values the Holm adjustment is bounded by its worst
    case, ``min(1, family_size * p)``, which is what is applied here.  The
    benchmark re-audits the whole family with :func:`audit_family` for the
    exact Holm-adjusted p-values.
    """
    cfg = cfg or GateConfig()
    hard = check_provenance(candidate, baseline, cfg)
    st: Optional[Dict[str, Any]] = None
    if MALFORMED not in hard:
        try:
            st = paired_subject_stats(candidate, baseline, cfg)
        except (ValueError, KeyError):
            hard.append(MALFORMED)
    p_raw = st["p_ttest"] if st else 1.0
    return _verdict(candidate, hard, st, min(1.0, p_raw * max(1, family_size)), max(1, family_size), cfg)
