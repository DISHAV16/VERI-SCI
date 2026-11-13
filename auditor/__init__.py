"""Independent statistical auditing gate."""

from .stats_gate import (
    AuditReport,
    ExperimentRecord,
    FoldManifest,
    GateConfig,
    audit_candidate,
    audit_family,
    build_loso_manifest,
    holm_bonferroni,
    paired_subject_stats,
    to_jsonable,
    variance_decomposition,
)

__all__ = [
    "AuditReport",
    "ExperimentRecord",
    "FoldManifest",
    "GateConfig",
    "audit_candidate",
    "audit_family",
    "build_loso_manifest",
    "holm_bonferroni",
    "paired_subject_stats",
    "to_jsonable",
    "variance_decomposition",
]
