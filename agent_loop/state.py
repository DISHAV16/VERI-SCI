"""Shared state of the cyclic research-agent graph."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, TypedDict


class AgentState(TypedDict, total=False):
    """State passed between LangGraph nodes.

    Attributes:
        trial_id: Index of the automated research cycle.
        mode: ``"self_debug"`` (retries enabled) or ``"zero_shot"``.
        max_retries: Debugger retries allowed (0 for zero-shot).
        hypothesis: Natural-language hypothesis proposed by the Planner.
        generated_code: Current candidate source (``preprocess_train`` / ``build_model``).
        sandbox_logs: Latest sandbox / AST-guard log (traceback on failure).
        retry_count: Number of Debugger invocations so far.
        executable: Whether the latest candidate executed successfully.
        first_attempt_ok: Whether the very first generated candidate executed.
        attempts: Per-attempt outcome log ``{"attempt", "ok", "log"}``.
        subject_f1_scores: Seed-averaged macro-F1 per LOSO subject.
        seed_variance: Within-subject seed variance vs between-subject variance.
        audit_verdict: Gate report (``VERIFIED`` / ``REJECTED`` / ``NOT_EVALUATED``).
        record: The trusted ``ExperimentRecord`` of the evaluated candidate.
    """

    trial_id: int
    mode: str
    max_retries: int
    hypothesis: str
    generated_code: str
    sandbox_logs: str
    retry_count: int
    executable: bool
    first_attempt_ok: Optional[bool]
    attempts: List[Dict[str, Any]]
    subject_f1_scores: List[float]
    seed_variance: Dict[str, float]
    audit_verdict: Dict[str, Any]
    record: Any
