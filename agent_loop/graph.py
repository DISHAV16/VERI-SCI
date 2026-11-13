"""Cyclic LangGraph state machine of the self-verifying research agent.

::

    Planner -> Coder -> SandboxRunner --ok--> StatisticalAuditor -> MemoryLogger -> END
                            |  ^
                   traceback|  |fixed code
                            v  |
                          Debugger   (at most ``max_retries`` = 3 rounds)

A candidate that still fails after the retry budget goes straight to the
MemoryLogger with verdict ``NOT_EVALUATED``.  Only candidates that execute are
sent to the StatisticalAuditor, which runs the full LOSO x seeds evaluation in
the sandbox and applies the independent statistical gate.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from langgraph.graph import END, StateGraph

from auditor.stats_gate import (
    ExperimentRecord,
    GateConfig,
    audit_candidate,
    variance_decomposition,
)
from sandbox.runner import CandidateEvaluator

from .llm import LLMClient
from .prompts import (
    CODER_SYSTEM,
    DEBUGGER_SYSTEM,
    PLANNER_SYSTEM,
    build_coder_prompt,
    build_debugger_prompt,
    build_planner_prompt,
    extract_code,
)
from .state import AgentState


@dataclass
class AgentConfig:
    """Dependencies and settings shared by all graph nodes."""

    llm: LLMClient
    evaluator: CandidateEvaluator
    baseline: ExperimentRecord
    max_retries: int = 3
    family_size: int = 15
    gate: GateConfig = field(default_factory=GateConfig)
    memory_path: Optional[Path] = None
    log_fn: Optional[Callable[[Dict[str, Any]], None]] = None
    memory: List[Dict[str, Any]] = field(default_factory=list)


def build_graph(cfg: AgentConfig) -> Any:
    """Compile the Planner -> Coder -> SandboxRunner -> Debugger/Auditor -> MemoryLogger graph."""

    def planner(state: AgentState) -> Dict[str, Any]:
        reply = cfg.llm.complete(PLANNER_SYSTEM, build_planner_prompt(state["trial_id"], cfg.memory), "planner")
        return {"hypothesis": reply.strip(), "retry_count": 0, "attempts": [], "first_attempt_ok": None}

    def coder(state: AgentState) -> Dict[str, Any]:
        reply = cfg.llm.complete(CODER_SYSTEM, build_coder_prompt(state["trial_id"], state["hypothesis"]), "coder")
        return {"generated_code": extract_code(reply)}

    def sandbox_runner(state: AgentState) -> Dict[str, Any]:
        result = cfg.evaluator.smoke(state["generated_code"])
        attempts = list(state.get("attempts", []))
        attempts.append({"attempt": len(attempts), "ok": result.ok, "log": result.log[-500:]})
        update: Dict[str, Any] = {
            "sandbox_logs": result.log, "executable": result.ok, "attempts": attempts,
        }
        if state.get("first_attempt_ok") is None:
            update["first_attempt_ok"] = result.ok
        return update

    def route_after_sandbox(state: AgentState) -> str:
        if state["executable"]:
            return "auditor"
        if state["retry_count"] < state["max_retries"]:
            return "debugger"
        return "logger"

    def debugger(state: AgentState) -> Dict[str, Any]:
        prompt = build_debugger_prompt(
            state["trial_id"], state["hypothesis"], state["generated_code"], state["sandbox_logs"])
        reply = cfg.llm.complete(DEBUGGER_SYSTEM, prompt, "debugger")
        return {"generated_code": extract_code(reply), "retry_count": state["retry_count"] + 1}

    def auditor(state: AgentState) -> Dict[str, Any]:
        name = f"trial_{state['trial_id']:02d}"
        outcome = cfg.evaluator.evaluate(state["generated_code"], name)
        if not outcome.ok or outcome.record is None:
            return {
                "executable": False, "sandbox_logs": outcome.logs,
                "audit_verdict": {"verdict": "NOT_EVALUATED", "flags": ["FULL_EVALUATION_FAILED"]},
            }
        record = outcome.record
        report = audit_candidate(record, cfg.baseline, family_size=cfg.family_size, cfg=cfg.gate)
        return {
            "record": record,
            "subject_f1_scores": [record.subject_means()[s] for s in record.subject_ids],
            "seed_variance": variance_decomposition(record.f1),
            "audit_verdict": report.to_dict(),
        }

    def memory_logger(state: AgentState) -> Dict[str, Any]:
        verdict = state.get("audit_verdict") or {"verdict": "NOT_EVALUATED", "flags": ["NOT_EXECUTABLE"]}
        scores = state.get("subject_f1_scores")
        entry = {
            "trial_id": state["trial_id"],
            "hypothesis": state["hypothesis"],
            "executable": bool(state.get("executable")),
            "first_attempt_ok": state.get("first_attempt_ok"),
            "retries": state["retry_count"],
            "mean_f1": float(sum(scores) / len(scores)) if scores else None,
            "verdict": verdict["verdict"],
            "flags": verdict.get("flags", []),
        }
        cfg.memory.append(entry)
        if cfg.memory_path is not None:
            cfg.memory_path.parent.mkdir(parents=True, exist_ok=True)
            with cfg.memory_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(entry) + "\n")
        if cfg.log_fn is not None:
            cfg.log_fn({"event": "trial", **entry})
        return {"audit_verdict": verdict}

    graph = StateGraph(AgentState)
    graph.add_node("planner", planner)
    graph.add_node("coder", coder)
    graph.add_node("sandbox_runner", sandbox_runner)
    graph.add_node("debugger", debugger)
    graph.add_node("statistical_auditor", auditor)
    graph.add_node("memory_logger", memory_logger)
    graph.set_entry_point("planner")
    graph.add_edge("planner", "coder")
    graph.add_edge("coder", "sandbox_runner")
    graph.add_conditional_edges(
        "sandbox_runner", route_after_sandbox,
        {"auditor": "statistical_auditor", "debugger": "debugger", "logger": "memory_logger"},
    )
    graph.add_edge("debugger", "sandbox_runner")
    graph.add_edge("statistical_auditor", "memory_logger")
    graph.add_edge("memory_logger", END)
    return graph.compile()


def run_trial(graph: Any, trial_id: int, mode: str = "self_debug", max_retries: int = 3) -> AgentState:
    """Run one automated research cycle and return the final state.

    Args:
        graph: Compiled graph from :func:`build_graph`.
        trial_id: Research-cycle index.
        mode: ``"self_debug"`` or ``"zero_shot"`` (forces zero retries).
        max_retries: Debugger rounds allowed in self-debug mode.
    """
    retries = 0 if mode == "zero_shot" else max_retries
    initial: AgentState = {"trial_id": trial_id, "mode": mode, "max_retries": retries}
    return graph.invoke(initial, config={"recursion_limit": 50})
