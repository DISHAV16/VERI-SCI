"""Prompt templates for the Planner, Coder and Debugger.

The Coder and Debugger are instructed to emit **only** the two functions
``preprocess_train(X_train, y_train)`` and ``build_model(in_channels, n_times)``
(plus tiny helper classes/functions they need) inside one ``python`` block.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List

CODE_CONTRACT = """\
You write candidate code for an EEG left-vs-right-hand motor-imagery classifier.
Data: X float32 array of shape (n_trials, 64, 320) (8-30 Hz band-passed, 80 Hz), y int labels in {0, 1}.

Output ONLY one ```python code block that defines exactly these two public functions:

  def preprocess_train(X_train, y_train):
      # Return (X_out, y_out, transform).
      # X_out: float32 (n, channels, times); y_out: labels aligned with X_out.
      # transform: a callable applied to the held-out test trials, or None. It must be
      # per-trial (stateless) or use only statistics computed from X_train. It must NEVER
      # use statistics of the data it is applied to across trials (that is test leakage).

  def build_model(in_channels, n_times):
      # Return a torch.nn.Module mapping (batch, in_channels, n_times) -> (batch, 2) logits.

Rules (violations are rejected by a static AST guard before execution):
- Allowed imports: numpy, torch, torch.nn, torch.nn.functional, math, scipy, sklearn, typing.
- Forbidden: os, subprocess, sys, shutil, socket, any file/network I/O, eval/exec/open,
  dunder tricks, and ANY reference to the evaluation package `trusted_eval`.
- No top-level code other than imports, function/class definitions and constants.
- Do NOT write a training loop, data loading or evaluation code: the trusted harness trains
  (Adam, cross-entropy, fixed epochs) and evaluates with Leave-One-Subject-Out.
- No explanations, no prose, no extra code blocks.
"""

PLANNER_SYSTEM = (
    "You are the Planner of an autonomous research agent for cross-subject EEG motor-imagery "
    "decoding. Propose ONE new, concrete, testable hypothesis (architecture and/or preprocessing "
    "change) that could improve Leave-One-Subject-Out macro-F1 over an EEGNet baseline. "
    "Reply with a single short paragraph and nothing else."
)
CODER_SYSTEM = "You are the Coder of an autonomous research agent.\n" + CODE_CONTRACT
DEBUGGER_SYSTEM = (
    "You are the Debugger of an autonomous research agent. Fix the failing candidate so that it "
    "satisfies the contract below and executes successfully. Keep the scientific idea unchanged.\n"
    + CODE_CONTRACT
)


def build_planner_prompt(trial_id: int, memory: List[Dict[str, Any]]) -> str:
    """Planner user prompt including the most recent audited hypotheses."""
    lines = [f"Trial: {trial_id}", "Recent audited hypotheses (most recent last):"]
    if not memory:
        lines.append("- none yet")
    for item in memory[-5:]:
        f1 = item.get("mean_f1")
        f1_txt = "n/a" if f1 is None else f"{f1:.3f}"
        lines.append(f"- {item.get('hypothesis', '')[:160]} -> verdict={item.get('verdict')} meanF1={f1_txt}")
    lines.append("Propose a hypothesis that has not been tried yet.")
    return "\n".join(lines)


def build_coder_prompt(trial_id: int, hypothesis: str) -> str:
    """Coder user prompt for a hypothesis."""
    return f"Trial: {trial_id}\nHypothesis: {hypothesis}\nWrite the candidate code now."


def build_debugger_prompt(trial_id: int, hypothesis: str, code: str, logs: str) -> str:
    """Debugger user prompt carrying the failing code and its sandbox log."""
    return (
        f"Trial: {trial_id}\nHypothesis: {hypothesis}\n"
        f"The candidate below failed.\nCurrent code:\n```python\n{code}\n```\n"
        f"Execution log:\n<log>\n{logs}\n</log>\nReturn the corrected code."
    )


_FENCE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL)


def extract_code(text: str) -> str:
    """Extract the first fenced Python block, or the raw text if no fence is present."""
    match = _FENCE.search(text)
    return (match.group(1) if match else text).strip() + "\n"
