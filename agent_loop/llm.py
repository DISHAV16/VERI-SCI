"""LLM back-ends for the research agent.

Two interchangeable clients implement :class:`LLMClient`:

* :class:`AnthropicLLM` - real model calls (needs ``ANTHROPIC_API_KEY`` and the
  ``anthropic`` package).  Reliability numbers obtained with it are genuine
  measurements of that model.
* :class:`OfflineScriptedLLM` - a deterministic, network-free simulator with a
  library of 15 hypotheses and *injected* coding faults.  It exists so the
  whole loop (sandbox, debugger, auditor) can be exercised and tested without
  an API key.  Its zero-shot vs self-debugging reliability is a stress test of
  the loop, **not** a measurement of any real LLM.
"""

from __future__ import annotations

import os
import random
import re
from typing import Any, Dict, List, Optional, Protocol, Tuple

from .prompts import extract_code


class LLMClient(Protocol):
    """Minimal chat interface used by the graph nodes."""

    name: str

    def complete(self, system: str, user: str, role: str) -> str:
        """Return the model reply for ``role`` in ``{"planner", "coder", "debugger"}``."""


class AnthropicLLM:
    """Claude via the Anthropic Messages API."""

    def __init__(self, model: Optional[str] = None, max_tokens: int = 4000) -> None:
        import anthropic  # imported lazily so the offline path needs no SDK

        self._client = anthropic.Anthropic()
        self.model = model or os.environ.get("VERI_SCI_MODEL", "claude-sonnet-5-5")
        self.max_tokens = max_tokens
        self.name = f"anthropic:{self.model}"

    def complete(self, system: str, user: str, role: str) -> str:
        """Call the API and return the concatenated text blocks."""
        reply = self._client.messages.create(
            model=self.model, max_tokens=self.max_tokens, system=system,
            messages=[{"role": "user", "content": user}],
        )
        return "".join(block.text for block in reply.content if getattr(block, "type", "") == "text")


# --------------------------------------------------------------------------- #
# Offline scripted simulator
# --------------------------------------------------------------------------- #

_PRE_TEMPLATE = '''import numpy as np
import torch
import torch.nn as nn


def _norm(X):
    X = np.asarray(X, dtype=np.float32)
{car}    X = X - X.mean(axis=2, keepdims=True)
    std = X.std(axis=(1, 2), keepdims=True) + 1e-6
    return X / std


def preprocess_train(X_train, y_train):
    X = _norm(X_train)
    y = np.asarray(y_train)
{aug}    return X, y, _norm
'''

_CAR_LINE = "    X = X - X.mean(axis=1, keepdims=True)\n"
_AUG_BLOCK = '''    rng = np.random.default_rng(0)
    shifts = rng.integers(-8, 9, size=len(X))
    shifted = np.stack([np.roll(x, int(s), axis=1) for x, s in zip(X, shifts)])
    shifted = shifted + 0.1 * rng.standard_normal(X.shape).astype(np.float32)
    X = np.concatenate([X, shifted])
    y = np.concatenate([y, y])
'''

_NET_HEAD = '''

class Net(nn.Module):
    def __init__(self, c, t):
        super().__init__()
        self.features = nn.Sequential(
{layers}
        )
        with torch.no_grad():
            self.features.eval()
            self.n_feat = int(self.features(torch.zeros({dummy})).numel())
            self.features.train()
        self.head = nn.Linear(self.n_feat, 2)

    def forward(self, x):
{unsqueeze}        return self.head(self.features(x).flatten(1))


def build_model(in_channels, n_times):
    return Net(in_channels, n_times)
'''

_LAYERS = {
    "eegnet": '''            nn.Conv2d(1, {F}, (1, {K}), padding="same", bias=False),
            nn.BatchNorm2d({F}),
            nn.Conv2d({F}, {F} * {D}, (c, 1), groups={F}, bias=False),
            nn.BatchNorm2d({F} * {D}),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout({DROP}),
            nn.Conv2d({F} * {D}, {F} * {D}, (1, 16), groups={F} * {D}, padding="same", bias=False),
            nn.Conv2d({F} * {D}, {F} * {D}, 1, bias=False),
            nn.BatchNorm2d({F} * {D}),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout({DROP}),''',
    "shallow": '''            nn.Conv2d(1, {F}, (1, {K}), padding="same"),
            nn.Conv2d({F}, {F}, (c, 1), bias=False),
            nn.BatchNorm2d({F}),
            nn.ELU(),
            nn.AvgPool2d((1, {P}), stride=(1, {S})),
            nn.Dropout({DROP}),''',
    "cnn1d": '''            nn.Conv1d(c, {F}, {K}, padding="same"),
            nn.BatchNorm1d({F}),
            nn.ELU(),
            nn.AvgPool1d(4),
            nn.Conv1d({F}, {F}, {K2}, padding="same"),
            nn.BatchNorm1d({F}),
            nn.ELU(),
            nn.AdaptiveAvgPool1d(4),
            nn.Dropout({DROP}),''',
}

# (arch, params, car, aug, description)
_HYPOTHESES: List[Tuple[str, Dict[str, Any], bool, bool, str]] = [
    ("eegnet", dict(F=8, D=2, K=32, DROP=0.25), False, False, "EEGNet-8,2 with per-trial global scaling"),
    ("eegnet", dict(F=8, D=2, K=32, DROP=0.5), False, True, "EEGNet-8,2 with heavy dropout and time-shift/noise augmentation"),
    ("eegnet", dict(F=16, D=2, K=16, DROP=0.3), True, False, "wider EEGNet with short temporal kernel and common-average reference"),
    ("shallow", dict(F=20, K=25, P=30, S=7, DROP=0.5), False, False, "shallow temporal-spatial CNN with long pooling"),
    ("shallow", dict(F=16, K=15, P=25, S=5, DROP=0.4), True, False, "shallow CNN with common-average reference"),
    ("shallow", dict(F=24, K=25, P=40, S=8, DROP=0.5), False, True, "shallow CNN with augmentation"),
    ("cnn1d", dict(F=32, K=15, K2=7, DROP=0.3), False, False, "channel-as-feature 1D-CNN"),
    ("cnn1d", dict(F=48, K=25, K2=9, DROP=0.5), True, False, "larger 1D-CNN with common-average reference"),
    ("cnn1d", dict(F=32, K=15, K2=7, DROP=0.4), False, True, "1D-CNN with augmentation"),
    ("eegnet", dict(F=4, D=2, K=64, DROP=0.25), False, False, "narrow EEGNet with long temporal kernel"),
    ("eegnet", dict(F=8, D=4, K=32, DROP=0.4), True, True, "EEGNet with depth multiplier 4, CAR and augmentation"),
    ("shallow", dict(F=12, K=35, P=30, S=6, DROP=0.5), False, False, "shallow CNN with long temporal filters"),
    ("cnn1d", dict(F=16, K=11, K2=5, DROP=0.3), True, True, "small 1D-CNN with CAR and augmentation"),
    ("eegnet", dict(F=12, D=2, K=24, DROP=0.35), False, True, "EEGNet-12,2 with augmentation"),
    ("shallow", dict(F=32, K=20, P=35, S=10, DROP=0.5), True, True, "wide shallow CNN with CAR and augmentation"),
]

_FAULT_KINDS = ("forbidden_import", "bad_return", "undefined_name", "shape_mismatch")


def hypothesis_count() -> int:
    """Number of scripted hypotheses available to the offline Planner."""
    return len(_HYPOTHESES)


def _clean_source(index: int) -> str:
    arch, params, car, aug, _ = _HYPOTHESES[index % len(_HYPOTHESES)]
    layers = _LAYERS[arch].format(**params)
    two_d = arch != "cnn1d"
    head = _NET_HEAD.format(
        layers=layers,
        dummy="1, 1, c, t" if two_d else "1, c, t",
        unsqueeze="        x = x.unsqueeze(1)\n" if two_d else "",
    )
    pre = _PRE_TEMPLATE.format(car=_CAR_LINE if car else "", aug=_AUG_BLOCK if aug else "")
    return pre + head


def _apply_fault(code: str, kind: str) -> str:
    if kind == "forbidden_import":
        return "import os\n_N_THREADS = os.cpu_count()\n" + code
    if kind == "bad_return":
        return code.replace("    return X, y, _norm\n", "    return X\n")
    if kind == "undefined_name":
        return code.replace("return Net(in_channels, n_times)", "return NetModel(in_channels, n_times)")
    if kind == "shape_mismatch":
        return code.replace("nn.Linear(self.n_feat, 2)", "nn.Linear(self.n_feat + 1, 2)")
    raise ValueError(kind)


def _fault_plan(trial_id: int) -> List[str]:
    """Deterministic list of faults injected into the first attempt of a trial."""
    rng = random.Random(1000 + trial_id)
    roll = rng.random()
    if roll < 0.55:
        return []
    if roll < 0.85:
        n = 1
    elif roll < 0.95:
        n = 2
    else:
        n = len(_FAULT_KINDS)
    return rng.sample(list(_FAULT_KINDS), n)


class OfflineScriptedLLM:
    """Deterministic simulator of a fallible code-generating LLM (no network)."""

    name = "offline-scripted-simulator"

    def complete(self, system: str, user: str, role: str) -> str:
        """Dispatch on ``role``; see class docstring for the simulation semantics."""
        trial = int(re.search(r"Trial:\s*(\d+)", user).group(1))  # type: ignore[union-attr]
        if role == "planner":
            idx = trial % len(_HYPOTHESES)
            return f"[H{idx}] Test {_HYPOTHESES[idx][4]}."
        hyp_idx = int(re.search(r"\[H(\d+)\]", user).group(1))  # type: ignore[union-attr]
        if role == "coder":
            code = _clean_source(hyp_idx)
            for kind in _fault_plan(trial):
                code = _apply_fault(code, kind)
            return f"```python\n{code}```"
        if role == "debugger":
            code = extract_code(re.search(r"```python\n(.*?)```", user, re.DOTALL).group(1))  # type: ignore[union-attr]
            logs = re.search(r"<log>\n(.*?)</log>", user, re.DOTALL).group(1)  # type: ignore[union-attr]
            return f"```python\n{self._fix(code, logs)}```"
        raise ValueError(f"unknown role {role!r}")

    @staticmethod
    def _fix(code: str, logs: str) -> str:
        """Repair the first fault visible in the log, mimicking a one-bug-per-round debugger."""
        if "forbidden import" in logs or "SecurityViolation" in logs:
            kept = [ln for ln in code.splitlines() if "import os" not in ln and "os." not in ln]
            return "\n".join(kept) + "\n"
        if "must return a tuple" in logs:
            return code.replace("    return X\n", "    return X, y, _norm\n")
        if "NetModel" in logs:
            return code.replace("NetModel", "Net")
        if "shapes cannot be multiplied" in logs or "size mismatch" in logs or "mat1 and mat2" in logs:
            return code.replace("self.n_feat + 1", "self.n_feat")
        return code
