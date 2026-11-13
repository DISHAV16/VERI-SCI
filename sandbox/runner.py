"""Isolated subprocess execution of candidate PyTorch code.

Design (what keeps the evaluation trustworthy):

* The candidate only supplies ``preprocess_train`` and ``build_model``.  The
  *training loop is owned by the trusted harness* below, so a candidate cannot
  alter optimisation, evaluation or metrics.
* The subprocess receives ``X_train``, ``y_train`` and **unlabeled** ``X_test``.
  Test labels never leave the parent process; predictions come back as JSON
  and the parent computes macro-F1.
* The harness verifies that the test-time ``transform`` is per-sample
  consistent, which detects transductive normalisation that peeks at test
  statistics.
* Each fold run has a strict wall-clock timeout (default 120 s), a cleaned
  environment, a throw-away working directory and (on POSIX) CPU / file-size
  rlimits.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.metrics import f1_score

from auditor.stats_gate import ExperimentRecord, FoldManifest
from trusted_eval.data_loader import EEGDataset, LOSOSplit, loso_splits

from .ast_guard import check_code

DEFAULT_TIMEOUT_S = 120.0
DEFAULT_EPOCHS = 30
RESULT_PREFIX = "HARNESS_RESULT:"
LOG_TAIL_CHARS = 3000

HARNESS_SOURCE = r'''
import json
import random
import sys
import time

import numpy as np


def _limit_resources(cpu_seconds):
    try:
        import resource
        resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds))
        resource.setrlimit(resource.RLIMIT_FSIZE, (256 * 1024 ** 2, 256 * 1024 ** 2))
    except Exception:
        pass


def main():
    with open("job.json") as fh:
        job = json.load(fh)
    _limit_resources(int(job["cpu_limit_s"]))
    seed = int(job["seed"])
    data = np.load("data.npz")
    X_train, y_train, X_test = data["X_train"], data["y_train"], data["X_test"]
    with open("candidate.py") as fh:
        source = fh.read()

    import torch
    import torch.nn.functional as F
    torch.set_num_threads(1)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass

    namespace = {"__name__": "candidate"}
    exec(compile(source, "candidate.py", "exec"), namespace)
    out = namespace["preprocess_train"](X_train.copy(), y_train.copy())
    if not isinstance(out, tuple) or len(out) not in (2, 3):
        raise TypeError("preprocess_train must return a tuple (X_train, y_train, transform)")
    X_tr, y_tr = np.asarray(out[0], dtype=np.float32), np.asarray(out[1]).astype(np.int64)
    transform = out[2] if len(out) == 3 else None
    if X_tr.ndim != 3 or len(X_tr) != len(y_tr) or len(X_tr) == 0:
        raise ValueError("preprocess_train must return X of shape (n, channels, times) and matching y")
    if not np.all(np.isfinite(X_tr)):
        raise ValueError("preprocess_train returned non-finite values")
    if set(np.unique(y_tr).tolist()) - {0, 1}:
        raise ValueError("labels must stay in {0, 1}")

    if transform is not None:
        X_te = np.asarray(transform(X_test.copy()), dtype=np.float32)
        for i in range(min(8, len(X_test))):
            single = np.asarray(transform(X_test[i:i + 1].copy()), dtype=np.float32)
            if single.shape[0] != 1 or not np.allclose(single[0], X_te[i], rtol=1e-3, atol=1e-4):
                raise ValueError(
                    "TransductiveTransformError: transform output for a test trial depends on other "
                    "test trials (test-set statistics leakage). The transform must be per-trial or "
                    "use only statistics fitted on the training data."
                )
    else:
        X_te = np.asarray(X_test, dtype=np.float32)
    if X_te.ndim != 3 or X_te.shape[1:] != X_tr.shape[1:]:
        raise ValueError("transform must map test data to the same (channels, times) shape as training data")

    Xt = torch.from_numpy(np.ascontiguousarray(X_tr))
    yt = torch.from_numpy(y_tr)
    model = namespace["build_model"](int(Xt.shape[1]), int(Xt.shape[2]))
    if not isinstance(model, torch.nn.Module):
        raise TypeError("build_model must return a torch.nn.Module")
    optimizer = torch.optim.Adam(model.parameters(), lr=float(job["lr"]), weight_decay=1e-4)
    generator = torch.Generator().manual_seed(seed)
    batch_size = int(job["batch_size"])
    t0 = time.time()
    last_loss = float("nan")
    for _ in range(int(job["epochs"])):
        model.train()
        perm = torch.randperm(len(Xt), generator=generator)
        for start in range(0, len(Xt), batch_size):
            idx = perm[start:start + batch_size]
            if len(idx) < 2:
                continue
            optimizer.zero_grad()
            logits = model(Xt[idx])
            if logits.ndim != 2 or logits.shape[1] != 2:
                raise ValueError("model must output logits of shape (batch, 2), got %s" % (tuple(logits.shape),))
            loss = F.cross_entropy(logits, yt[idx])
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite training loss")
            loss.backward()
            optimizer.step()
            last_loss = float(loss.item())

    model.eval()
    preds = []
    with torch.no_grad():
        Xe = torch.from_numpy(np.ascontiguousarray(X_te))
        for start in range(0, len(Xe), 64):
            preds.extend(model(Xe[start:start + 64]).argmax(dim=1).tolist())
    print("HARNESS_RESULT:" + json.dumps({
        "preds": preds, "train_seconds": time.time() - t0, "final_loss": last_loss,
    }))


if __name__ == "__main__":
    main()
'''


@dataclass
class FoldResult:
    """Outcome of one (test subject, seed) sandbox run."""

    subject: int
    seed: int
    ok: bool
    f1: Optional[float]
    log: str
    seconds: float


@dataclass
class EvaluationOutcome:
    """Outcome of a full candidate evaluation (all folds, all seeds, replay)."""

    ok: bool
    record: Optional[ExperimentRecord]
    logs: str
    folds: List[FoldResult] = field(default_factory=list)


def _tail(text: str) -> str:
    return text if len(text) <= LOG_TAIL_CHARS else "...[truncated]...\n" + text[-LOG_TAIL_CHARS:]


def _clean_env() -> Dict[str, str]:
    env = {"PYTHONHASHSEED": "0", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
           "CUDA_VISIBLE_DEVICES": "", "PATH": os.environ.get("PATH", "")}
    for key in ("SYSTEMROOT", "TEMP", "TMP", "HOME", "USERPROFILE", "LD_LIBRARY_PATH"):
        if key in os.environ:
            env[key] = os.environ[key]
    return env


def run_fold(
    source: str,
    split: LOSOSplit,
    seed: int,
    *,
    epochs: int = DEFAULT_EPOCHS,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    batch_size: int = 32,
    lr: float = 1e-3,
) -> FoldResult:
    """Execute candidate code on one LOSO fold inside an isolated subprocess.

    The AST guard runs first; rejected code is never executed.  Test labels are
    kept in this (parent) process and used only to score returned predictions.
    """
    start = time.time()
    report = check_code(source)
    if not report.ok:
        return FoldResult(split.test_subject, seed, False, None, report.format(), time.time() - start)
    with tempfile.TemporaryDirectory(prefix="veri_sci_") as tmp:
        work = Path(tmp)
        (work / "candidate.py").write_text(source, encoding="utf-8")
        (work / "harness.py").write_text(HARNESS_SOURCE, encoding="utf-8")
        np.savez(work / "data.npz", X_train=split.X_train, y_train=split.y_train, X_test=split.X_test)
        (work / "job.json").write_text(json.dumps({
            "seed": seed, "epochs": epochs, "batch_size": batch_size, "lr": lr,
            "cpu_limit_s": int(timeout_s) + 10,
        }), encoding="utf-8")
        try:
            proc = subprocess.run(
                [sys.executable, "-I", "harness.py"], cwd=work, env=_clean_env(),
                capture_output=True, text=True, timeout=timeout_s,
            )
        except subprocess.TimeoutExpired:
            return FoldResult(split.test_subject, seed, False, None,
                              f"TimeoutError: candidate exceeded the {timeout_s:.0f}s limit", time.time() - start)
    elapsed = time.time() - start
    line = next((ln for ln in proc.stdout.splitlines() if ln.startswith(RESULT_PREFIX)), None)
    if proc.returncode != 0 or line is None:
        return FoldResult(split.test_subject, seed, False, None,
                          _tail(proc.stderr or proc.stdout or "no output"), elapsed)
    preds = np.asarray(json.loads(line[len(RESULT_PREFIX):])["preds"], dtype=int)
    if preds.shape != split.y_test.shape or not set(np.unique(preds).tolist()) <= {0, 1}:
        return FoldResult(split.test_subject, seed, False, None, "invalid prediction vector", elapsed)
    f1 = float(f1_score(split.y_test, preds, average="macro", labels=[0, 1], zero_division=0))
    return FoldResult(split.test_subject, seed, True, f1, "ok", elapsed)


class CandidateEvaluator:
    """Evaluates candidate source code over all LOSO folds and seeds."""

    def __init__(
        self,
        dataset: EEGDataset,
        seeds: Sequence[int] = (0, 1, 2),
        epochs: int = DEFAULT_EPOCHS,
        timeout_s: float = DEFAULT_TIMEOUT_S,
        max_workers: int = 2,
    ) -> None:
        self.dataset = dataset
        self.seeds = tuple(seeds)
        self.epochs = epochs
        self.timeout_s = timeout_s
        self.max_workers = max(1, max_workers)
        self._splits = list(loso_splits(dataset))

    def smoke(self, source: str) -> FoldResult:
        """Run a single fold (first subject, first seed) to check the candidate executes."""
        return run_fold(source, self._splits[0], self.seeds[0], epochs=self.epochs, timeout_s=self.timeout_s)

    def evaluate(self, source: str, name: str) -> EvaluationOutcome:
        """Evaluate every (subject, seed) fold plus one replay of the first fold."""
        jobs: List[Tuple[int, int]] = [(j, seed) for seed in self.seeds for j in range(len(self._splits))]
        jobs.append((0, self.seeds[0]))  # replay of the very first run

        def work(job: Tuple[int, int]) -> FoldResult:
            return run_fold(source, self._splits[job[0]], job[1], epochs=self.epochs, timeout_s=self.timeout_s)

        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            results = list(pool.map(work, jobs))
        failed = [r for r in results if not r.ok]
        if failed:
            return EvaluationOutcome(False, None, failed[0].log, results)
        n_sub = len(self._splits)
        f1 = np.zeros((len(self.seeds), n_sub))
        for (j, seed), res in zip(jobs[:-1], results[:-1]):
            f1[self.seeds.index(seed), j] = res.f1
        folds = [FoldManifest(s.test_subject, s.train_subjects, s.train_subjects, "LOSO") for s in self._splits]
        record = ExperimentRecord(
            name=name,
            subject_ids=[s.test_subject for s in self._splits],
            seeds=list(self.seeds),
            f1=f1,
            folds=folds,
            n_runs_logged=len(self.seeds),
            replay_pairs=[(float(results[0].f1), float(results[-1].f1))],
            metadata={"epochs": self.epochs, "mean_fold_seconds": float(np.mean([r.seconds for r in results]))},
        )
        return EvaluationOutcome(True, record, "ok", results)
