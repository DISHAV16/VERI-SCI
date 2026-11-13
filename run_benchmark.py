"""VERI-SCI benchmark: 15 autonomous trials + 5 injected-trap trials (N = 20 cycles).

Phases
------
A. Baseline EEGNet over 5 LOSO subjects x 3 seeds (trusted harness).
B. 15 autonomous research cycles.  The first generated candidate of each cycle
   is the *zero-shot* outcome; the final outcome after up to 3 debugger rounds
   is the *self-debugging* outcome (paired design: both share the same first
   LLM sample).
C. 5 injected invalid-experiment traps audited by the statistical gate.

Outputs: ``results/results_table.md`` (also copied to ``./results_table.md``),
``results/benchmark_results.json``, ``results/agent_memory.jsonl``, W&B logs
(or ``results/wandb_offline_log.jsonl``), and the README benchmark section.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from agent_loop.graph import AgentConfig, build_graph, run_trial
from agent_loop.llm import LLMClient, OfflineScriptedLLM
from auditor.stats_gate import (
    ExperimentRecord,
    GateConfig,
    audit_family,
    to_jsonable,
    variance_decomposition,
)
from baselines.baseline_eegnet import BASELINE_SOURCE
from sandbox.runner import CandidateEvaluator
from trusted_eval.data_loader import DataUnavailableError, load_dataset
from trusted_eval.trap_generator import generate_traps

ROOT = Path(__file__).resolve().parent
README_START, README_END = "<!-- BENCHMARK_TABLE_START -->", "<!-- BENCHMARK_TABLE_END -->"
LOGGER = logging.getLogger("veri_sci")


class ExperimentLogger:
    """Weights & Biases logger with an always-on offline JSON-lines mirror."""

    def __init__(self, out_dir: Path, project: str = "veri-sci", use_wandb: bool = True) -> None:
        self.path = out_dir / "wandb_offline_log.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text("", encoding="utf-8")
        self._run = None
        self.backend = "json-offline"
        if use_wandb:
            try:
                import wandb

                mode = "online" if os.environ.get("WANDB_API_KEY") else "offline"
                self._run = wandb.init(project=project, mode=mode, dir=str(out_dir), reinit=True)
                self.backend = f"wandb-{mode}+json"
            except Exception as exc:  # wandb missing or misconfigured -> JSON only
                LOGGER.warning("W&B unavailable (%s); logging to %s only", exc, self.path)

    def log(self, payload: Dict[str, Any]) -> None:
        """Log one event to W&B (if active) and to the JSON-lines file."""
        clean = to_jsonable(payload)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(clean) + "\n")
        if self._run is not None:
            try:
                self._run.log({k: v for k, v in clean.items() if isinstance(v, (int, float, str, bool))})
            except Exception:  # pragma: no cover - never let logging break a benchmark
                pass

    def finish(self) -> None:
        """Close the W&B run if one is open."""
        if self._run is not None:
            try:
                self._run.finish()
            except Exception:  # pragma: no cover
                pass


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--synthetic", action="store_true", help="use synthetic EEG (smoke test only, NOT real results)")
    p.add_argument("--allow-synthetic-fallback", action="store_true",
                   help="fall back to synthetic data if the EEGBCI download fails")
    p.add_argument("--data-dir", type=Path, default=None, help="MNE data cache directory")
    p.add_argument("--llm", choices=["auto", "anthropic", "offline"], default="auto")
    p.add_argument("--trials", type=int, default=15, help="number of autonomous trials")
    p.add_argument("--epochs", type=int, default=30, help="training epochs per fold run")
    p.add_argument("--workers", type=int, default=2, help="parallel sandbox subprocesses")
    p.add_argument("--timeout", type=float, default=120.0, help="per-run sandbox timeout in seconds")
    p.add_argument("--output-dir", type=Path, default=ROOT / "results")
    p.add_argument("--no-wandb", action="store_true")
    p.add_argument("--no-readme", action="store_true", help="do not update the README benchmark section")
    return p.parse_args()


def make_llm(choice: str) -> LLMClient:
    """Select the LLM backend (real Anthropic model if a key is present, else offline simulator)."""
    if choice in ("auto", "anthropic") and os.environ.get("ANTHROPIC_API_KEY"):
        try:
            from agent_loop.llm import AnthropicLLM

            return AnthropicLLM()
        except ImportError:
            if choice == "anthropic":
                raise SystemExit("pip install anthropic to use --llm anthropic")
    elif choice == "anthropic":
        raise SystemExit("--llm anthropic requires the ANTHROPIC_API_KEY environment variable")
    return OfflineScriptedLLM()


def _fmt(value: Optional[float], digits: int = 4) -> str:
    return "n/a" if value is None or not np.isfinite(value) else f"{value:.{digits}f}"


def _fmt_p(p: Optional[float]) -> str:
    if p is None or not np.isfinite(p):
        return "n/a"
    return "<0.0001" if p < 1e-4 else f"{p:.4f}"


def build_markdown(res: Dict[str, Any]) -> str:
    """Render the results table markdown from the benchmark result dictionary."""
    best = res["best"]
    lines = [
        "# VERI-SCI benchmark results",
        "",
        f"*Generated {res['timestamp']} - data source: **{res['data_source']}** - "
        f"LLM backend: **{res['llm']}** - epochs/run: {res['epochs']} - logging: {res['logger']}*",
        "",
    ]
    if res["data_source"] != "eegbci":
        lines += ["> **WARNING: synthetic data. These numbers are a software smoke test, NOT real EEG results.**", ""]
    if res["llm"].startswith("offline"):
        lines += ["> **NOTE: offline-scripted LLM simulator with injected coding faults.** The zero-shot vs "
                  "self-debugging reliability below stress-tests the loop; it is not a measurement of a real LLM. "
                  "Re-run with `ANTHROPIC_API_KEY` set for real-LLM numbers.", ""]
    lines += [
        "## Headline metrics",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Total automated research cycles (N) | **{res['n_cycles']}** ({res['n_trials']} autonomous + {res['n_traps']} injected traps) |",
        f"| Executable run reliability - Zero-Shot | **{res['zero_shot_pct']:.1f}%** ({res['zero_shot_ok']}/{res['n_trials']}) |",
        f"| Executable run reliability - Self-Debugging Agent | **{res['self_debug_pct']:.1f}%** ({res['self_debug_ok']}/{res['n_trials']}) |",
        f"| Invalid-experiment trap detection rate | **{res['trap_rate_pct']:.0f}%** ({res['traps_rejected']}/{res['n_traps']} rejected) |",
        f"| Mean LOSO F1 - Baseline (EEGNet) | **{_fmt(res['baseline_mean_f1'])}** |",
    ]
    if best is None:
        lines += ["| Mean LOSO F1 - Best verified discovery | **none** (no executable candidate) |", ""]
    else:
        status = "Best verified discovery" if best["verified"] else "Best candidate (**NOT verified**)"
        st = best["stats"]
        lines += [
            f"| Mean LOSO F1 - {status} | **{_fmt(st['mean_f1_candidate'])}** (`{best['name']}`) |",
            f"| Holm-Bonferroni adjusted p-value (paired t-test, family m={best['family_size']}) | **{_fmt_p(best['p_holm'])}** |",
            f"| Cohen's d_z (paired, subject-level) | **{_fmt(st['cohens_dz'], 3)}** |",
            f"| 95% CI of mean F1 improvement | **[{_fmt(st['ci_low'])}, {_fmt(st['ci_high'])}]** |",
            f"| Wilcoxon p (n={st['n_subjects']} subjects, informational) | {_fmt_p(st['p_wilcoxon'])} |",
            "",
        ]
        if not best["verified"]:
            lines += [f"Gate flags for this candidate: `{', '.join(best['flags'])}`.", ""]
    n_ver = sum(1 for t in res["trials"] if t["verdict"] == "VERIFIED")
    lines += [f"Candidates verified by the gate: **{n_ver}/{res['n_trials']}**.", ""]

    lines += ["## Autonomous trials", "",
              "| Trial | Hypothesis | 1st attempt ok | Debug rounds | Executable | Mean F1 | Holm p | d_z | Verdict | Flags |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    for t in res["trials"]:
        lines.append(
            f"| {t['trial_id']} | {t['hypothesis'][:60].replace('|', '/')} | {'yes' if t['first_attempt_ok'] else 'no'} | "
            f"{t['retries']} | {'yes' if t['executable'] else 'no'} | {_fmt(t['mean_f1'], 3)} | "
            f"{_fmt_p(t['p_holm'])} | {_fmt(t['dz'], 2)} | {t['verdict']} | {', '.join(t['flags']) or '-'} |")
    lines += ["", "## Injected invalid-experiment traps", "",
              "| Trap | Expected flag | Flags raised | Rejected |", "|---|---|---|---|"]
    for t in res["traps"]:
        lines.append(f"| {t['name']} | `{t['expected_flag']}` | {', '.join(t['flags']) or '-'} | "
                     f"{'yes' if t['rejected'] else '**NO**'} |")
    bv = res["baseline_variance"]
    lines += ["", "## Baseline variance decomposition", "",
              f"- Mean within-subject **seed** variance: {_fmt(bv['seed_variance_within_subject'], 6)}",
              f"- **Between-subject** variance of seed-averaged F1: {_fmt(bv['subject_variance_between'], 6)}",
              f"- Seed-to-subject variance ratio: {_fmt(bv['seed_to_subject_variance_ratio'], 4)}",
              "",
              "Limitations: only 5 subjects, so statistical power is low and the Wilcoxon test cannot reach "
              "p < 0.05 (minimum exact two-sided p = 0.0625); the paired t-test with Holm-Bonferroni correction "
              "is the gating test.", ""]
    return "\n".join(lines)


def update_readme(markdown: str) -> None:
    """Replace the benchmark section of README.md between the marker comments."""
    readme = ROOT / "README.md"
    if not readme.exists():
        return
    text = readme.read_text(encoding="utf-8")
    pattern = re.compile(re.escape(README_START) + r".*?" + re.escape(README_END), re.DOTALL)
    body = markdown.replace("# VERI-SCI benchmark results", "").strip()
    if pattern.search(text):
        readme.write_text(pattern.sub(lambda _m: f"{README_START}\n{body}\n{README_END}", text), encoding="utf-8")


def main() -> int:
    """Run the full benchmark and write all result artefacts."""
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    out_dir: Path = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    tracker = ExperimentLogger(out_dir, use_wandb=not args.no_wandb)
    started = time.time()

    try:
        dataset = load_dataset(args.data_dir, synthetic=args.synthetic,
                               allow_synthetic_fallback=args.allow_synthetic_fallback)
    except DataUnavailableError as exc:
        print(f"ERROR: {exc}\nTip: check your internet connection, or add --synthetic for a smoke test.",
              file=sys.stderr)
        return 1
    LOGGER.info("data source=%s subjects=%s trials=%d fingerprint=%s", dataset.source,
                dataset.subject_ids, len(dataset.y), dataset.fingerprint()[:12])
    evaluator = CandidateEvaluator(dataset, seeds=(0, 1, 2), epochs=args.epochs,
                                   timeout_s=args.timeout, max_workers=args.workers)
    gate_cfg = GateConfig()
    llm = make_llm(args.llm)

    # ---- Phase A: baseline ------------------------------------------------
    LOGGER.info("Phase A: baseline EEGNet (5 subjects x 3 seeds)")
    base_out = evaluator.evaluate(BASELINE_SOURCE, "baseline_eegnet")
    if not base_out.ok or base_out.record is None:
        print(f"ERROR: baseline failed in the sandbox:\n{base_out.logs}", file=sys.stderr)
        return 1
    baseline: ExperimentRecord = base_out.record
    tracker.log({"event": "baseline", "mean_f1": baseline.mean_f1(),
                 "subject_f1": baseline.subject_means(), "data_source": dataset.source})

    # ---- Phase B: autonomous trials --------------------------------------
    memory_path = out_dir / "agent_memory.jsonl"
    memory_path.write_text("", encoding="utf-8")
    cfg = AgentConfig(llm=llm, evaluator=evaluator, baseline=baseline, max_retries=3,
                      family_size=args.trials, gate=gate_cfg, memory_path=memory_path, log_fn=tracker.log)
    graph = build_graph(cfg)
    states = []
    for trial in range(args.trials):
        LOGGER.info("Phase B: trial %d/%d", trial + 1, args.trials)
        states.append(run_trial(graph, trial, mode="self_debug", max_retries=cfg.max_retries))

    executed = [(i, s["record"]) for i, s in enumerate(states) if s.get("record") is not None]
    reports = audit_family([r for _, r in executed], baseline, gate_cfg,
                           n_untested=args.trials - len(executed))
    report_by_trial = {i: rep for (i, _), rep in zip(executed, reports)}

    trial_rows: List[Dict[str, Any]] = []
    for i, state in enumerate(states):
        rep = report_by_trial.get(i)
        rec = state.get("record")
        trial_rows.append({
            "trial_id": i,
            "hypothesis": state["hypothesis"],
            "first_attempt_ok": bool(state.get("first_attempt_ok")),
            "retries": state["retry_count"],
            "executable": rec is not None,
            "mean_f1": rec.mean_f1() if rec is not None else None,
            "p_holm": rep.p_holm if rep else None,
            "dz": rep.stats.get("cohens_dz") if rep else None,
            "verdict": rep.verdict if rep else "NOT_EVALUATED",
            "flags": rep.flags if rep else ["NOT_EXECUTABLE"],
        })

    best = None
    if reports:
        verified = [r for r in reports if r.verified]
        pool = verified or reports
        top = max(pool, key=lambda r: r.stats["mean_f1_candidate"] if r.stats else -1.0)
        best = {"name": top.name, "verified": top.verified, "stats": top.stats, "p_holm": top.p_holm,
                "family_size": top.family_size, "flags": top.flags}

    # ---- Phase C: injected traps -----------------------------------------
    LOGGER.info("Phase C: 5 injected invalid-experiment traps")
    traps = generate_traps(dataset)
    trap_reports = audit_family([t.record for t in traps], baseline, gate_cfg)
    trap_rows = [{"name": t.name, "expected_flag": t.expected_flag, "flags": r.flags,
                  "rejected": (not r.verified)} for t, r in zip(traps, trap_reports)]
    for row in trap_rows:
        tracker.log({"event": "trap", **row})

    n_ok_first = sum(1 for t in trial_rows if t["first_attempt_ok"])
    n_ok_final = sum(1 for t in trial_rows if t["executable"])
    rejected = sum(1 for t in trap_rows if t["rejected"])
    result: Dict[str, Any] = {
        "timestamp": dt.datetime.now().strftime("%Y-%m-%d %H:%M"),
        "data_source": dataset.source,
        "llm": llm.name,
        "epochs": args.epochs,
        "logger": tracker.backend,
        "n_trials": args.trials,
        "n_traps": len(traps),
        "n_cycles": args.trials + len(traps),
        "zero_shot_ok": n_ok_first, "zero_shot_pct": 100.0 * n_ok_first / args.trials,
        "self_debug_ok": n_ok_final, "self_debug_pct": 100.0 * n_ok_final / args.trials,
        "traps_rejected": rejected, "trap_rate_pct": 100.0 * rejected / len(traps),
        "baseline_mean_f1": baseline.mean_f1(),
        "baseline_variance": variance_decomposition(baseline.f1),
        "best": best, "trials": trial_rows, "traps": trap_rows,
        "runtime_seconds": time.time() - started,
    }
    markdown = build_markdown(result)
    (out_dir / "results_table.md").write_text(markdown, encoding="utf-8")
    (ROOT / "results_table.md").write_text(markdown, encoding="utf-8")
    (out_dir / "benchmark_results.json").write_text(json.dumps(to_jsonable(result), indent=2), encoding="utf-8")
    tracker.log({"event": "summary", **{k: v for k, v in result.items() if isinstance(v, (int, float, str))}})
    tracker.finish()
    if not args.no_readme:
        update_readme(markdown)
    print(markdown)
    return 0


if __name__ == "__main__":
    sys.exit(main())
