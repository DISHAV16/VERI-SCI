
"""Tests for the AST guard, the LOSO harness and the statistical gate."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from auditor.stats_gate import (  # noqa: E402
    ExperimentRecord,
    GateConfig,
    audit_family,
    build_loso_manifest,
    holm_bonferroni,
    variance_decomposition,
)
from sandbox.ast_guard import check_code  # noqa: E402
from trusted_eval.data_loader import loso_splits, make_synthetic_dataset  # noqa: E402
from trusted_eval.trap_generator import generate_traps  # noqa: E402

SUBJECTS = [1, 2, 3, 4, 5]

VALID_CODE = '''import numpy as np
import torch.nn as nn


def preprocess_train(X_train, y_train):
    return X_train, y_train, None


def build_model(in_channels, n_times):
    return nn.Linear(in_channels * n_times, 2)
'''


def _record(
    name: str,
    mean: float,
    seed_sd: float = 0.01,
    subject_effects: List[float] = None,
    rng_seed: int = 0,
) -> ExperimentRecord:
    rng = np.random.default_rng(rng_seed)
    effects = np.array(
        subject_effects if subject_effects is not None else [0.0] * 5
    )
    f1 = mean + effects[None, :] + rng.normal(0, seed_sd, size=(3, 5))

    return ExperimentRecord(
        name=name,
        subject_ids=SUBJECTS,
        seeds=[0, 1, 2],
        f1=f1,
        folds=build_loso_manifest(SUBJECTS),
        n_runs_logged=3,
        replay_pairs=[(float(f1[0, 0]), float(f1[0, 0]))],
    )


@pytest.fixture(scope="module")
def baseline() -> ExperimentRecord:
    return _record(
        "baseline",
        0.60,
        subject_effects=[0.05, -0.04, 0.02, -0.03, 0.0],
        rng_seed=1,
    )


# ---------------------------------------------------------------------
# AST guard
# ---------------------------------------------------------------------

def test_ast_guard_blocks_os_system() -> None:
    report = check_code(
        "import os\nos.system('echo pwned')\n",
        require_interface=False,
    )
    assert not report.ok
    assert any("os" in v for v in report.violations)
    assert any("system" in v for v in report.violations)


@pytest.mark.parametrize(
    "module",
    ["os", "subprocess", "sys", "shutil", "socket"],
)
def test_ast_guard_blocks_forbidden_imports(module: str) -> None:
    assert not check_code(
        f"import {module}\n",
        require_interface=False,
    ).ok

    assert not check_code(
        f"from {module} import path\n",
        require_interface=False,
    ).ok


@pytest.mark.parametrize(
    "snippet",
    [
        "import trusted_eval\n",
        "from trusted_eval.data_loader import load_eegbci\n",
        "x = 'trusted_eval/data_loader.py'\n",
        "def f():\n    return trusted_eval\n",
    ],
)
def test_ast_guard_blocks_trusted_eval_reference(snippet: str) -> None:
    assert not check_code(
        snippet,
        require_interface=False,
    ).ok


@pytest.mark.parametrize(
    "snippet",
    [
        "def f():\n    return eval('1')\n",
        "def f():\n    return open('x')\n",
        "def f(m):\n    return m.__class__.__subclasses__()\n",
        "print('top-level code')\n",
    ],
)
def test_ast_guard_blocks_escape_primitives(snippet: str) -> None:
    assert not check_code(
        snippet,
        require_interface=False,
    ).ok


def test_ast_guard_accepts_valid_candidate_and_checks_interface() -> None:
    assert check_code(VALID_CODE).ok

    missing = check_code(
        "import numpy as np\n\n"
        "def preprocess_train(X, y):\n"
        "    return X, y, None\n"
    )

    assert not missing.ok
    assert any("build_model" in e for e in missing.interface_errors)



def test_ast_guard_accepts_baseline_source() -> None:
    import ast
    from pathlib import Path

    baseline_path = (
        Path(__file__).resolve().parents[1]
        / "baselines"
        / "baseline_eegnet.py"
    )
    source = baseline_path.read_text(encoding="utf-8")
    tree = ast.parse(source)

    names = {"_zscore", "EEGNet", "preprocess_train", "build_model"}
    nodes = [
        node for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.ClassDef))
        and node.name in names
    ]

    assert {node.name for node in nodes} == names

    header = (
        "from __future__ import annotations\n"
        "from typing import Callable, Optional, Tuple\n"
        "import numpy as np\n"
        "import torch\n"
        "import torch.nn as nn\n\n"
    )

    baseline_source = header + "\n\n".join(
        ast.get_source_segment(source, node) for node in nodes
    )

    report = check_code(baseline_source)
    assert report.ok, report.format()


# ---------------------------------------------------------------------
# LOSO harness
# ---------------------------------------------------------------------

def test_loso_splits_are_disjoint_and_read_only() -> None:
    ds = make_synthetic_dataset(
        trials_per_subject=10,
        n_channels=8,
        n_times=64,
    )

    splits = list(loso_splits(ds))

    assert len(splits) == 5

    for split in splits:
        assert split.test_subject not in split.train_subjects
        assert len(split.X_test) == 10
        assert len(split.X_train) == 40

        with pytest.raises(ValueError):
            split.X_train[0, 0, 0] = 1.0

    with pytest.raises(ValueError):
        ds.X[0, 0, 0] = 1.0


# ---------------------------------------------------------------------
# Statistical gate
# ---------------------------------------------------------------------

def test_holm_bonferroni_known_values() -> None:
    _, adj = holm_bonferroni(
        [0.01, 0.04, 0.03],
        alpha=0.05,
    )

    assert np.allclose(adj, [0.03, 0.06, 0.06])


def test_variance_decomposition_separates_seed_and_subject_variance() -> None:
    f1 = np.array(
        [
            [0.5, 0.7, 0.9],
            [0.52, 0.68, 0.92],
            [0.48, 0.72, 0.88],
        ]
    )

    out = variance_decomposition(f1)

    assert out["seed_variance_within_subject"] < 0.001
    assert out["subject_variance_between"] > 0.03
    assert out["seed_to_subject_variance_ratio"] < 0.05


def test_stats_gate_rejects_all_five_injected_traps(
    baseline: ExperimentRecord,
) -> None:
    ds = make_synthetic_dataset(
        trials_per_subject=30,
        n_channels=16,
        n_times=160,
        seed=3,
    )

    traps = generate_traps(ds)

    assert len(traps) == 5

    reports = audit_family(
        [trap.record for trap in traps],
        baseline,
    )

    for trap, report in zip(traps, reports):
        assert not report.verified, (
            f"trap {trap.name} was NOT rejected"
        )
        assert trap.expected_flag in report.flags, (
            trap.name,
            report.flags,
        )

    assert sum(not report.verified for report in reports) == 5


def test_stats_gate_verifies_genuine_improvement(
    baseline: ExperimentRecord,
) -> None:
    good = _record(
        "good",
        0.72,
        subject_effects=[0.05, -0.04, 0.02, -0.03, 0.0],
        rng_seed=2,
    )

    report = audit_family(
        [good],
        baseline,
        GateConfig(),
    )[0]

    assert report.verified, report.flags
    assert report.stats["cohens_dz"] >= 0.5
    assert report.stats["ci_low"] > 0
    assert report.p_holm < 0.05


def test_stats_gate_rejects_no_improvement_and_multiple_comparison_inflation(
    baseline: ExperimentRecord,
) -> None:
    same = _record(
        "same",
        0.60,
        subject_effects=[0.05, -0.04, 0.02, -0.03, 0.0],
        rng_seed=5,
    )

    assert not audit_family([same], baseline)[0].verified

    # A borderline improvement survives alone but not once the family is large.
    marginal = _record(
        "marginal",
        0.64,
        seed_sd=0.005,
        subject_effects=[0.05, -0.04, 0.02, -0.03, 0.0],
        rng_seed=6,
    )

    marginal.f1 = marginal.f1 + np.array(
        [[0.03, -0.03, 0.02, -0.02, 0.0]] * 3
    )

    alone = audit_family([marginal], baseline)[0]
    crowded = audit_family(
        [marginal],
        baseline,
        n_untested=50,
    )[0]

    assert crowded.p_holm >= alone.p_holm
    assert not crowded.verified
