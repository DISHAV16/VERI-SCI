"""Compact EEGNet baseline (Lawhern et al., 2018) in the VERI-SCI candidate interface.

The functions here follow exactly the contract imposed on agent-generated
code (``preprocess_train`` and ``build_model``), so the baseline is trained and
scored by the *same* trusted harness, across all LOSO subjects and 3 seeds.
``BASELINE_SOURCE`` is a self-contained, AST-guard-clean source string.

Run standalone::

    python -m baselines.baseline_eegnet --synthetic --epochs 10
"""

from __future__ import annotations

import argparse
import inspect
from typing import Callable, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn


def _zscore(X: np.ndarray) -> np.ndarray:
    """Per-trial demeaning and one global scale per trial.

    The scale is a single scalar per trial (not per channel) so relative
    channel power, i.e. the ERD/ERS lateralisation cue, is preserved.  The
    transform is stateless, hence leakage-free.
    """
    X = np.asarray(X, dtype=np.float32)
    X = X - X.mean(axis=2, keepdims=True)
    std = X.std(axis=(1, 2), keepdims=True) + 1e-6
    return X / std


class EEGNet(nn.Module):
    """EEGNet-F1,D,F2 for inputs of shape ``(batch, channels, times)``."""

    def __init__(self, n_channels: int, n_times: int, f1: int = 8, depth: int = 2,
                 kernel_length: int = 32, dropout: float = 0.25) -> None:
        super().__init__()
        f2 = f1 * depth
        self.features = nn.Sequential(
            nn.Conv2d(1, f1, (1, kernel_length), padding="same", bias=False),
            nn.BatchNorm2d(f1),
            nn.Conv2d(f1, f1 * depth, (n_channels, 1), groups=f1, bias=False),
            nn.BatchNorm2d(f1 * depth),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(dropout),
            nn.Conv2d(f1 * depth, f1 * depth, (1, 16), groups=f1 * depth, padding="same", bias=False),
            nn.Conv2d(f1 * depth, f2, 1, bias=False),
            nn.BatchNorm2d(f2),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(dropout),
        )
        with torch.no_grad():
            self.features.eval()
            self.n_feat = int(self.features(torch.zeros(1, 1, n_channels, n_times)).numel())
            self.features.train()
        self.head = nn.Linear(self.n_feat, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return class logits of shape ``(batch, 2)``."""
        x = x.unsqueeze(1)
        return self.head(self.features(x).flatten(1))


def preprocess_train(X_train: np.ndarray, y_train: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Optional[Callable]]:
    """Normalise training trials; return ``(X, y, test_time_transform)``."""
    return _zscore(X_train), np.asarray(y_train), _zscore


def build_model(in_channels: int, n_times: int) -> nn.Module:
    """Build the EEGNet-8,2 baseline."""
    return EEGNet(in_channels, n_times)


_HEADER = (
    "from __future__ import annotations\n"
    "from typing import Callable, Optional, Tuple\n"
    "import numpy as np\n"
    "import torch\n"
    "import torch.nn as nn\n\n"
)

BASELINE_SOURCE: str = _HEADER + "\n\n".join(
    inspect.getsource(obj) for obj in (_zscore, EEGNet, preprocess_train, build_model)
)


def main() -> None:
    """CLI: evaluate the baseline over all LOSO subjects x 3 seeds."""
    from auditor.stats_gate import variance_decomposition
    from sandbox.runner import CandidateEvaluator
    from trusted_eval.data_loader import load_dataset

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--synthetic", action="store_true", help="use synthetic data (smoke test only)")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    dataset = load_dataset(synthetic=args.synthetic)
    outcome = CandidateEvaluator(dataset, epochs=args.epochs, max_workers=args.workers).evaluate(
        BASELINE_SOURCE, "baseline_eegnet")
    if not outcome.ok or outcome.record is None:
        raise SystemExit(f"baseline failed: {outcome.logs}")
    print(f"data source: {dataset.source}")
    print("per-subject mean F1:", {k: round(v, 3) for k, v in outcome.record.subject_means().items()})
    print(f"mean LOSO F1: {outcome.record.mean_f1():.4f}")
    print("variance decomposition:", variance_decomposition(outcome.record.f1))


if __name__ == "__main__":
    main()
