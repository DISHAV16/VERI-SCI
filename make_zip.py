"""Package the clean VERI-SCI directory into ``VERI-SCI.zip`` (VS Code ready).

Usage::

    python make_zip.py                 # writes ../VERI-SCI.zip
    python make_zip.py --output out.zip

Caches, virtual environments, datasets, benchmark outputs and existing zip
files are excluded so the archive contains only source files.
"""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path
from typing import List

ROOT = Path(__file__).resolve().parent
TOP_LEVEL = "VERI-SCI"
EXCLUDED_DIRS = {"__pycache__", ".pytest_cache", ".git", ".venv", "venv", "env", "wandb",
                 "results", "runs", "data", "mne_data", ".mypy_cache", ".idea", ".vscode"}
EXCLUDED_SUFFIXES = {".pyc", ".pyo", ".zip", ".log"}
EXCLUDED_NAMES = {".DS_Store", "results_table.md"}


def collect_files(root: Path = ROOT) -> List[Path]:
    """Return every source file that belongs in the archive, sorted."""
    files = []
    for path in sorted(root.rglob("*")):
        rel = path.relative_to(root)
        if not path.is_file():
            continue
        if any(part in EXCLUDED_DIRS for part in rel.parts[:-1]):
            continue
        if path.suffix in EXCLUDED_SUFFIXES or path.name in EXCLUDED_NAMES:
            continue
        files.append(path)
    return files


def make_zip(output: Path) -> Path:
    """Write the archive and return its path."""
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in collect_files():
            zf.write(path, arcname=f"{TOP_LEVEL}/{path.relative_to(ROOT).as_posix()}")
    return output


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT.parent / "VERI-SCI.zip")
    args = parser.parse_args()
    out = make_zip(args.output.resolve())
    with zipfile.ZipFile(out) as zf:
        names = zf.namelist()
    print(f"Created {out} with {len(names)} files:")
    for name in names:
        print(f"  {name}")


if __name__ == "__main__":
    main()
