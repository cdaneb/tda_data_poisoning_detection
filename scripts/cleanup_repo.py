#!/usr/bin/env python
"""
Repository cleanup utility for TDA_CODE.

This script removes generated artifacts (caches, bytecode, duplicate outputs)
while leaving source code, notebooks, and user data intact by default.

Usage (from repo root):
    python scripts/cleanup_repo.py

See README for flag documentation.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple


def compute_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Compute SHA256 hash of a file."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def delete_dir(path: Path, category: str, summary: Dict[str, int], deleted: List[str]) -> None:
    """Delete a directory tree if it exists."""
    if not path.exists():
        return
    try:
        shutil.rmtree(path)
        summary[category] = summary.get(category, 0) + 1
        deleted.append(str(path))
    except Exception as exc:  # pragma: no cover - defensive logging
        print(f"[WARN] Failed to delete directory {path}: {exc}", file=sys.stderr)


def delete_file(path: Path, category: str, summary: Dict[str, int], deleted: List[str]) -> None:
    """Delete a single file if it exists."""
    if not path.exists():
        return
    try:
        path.unlink()
        summary[category] = summary.get(category, 0) + 1
        deleted.append(str(path))
    except Exception as exc:  # pragma: no cover - defensive logging
        print(f"[WARN] Failed to delete file {path}: {exc}", file=sys.stderr)


def clean_caches_and_bytecode(root: Path, summary: Dict[str, int], deleted: List[str]) -> None:
    """
    Remove Python caches and bytecode:
      - .pytest_cache directories
      - __pycache__ directories
      - .ipynb_checkpoints directories
      - *.pyc files

    Skips common virtual environment directories to avoid touching them:
      - .venv, venv, env
    """
    skip_dirs = {".venv", "venv", "env"}

    for dirpath, dirnames, filenames in os.walk(root):
        current = Path(dirpath)

        # Prune virtual environments from traversal
        dirnames[:] = [d for d in dirnames if d not in skip_dirs]

        # Remove cache-style directories
        for cache_name, category in [
            (".pytest_cache", "pytest_cache_dirs_deleted"),
            ("__pycache__", "pycache_dirs_deleted"),
            (".ipynb_checkpoints", "ipynb_checkpoints_dirs_deleted"),
        ]:
            if cache_name in dirnames:
                cache_path = current / cache_name
                delete_dir(cache_path, category, summary, deleted)
                # Remove from dirnames to prevent os.walk from descending
                dirnames.remove(cache_name)

        # Remove *.pyc files
        for filename in filenames:
            if filename.endswith(".pyc"):
                file_path = current / filename
                delete_file(file_path, "pyc_files_deleted", summary, deleted)


def prune_outputs_duplicates(
    root: Path, prune_outputs: bool, summary: Dict[str, int], deleted: List[str]
) -> None:
    """
    Remove duplicate top-level files in outputs/ that are identical to
    corresponding files under outputs/clean/seed_0/.

    A file is considered a duplicate if:
      - It lives directly under outputs/ (not in a subdirectory), and
      - A file with the same filename exists under outputs/clean/seed_0/, and
      - The SHA256 hashes of the two files match.
    """
    if not prune_outputs:
        return

    outputs_dir = root / "outputs"
    canonical_dir = outputs_dir / "clean" / "seed_0"

    if not outputs_dir.is_dir() or not canonical_dir.is_dir():
        return

    # Build filename -> hash map for canonical copies
    canonical_hashes: Dict[str, str] = {}
    for path in canonical_dir.rglob("*"):
        if path.is_file():
            canonical_hashes[path.name] = compute_sha256(path)

    if not canonical_hashes:
        return

    # Only consider files that are direct children of outputs/
    for path in outputs_dir.iterdir():
        if not path.is_file():
            continue
        filename = path.name
        if filename not in canonical_hashes:
            continue
        try:
            candidate_hash = compute_sha256(path)
        except Exception as exc:  # pragma: no cover - defensive
            print(f"[WARN] Failed to hash {path}: {exc}", file=sys.stderr)
            continue
        if candidate_hash == canonical_hashes[filename]:
            delete_file(path, "outputs_duplicates_deleted", summary, deleted)


def handle_outputs_and_results(
    root: Path,
    prune_outputs: bool,
    remove_results: bool,
    remove_outputs: bool,
    summary: Dict[str, int],
    deleted: List[str],
) -> None:
    """Apply outputs/results cleanup according to flags."""
    outputs_dir = root / "outputs"
    results_dir = root / "results"

    # Most destructive: remove entire outputs/ tree
    if remove_outputs and outputs_dir.exists():
        delete_dir(outputs_dir, "outputs_dir_removed", summary, deleted)
        # If the whole directory is gone, no further outputs-related work
        outputs_dir = None  # type: ignore[assignment]
    else:
        prune_outputs_duplicates(root, prune_outputs=prune_outputs, summary=summary, deleted=deleted)

    if remove_results and results_dir.exists():
        delete_dir(results_dir, "results_dir_removed", summary, deleted)


def maybe_remove_vscode(root: Path, remove_vscode: bool, summary: Dict[str, int], deleted: List[str]) -> None:
    """
    Optionally remove the .vscode directory.

    Default behavior is to *not* remove .vscode unless explicitly requested.
    """
    if not remove_vscode:
        return
    vscode_dir = root / ".vscode"
    if vscode_dir.is_dir():
        delete_dir(vscode_dir, "vscode_dir_removed", summary, deleted)


def write_log(root: Path, summary: Dict[str, int], deleted: List[str]) -> None:
    """Write a detailed log of deletions to logs/cleanup.log."""
    if not deleted:
        return

    logs_dir = root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = logs_dir / "cleanup.log"

    timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")

    with log_path.open("a", encoding="utf-8") as f:
        f.write(f"=== Cleanup run at {timestamp} ===\n")
        f.write("Summary:\n")
        for key in sorted(summary.keys()):
            f.write(f"  {key}: {summary[key]}\n")
        f.write("Deleted paths:\n")
        for path in deleted:
            f.write(f"  {path}\n")
        f.write("\n")


def parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Clean up generated artifacts in the TDA_CODE repository. "
            "By default, removes Python caches/bytecode and prunes duplicate "
            "outputs that match copies under outputs/clean/seed_0/."
        )
    )

    parser.add_argument(
        "--root",
        type=str,
        default=None,
        help="Repository root (default: parent directory of this script).",
    )

    # Default True: allow disabling via --no-prune-outputs on modern Python.
    try:
        from argparse import BooleanOptionalAction  # type: ignore

        parser.add_argument(
            "--prune-outputs",
            dest="prune_outputs",
            action=BooleanOptionalAction,
            default=True,
            help=(
                "Prune duplicate top-level files in outputs/ that are identical "
                "to outputs/clean/seed_0/ copies (default: True)."
            ),
        )
    except ImportError:  # pragma: no cover - older Python fallback
        parser.add_argument(
            "--prune-outputs",
            dest="prune_outputs",
            action="store_true",
            help=(
                "Prune duplicate top-level files in outputs/ that are identical "
                "to outputs/clean/seed_0/ copies (enabled when this flag is set)."
            ),
        )

    parser.add_argument(
        "--remove-results",
        action="store_true",
        help="Remove the entire results/ directory (default: False).",
    )

    parser.add_argument(
        "--remove-outputs",
        action="store_true",
        help="Remove the entire outputs/ directory, including all runs (default: False).",
    )

    parser.add_argument(
        "--remove-vscode",
        action="store_true",
        help="Also delete the .vscode/ directory (default: False).",
    )

    return parser.parse_args(argv)


def main(argv: List[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])

    script_path = Path(__file__).resolve()
    default_root = script_path.parent.parent
    root = Path(args.root).resolve() if args.root is not None else default_root

    if not root.is_dir():
        print(f"[ERROR] Root path does not exist or is not a directory: {root}", file=sys.stderr)
        return 1

    print(f"[INFO] Running cleanup in repository root: {root}")

    summary: Dict[str, int] = {}
    deleted: List[str] = []

    # A) Caches and bytecode
    clean_caches_and_bytecode(root, summary, deleted)

    # B) outputs/ and results/ handling
    handle_outputs_and_results(
        root=root,
        prune_outputs=bool(getattr(args, "prune_outputs", True)),
        remove_results=args.remove_results,
        remove_outputs=args.remove_outputs,
        summary=summary,
        deleted=deleted,
    )

    # C) Optional .vscode removal
    maybe_remove_vscode(root, args.remove_vscode, summary, deleted)

    # D) Log and human-readable summary
    write_log(root, summary, deleted)

    print("\nCleanup summary:")
    if summary:
        for key in sorted(summary.keys()):
            print(f"  {key}: {summary[key]}")
    else:
        print("  No files or directories were removed.")

    if deleted:
        print(f"\nDetailed log written to: {root / 'logs' / 'cleanup.log'}")
    else:
        print("\nNo deletions were necessary; logs/cleanup.log not created.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

