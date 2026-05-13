"""Canonical on-disk layout."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RunPaths:
    run_dir: Path
    rows_jsonl: Path
    rows_jsonl_partial: Path
    rows_parquet: Path
    manifest_json: Path
    manifest_sha256: Path
    lock_file: Path
    activations_dir: Path
    prompts_dir: Path
    logs_dir: Path
    aggregates_dir: Path
    figures_dir: Path
    summary_md: Path


def run_paths(root: Path, run_id: str) -> RunPaths:
    run_dir = root / run_id
    return RunPaths(
        run_dir=run_dir,
        rows_jsonl=run_dir / "rows.jsonl",
        rows_jsonl_partial=run_dir / "rows.jsonl.partial",
        rows_parquet=run_dir / "rows.parquet",
        manifest_json=run_dir / "manifest.json",
        manifest_sha256=run_dir / "manifest.sha256",
        lock_file=run_dir / "run.lock",
        activations_dir=run_dir / "activations",
        prompts_dir=run_dir / "prompts",
        logs_dir=run_dir / "logs",
        aggregates_dir=run_dir / "aggregates",
        figures_dir=run_dir / "figures",
        summary_md=run_dir / "summary.md",
    )


__all__ = ["RunPaths", "run_paths"]
