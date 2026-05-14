"""Canonical on-disk layout."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal


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
    # Phase 2/3 additions (C09).
    pilot_manifest_json: Path
    pilot_manifest_sha256: Path
    preregistration_md: Path
    pilot_dir: Path
    holdout_dir: Path

    def fold_dir(self, fold: Literal["pilot", "holdout"]) -> Path:
        """Return the per-fold root under ``runs/<run_id>/`` (``pilot/`` or ``holdout/``)."""
        if fold == "pilot":
            return self.pilot_dir
        if fold == "holdout":
            return self.holdout_dir
        raise ValueError(f"fold must be 'pilot' or 'holdout'; got {fold!r}")


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
        pilot_manifest_json=run_dir / "pilot_manifest.json",
        pilot_manifest_sha256=run_dir / "pilot_manifest.json.sha256",
        preregistration_md=run_dir / "preregistration.md",
        pilot_dir=run_dir / "pilot",
        holdout_dir=run_dir / "holdout",
    )


__all__ = ["RunPaths", "run_paths"]
