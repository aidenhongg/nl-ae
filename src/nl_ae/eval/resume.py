"""Resume-state reconciliation.

Reads existing rows via ``ResultsReader.iter_keys()`` (CI.13) and the prior
``RunManifest``. Mismatches in run_id / schema_version / config_digest raise
``ResumeMismatchError`` (loud failure — silent drift across the resume
boundary is the worst class of bug).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from nl_ae.schema.models import RunManifest, SCHEMA_VERSION
from nl_ae.schema.reader import ResultsReader, load_manifest
from nl_ae.schema.paths import run_paths

from .plan import EvalVisit

VisitKey = tuple[str, int, str]


class ResumeMismatchError(RuntimeError):
    """The on-disk manifest is incompatible with the resuming config."""


@dataclass(frozen=True)
class ResumeState:
    run_dir: Path
    rows_already_written: int
    completed_visits: frozenset[VisitKey]
    prior_manifest: RunManifest | None

    @property
    def is_fresh(self) -> bool:
        return self.rows_already_written == 0 and self.prior_manifest is None

    def is_completed(self, visit: EvalVisit) -> bool:
        return (
            visit.item.item_id,
            visit.permutation_id,
            visit.template_id,
        ) in self.completed_visits


def scan_resume_state(
    run_dir: Path,
    *,
    expected_run_id: str,
    expected_config_digest: str,
) -> ResumeState:
    paths = run_paths(run_dir.parent, run_dir.name)
    if not run_dir.exists():
        return ResumeState(
            run_dir=run_dir,
            rows_already_written=0,
            completed_visits=frozenset(),
            prior_manifest=None,
        )

    prior: RunManifest | None = None
    if paths.manifest_json.exists():
        prior = load_manifest(paths.manifest_json)
        if prior.run_id != expected_run_id:
            raise ResumeMismatchError(
                f"run_id mismatch: on-disk={prior.run_id!r} expected={expected_run_id!r}"
            )
        if prior.schema_version != SCHEMA_VERSION:
            raise ResumeMismatchError(
                f"schema_version mismatch: on-disk={prior.schema_version!r} "
                f"expected={SCHEMA_VERSION!r}"
            )
        if prior.config_digest != expected_config_digest:
            raise ResumeMismatchError(
                f"config_digest mismatch: on-disk={prior.config_digest} "
                f"expected={expected_config_digest}"
            )

    completed: set[VisitKey] = set()
    if paths.rows_jsonl.exists() or paths.rows_jsonl_partial.exists():
        reader = ResultsReader(run_dir, strict=False)
        for key in reader.iter_keys():
            completed.add((key.item_id, key.permutation_id, key.template_id))
    return ResumeState(
        run_dir=run_dir,
        rows_already_written=len(completed),
        completed_visits=frozenset(completed),
        prior_manifest=prior,
    )


__all__ = ["ResumeMismatchError", "ResumeState", "VisitKey", "scan_resume_state"]
