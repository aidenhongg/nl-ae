"""Readers for JSONL rows and the JSON manifest.

``strict=False`` tolerates a torn last line (UR-R2.5) so the aggregator can be
run mid-flight under ``--include-partial``.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from .models import ResultRow, ResultRowKey, RunManifest, SCHEMA_VERSION
from .paths import run_paths

if TYPE_CHECKING:  # pragma: no cover
    import pyarrow as pa

LOG = logging.getLogger(__name__)


class _SchemaCompat:
    """Encode the SemVer policy: same MAJOR, warn on newer MINOR, raise on MAJOR mismatch."""

    @staticmethod
    def check(version: str) -> None:
        try:
            major_s, minor_s, _patch_s = version.split(".")
            major, minor = int(major_s), int(minor_s)
        except (ValueError, AttributeError) as exc:
            raise ValueError(f"unparseable schema_version: {version!r}") from exc
        cur_major, cur_minor, _ = (int(p) for p in SCHEMA_VERSION.split("."))
        if major != cur_major:
            raise ValueError(
                f"incompatible schema MAJOR: got {version!r}, expected {SCHEMA_VERSION!r}"
            )
        if minor > cur_minor:
            LOG.warning(
                "Reading newer minor schema %s (current %s); unknown fields may be dropped.",
                version,
                SCHEMA_VERSION,
            )


class ResultsReader:
    """Validate and stream rows + load the manifest.

    Use ``strict=False`` when reading a run that is still being written
    (mid-flight aggregator); the last line may be torn and will be skipped
    with a debug log.
    """

    def __init__(self, run_dir: Path, *, strict: bool = True) -> None:
        self._paths = run_paths(run_dir.parent, run_dir.name)
        self._strict = strict
        self._manifest: RunManifest | None = None

    @property
    def manifest(self) -> RunManifest:
        if self._manifest is None:
            self._manifest = load_manifest(self._paths.manifest_json)
            _SchemaCompat.check(self._manifest.schema_version)
        return self._manifest

    @property
    def schema_version(self) -> str:
        return self.manifest.schema_version

    @property
    def rows_path(self) -> Path:
        """Return whichever rows file actually exists (final preferred, else partial)."""
        if self._paths.rows_jsonl.exists():
            return self._paths.rows_jsonl
        return self._paths.rows_jsonl_partial

    def iter_rows(self) -> Iterator[ResultRow]:
        path = self.rows_path
        if not path.exists():
            return
        yield from load_rows(path, strict=self._strict)

    def iter_keys(self) -> Iterator[ResultRowKey]:
        """CI.13 — key-only fast path; avoids full validation when only resume keys are needed."""
        path = self.rows_path
        if not path.exists():
            return
        with path.open("rb") as f:
            for raw in f:
                line = raw.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    if self._strict:
                        raise
                    LOG.debug("skipping torn JSONL line in %s", path)
                    continue
                yield ResultRowKey(
                    run_id=obj["run_id"],
                    item_id=obj["item_id"],
                    permutation_id=obj["permutation_id"],
                    template_id=obj["template_id"],
                )

    def row_count(self) -> int:
        path = self.rows_path
        if not path.exists():
            return 0
        with path.open("rb") as f:
            return sum(1 for line in f if line.strip())

    def to_arrow_table(self) -> "pa.Table":  # type: ignore[name-defined]
        import pyarrow as pa  # noqa: PLC0415

        return pa.Table.from_pylist([r.model_dump() for r in self.iter_rows()])


def load_manifest(path: Path) -> RunManifest:
    payload = path.read_bytes()
    manifest = RunManifest.model_validate_json(payload)
    _SchemaCompat.check(manifest.schema_version)
    return manifest


def load_rows(path: Path, *, strict: bool = True) -> Iterator[ResultRow]:
    if not path.exists():
        return
    with path.open("rb") as f:
        for idx, raw in enumerate(f):
            line = raw.strip()
            if not line:
                continue
            try:
                yield ResultRow.model_validate_json(line)
            except (json.JSONDecodeError, ValueError) as exc:
                if strict:
                    raise
                LOG.debug("skipping malformed JSONL line %d in %s: %s", idx, path, exc)


def detect_status(
    run_dir: Path,
) -> Literal["in_progress", "completed", "failed", "aborted", "unknown"]:
    paths = run_paths(run_dir.parent, run_dir.name)
    if not paths.manifest_json.exists():
        return "unknown"
    try:
        manifest = load_manifest(paths.manifest_json)
    except (OSError, ValueError):
        return "unknown"
    declared = manifest.completion_status
    if declared == "in_progress" and paths.rows_jsonl_partial.exists():
        return "in_progress"
    return declared


__all__ = [
    "ResultsReader",
    "detect_status",
    "load_manifest",
    "load_rows",
]
