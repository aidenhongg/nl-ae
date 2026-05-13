"""Context-managed JSONL row + manifest writer.

Crash-tolerant: writes to ``rows.jsonl.partial`` then renames to ``rows.jsonl``
on ``finalize()``. The manifest is atomically replaced via temp-file + ``os.replace``.
"""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Iterable
from contextlib import AbstractContextManager
from pathlib import Path
from types import TracebackType
from typing import Any, Literal

from .hashing import hash_json_bytes, now_utc_iso
from .models import ResultRow, RunManifest, Sha256Hex
from .paths import RunPaths, run_paths


class ResultsWriter(AbstractContextManager["ResultsWriter"]):
    """Append-safe writer. One process; one run dir; one open at a time."""

    def __init__(
        self,
        run_dir: Path,
        manifest: RunManifest,
        *,
        flush_every: int = 1,
        fsync_every: int = 32,
        on_existing: Literal["resume", "overwrite", "error"] = "error",
    ) -> None:
        if run_dir.name != manifest.run_id:
            raise ValueError(
                f"run_dir name {run_dir.name!r} must match manifest.run_id {manifest.run_id!r}"
            )
        self._paths: RunPaths = run_paths(run_dir.parent, manifest.run_id)
        self._manifest: RunManifest = manifest
        self._flush_every = flush_every
        self._fsync_every = fsync_every
        self._on_existing = on_existing
        self._rows_written = 0
        self._handle: Any | None = None
        self._finalized = False
        self._opened = False

    # --- properties -------------------------------------------------------

    @property
    def run_dir(self) -> Path:
        return self._paths.run_dir

    @property
    def rows_written(self) -> int:
        return self._rows_written

    @property
    def manifest(self) -> RunManifest:
        return self._manifest

    @property
    def paths(self) -> RunPaths:
        return self._paths

    # --- context manager --------------------------------------------------

    def __enter__(self) -> ResultsWriter:
        self._open()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        if self._finalized:
            return
        status: Literal["failed", "aborted"] = "failed" if exc is not None else "aborted"
        reason = repr(exc) if exc is not None else "writer exited without finalize()"
        try:
            self.finalize(status=status, failure_reason=reason, emit_parquet=False)
        except Exception:
            self._close_handle()
            raise

    # --- lifecycle --------------------------------------------------------

    def _open(self) -> None:
        if self._opened:
            return
        self._paths.run_dir.mkdir(parents=True, exist_ok=True)
        self._paths.activations_dir.mkdir(parents=True, exist_ok=True)
        self._paths.prompts_dir.mkdir(parents=True, exist_ok=True)
        self._paths.logs_dir.mkdir(parents=True, exist_ok=True)

        partial = self._paths.rows_jsonl_partial
        final = self._paths.rows_jsonl
        if final.exists():
            if self._on_existing == "error":
                raise FileExistsError(f"refusing to overwrite finalized run: {final}")
            if self._on_existing == "overwrite":
                final.unlink()
        if partial.exists():
            if self._on_existing == "error":
                raise FileExistsError(f"partial run already present: {partial}")
            if self._on_existing == "overwrite":
                partial.unlink()
            # "resume" keeps the partial file: we'll append to it. Caller is expected to
            # have reconciled completed visits via ResultsReader(strict=False).
            if self._on_existing == "resume" and partial.exists():
                with partial.open("rb") as f:
                    self._rows_written = sum(1 for _ in f)

        # Lock first.
        if self._paths.lock_file.exists():
            raise FileExistsError(
                f"lock file present at {self._paths.lock_file} — another writer is open"
            )
        self._paths.lock_file.write_text(f"pid={os.getpid()}\n", encoding="utf-8")

        self._handle = partial.open("ab")
        write_manifest_atomic(self._paths.manifest_json, self._manifest)
        self._opened = True

    def _close_handle(self) -> None:
        if self._handle is not None:
            try:
                self._handle.flush()
                self._handle.close()
            finally:
                self._handle = None
        if self._paths.lock_file.exists():
            try:
                self._paths.lock_file.unlink()
            except OSError:
                pass

    # --- writes -----------------------------------------------------------

    def write_row(self, row: ResultRow) -> None:
        if not self._opened or self._handle is None:
            raise RuntimeError("writer is not open; use as a context manager")
        if self._finalized:
            raise RuntimeError("writer has been finalized")
        if row.run_id != self._manifest.run_id:
            raise ValueError(
                f"row.run_id={row.run_id!r} does not match manifest.run_id={self._manifest.run_id!r}"
            )
        payload = row.model_dump_json(exclude_none=False).encode("utf-8") + b"\n"
        self._handle.write(payload)
        self._rows_written += 1
        if self._rows_written % self._flush_every == 0:
            self._handle.flush()
        if self._rows_written % self._fsync_every == 0:
            self._handle.flush()
            try:
                os.fsync(self._handle.fileno())
            except OSError:
                # fsync is best-effort on some platforms; never block the run on it.
                pass

    def write_rows(self, rows: Iterable[ResultRow]) -> int:
        n = 0
        for row in rows:
            self.write_row(row)
            n += 1
        return n

    def update_manifest(self, **fields: object) -> None:
        if self._finalized:
            raise RuntimeError("writer has been finalized")
        data = self._manifest.model_dump()
        data.update(fields)
        data["rows_written"] = self._rows_written
        self._manifest = RunManifest.model_validate(data)
        write_manifest_atomic(self._paths.manifest_json, self._manifest)

    def finalize(
        self,
        *,
        status: Literal["completed", "failed", "aborted"] = "completed",
        failure_reason: str | None = None,
        emit_parquet: bool = True,
    ) -> None:
        if self._finalized:
            return
        try:
            if self._handle is not None:
                self._handle.flush()
                try:
                    os.fsync(self._handle.fileno())
                except OSError:
                    pass
                self._handle.close()
                self._handle = None

            partial = self._paths.rows_jsonl_partial
            final = self._paths.rows_jsonl
            if partial.exists():
                if final.exists():
                    final.unlink()
                os.replace(partial, final)

            # Manifest finalization.
            updated = self._manifest.model_copy(
                update={
                    "ended_at": now_utc_iso(),
                    "completion_status": status,
                    "failure_reason": failure_reason,
                    "rows_written": self._rows_written,
                }
            )
            self._manifest = updated
            write_manifest_atomic(self._paths.manifest_json, self._manifest)

            if emit_parquet and status == "completed" and final.exists():
                derive_parquet_from_jsonl(final, self._paths.rows_parquet)
        finally:
            self._finalized = True
            if self._paths.lock_file.exists():
                try:
                    self._paths.lock_file.unlink()
                except OSError:
                    pass


def write_manifest_atomic(path: Path, manifest: RunManifest) -> Sha256Hex:
    """Atomic-replace ``path`` with the manifest. Returns the SHA-256 hex of the bytes."""
    payload = manifest.model_dump_json(indent=2, exclude_none=False).encode("utf-8")
    sidecar = path.with_suffix(path.suffix + ".sha256")
    digest = hash_json_bytes(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=".manifest.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(payload)
            f.flush()
            try:
                os.fsync(f.fileno())
            except OSError:
                pass
        os.replace(tmp_name, path)
    except BaseException:
        try:
            Path(tmp_name).unlink(missing_ok=True)
        finally:
            raise
    sidecar.write_text(digest + "\n", encoding="utf-8")
    return digest


def derive_parquet_from_jsonl(rows_jsonl: Path, out_parquet: Path) -> int:
    """Stream-validate every JSONL line through ``ResultRow`` and write a parquet file.

    Returns the number of rows written. Imports ``pyarrow`` lazily so the schema
    package stays light when callers don't need parquet (e.g., tests).
    """
    import pyarrow as pa  # noqa: PLC0415
    import pyarrow.parquet as pq  # noqa: PLC0415

    if not rows_jsonl.exists():
        out_parquet.write_bytes(b"")
        return 0

    letter_softmax_struct = pa.struct(
        [
            pa.field("letter", pa.string(), nullable=False),
            pa.field("token_id", pa.int64(), nullable=False),
            pa.field("prob", pa.float64(), nullable=True),
            pa.field("prob_valid", pa.bool_(), nullable=False),
            pa.field("logit", pa.float64(), nullable=False),
        ]
    )
    schema = pa.schema(
        [
            pa.field("schema_version", pa.string()),
            pa.field("run_id", pa.string()),
            pa.field("item_id", pa.string()),
            pa.field("dataset_name", pa.string()),
            pa.field("dataset_split", pa.string()),
            pa.field("subject", pa.string()),
            pa.field("template_id", pa.string()),
            pa.field("permutation_id", pa.int64()),
            pa.field("prompt_hash", pa.string()),
            pa.field("rendered_prompt_ref", pa.string()),
            pa.field("gold_letter", pa.string()),
            pa.field("first_token_letter", pa.string()),
            pa.field("free_text_letter", pa.string()),
            pa.field("free_text_raw", pa.string()),
            pa.field("free_text_truncated", pa.bool_()),
            pa.field("agreement_flag", pa.bool_()),
            pa.field("letter_softmax", pa.list_(letter_softmax_struct)),
            pa.field("n_options", pa.int64()),
            pa.field("free_text_seed", pa.int64()),
            pa.field("decode_strategy", pa.string()),
            pa.field("activation_ref", pa.string()),
            pa.field("wall_time_ms", pa.float64()),
            pa.field("created_at", pa.string()),
            pa.field("extractor_id", pa.string()),
            pa.field("extractor_match_rule", pa.string()),
            pa.field("first_token_scoring_math", pa.string()),
            pa.field("total_letter_mass", pa.float64()),
        ]
    )
    out_parquet.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    batch_rows: list[dict[str, Any]] = []
    BATCH = 512
    with rows_jsonl.open("rb") as f, pq.ParquetWriter(out_parquet, schema) as writer:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            # Defensive re-validation; raises if the JSONL is malformed.
            ResultRow.model_validate(data)
            batch_rows.append(data)
            written += 1
            if len(batch_rows) >= BATCH:
                writer.write_table(pa.Table.from_pylist(batch_rows, schema=schema))
                batch_rows.clear()
        if batch_rows:
            writer.write_table(pa.Table.from_pylist(batch_rows, schema=schema))
    return written


__all__ = ["ResultsWriter", "derive_parquet_from_jsonl", "write_manifest_atomic"]
