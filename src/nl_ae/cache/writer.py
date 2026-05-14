"""Per-layer parquet writer + manifest atomic writer for the activation cache.

Shard discipline: each layer's buffer flushes to ``L<NN>/shard-NNNN.partial.parquet``
then renames atomically to ``L<NN>/shard-NNNN.parquet``. The manifest is rewritten
atomically (temp-file + ``os.replace``) after every shard rotation so a crash at
any point still leaves a readable, internally consistent manifest pointing at
finalized shards only.

All layers share the same flush trigger (``shard_rows`` rows in the buffer);
since every visit contributes exactly one row per requested layer, the layers
rotate in lockstep.
"""

from __future__ import annotations

import contextlib
import hashlib
import os
import tempfile
from collections.abc import Sequence
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from pathlib import Path
from types import TracebackType
from typing import TYPE_CHECKING, Any, Literal

from nl_ae.schema.hashing import now_utc_iso
from nl_ae.schema.models import Sha256Hex

from .errors import CacheLockError, DuplicateActivationRowError
from .models import (
    ActivationManifest,
    LayerShardManifest,
    ShardRecord,
)

if TYPE_CHECKING:  # pragma: no cover
    import pyarrow as pa


def _layer_dir_name(layer: int) -> str:
    return f"L{layer:02d}"


def _shard_file_name(shard_index: int, *, partial: bool = False) -> str:
    suffix = ".partial.parquet" if partial else ".parquet"
    return f"shard-{shard_index:04d}{suffix}"


def _build_parquet_schema(activation_dim: int) -> pa.Schema:
    import pyarrow as pa

    return pa.schema(
        [
            pa.field("run_id", pa.string(), nullable=False),
            pa.field("fold", pa.string(), nullable=False),
            pa.field("layer", pa.int16(), nullable=False),
            pa.field("item_id", pa.string(), nullable=False),
            pa.field("permutation_id", pa.int32(), nullable=False),
            pa.field("template_id", pa.string(), nullable=False),
            pa.field("prompt_hash", pa.string(), nullable=False),
            pa.field("position_policy", pa.string(), nullable=False),
            pa.field("activation_dtype", pa.string(), nullable=False),
            pa.field("activation_dim", pa.int32(), nullable=False),
            pa.field(
                "activation",
                pa.list_(pa.float16(), activation_dim),
                nullable=False,
            ),
            pa.field("extracted_at", pa.string(), nullable=False),
        ]
    )


def _hash_file(path: Path, *, chunk: int = 1 << 20) -> Sha256Hex:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            block = f.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


@dataclass
class _LayerBuffer:
    """In-memory accumulator for one layer between shard rotations."""

    layer: int
    rows: list[dict[str, Any]] = field(default_factory=list)
    shards: list[ShardRecord] = field(default_factory=list)

    def append(self, row: dict[str, Any]) -> None:
        self.rows.append(row)

    @property
    def total_rows(self) -> int:
        return sum(s.rows for s in self.shards) + len(self.rows)

    @property
    def finalized_rows(self) -> int:
        return sum(s.rows for s in self.shards)

    def consume(self) -> list[dict[str, Any]]:
        out = self.rows
        self.rows = []
        return out


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(payload)
            f.flush()
            with contextlib.suppress(OSError):
                os.fsync(f.fileno())
        os.replace(tmp_name, path)
    except BaseException:
        try:
            Path(tmp_name).unlink(missing_ok=True)
        finally:
            raise


def write_activation_manifest_atomic(path: Path, manifest: ActivationManifest) -> Sha256Hex:
    """Atomic-replace ``path`` with the manifest. Returns SHA-256 of the bytes."""
    payload = manifest.model_dump_json(indent=2, exclude_none=False).encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()
    _atomic_write(path, payload)
    sidecar = path.with_suffix(path.suffix + ".sha256")
    sidecar.write_text(digest + "\n", encoding="utf-8")
    return digest


class ActivationCacheWriter(AbstractContextManager["ActivationCacheWriter"]):
    """Layer-major parquet writer with atomic shard rotation + manifest updates.

    One writer per ``(run_dir, fold)``; the lock file at
    ``<fold>/activations/.run.lock`` prevents concurrent extractors.
    """

    def __init__(
        self,
        *,
        cache_dir: Path,
        manifest: ActivationManifest,
        seed_shards: Sequence[LayerShardManifest] | None = None,
    ) -> None:
        if cache_dir.name != "activations":
            raise ValueError(
                f"cache_dir must be named 'activations'; got {cache_dir.name!r}"
            )
        if len(manifest.layers) == 0:
            raise ValueError("manifest must declare at least one layer")
        self._cache_dir = cache_dir
        self._manifest = manifest
        self._buffers: dict[int, _LayerBuffer] = {
            layer: _LayerBuffer(layer=layer) for layer in manifest.layers
        }
        if seed_shards:
            seen_layers = set()
            for ls in seed_shards:
                if ls.layer in seen_layers:
                    raise ValueError(f"duplicate seed shard layer {ls.layer}")
                if ls.layer not in self._buffers:
                    raise ValueError(
                        f"seed shard layer {ls.layer} not declared in manifest.layers"
                    )
                self._buffers[ls.layer].shards.extend(ls.shards)
                seen_layers.add(ls.layer)
        self._rows_written = self._finalized_visit_count()
        self._lock_path = cache_dir / ".run.lock"
        self._manifest_path = cache_dir / "activation_manifest.json"
        self._partial_manifest_path = cache_dir / "activation_manifest.json.partial"
        self._opened = False
        self._finalized = False
        # Cross-shard duplicate guard: per-shard we additionally enforce that no
        # ``(item_id, perm_id, template_id)`` appears twice in the current buffer.
        # Across shards the resume scan filters duplicates upstream.
        self._buffer_visit_keys: set[tuple[str, int, str]] = set()

    # --- properties -----------------------------------------------------

    @property
    def cache_dir(self) -> Path:
        return self._cache_dir

    @property
    def manifest(self) -> ActivationManifest:
        return self._manifest

    @property
    def rows_written(self) -> int:
        """Number of visits (one per layer per visit) durably written to shards."""
        return self._rows_written

    @property
    def buffered_visits(self) -> int:
        """Visits in the in-memory buffer not yet flushed to shards."""
        first_layer = self._manifest.layers[0]
        return len(self._buffers[first_layer].rows)

    # --- lifecycle ------------------------------------------------------

    def __enter__(self) -> ActivationCacheWriter:
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
            self.finalize(status=status, failure_reason=reason)
        finally:
            self._release_lock()

    def _open(self) -> None:
        if self._opened:
            return
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        if self._lock_path.exists():
            raise CacheLockError(
                f"activation cache lock present at {self._lock_path}; another extractor "
                "is open or a prior run crashed (delete to recover)"
            )
        self._lock_path.write_text(f"pid={os.getpid()}\n", encoding="utf-8")
        # Stamp the initial manifest so partial readers see a valid header.
        self._persist_manifest(status="in_progress", failure_reason=None, ended_at=None)
        self._opened = True

    def _release_lock(self) -> None:
        if self._lock_path.exists():
            with contextlib.suppress(OSError):
                self._lock_path.unlink()

    # --- write path -----------------------------------------------------

    def write_visit(
        self,
        *,
        item_id: str,
        permutation_id: int,
        template_id: str,
        prompt_hash: str,
        per_layer_vectors: dict[int, Sequence[float]],
        extracted_at: str | None = None,
    ) -> None:
        """Append one visit's per-layer vectors to the in-memory buffer.

        Raises :class:`DuplicateActivationRowError` if the ``(item_id, perm_id,
        template_id)`` is already in the current shard buffer. Auto-flushes when
        the buffer hits ``manifest.shard_rows``.
        """
        if not self._opened:
            raise RuntimeError("writer is not open; use as a context manager")
        if self._finalized:
            raise RuntimeError("writer has been finalized")
        visit_key = (item_id, permutation_id, template_id)
        if visit_key in self._buffer_visit_keys:
            raise DuplicateActivationRowError(
                f"duplicate activation row in current shard: {visit_key!r}"
            )
        if set(per_layer_vectors.keys()) != set(self._manifest.layers):
            missing = set(self._manifest.layers) - set(per_layer_vectors.keys())
            extra = set(per_layer_vectors.keys()) - set(self._manifest.layers)
            raise ValueError(
                f"per_layer_vectors layer set mismatch: missing={sorted(missing)}, "
                f"extra={sorted(extra)}"
            )
        timestamp = extracted_at or now_utc_iso()
        for layer in self._manifest.layers:
            vec = per_layer_vectors[layer]
            if len(vec) != self._manifest.activation_dim:
                raise ValueError(
                    f"layer {layer} vector dim {len(vec)} != manifest.activation_dim "
                    f"{self._manifest.activation_dim}"
                )
            self._buffers[layer].append(
                {
                    "run_id": self._manifest.run_id,
                    "fold": self._manifest.fold,
                    "layer": layer,
                    "item_id": item_id,
                    "permutation_id": permutation_id,
                    "template_id": template_id,
                    "prompt_hash": prompt_hash,
                    "position_policy": self._manifest.position_policy,
                    "activation_dtype": self._manifest.activation_dtype,
                    "activation_dim": self._manifest.activation_dim,
                    "activation": list(vec),
                    "extracted_at": timestamp,
                }
            )
        self._buffer_visit_keys.add(visit_key)
        if self._first_layer_buffer_size() >= self._manifest.shard_rows:
            self.flush()

    def _first_layer_buffer_size(self) -> int:
        return len(self._buffers[self._manifest.layers[0]].rows)

    def flush(self) -> None:
        """Rotate the current shard for every layer atomically; update the manifest."""
        if not self._opened or self._finalized:
            return
        if self._first_layer_buffer_size() == 0:
            return
        for layer in self._manifest.layers:
            self._flush_layer(layer)
        self._buffer_visit_keys.clear()
        self._rows_written = self._finalized_visit_count()
        self._persist_manifest(
            status="in_progress", failure_reason=None, ended_at=None
        )

    def _flush_layer(self, layer: int) -> None:
        import pyarrow as pa
        import pyarrow.parquet as pq

        buf = self._buffers[layer]
        if not buf.rows:
            return
        layer_dir = self._cache_dir / _layer_dir_name(layer)
        layer_dir.mkdir(parents=True, exist_ok=True)
        shard_index = len(buf.shards)
        partial = layer_dir / _shard_file_name(shard_index, partial=True)
        final = layer_dir / _shard_file_name(shard_index, partial=False)

        schema = _build_parquet_schema(self._manifest.activation_dim)
        table = pa.Table.from_pylist(buf.consume(), schema=schema)
        pq.write_table(
            table,
            partial.as_posix(),
            compression="zstd",
            compression_level=3,
            row_group_size=4_096,
        )
        os.replace(partial, final)
        digest = _hash_file(final)
        buf.shards.append(
            ShardRecord(
                shard_index=shard_index,
                relative_path=f"{layer_dir.name}/{final.name}",
                rows=table.num_rows,
                sha256=digest,
            )
        )

    def _finalized_visit_count(self) -> int:
        # Every layer carries the same visit set; pick the first.
        first = self._manifest.layers[0]
        return self._buffers[first].finalized_rows

    # --- manifest persistence -------------------------------------------

    def _persist_manifest(
        self,
        *,
        status: Literal["in_progress", "completed", "failed", "aborted"],
        failure_reason: str | None,
        ended_at: str | None,
    ) -> None:
        layer_shards = tuple(
            LayerShardManifest(
                layer=layer,
                shards=tuple(self._buffers[layer].shards),
                rows=self._buffers[layer].finalized_rows,
            )
            for layer in self._manifest.layers
        )
        updated = self._manifest.model_copy(
            update={
                "layer_shards": layer_shards,
                "rows_written": self._finalized_visit_count(),
                "completion_status": status,
                "failure_reason": failure_reason,
                "ended_at": ended_at,
            }
        )
        # Re-validate via model_validate to engage the cross-field invariants.
        validated = ActivationManifest.model_validate(updated.model_dump())
        self._manifest = validated
        write_activation_manifest_atomic(self._manifest_path, validated)

    def finalize(
        self,
        *,
        status: Literal["completed", "failed", "aborted"] = "completed",
        failure_reason: str | None = None,
    ) -> None:
        """Drain remaining buffers + stamp completion status on the manifest."""
        if self._finalized:
            return
        try:
            if status == "completed":
                self.flush()
            self._persist_manifest(
                status=status,
                failure_reason=failure_reason,
                ended_at=now_utc_iso(),
            )
        finally:
            self._finalized = True
            self._release_lock()


__all__ = [
    "ActivationCacheWriter",
    "write_activation_manifest_atomic",
]
