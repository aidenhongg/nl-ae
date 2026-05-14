"""Pydantic v2 models + digests for the per-fold activation cache (C06).

Schema 1.0.0. Two on-disk artifacts:

* The per-shard parquet rows (column schema in :data:`ACTIVATION_PARQUET_COLUMNS`).
* The ``activation_manifest.json`` (:class:`ActivationManifest`), atomically
  written + ``.sha256``-sided after every shard rotation and on finalize.

The per-row :class:`ActivationCacheKey` lives in process memory only; the
parquet shards carry the same fields as columns.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from typing import Annotated, Final, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

from nl_ae.pilot.models import LayerIndex
from nl_ae.schema.models import IsoUtcStr, ItemIdStr, QuantizationSpec, Sha256Hex

ACTIVATION_MANIFEST_SCHEMA_VERSION: Literal["1.0.0"] = "1.0.0"

# Field names that compose the per-row activation cache identity. Order is
# significant — the composition digest below is computed over the joined names.
CACHE_KEY_COMPOSITION: Final[tuple[str, ...]] = (
    "layer",
    "fold",
    "item_id",
    "permutation_id",
    "template_id",
    "run_id",
    "position_policy",
    "quantization_spec_digest",
    "prompt_hash",
    "model_commit",
)

PositionPolicy = Literal["last_prompt_token"]
ActivationDtype = Literal["fp16"]
ModelCommitField = Annotated[
    str, StringConstraints(min_length=1, max_length=64, pattern=r"^[0-9a-fA-F]{7,40}$|^unknown$")
]
TemplateIdField = Annotated[str, StringConstraints(min_length=1, max_length=64)]


# --- digests ----------------------------------------------------------------


def compute_cache_key_composition_digest(composition: Iterable[str]) -> Sha256Hex:
    """SHA-256 over the ``|``-joined cache-key field names.

    A reader that recomputes this digest from its *expected* composition and
    compares against the manifest catches "we added a new key field but the
    on-disk cache predates it".
    """
    payload = "|".join(composition).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def compute_quantization_spec_digest(spec: QuantizationSpec) -> Sha256Hex:
    """SHA-256 over the canonical JSON of a :class:`QuantizationSpec`."""
    payload = json.dumps(
        spec.model_dump(mode="json"),
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def compute_prompt_hash_set_digest(prompt_hashes: Iterable[str]) -> Sha256Hex:
    """SHA-256 over the sorted, deduplicated, newline-joined ``prompt_hash`` set.

    Pilot and holdout caches over the same Phase 1 run share a base composition
    (run_id, model_commit, etc.) and differ in their prompt-hash-set fingerprint;
    that's the proof that the two caches are over disjoint row sets.
    """
    sorted_hashes = sorted({h for h in prompt_hashes})
    payload = "\n".join(sorted_hashes).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


# --- structured records -----------------------------------------------------


class ActivationCacheKey(BaseModel):
    """Per-row cache identity (in-memory; never serialized as a unit).

    The parquet rows carry these fields as columns. Two rows in the same fold
    sharing ``(layer, item_id, permutation_id, template_id)`` is a loud error.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    layer: LayerIndex
    fold: Literal["pilot", "holdout"]
    item_id: ItemIdStr
    permutation_id: Annotated[int, Field(ge=0)]
    template_id: TemplateIdField
    run_id: Annotated[str, StringConstraints(min_length=1, max_length=64)]
    position_policy: PositionPolicy
    quantization_spec_digest: Sha256Hex
    prompt_hash: Sha256Hex
    model_commit: ModelCommitField


class ShardRecord(BaseModel):
    """One per-layer parquet shard's on-disk identity + accounting."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    shard_index: Annotated[int, Field(ge=0)]
    relative_path: Annotated[str, StringConstraints(min_length=1, max_length=128)]
    rows: Annotated[int, Field(ge=0)]
    sha256: Sha256Hex


class LayerShardManifest(BaseModel):
    """The shard inventory for one layer."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    layer: LayerIndex
    shards: tuple[ShardRecord, ...] = ()
    rows: Annotated[int, Field(ge=0)] = 0

    @model_validator(mode="after")
    def _check_indices_and_sum(self) -> Self:
        for expected, rec in enumerate(self.shards):
            if rec.shard_index != expected:
                raise ValueError(
                    f"layer {self.layer}: shard_index {rec.shard_index} not contiguous "
                    f"(expected {expected})"
                )
        if sum(s.rows for s in self.shards) != self.rows:
            raise ValueError(
                f"layer {self.layer}: rows={self.rows} disagrees with sum-of-shards "
                f"({sum(s.rows for s in self.shards)})"
            )
        return self


class ActivationManifest(BaseModel):
    """Per-fold cache manifest.

    Carries the run-level constants of every cached row (model commit,
    quantization spec digest, chat-template hash, position policy) plus the
    layer-major shard inventory and the cache-key composition + digest. The
    reader cross-checks the composition digest against its own expectation;
    drift refuses reads.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["1.0.0"] = ACTIVATION_MANIFEST_SCHEMA_VERSION
    run_id: Annotated[str, StringConstraints(min_length=1, max_length=64)]
    fold: Literal["pilot", "holdout"]
    layers: tuple[LayerIndex, ...] = Field(min_length=1)
    position_policy: PositionPolicy
    activation_dtype: ActivationDtype
    activation_dim: Annotated[int, Field(ge=1, le=65_536)]
    model_commit: ModelCommitField
    quantization_spec_digest: Sha256Hex
    quantization_kind: Annotated[str, StringConstraints(min_length=1, max_length=32)]
    chat_template_hash: Sha256Hex
    shard_rows: Annotated[int, Field(ge=1)]
    layer_shards: tuple[LayerShardManifest, ...] = Field(min_length=1)
    rows_written: Annotated[int, Field(ge=0)]
    rows_expected: Annotated[int, Field(ge=0)]
    prompt_hash_set_digest: Sha256Hex
    cache_key_composition: tuple[str, ...] = Field(min_length=1)
    cache_key_composition_digest: Sha256Hex
    pilot_manifest_digest: Sha256Hex
    preregistration_digest: Sha256Hex | None = None
    completion_status: Literal["in_progress", "completed", "failed", "aborted"]
    failure_reason: str | None = None
    started_at: IsoUtcStr
    ended_at: IsoUtcStr | None = None

    @model_validator(mode="after")
    def _check_invariants(self) -> Self:
        if len(set(self.layers)) != len(self.layers):
            raise ValueError("activation_manifest.layers must be unique")
        if tuple(sorted(self.layers)) != self.layers:
            raise ValueError("activation_manifest.layers must be sorted ascending")
        seen_layers = {ls.layer for ls in self.layer_shards}
        if seen_layers != set(self.layers):
            raise ValueError(
                "activation_manifest.layer_shards layer set disagrees with .layers"
            )
        expected_digest = compute_cache_key_composition_digest(self.cache_key_composition)
        if expected_digest != self.cache_key_composition_digest:
            raise ValueError(
                "cache_key_composition_digest does not match composition fields "
                f"({expected_digest} vs {self.cache_key_composition_digest})"
            )
        total_rows = sum(ls.rows for ls in self.layer_shards)
        # rows_written counts visits (one per layer per row), so per-layer rows
        # times len(layers) must equal total parquet rows across layers.
        if total_rows != self.rows_written * len(self.layers):
            raise ValueError(
                f"rows_written * n_layers = {self.rows_written * len(self.layers)} "
                f"disagrees with sum-of-layer-rows = {total_rows}"
            )
        if self.fold == "pilot" and self.preregistration_digest is not None:
            raise ValueError("preregistration_digest must be null for pilot fold")
        if self.fold == "holdout" and self.preregistration_digest is None:
            raise ValueError("preregistration_digest is required for holdout fold")
        if self.completion_status == "completed" and self.ended_at is None:
            raise ValueError("ended_at is required when completion_status='completed'")
        return self


# --- parquet schema ---------------------------------------------------------


ACTIVATION_PARQUET_COLUMNS: Final[tuple[str, ...]] = (
    "run_id",
    "fold",
    "layer",
    "item_id",
    "permutation_id",
    "template_id",
    "prompt_hash",
    "position_policy",
    "activation_dtype",
    "activation_dim",
    "activation",
    "extracted_at",
)


__all__ = [
    "ACTIVATION_MANIFEST_SCHEMA_VERSION",
    "ACTIVATION_PARQUET_COLUMNS",
    "CACHE_KEY_COMPOSITION",
    "ActivationCacheKey",
    "ActivationDtype",
    "ActivationManifest",
    "LayerShardManifest",
    "ModelCommitField",
    "PositionPolicy",
    "ShardRecord",
    "TemplateIdField",
    "compute_cache_key_composition_digest",
    "compute_prompt_hash_set_digest",
    "compute_quantization_spec_digest",
]
