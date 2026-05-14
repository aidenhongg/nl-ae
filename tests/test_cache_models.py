"""ActivationManifest + digest helpers — frozen, validated, deterministic."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from nl_ae.cache.models import (
    CACHE_KEY_COMPOSITION,
    ActivationManifest,
    LayerShardManifest,
    ShardRecord,
    compute_cache_key_composition_digest,
    compute_prompt_hash_set_digest,
    compute_quantization_spec_digest,
)
from nl_ae.schema.hashing import now_utc_iso
from nl_ae.schema.models import QuantizationSpec


def _shard(idx: int = 0, rows: int = 10) -> ShardRecord:
    return ShardRecord(
        shard_index=idx,
        relative_path=f"L00/shard-{idx:04d}.parquet",
        rows=rows,
        sha256="a" * 64,
    )


def _minimal_manifest(
    *,
    fold: str = "pilot",
    layers: tuple[int, ...] = (0, 1),
    rows_per_layer: int = 0,
    preregistration_digest: str | None = None,
) -> ActivationManifest:
    layer_shards = tuple(
        LayerShardManifest(
            layer=layer,
            shards=(_shard(0, rows=rows_per_layer),) if rows_per_layer > 0 else (),
            rows=rows_per_layer,
        )
        for layer in layers
    )
    composition = CACHE_KEY_COMPOSITION
    return ActivationManifest(
        run_id="r-1",
        fold=fold,  # type: ignore[arg-type]
        layers=layers,
        position_policy="last_prompt_token",
        activation_dtype="fp16",
        activation_dim=8,
        model_commit="unknown",
        quantization_spec_digest="b" * 64,
        quantization_kind="fp16",
        chat_template_hash="c" * 64,
        shard_rows=50_000,
        layer_shards=layer_shards,
        rows_written=rows_per_layer,
        rows_expected=rows_per_layer,
        prompt_hash_set_digest="d" * 64,
        cache_key_composition=composition,
        cache_key_composition_digest=compute_cache_key_composition_digest(composition),
        pilot_manifest_digest="e" * 64,
        preregistration_digest=preregistration_digest,
        completion_status="completed",
        started_at=now_utc_iso(),
        ended_at=now_utc_iso(),
    )


def test_cache_key_composition_digest_is_stable() -> None:
    d1 = compute_cache_key_composition_digest(CACHE_KEY_COMPOSITION)
    d2 = compute_cache_key_composition_digest(CACHE_KEY_COMPOSITION)
    assert d1 == d2
    # Order matters; reordered composition produces a different digest.
    reordered = CACHE_KEY_COMPOSITION[::-1]
    assert compute_cache_key_composition_digest(reordered) != d1


def test_prompt_hash_set_digest_is_order_independent() -> None:
    hashes = ["h1", "h2", "h3"]
    d1 = compute_prompt_hash_set_digest(hashes)
    d2 = compute_prompt_hash_set_digest(reversed(hashes))
    # Sorted internally, so order doesn't matter.
    assert d1 == d2
    # Duplicates collapse.
    assert compute_prompt_hash_set_digest(hashes * 3) == d1


def test_quantization_spec_digest_matches_for_equal_specs() -> None:
    a = QuantizationSpec(kind="fp16")
    b = QuantizationSpec(kind="fp16")
    assert compute_quantization_spec_digest(a) == compute_quantization_spec_digest(b)
    c = QuantizationSpec(kind="int4-nf4")
    assert compute_quantization_spec_digest(c) != compute_quantization_spec_digest(a)


def test_activation_manifest_is_frozen() -> None:
    m = _minimal_manifest()
    with pytest.raises(ValidationError):
        m.run_id = "other"  # type: ignore[misc]


def test_activation_manifest_round_trips_through_json() -> None:
    m = _minimal_manifest()
    payload = m.model_dump_json()
    again = ActivationManifest.model_validate_json(payload)
    assert again == m


def test_activation_manifest_rejects_unsorted_layers() -> None:
    composition = CACHE_KEY_COMPOSITION
    with pytest.raises(ValidationError):
        ActivationManifest(
            run_id="r-1",
            fold="pilot",
            layers=(1, 0),
            position_policy="last_prompt_token",
            activation_dtype="fp16",
            activation_dim=8,
            model_commit="unknown",
            quantization_spec_digest="b" * 64,
            quantization_kind="fp16",
            chat_template_hash="c" * 64,
            shard_rows=50_000,
            layer_shards=(
                LayerShardManifest(layer=0),
                LayerShardManifest(layer=1),
            ),
            rows_written=0,
            rows_expected=0,
            prompt_hash_set_digest="d" * 64,
            cache_key_composition=composition,
            cache_key_composition_digest=compute_cache_key_composition_digest(composition),
            pilot_manifest_digest="e" * 64,
            completion_status="completed",
            started_at=now_utc_iso(),
            ended_at=now_utc_iso(),
        )


def test_activation_manifest_rejects_rows_written_vs_layer_shards_drift() -> None:
    composition = CACHE_KEY_COMPOSITION
    # rows_written says 5 but layer shards each have 10 rows → 10 * 2 != 5 * 2.
    with pytest.raises(ValidationError):
        ActivationManifest(
            run_id="r-1",
            fold="pilot",
            layers=(0, 1),
            position_policy="last_prompt_token",
            activation_dtype="fp16",
            activation_dim=8,
            model_commit="unknown",
            quantization_spec_digest="b" * 64,
            quantization_kind="fp16",
            chat_template_hash="c" * 64,
            shard_rows=50_000,
            layer_shards=(
                LayerShardManifest(layer=0, shards=(_shard(0, 10),), rows=10),
                LayerShardManifest(layer=1, shards=(_shard(0, 10),), rows=10),
            ),
            rows_written=5,
            rows_expected=10,
            prompt_hash_set_digest="d" * 64,
            cache_key_composition=composition,
            cache_key_composition_digest=compute_cache_key_composition_digest(composition),
            pilot_manifest_digest="e" * 64,
            completion_status="completed",
            started_at=now_utc_iso(),
            ended_at=now_utc_iso(),
        )


def test_activation_manifest_requires_preregistration_digest_for_holdout() -> None:
    with pytest.raises(ValidationError):
        _minimal_manifest(fold="holdout", preregistration_digest=None)
    # Setting it works.
    m = _minimal_manifest(fold="holdout", preregistration_digest="f" * 64)
    assert m.preregistration_digest == "f" * 64


def test_activation_manifest_forbids_preregistration_digest_for_pilot() -> None:
    with pytest.raises(ValidationError):
        _minimal_manifest(fold="pilot", preregistration_digest="f" * 64)


def test_activation_manifest_rejects_layer_shards_layer_mismatch() -> None:
    composition = CACHE_KEY_COMPOSITION
    with pytest.raises(ValidationError):
        ActivationManifest(
            run_id="r-1",
            fold="pilot",
            layers=(0, 1),
            position_policy="last_prompt_token",
            activation_dtype="fp16",
            activation_dim=8,
            model_commit="unknown",
            quantization_spec_digest="b" * 64,
            quantization_kind="fp16",
            chat_template_hash="c" * 64,
            shard_rows=50_000,
            layer_shards=(  # only one layer when two declared
                LayerShardManifest(layer=0),
            ),
            rows_written=0,
            rows_expected=0,
            prompt_hash_set_digest="d" * 64,
            cache_key_composition=composition,
            cache_key_composition_digest=compute_cache_key_composition_digest(composition),
            pilot_manifest_digest="e" * 64,
            completion_status="completed",
            started_at=now_utc_iso(),
            ended_at=now_utc_iso(),
        )


def test_layer_shard_manifest_enforces_contiguous_shard_indices() -> None:
    with pytest.raises(ValidationError):
        LayerShardManifest(
            layer=0,
            shards=(_shard(0, 5), _shard(2, 5)),  # gap at index 1
            rows=10,
        )
