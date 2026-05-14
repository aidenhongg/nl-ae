"""ActivationCacheWriter + ActivationCacheReader — atomic shards, fold-aware reads."""

from __future__ import annotations

from pathlib import Path

import pytest

from nl_ae.cache.errors import (
    ActivationManifestMissingError,
    CacheKeyMismatchError,
    CacheLockError,
    DuplicateActivationRowError,
)
from nl_ae.cache.models import (
    CACHE_KEY_COMPOSITION,
    ActivationManifest,
    LayerShardManifest,
    compute_cache_key_composition_digest,
)
from nl_ae.cache.reader import ActivationCacheReader
from nl_ae.cache.writer import (
    ActivationCacheWriter,
    write_activation_manifest_atomic,
)
from nl_ae.schema.hashing import now_utc_iso
from nl_ae.schema.paths import run_paths

DIM = 8
LAYERS = (0, 1, 2)


def _make_manifest(
    *,
    fold: str = "pilot",
    rows_expected: int = 100,
    shard_rows: int = 4,
    preregistration_digest: str | None = None,
) -> ActivationManifest:
    composition = CACHE_KEY_COMPOSITION
    return ActivationManifest(
        run_id="20260101T000000Z-deadbee-test",
        fold=fold,  # type: ignore[arg-type]
        layers=LAYERS,
        position_policy="last_prompt_token",
        activation_dtype="fp16",
        activation_dim=DIM,
        model_commit="unknown",
        quantization_spec_digest="b" * 64,
        quantization_kind="fp16",
        chat_template_hash="c" * 64,
        shard_rows=shard_rows,
        layer_shards=tuple(LayerShardManifest(layer=layer) for layer in LAYERS),
        rows_written=0,
        rows_expected=rows_expected,
        prompt_hash_set_digest="d" * 64,
        cache_key_composition=composition,
        cache_key_composition_digest=compute_cache_key_composition_digest(composition),
        pilot_manifest_digest="e" * 64,
        preregistration_digest=preregistration_digest,
        completion_status="in_progress",
        started_at=now_utc_iso(),
    )


def _vec(seed: int) -> list[float]:
    return [float((seed + i) % 7) for i in range(DIM)]


def _write_visit(writer: ActivationCacheWriter, *, idx: int, item: str | None = None) -> None:
    writer.write_visit(
        item_id=item or f"mmlu/v1/x/q-{idx:08d}",
        permutation_id=idx % 4,
        template_id="mcq_flat_v1",
        prompt_hash=f"{idx:064x}",
        per_layer_vectors={layer: _vec(idx + layer) for layer in LAYERS},
    )


def _writer_for(tmp_path: Path, fold: str = "pilot", **kw: object) -> tuple[ActivationCacheWriter, Path]:
    run_dir = tmp_path / "20260101T000000Z-deadbee-test"
    paths = run_paths(run_dir.parent, run_dir.name)
    cache_dir = paths.fold_activations_dir(fold)  # type: ignore[arg-type]
    cache_dir.mkdir(parents=True, exist_ok=True)
    writer = ActivationCacheWriter(
        cache_dir=cache_dir,
        manifest=_make_manifest(fold=fold, **kw),  # type: ignore[arg-type]
    )
    return writer, run_dir


def test_writer_round_trip_one_shard(tmp_path: Path) -> None:
    writer, run_dir = _writer_for(tmp_path, shard_rows=10)
    with writer:
        for i in range(3):
            _write_visit(writer, idx=i)
        writer.finalize(status="completed")

    reader = ActivationCacheReader.open(run_dir, "pilot")
    assert reader.manifest.completion_status == "completed"
    assert reader.manifest.rows_written == 3
    assert reader.completed_visit_keys() == {
        (f"mmlu/v1/x/q-{i:08d}", i % 4, "mcq_flat_v1") for i in range(3)
    }
    rows = list(reader.iter_rows(layer=0))
    assert len(rows) == 3
    assert rows[0]["activation_dim"] == DIM
    assert len(rows[0]["activation"]) == DIM
    # Every layer has identical visit count.
    for layer in LAYERS:
        assert len(list(reader.iter_rows(layer=layer))) == 3


def test_writer_rotates_shards_at_threshold(tmp_path: Path) -> None:
    writer, run_dir = _writer_for(tmp_path, shard_rows=4)
    with writer:
        for i in range(10):
            _write_visit(writer, idx=i)
        writer.finalize(status="completed")

    reader = ActivationCacheReader.open(run_dir, "pilot")
    # 10 visits / 4 per shard = 3 shards (4 + 4 + 2 finalized on completed flush).
    for ls in reader.manifest.layer_shards:
        assert ls.rows == 10
        assert len(ls.shards) == 3
        assert [s.rows for s in ls.shards] == [4, 4, 2]
    assert reader.manifest.rows_written == 10


def test_writer_refuses_duplicate_within_shard(tmp_path: Path) -> None:
    writer, _ = _writer_for(tmp_path, shard_rows=100)
    with pytest.raises(DuplicateActivationRowError), writer:
        _write_visit(writer, idx=0)
        _write_visit(writer, idx=0)  # same visit key
        writer.finalize(status="completed")


def test_writer_refuses_dim_mismatch(tmp_path: Path) -> None:
    writer, _ = _writer_for(tmp_path)
    with pytest.raises(ValueError, match="dim"), writer:
        writer.write_visit(
            item_id="mmlu/v1/x/q-00",
            permutation_id=0,
            template_id="mcq_flat_v1",
            prompt_hash="0" * 64,
            per_layer_vectors={layer: [0.0] * (DIM + 1) for layer in LAYERS},
        )


def test_writer_refuses_missing_layer(tmp_path: Path) -> None:
    writer, _ = _writer_for(tmp_path)
    with pytest.raises(ValueError, match="layer set mismatch"), writer:
        writer.write_visit(
            item_id="mmlu/v1/x/q-00",
            permutation_id=0,
            template_id="mcq_flat_v1",
            prompt_hash="0" * 64,
            per_layer_vectors={0: _vec(0), 1: _vec(1)},  # missing layer 2
        )


def test_writer_lock_file_blocks_concurrent_open(tmp_path: Path) -> None:
    writer_a, run_dir = _writer_for(tmp_path)
    with writer_a:
        writer_b, _ = _writer_for(tmp_path)
        with pytest.raises(CacheLockError), writer_b:
            pass
        writer_a.finalize(status="completed")
    # Lock cleared after finalize.
    assert not (run_dir / "pilot" / "activations" / ".run.lock").exists()


def test_reader_open_refuses_missing_manifest(tmp_path: Path) -> None:
    run_dir = tmp_path / "20260101T000000Z-deadbee-test"
    run_dir.mkdir()
    with pytest.raises(ActivationManifestMissingError):
        ActivationCacheReader.open(run_dir, "pilot")


def test_reader_refuses_fold_dir_mismatch(tmp_path: Path) -> None:
    """A holdout manifest left in a pilot directory should refuse loading."""
    writer, run_dir = _writer_for(tmp_path, fold="pilot")
    with writer:
        _write_visit(writer, idx=0)
        writer.finalize(status="completed")
    # Move pilot manifest to holdout dir (simulate corruption).
    pilot_manifest = run_dir / "pilot" / "activations" / "activation_manifest.json"
    holdout_dir = run_dir / "holdout" / "activations"
    holdout_dir.mkdir(parents=True, exist_ok=True)
    pilot_manifest_text = pilot_manifest.read_text(encoding="utf-8")
    (holdout_dir / "activation_manifest.json").write_text(pilot_manifest_text, encoding="utf-8")
    with pytest.raises(CacheKeyMismatchError, match=r"manifest\.fold"):
        ActivationCacheReader.open(run_dir, "holdout")


def test_reader_refuses_composition_drift(tmp_path: Path) -> None:
    writer, run_dir = _writer_for(tmp_path)
    with writer:
        _write_visit(writer, idx=0)
        writer.finalize(status="completed")

    # Mutate cache_key_composition_digest on disk to simulate a future-schema
    # cache that the current reader doesn't understand. Pydantic re-validates
    # at load and rejects the now-inconsistent (composition, digest) pair.
    from pydantic import ValidationError

    manifest_path = run_dir / "pilot" / "activations" / "activation_manifest.json"
    text = manifest_path.read_text(encoding="utf-8")
    text = text.replace(
        compute_cache_key_composition_digest(CACHE_KEY_COMPOSITION),
        compute_cache_key_composition_digest((*CACHE_KEY_COMPOSITION, "foo")),
    )
    manifest_path.write_text(text, encoding="utf-8")
    with pytest.raises(ValidationError):
        ActivationCacheReader.open(run_dir, "pilot")


def test_reader_refuses_partial_unless_opted_in(tmp_path: Path) -> None:
    writer, run_dir = _writer_for(tmp_path)
    with writer:
        _write_visit(writer, idx=0)
        # Don't finalize — context manager will mark as aborted on exit.
    with pytest.raises(CacheKeyMismatchError, match="completion_status"):
        ActivationCacheReader.open(run_dir, "pilot")
    # allow_partial bypasses.
    reader = ActivationCacheReader.open(run_dir, "pilot", allow_partial=True)
    assert reader.manifest.completion_status in ("aborted", "failed")


def test_write_activation_manifest_atomic_creates_sidecar(tmp_path: Path) -> None:
    manifest = _make_manifest()
    manifest = manifest.model_copy(
        update={"completion_status": "completed", "ended_at": now_utc_iso()}
    )
    target = tmp_path / "activation_manifest.json"
    digest = write_activation_manifest_atomic(target, manifest)
    assert target.exists()
    sidecar = tmp_path / "activation_manifest.json.sha256"
    assert sidecar.exists()
    assert sidecar.read_text().strip() == digest
