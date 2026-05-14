"""Atomic writers for probe-cell artifacts + per-label aggregates (C07).

Conventions:

* Every public ``write_*_atomic`` writes to ``<path>.partial`` (or a sibling
  temporary file) and renames into place via ``os.replace`` (atomic on POSIX
  and Windows for same-volume moves).
* ``write_metrics_atomic`` / ``write_coef_atomic`` / ``write_probe_manifest_atomic``
  return the SHA-256 of the bytes actually placed on disk so the trainer can
  pin them on the manifest record.
* Per-cell prediction parquets live at ``<label>/predictions.L<NN>.parquet``
  while a label is in flight; :func:`finalize_label_predictions` concatenates
  them into ``<label>/predictions.parquet`` and unlinks the per-layer shards
  once the label is done.

pyarrow / numpy are lazy-imported inside the function bodies so importing the
``nl_ae.probes`` package does not require the ``[probes]`` extra unless the
caller actually fits.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from nl_ae.schema.models import Sha256Hex

from .labels import LabelExtraction
from .models import ProbeManifest

if TYPE_CHECKING:  # pragma: no cover
    import numpy as np
    import pyarrow as pa

    from .fitter import FitResult


SubFoldLiteral = Literal["train", "val", "test"]


# --- low-level atomic primitives -------------------------------------------


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=str(path.parent)
    )
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


def _atomic_replace_file(src: Path, dest: Path) -> None:
    """Rename ``src`` over ``dest`` atomically (same directory)."""
    os.replace(src, dest)


# --- per-cell writers ------------------------------------------------------


def write_metrics_atomic(path: Path, metrics: dict[str, Any]) -> Sha256Hex:
    """Atomically write ``metrics.json``. Returns the SHA-256 of the bytes."""
    payload = (
        json.dumps(metrics, sort_keys=True, ensure_ascii=False, indent=2).encode("utf-8")
    )
    _atomic_write_bytes(path, payload)
    return hashlib.sha256(payload).hexdigest()


def write_coef_atomic(
    path: Path, coef: np.ndarray, intercept: np.ndarray
) -> Sha256Hex:
    """Atomically write ``coef.npy`` (actually ``np.savez`` with ``coef`` +
    ``intercept`` arrays). Returns the SHA-256 of the on-disk bytes."""
    import numpy as np

    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(fd, "wb") as f:
            np.savez(f, coef=coef, intercept=intercept)
            f.flush()
            with contextlib.suppress(OSError):
                os.fsync(f.fileno())
        os.replace(tmp_name, path)
    except BaseException:
        try:
            Path(tmp_name).unlink(missing_ok=True)
        finally:
            raise
    return _hash_file(path)


def _hash_file(path: Path, *, chunk: int = 1 << 20) -> Sha256Hex:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            block = f.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


# --- predictions parquet (per-layer + per-label) ---------------------------


_PREDICTIONS_LAYER_TEMPLATE = "predictions.L{layer:02d}.parquet"
_PREDICTIONS_FINAL = "predictions.parquet"


def _predictions_schema() -> pa.Schema:
    import pyarrow as pa

    return pa.schema(
        [
            pa.field("run_id", pa.string(), nullable=False),
            pa.field("fold", pa.string(), nullable=False),
            pa.field("item_id", pa.string(), nullable=False),
            pa.field("permutation_id", pa.int32(), nullable=False),
            pa.field("template_id", pa.string(), nullable=False),
            pa.field("sub_fold", pa.string(), nullable=False),
            pa.field("layer", pa.int16(), nullable=False),
            pa.field("gold", pa.string(), nullable=False),
            pa.field("pred", pa.string(), nullable=False),
            pa.field("class", pa.string(), nullable=False),
            pa.field("prob", pa.float64(), nullable=False),
        ]
    )


def append_predictions_long(
    label_dir: Path,
    *,
    run_id: str,
    fold: str,
    layer: int,
    item_ids: Sequence[str],
    permutation_ids: Sequence[int],
    template_ids: Sequence[str],
    sub_folds: Sequence[str],
    gold_values: Sequence[str],
    fit_result: FitResult,
    extraction: LabelExtraction,
) -> Path:
    """Write the per-layer slice of the long-layout predictions parquet.

    Lands at ``label_dir / predictions.L<NN>.parquet``. Atomic via
    ``.partial`` → ``os.replace``. For binary cells, ``len(rows)=N``; for
    multi-class, ``len(rows)=N * len(fit_result.classes)``.

    The columns ``gold`` / ``pred`` / ``class`` are stringified at write time
    (sklearn's ``predict`` already returns the train-time class dtype, which
    we mirror in ``LabelExtraction`` — both are strings).
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    n_valid = len(item_ids)
    if any(len(seq) != n_valid for seq in (permutation_ids, template_ids, sub_folds, gold_values)):
        raise ValueError("visit-side arrays must have equal length")
    if fit_result.test_pred.shape[0] + fit_result.val_pred.shape[0] + fit_result.train_pred.shape[0] != n_valid:
        raise ValueError(
            "sum of train/val/test predictions disagrees with visit count: "
            f"{fit_result.train_pred.shape[0]} + {fit_result.val_pred.shape[0]} + "
            f"{fit_result.test_pred.shape[0]} != {n_valid}"
        )

    pred_by_sub: dict[str, np.ndarray] = {
        "train": fit_result.train_pred,
        "val": fit_result.val_pred,
        "test": fit_result.test_pred,
    }
    proba_by_sub: dict[str, np.ndarray] = {
        "train": fit_result.train_proba,
        "val": fit_result.val_proba,
        "test": fit_result.test_proba,
    }
    cursor: dict[str, int] = {"train": 0, "val": 0, "test": 0}

    rows: list[dict[str, Any]] = []
    classes = fit_result.classes
    is_binary = extraction.is_binary

    for i in range(n_valid):
        sub = sub_folds[i]
        idx = cursor[sub]
        cursor[sub] = idx + 1
        pred = str(pred_by_sub[sub][idx])
        if is_binary:
            prob_pos = float(proba_by_sub[sub][idx])
            rows.append(
                {
                    "run_id": run_id,
                    "fold": fold,
                    "item_id": str(item_ids[i]),
                    "permutation_id": int(permutation_ids[i]),
                    "template_id": str(template_ids[i]),
                    "sub_fold": sub,
                    "layer": int(layer),
                    "gold": str(gold_values[i]),
                    "pred": pred,
                    "class": "1",
                    "prob": prob_pos,
                }
            )
        else:
            probs_row = proba_by_sub[sub][idx]
            for c_idx, c_name in enumerate(classes):
                rows.append(
                    {
                        "run_id": run_id,
                        "fold": fold,
                        "item_id": str(item_ids[i]),
                        "permutation_id": int(permutation_ids[i]),
                        "template_id": str(template_ids[i]),
                        "sub_fold": sub,
                        "layer": int(layer),
                        "gold": str(gold_values[i]),
                        "pred": pred,
                        "class": str(c_name),
                        "prob": float(probs_row[c_idx]),
                    }
                )

    schema = _predictions_schema()
    table = pa.Table.from_pylist(rows, schema=schema)
    out = label_dir / _PREDICTIONS_LAYER_TEMPLATE.format(layer=layer)
    label_dir.mkdir(parents=True, exist_ok=True)
    partial = out.with_suffix(out.suffix + ".partial")
    pq.write_table(table, partial.as_posix(), compression="zstd", compression_level=3)
    _atomic_replace_file(partial, out)
    return out


def finalize_label_predictions(label_dir: Path, *, layers: Sequence[int]) -> Path | None:
    """Concat ``predictions.L<NN>.parquet`` files into ``predictions.parquet``.

    Idempotent: if no per-layer files exist and ``predictions.parquet`` is
    present, returns the existing path without rewriting. If per-layer files
    are present, concatenates them in layer-sorted order, writes the final
    file atomically, and unlinks the per-layer shards.

    Returns the final path or ``None`` if neither per-layer files nor an
    existing final file are present.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    final = label_dir / _PREDICTIONS_FINAL
    per_layer = sorted(
        (label_dir / _PREDICTIONS_LAYER_TEMPLATE.format(layer=layer) for layer in layers),
        key=lambda p: p.name,
    )
    present = [p for p in per_layer if p.exists()]
    if not present:
        return final if final.exists() else None

    schema = _predictions_schema()
    tables = [pq.read_table(p.as_posix()) for p in present]
    combined = pa.concat_tables(tables, promote_options="default").cast(schema)
    partial = final.with_suffix(final.suffix + ".partial")
    pq.write_table(combined, partial.as_posix(), compression="zstd", compression_level=3)
    _atomic_replace_file(partial, final)
    for p in present:
        with contextlib.suppress(OSError):
            p.unlink()
    return final


# --- per-label summary parquet --------------------------------------------


def write_summary(path: Path, *, cell_rows: list[dict[str, Any]]) -> None:
    """Write the per-layer summary parquet (one row per layer). Atomic."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(cell_rows)
    partial = path.with_suffix(path.suffix + ".partial")
    pq.write_table(table, partial.as_posix(), compression="zstd", compression_level=3)
    _atomic_replace_file(partial, path)


# --- manifest --------------------------------------------------------------


def write_probe_manifest_atomic(path: Path, manifest: ProbeManifest) -> Sha256Hex:
    """Atomic-replace ``probe_manifest.json`` + ``.sha256`` sidecar."""
    payload = manifest.model_dump_json(indent=2, exclude_none=False).encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()
    _atomic_write_bytes(path, payload)
    sidecar = path.with_suffix(path.suffix + ".sha256")
    sidecar.write_text(digest + "\n", encoding="utf-8")
    return digest


__all__ = [
    "append_predictions_long",
    "finalize_label_predictions",
    "write_coef_atomic",
    "write_metrics_atomic",
    "write_probe_manifest_atomic",
    "write_summary",
]
