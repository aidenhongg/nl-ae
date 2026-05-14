"""Probe-training orchestration (C07).

Outer loop = layer (amortizes one parquet read per layer), inner loop = label.
Each cell is independent; a single cell's failure is recorded with
``status="failed"`` and the loop continues unless ``fail_fast=True``. The
manifest is persisted atomically after every cell so a crash leaves a
consistent, replayable on-disk state.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from nl_ae.cache.errors import ActivationManifestMissingError
from nl_ae.cache.models import ActivationManifest
from nl_ae.cache.reader import ActivationCacheReader
from nl_ae.pilot.manifest import load_pilot_manifest
from nl_ae.pilot.models import (
    CANDIDATE_LABELS,
    PilotManifest,
    Preregistration,
    ProbeLabel,
)
from nl_ae.pilot.preregistration import require_preregistration
from nl_ae.runtime.seeds import derive_child_seed
from nl_ae.schema.hashing import now_utc_iso
from nl_ae.schema.models import RunManifest, Sha256Hex
from nl_ae.schema.paths import RunPaths, run_paths
from nl_ae.schema.reader import ResultsReader, load_manifest

from .errors import (
    ActivationManifestNotCompletedError,
    FitFailedError,
    InsufficientLabelDataError,
    LabelOutOfScopeError,
    LayerOutOfScopeError,
    ProbeManifestStaleError,
)
from .fitter import ProbeFitter
from .labels import LabelExtraction, extract_label
from .models import (
    ProbeCellKey,
    ProbeCellRecord,
    ProbeManifest,
    SklearnKwargs,
    compute_probe_manifest_digest,
    compute_sklearn_kwargs_digest,
)
from .splits import split_by_item_within_fold
from .writer import (
    append_predictions_long,
    finalize_label_predictions,
    write_coef_atomic,
    write_metrics_atomic,
    write_probe_manifest_atomic,
    write_summary,
)

if TYPE_CHECKING:  # pragma: no cover
    import numpy as np

LOG = logging.getLogger(__name__)

Fold = Literal["pilot", "holdout"]
VisitKey = tuple[str, int, str]


# --- preflight bundle -------------------------------------------------------


@dataclass(frozen=True)
class TrainerInputs:
    """Cross-validated inputs ready for ``ProbeTrainer``."""

    run_dir: Path
    paths: RunPaths
    run_manifest: RunManifest
    pilot_manifest: PilotManifest
    preregistration: Preregistration | None
    activation_manifest: ActivationManifest
    fold: Fold
    in_scope_item_ids: frozenset[str]


@dataclass(frozen=True)
class TrainerOutcome:
    cells_completed: int
    cells_skipped_resume: int
    cells_failed: int
    cells_expected: int
    status: Literal["completed", "failed", "aborted"]
    failure_reason: str | None


def load_trainer_inputs(
    run_dir: Path,
    fold: Fold,
    *,
    labels_override: tuple[ProbeLabel, ...] | None = None,
    layers_override: tuple[int, ...] | None = None,
) -> tuple[TrainerInputs, tuple[ProbeLabel, ...], tuple[int, ...]]:
    """Validate every preflight invariant before any sklearn import.

    Returns ``(inputs, resolved_labels, resolved_layers)``. The resolved label
    and layer tuples are sorted and deduplicated; holdout always returns the
    preregistered sets verbatim (after defense-in-depth scope checks).
    """
    if fold not in ("pilot", "holdout"):
        raise ValueError(f"fold must be 'pilot' or 'holdout'; got {fold!r}")
    paths = run_paths(run_dir.parent, run_dir.name)
    if not paths.manifest_json.exists():
        raise FileNotFoundError(f"manifest.json missing: {paths.manifest_json}")
    run_manifest = load_manifest(paths.manifest_json)
    if run_manifest.completion_status != "completed":
        raise RuntimeError(
            f"Phase 1 manifest completion_status={run_manifest.completion_status!r}; "
            "probe-train requires a completed run"
        )
    if not paths.rows_jsonl.exists():
        raise FileNotFoundError(f"rows.jsonl missing: {paths.rows_jsonl}")
    pilot_manifest = load_pilot_manifest(paths.pilot_manifest_json)

    preregistration: Preregistration | None = None
    if fold == "holdout":
        preregistration = require_preregistration(paths)

    cache_manifest_path = paths.fold_activations_dir(fold) / "activation_manifest.json"
    if not cache_manifest_path.exists():
        raise ActivationManifestMissingError(
            f"activation_manifest.json not found for fold={fold!r}: {cache_manifest_path}"
        )
    cache_reader = ActivationCacheReader.open(run_dir, fold)
    activation_manifest = cache_reader.manifest
    if activation_manifest.completion_status != "completed":
        raise ActivationManifestNotCompletedError(
            f"activation cache completion_status={activation_manifest.completion_status!r}; "
            "probe-train requires a completed cache"
        )

    candidate_pool: frozenset[ProbeLabel] = frozenset(CANDIDATE_LABELS)  # type: ignore[arg-type]
    cache_layers = frozenset(activation_manifest.layers)

    if fold == "pilot":
        if labels_override is not None:
            if not set(labels_override).issubset(candidate_pool):
                raise LabelOutOfScopeError(
                    f"labels {sorted(set(labels_override) - candidate_pool)!r} "
                    f"not in CANDIDATE_LABELS={sorted(candidate_pool)!r}"
                )
            labels = tuple(sorted(set(labels_override)))
        else:
            labels = tuple(sorted(candidate_pool))
        if layers_override is not None:
            if not set(layers_override).issubset(cache_layers):
                raise LayerOutOfScopeError(
                    f"layers {sorted(set(layers_override) - cache_layers)!r} "
                    f"not in cache.layers={sorted(cache_layers)!r}"
                )
            layers = tuple(sorted(set(layers_override)))
        else:
            layers = tuple(activation_manifest.layers)
    else:
        # Holdout: preregistration is authoritative.
        assert preregistration is not None
        prereg_labels = tuple(preregistration.labels)
        prereg_layers = tuple(preregistration.layers)
        if not set(prereg_labels).issubset(candidate_pool):
            raise LabelOutOfScopeError(
                f"preregistration.labels {sorted(set(prereg_labels) - candidate_pool)!r} "
                f"not in CANDIDATE_LABELS"
            )
        if not set(prereg_layers).issubset(cache_layers):
            raise LayerOutOfScopeError(
                f"preregistration.layers {sorted(set(prereg_layers) - cache_layers)!r} "
                f"not present in activation cache (layers={sorted(cache_layers)!r})"
            )
        if labels_override is not None and tuple(sorted(set(labels_override))) != tuple(
            sorted(set(prereg_labels))
        ):
            raise LabelOutOfScopeError(
                "--labels disagrees with preregistration.labels; holdout is locked by "
                "the preregistration"
            )
        if layers_override is not None and tuple(sorted(set(layers_override))) != tuple(
            sorted(set(prereg_layers))
        ):
            raise LayerOutOfScopeError(
                "--layers disagrees with preregistration.layers; holdout is locked by "
                "the preregistration"
            )
        labels = tuple(sorted(set(prereg_labels)))
        layers = tuple(sorted(set(prereg_layers)))

    in_fold_visits = cache_reader.completed_visit_keys()
    in_fold_items = frozenset({iid for (iid, _, _) in in_fold_visits})
    if fold == "pilot":
        pilot_ids = frozenset(pilot_manifest.pilot_item_ids)
        if in_fold_items != pilot_ids:
            LOG.warning(
                "pilot cache covers %d items but pilot_manifest declares %d; "
                "training over the intersection",
                len(in_fold_items),
                len(pilot_ids),
            )

    inputs = TrainerInputs(
        run_dir=paths.run_dir,
        paths=paths,
        run_manifest=run_manifest,
        pilot_manifest=pilot_manifest,
        preregistration=preregistration,
        activation_manifest=activation_manifest,
        fold=fold,
        in_scope_item_ids=in_fold_items,
    )
    return inputs, labels, layers


def hash_preregistration(prereg: Preregistration) -> Sha256Hex:
    """Canonical-JSON SHA-256 of a ``Preregistration`` model (defense-in-depth
    against the YAML round-trip changing whitespace). Mirrors the helper used
    in :mod:`nl_ae.cache.extractor`."""
    import hashlib

    payload = json.dumps(
        prereg.model_dump(mode="json"),
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def scan_resume_state(
    *,
    paths: RunPaths,
    fold: Fold,
    expected_manifest_digest: Sha256Hex,
) -> tuple[ProbeManifest | None, frozenset[ProbeCellKey]]:
    """Inspect any existing manifest + per-cell artifacts for resume.

    Returns ``(existing_manifest, completed_cell_keys)``. A cell is considered
    completed only when (a) the manifest record's ``status == "completed"``,
    (b) ``coef.npy`` exists and its SHA-256 matches the record, and (c)
    ``metrics.json`` exists and its SHA-256 matches the record. Raises
    :class:`ProbeManifestStaleError` if the on-disk manifest's
    ``probe_manifest_digest`` disagrees with ``expected_manifest_digest``.
    """
    manifest_path = paths.fold_probes_dir(fold) / "probe_manifest.json"
    if not manifest_path.exists():
        return None, frozenset()
    manifest = ProbeManifest.model_validate_json(manifest_path.read_bytes())
    if manifest.probe_manifest_digest != expected_manifest_digest:
        raise ProbeManifestStaleError(
            "probe manifest digest drift: on-disk="
            f"{manifest.probe_manifest_digest} expected={expected_manifest_digest}"
        )
    if manifest.fold != fold:
        raise ProbeManifestStaleError(
            f"on-disk manifest.fold={manifest.fold!r} but resume requested fold={fold!r}"
        )

    completed: set[ProbeCellKey] = set()
    fold_probes = paths.fold_probes_dir(fold)
    for cell in manifest.cells:
        if cell.status != "completed":
            continue
        cell_dir = fold_probes / cell.label / f"L{cell.layer:02d}"
        coef_path = cell_dir / "coef.npy"
        metrics_path = cell_dir / "metrics.json"
        if not coef_path.exists() or not metrics_path.exists():
            continue
        if cell.coef_sha256 is None:
            continue
        actual_coef = _hash_file(coef_path)
        actual_metrics = _hash_file(metrics_path)
        if cell.coef_sha256 != actual_coef or cell.metrics_sha256 != actual_metrics:
            continue
        completed.add(ProbeCellKey(label=cell.label, layer=cell.layer))
    return manifest, frozenset(completed)


def _hash_file(path: Path, *, chunk: int = 1 << 20) -> Sha256Hex:
    import hashlib

    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            block = f.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


# --- trainer ---------------------------------------------------------------


class ProbeTrainer:
    """Fit one ``(label, layer)`` cell at a time over a fold's activation cache."""

    def __init__(
        self,
        *,
        inputs: TrainerInputs,
        fitter: ProbeFitter,
        labels: tuple[ProbeLabel, ...],
        layers: tuple[int, ...],
        sklearn_kwargs: SklearnKwargs,
        split_seed: int,
        split_frac: tuple[float, float, float],
        completed: frozenset[ProbeCellKey] = frozenset(),
        existing_manifest: ProbeManifest | None = None,
        fail_fast: bool = False,
    ) -> None:
        if not labels:
            raise ValueError("labels must be non-empty")
        if not layers:
            raise ValueError("layers must be non-empty")
        self._inputs = inputs
        self._fitter = fitter
        self._labels = tuple(sorted(set(labels)))
        self._layers = tuple(sorted(set(layers)))
        self._sklearn_kwargs = sklearn_kwargs
        self._split_seed = int(split_seed)
        self._split_frac = (
            float(split_frac[0]),
            float(split_frac[1]),
            float(split_frac[2]),
        )
        self._completed = completed
        self._fail_fast = fail_fast
        self._cell_records: dict[tuple[str, int], ProbeCellRecord] = {}
        if existing_manifest is not None:
            for cell in existing_manifest.cells:
                self._cell_records[(cell.label, cell.layer)] = cell

        # Pre-compute fixed manifest fields.
        kwargs_digest = compute_sklearn_kwargs_digest(sklearn_kwargs)
        prereg_digest = (
            hash_preregistration(inputs.preregistration)
            if inputs.preregistration is not None
            else None
        )
        manifest_digest = compute_probe_manifest_digest(
            run_id=inputs.run_manifest.run_id,
            fold=inputs.fold,
            labels=self._labels,
            layers=self._layers,
            split_seed=self._split_seed,
            split_frac=self._split_frac,
            sklearn_kwargs=sklearn_kwargs,
            sklearn_kwargs_digest=kwargs_digest,
            source_run_id=inputs.run_manifest.run_id,
            source_cache_key_digest=inputs.activation_manifest.cache_key_composition_digest,
            source_pilot_manifest_digest=inputs.pilot_manifest.pilot_manifest_digest,
            source_preregistration_digest=prereg_digest,
        )
        self._manifest = ProbeManifest(
            run_id=inputs.run_manifest.run_id,
            fold=inputs.fold,
            labels=self._labels,
            layers=self._layers,
            split_seed=self._split_seed,
            split_frac=self._split_frac,
            sklearn_kwargs=sklearn_kwargs,
            sklearn_kwargs_digest=kwargs_digest,
            source_run_id=inputs.run_manifest.run_id,
            source_cache_key_digest=inputs.activation_manifest.cache_key_composition_digest,
            source_pilot_manifest_digest=inputs.pilot_manifest.pilot_manifest_digest,
            source_preregistration_digest=prereg_digest,
            cells=self._cells_tuple(),
            completion_status="in_progress",
            failure_reason=None,
            started_at=now_utc_iso(),
            ended_at=None,
            probe_manifest_digest=manifest_digest,
        )

    # --- public API ----------------------------------------------------

    @property
    def manifest(self) -> ProbeManifest:
        return self._manifest

    def run(self) -> TrainerOutcome:
        import numpy as np

        paths = self._inputs.paths
        fold = self._inputs.fold
        probes_dir = paths.fold_probes_dir(fold)
        manifest_path = probes_dir / "probe_manifest.json"
        cache_reader = ActivationCacheReader.open(self._inputs.run_dir, fold)

        # Initial manifest stamp.
        probes_dir.mkdir(parents=True, exist_ok=True)
        self._persist_manifest(manifest_path, status="in_progress")

        # Build row index once (one Phase 1 row per visit in the cache).
        cache_visits = cache_reader.completed_visit_keys()
        rows_by_key: dict[VisitKey, Any] = {}
        for source_row in ResultsReader(self._inputs.run_dir).iter_rows():
            key: VisitKey = (
                source_row.item_id,
                source_row.permutation_id,
                source_row.template_id,
            )
            if key not in cache_visits:
                continue
            rows_by_key[key] = source_row

        items_in_cache = frozenset({key[0] for key in rows_by_key})
        sub_fold_map = split_by_item_within_fold(
            items_in_cache,
            probe_seed=self._split_seed,
            frac=self._split_frac,
        )

        cells_completed = 0
        cells_skipped = 0
        cells_failed = 0
        cells_expected = len(self._labels) * len(self._layers)
        overall_status: Literal["completed", "failed", "aborted"] = "completed"
        failure_reason: str | None = None
        abort = False

        try:
            for layer in self._layers:
                if abort:
                    break
                arrow_table = cache_reader.load_all(layer)
                ordered_keys, X = self._extract_features(arrow_table)
                # Drop keys not present in rows_by_key (defensive).
                aligned_keys: list[VisitKey] = []
                aligned_rows: list[Any] = []
                aligned_idx: list[int] = []
                for idx, key in enumerate(ordered_keys):
                    row = rows_by_key.get(key)
                    if row is None:
                        continue
                    aligned_keys.append(key)
                    aligned_rows.append(row)
                    aligned_idx.append(idx)
                X_aligned = X[np.asarray(aligned_idx, dtype=np.int64)] if aligned_idx else X[:0]

                for label in self._labels:
                    if abort:
                        break
                    cell_key = ProbeCellKey(label=label, layer=layer)
                    if cell_key in self._completed:
                        cells_skipped += 1
                        continue
                    started_at = now_utc_iso()
                    t_cell = time.perf_counter()
                    cell_dir = probes_dir / label / f"L{layer:02d}"
                    record: ProbeCellRecord | None = None
                    try:
                        extraction = extract_label(aligned_rows, label)
                        valid_idx = np.where(extraction.valid_mask)[0]
                        X_valid = X_aligned[extraction.valid_mask]
                        y_valid = extraction.y
                        valid_keys = [aligned_keys[i] for i in valid_idx]
                        sub_assign = np.asarray(
                            [sub_fold_map[k[0]] for k in valid_keys], dtype=object
                        )
                        train_mask = sub_assign == "train"
                        val_mask = sub_assign == "val"
                        test_mask = sub_assign == "test"

                        random_state = derive_child_seed(
                            self._split_seed,
                            f"{label}|L{layer:02d}".encode(),
                        )
                        fit_result = self._fitter.fit_predict(
                            X_train=X_valid[train_mask],
                            y_train=y_valid[train_mask],
                            X_val=X_valid[val_mask],
                            y_val=y_valid[val_mask],
                            X_test=X_valid[test_mask],
                            y_test=y_valid[test_mask],
                            is_binary=extraction.is_binary,
                            kwargs=self._sklearn_kwargs,
                            random_state=random_state,
                        )
                        fit_wall = time.perf_counter() - t_cell
                        metrics = _compute_metrics(
                            fit_result=fit_result,
                            extraction=extraction,
                            y_valid=y_valid,
                            train_mask=train_mask,
                            val_mask=val_mask,
                            test_mask=test_mask,
                            fit_wall_seconds=fit_wall,
                        )

                        coef_sha = write_coef_atomic(
                            cell_dir / "coef.npy", fit_result.coef, fit_result.intercept
                        )
                        metrics_sha = write_metrics_atomic(
                            cell_dir / "metrics.json", metrics
                        )
                        # Reorder valid_keys / sub_assign to match the natural
                        # order they were fed to the fitter? We feed in
                        # valid-row order; cursor-driven write recovers per-sub
                        # arrays. Pass valid-row order here.
                        append_predictions_long(
                            probes_dir / label,
                            run_id=self._inputs.run_manifest.run_id,
                            fold=fold,
                            layer=layer,
                            item_ids=[k[0] for k in valid_keys],
                            permutation_ids=[k[1] for k in valid_keys],
                            template_ids=[k[2] for k in valid_keys],
                            sub_folds=[str(sub_assign[i]) for i in range(len(valid_keys))],
                            gold_values=[str(y_valid[i]) for i in range(len(valid_keys))],
                            fit_result=fit_result,
                            extraction=extraction,
                        )

                        record = ProbeCellRecord(
                            label=label,
                            layer=layer,
                            status="completed",
                            n_train=int(train_mask.sum()),
                            n_val=int(val_mask.sum()),
                            n_test=int(test_mask.sum()),
                            n_dropped_label_null=extraction.n_dropped,
                            classes=fit_result.classes,
                            fit_wall_seconds=fit_wall,
                            metrics_sha256=metrics_sha,
                            coef_sha256=coef_sha,
                            failure_reason=None,
                            started_at=started_at,
                            ended_at=now_utc_iso(),
                        )
                        cells_completed += 1
                    except (FitFailedError, InsufficientLabelDataError) as exc:
                        # extraction / train_mask / etc. are always assigned before
                        # fit_predict; the InsufficientLabelDataError path inside the
                        # fitter runs after we've called extract_label + split here.
                        fit_wall = time.perf_counter() - t_cell
                        cell_dir.mkdir(parents=True, exist_ok=True)
                        failure_metrics = {
                            "status": "failed",
                            "failure_reason": repr(exc),
                            "fit_wall_seconds": fit_wall,
                        }
                        metrics_sha = write_metrics_atomic(
                            cell_dir / "metrics.json", failure_metrics
                        )
                        record = ProbeCellRecord(
                            label=label,
                            layer=layer,
                            status="failed",
                            n_train=int(train_mask.sum()),
                            n_val=int(val_mask.sum()),
                            n_test=int(test_mask.sum()),
                            n_dropped_label_null=extraction.n_dropped,
                            classes=extraction.classes,
                            fit_wall_seconds=fit_wall,
                            metrics_sha256=metrics_sha,
                            coef_sha256=None,
                            failure_reason=repr(exc),
                            started_at=started_at,
                            ended_at=now_utc_iso(),
                        )
                        cells_failed += 1
                        if self._fail_fast:
                            overall_status = "failed"
                            failure_reason = repr(exc)
                            abort = True
                    self._cell_records[(label, layer)] = record
                    self._persist_manifest(manifest_path, status="in_progress")

            # Per-label finalization runs in all cases (even after abort) so
            # whatever per-layer predictions did land get concatenated and a
            # partial summary is materialized for debugging.
            for label in self._labels:
                label_dir = probes_dir / label
                if not label_dir.exists():
                    continue
                finalize_label_predictions(label_dir, layers=self._layers)
                summary_rows = self._build_summary_rows(label, probes_dir)
                write_summary(label_dir / "summary.parquet", cell_rows=summary_rows)
        except KeyboardInterrupt:
            overall_status = "aborted"
            failure_reason = "KeyboardInterrupt"
            self._persist_manifest(
                manifest_path,
                status="aborted",
                failure_reason=failure_reason,
                ended_at=now_utc_iso(),
            )
            raise

        self._persist_manifest(
            manifest_path,
            status=overall_status,
            failure_reason=failure_reason,
            ended_at=now_utc_iso(),
        )
        return TrainerOutcome(
            cells_completed=cells_completed,
            cells_skipped_resume=cells_skipped,
            cells_failed=cells_failed,
            cells_expected=cells_expected,
            status=overall_status,
            failure_reason=failure_reason,
        )

    # --- helpers -------------------------------------------------------

    def _cells_tuple(self) -> tuple[ProbeCellRecord, ...]:
        return tuple(
            sorted(
                self._cell_records.values(),
                key=lambda r: (r.label, r.layer),
            )
        )

    def _persist_manifest(
        self,
        manifest_path: Path,
        *,
        status: Literal["in_progress", "completed", "failed", "aborted"],
        failure_reason: str | None = None,
        ended_at: str | None = None,
    ) -> None:
        updates: dict[str, Any] = {
            "cells": self._cells_tuple(),
            "completion_status": status,
            "failure_reason": failure_reason,
            "ended_at": ended_at,
        }
        merged = {**self._manifest.model_dump(), **updates}
        self._manifest = ProbeManifest.model_validate(merged)
        write_probe_manifest_atomic(manifest_path, self._manifest)

    def _extract_features(
        self, arrow_table: Any
    ) -> tuple[list[VisitKey], np.ndarray]:
        import numpy as np

        if arrow_table.num_rows == 0:
            return [], np.zeros((0, self._inputs.activation_manifest.activation_dim), dtype=np.float32)
        item_ids = arrow_table.column("item_id").to_pylist()
        perm_ids = arrow_table.column("permutation_id").to_pylist()
        template_ids = arrow_table.column("template_id").to_pylist()
        acts_col = arrow_table.column("activation").combine_chunks()
        flat = acts_col.values.to_numpy(zero_copy_only=False)
        n_rows = arrow_table.num_rows
        dim = self._inputs.activation_manifest.activation_dim
        if flat.size != n_rows * dim:
            raise RuntimeError(
                f"activation flat array size {flat.size} != n_rows * dim ({n_rows} * {dim})"
            )
        X = flat.reshape(n_rows, dim).astype(np.float32, copy=True)
        keys: list[VisitKey] = [
            (str(item_ids[i]), int(perm_ids[i]), str(template_ids[i])) for i in range(n_rows)
        ]
        return keys, X

    def _build_summary_rows(
        self, label: ProbeLabel, probes_dir: Path
    ) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for layer in self._layers:
            cell = self._cell_records.get((label, layer))
            if cell is None:
                rows.append(_empty_summary_row(label, layer, status="missing"))
                continue
            base = {
                "layer": layer,
                "n_train": cell.n_train,
                "n_val": cell.n_val,
                "n_test": cell.n_test,
                "n_dropped_label_null": cell.n_dropped_label_null,
                "classes_count": len(cell.classes),
                "status": cell.status,
                "fit_wall_seconds": cell.fit_wall_seconds,
            }
            if cell.status != "completed":
                rows.append({**base, **_empty_metrics()})
                continue
            metrics_path = probes_dir / label / f"L{layer:02d}" / "metrics.json"
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            base.update(_summary_metric_projection(metrics))
            rows.append(base)
        return rows


def _empty_metrics() -> dict[str, Any]:
    return {
        "train_accuracy": None,
        "val_accuracy": None,
        "test_accuracy": None,
        "balanced_accuracy_test": None,
        "roc_auc_test": None,
        "brier_score_test": None,
        "ece_test": None,
        "macro_f1_test": None,
        "converged": None,
        "n_iter": None,
    }


def _empty_summary_row(label: ProbeLabel, layer: int, *, status: str) -> dict[str, Any]:
    return {
        "layer": layer,
        "n_train": 0,
        "n_val": 0,
        "n_test": 0,
        "n_dropped_label_null": 0,
        "classes_count": 0,
        "status": status,
        "fit_wall_seconds": 0.0,
        **_empty_metrics(),
    }


def _summary_metric_projection(metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        "train_accuracy": metrics.get("accuracy_train"),
        "val_accuracy": metrics.get("accuracy_val"),
        "test_accuracy": metrics.get("accuracy_test"),
        "balanced_accuracy_test": metrics.get("balanced_accuracy_test"),
        "roc_auc_test": metrics.get("roc_auc_test"),
        "brier_score_test": metrics.get("brier_score_test"),
        "ece_test": metrics.get("ece_test"),
        "macro_f1_test": metrics.get("macro_f1_test"),
        "converged": metrics.get("converged"),
        "n_iter": metrics.get("n_iter"),
    }


# --- metric computation ---------------------------------------------------


def _compute_metrics(
    *,
    fit_result: Any,
    extraction: LabelExtraction,
    y_valid: np.ndarray,
    train_mask: np.ndarray,
    val_mask: np.ndarray,
    test_mask: np.ndarray,
    fit_wall_seconds: float,
) -> dict[str, Any]:
    import numpy as np
    from sklearn.metrics import (
        accuracy_score,
        balanced_accuracy_score,
        brier_score_loss,
        f1_score,
        roc_auc_score,
    )

    y_train = y_valid[train_mask]
    y_val = y_valid[val_mask]
    y_test = y_valid[test_mask]

    out: dict[str, Any] = {
        "accuracy_train": float(accuracy_score(y_train, fit_result.train_pred))
        if y_train.shape[0]
        else None,
        "accuracy_val": float(accuracy_score(y_val, fit_result.val_pred))
        if y_val.shape[0]
        else None,
        "accuracy_test": float(accuracy_score(y_test, fit_result.test_pred))
        if y_test.shape[0]
        else None,
        "balanced_accuracy_test": (
            float(balanced_accuracy_score(y_test, fit_result.test_pred))
            if y_test.shape[0]
            else None
        ),
        "classes": list(fit_result.classes),
        "intercept": fit_result.intercept.tolist(),
        "fit_wall_seconds": fit_wall_seconds,
        "converged": bool(fit_result.converged),
        "n_iter": int(fit_result.n_iter),
        "standardize_mean": (
            fit_result.standardize_mean.tolist()
            if fit_result.standardize_mean is not None
            else None
        ),
        "standardize_std": (
            fit_result.standardize_std.tolist()
            if fit_result.standardize_std is not None
            else None
        ),
        "n_classes_observed_test": len(np.unique(y_test)) if y_test.shape[0] else 0,
    }

    if extraction.is_binary:
        y_test_int = (y_test == "1").astype(np.int64) if y_test.shape[0] else np.empty((0,), dtype=np.int64)
        probs_test = fit_result.test_proba
        if y_test.shape[0] and np.unique(y_test).size >= 2:
            try:
                out["roc_auc_test"] = float(roc_auc_score(y_test_int, probs_test))
            except ValueError:
                out["roc_auc_test"] = None
        else:
            out["roc_auc_test"] = None
        out["brier_score_test"] = (
            float(brier_score_loss(y_test_int, probs_test))
            if y_test.shape[0]
            else None
        )
        out["ece_test"] = _ece(y_test_int, probs_test) if y_test.shape[0] else None
    else:
        if y_test.shape[0]:
            out["macro_f1_test"] = float(
                f1_score(
                    y_test, fit_result.test_pred, average="macro", zero_division=0
                )
            )
            per_class = f1_score(
                y_test,
                fit_result.test_pred,
                labels=list(fit_result.classes),
                average=None,
                zero_division=0,
            )
            out["per_class_f1_test"] = {
                cls: float(score)
                for cls, score in zip(fit_result.classes, per_class, strict=True)
            }
            out["out_of_classes_test_count"] = int(
                sum(1 for v in y_test if v not in fit_result.classes)
            )
        else:
            out["macro_f1_test"] = None
            out["per_class_f1_test"] = {}
            out["out_of_classes_test_count"] = 0

    return out


def _ece(
    y_true_int: np.ndarray, probs_pos: np.ndarray, *, n_buckets: int = 10
) -> float:
    import numpy as np

    if probs_pos.size == 0:
        return 0.0
    edges = np.linspace(0.0, 1.0, n_buckets + 1)
    indices = np.digitize(probs_pos, edges[1:-1])
    n = probs_pos.size
    ece = 0.0
    for b in range(n_buckets):
        mask = indices == b
        if not mask.any():
            continue
        emp_acc = float(y_true_int[mask].mean())
        mean_p = float(probs_pos[mask].mean())
        ece += (mask.sum() / n) * abs(emp_acc - mean_p)
    return float(ece)


__all__ = [
    "Fold",
    "ProbeTrainer",
    "TrainerInputs",
    "TrainerOutcome",
    "VisitKey",
    "hash_preregistration",
    "load_trainer_inputs",
    "scan_resume_state",
]
