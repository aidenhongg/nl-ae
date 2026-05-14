"""Pydantic v2 models + digests for ``probe_manifest.json`` (C07).

Schema 1.0.0. The manifest is content-only-digested: two probe runs with
identical inputs produce bit-identical ``probe_manifest_digest`` regardless of
when they ran. The digest excludes the growing per-cell record list and every
timestamp/status field — those mutate during the run.
"""

from __future__ import annotations

import hashlib
import json
from typing import Annotated, Final, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

from nl_ae.pilot.models import LayerIndex, ProbeLabel
from nl_ae.schema.models import IsoUtcStr, Sha256Hex

PROBE_MANIFEST_SCHEMA_VERSION: Literal["1.0.0"] = "1.0.0"

# The subset of sklearn kwargs that enters the kwargs digest. Documented as
# ordered for the spec; canonical-JSON sorts keys, so functionally order is
# irrelevant — but pin the field set here so an accidentally added kwarg can't
# silently change the digest.
SKLEARN_KWARGS_FIELDS: Final[tuple[str, ...]] = (
    "penalty",
    "C",
    "solver",
    "max_iter",
    "fit_intercept",
    "class_weight",
    "standardize",
)

CellStatus = Literal["completed", "failed", "skipped_resume", "in_progress"]
CompletionStatus = Literal["in_progress", "completed", "failed", "aborted"]
ClassWeight = Literal["balanced", "none"]


class SklearnKwargs(BaseModel):
    """The subset of sklearn ``LogisticRegression`` knobs we honor in v1."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    penalty: Literal["l2"]
    C: Annotated[float, Field(gt=0.0)]
    solver: Literal["lbfgs"]
    max_iter: Annotated[int, Field(ge=1)]
    fit_intercept: bool = True
    class_weight: ClassWeight = "none"
    standardize: bool = False


class ProbeCellKey(BaseModel):
    """Identifies one ``(label, layer)`` cell."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    label: ProbeLabel
    layer: LayerIndex


class ProbeCellRecord(BaseModel):
    """Per-``(label, layer)`` outcome record on the manifest.

    ``metrics_sha256`` is always set (a minimal metrics.json is written even
    for failed cells so debugging context survives a crash). ``coef_sha256`` is
    set only on ``status='completed'`` (no fit, no coef artifact).
    """

    model_config = ConfigDict(extra="forbid", frozen=True)
    label: ProbeLabel
    layer: LayerIndex
    status: CellStatus
    n_train: Annotated[int, Field(ge=0)]
    n_val: Annotated[int, Field(ge=0)]
    n_test: Annotated[int, Field(ge=0)]
    n_dropped_label_null: Annotated[int, Field(ge=0)]
    classes: tuple[str, ...]
    fit_wall_seconds: Annotated[float, Field(ge=0.0)]
    metrics_sha256: Sha256Hex
    coef_sha256: Sha256Hex | None = None
    failure_reason: str | None = None
    started_at: IsoUtcStr
    ended_at: IsoUtcStr | None = None

    @model_validator(mode="after")
    def _check(self) -> Self:
        if self.status == "completed" and self.coef_sha256 is None:
            raise ValueError("completed cell requires coef_sha256")
        if self.status == "completed" and self.ended_at is None:
            raise ValueError("completed cell requires ended_at")
        return self


class ProbeManifest(BaseModel):
    """Per-fold probe-run manifest. Content-only digest in ``probe_manifest_digest``."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["1.0.0"] = PROBE_MANIFEST_SCHEMA_VERSION
    run_id: Annotated[str, StringConstraints(min_length=1, max_length=64)]
    fold: Literal["pilot", "holdout"]
    labels: tuple[ProbeLabel, ...] = Field(min_length=1)
    layers: tuple[LayerIndex, ...] = Field(min_length=1)
    split_seed: Annotated[int, Field(ge=0, le=(1 << 32) - 1)]
    split_frac: tuple[float, float, float]
    sklearn_kwargs: SklearnKwargs
    sklearn_kwargs_digest: Sha256Hex
    source_run_id: Annotated[str, StringConstraints(min_length=1, max_length=64)]
    source_cache_key_digest: Sha256Hex
    source_pilot_manifest_digest: Sha256Hex
    source_preregistration_digest: Sha256Hex | None = None
    cells: tuple[ProbeCellRecord, ...] = ()
    completion_status: CompletionStatus
    failure_reason: str | None = None
    started_at: IsoUtcStr
    ended_at: IsoUtcStr | None = None
    probe_manifest_digest: Sha256Hex

    @model_validator(mode="after")
    def _check(self) -> Self:
        # split_frac
        a, b, c = self.split_frac
        if a <= 0.0 or b <= 0.0 or c <= 0.0:
            raise ValueError(f"split_frac entries must be > 0; got {self.split_frac}")
        if abs(a + b + c - 1.0) > 1e-6:
            raise ValueError(f"split_frac must sum to 1.0; got {a + b + c}")
        # preregistration digest binding to fold
        if self.fold == "holdout" and self.source_preregistration_digest is None:
            raise ValueError(
                "source_preregistration_digest is required when fold='holdout'"
            )
        if self.fold == "pilot" and self.source_preregistration_digest is not None:
            raise ValueError(
                "source_preregistration_digest must be null when fold='pilot'"
            )
        # labels / layers canonical form
        if tuple(sorted(set(self.labels))) != self.labels:
            raise ValueError("labels must be unique + sorted ascending")
        if tuple(sorted(set(self.layers))) != self.layers:
            raise ValueError("layers must be unique + sorted ascending")
        # completion invariants
        if self.completion_status == "completed" and self.ended_at is None:
            raise ValueError("ended_at required when completion_status='completed'")
        # sklearn kwargs digest binding
        expected_kwargs_digest = compute_sklearn_kwargs_digest(self.sklearn_kwargs)
        if expected_kwargs_digest != self.sklearn_kwargs_digest:
            raise ValueError(
                "sklearn_kwargs_digest does not match sklearn_kwargs: "
                f"expected={expected_kwargs_digest} got={self.sklearn_kwargs_digest}"
            )
        # content-only manifest digest
        expected_digest = compute_probe_manifest_digest(
            run_id=self.run_id,
            fold=self.fold,
            labels=self.labels,
            layers=self.layers,
            split_seed=self.split_seed,
            split_frac=self.split_frac,
            sklearn_kwargs=self.sklearn_kwargs,
            sklearn_kwargs_digest=self.sklearn_kwargs_digest,
            source_run_id=self.source_run_id,
            source_cache_key_digest=self.source_cache_key_digest,
            source_pilot_manifest_digest=self.source_pilot_manifest_digest,
            source_preregistration_digest=self.source_preregistration_digest,
        )
        if expected_digest != self.probe_manifest_digest:
            raise ValueError(
                "probe_manifest_digest does not match content fields: "
                f"expected={expected_digest} got={self.probe_manifest_digest}"
            )
        return self


# --- digests ----------------------------------------------------------------


def compute_sklearn_kwargs_digest(kw: SklearnKwargs) -> Sha256Hex:
    """SHA-256 over canonical JSON of the pinned ``SKLEARN_KWARGS_FIELDS`` subset."""
    payload = {name: getattr(kw, name) for name in SKLEARN_KWARGS_FIELDS}
    raw = json.dumps(
        payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def compute_probe_manifest_digest(
    *,
    run_id: str,
    fold: Literal["pilot", "holdout"],
    labels: tuple[ProbeLabel, ...],
    layers: tuple[int, ...],
    split_seed: int,
    split_frac: tuple[float, float, float],
    sklearn_kwargs: SklearnKwargs,
    sklearn_kwargs_digest: Sha256Hex,
    source_run_id: str,
    source_cache_key_digest: Sha256Hex,
    source_pilot_manifest_digest: Sha256Hex,
    source_preregistration_digest: Sha256Hex | None,
    schema_version: str = PROBE_MANIFEST_SCHEMA_VERSION,
) -> Sha256Hex:
    """Content-only digest. Cells, status, timestamps, and failure_reason excluded."""
    payload = {
        "schema_version": schema_version,
        "run_id": run_id,
        "fold": fold,
        "labels": list(labels),
        "layers": [int(layer) for layer in layers],
        "split_seed": int(split_seed),
        "split_frac": [float(x) for x in split_frac],
        "sklearn_kwargs": {name: getattr(sklearn_kwargs, name) for name in SKLEARN_KWARGS_FIELDS},
        "sklearn_kwargs_digest": sklearn_kwargs_digest,
        "source_run_id": source_run_id,
        "source_cache_key_digest": source_cache_key_digest,
        "source_pilot_manifest_digest": source_pilot_manifest_digest,
        "source_preregistration_digest": source_preregistration_digest,
    }
    raw = json.dumps(
        payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


__all__ = [
    "PROBE_MANIFEST_SCHEMA_VERSION",
    "SKLEARN_KWARGS_FIELDS",
    "CellStatus",
    "ClassWeight",
    "CompletionStatus",
    "ProbeCellKey",
    "ProbeCellRecord",
    "ProbeManifest",
    "SklearnKwargs",
    "compute_probe_manifest_digest",
    "compute_sklearn_kwargs_digest",
]
