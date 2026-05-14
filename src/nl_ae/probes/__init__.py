"""C07 — per-layer linear probes.

Public surface re-exported here so callers can ``from nl_ae.probes import ...``
without remembering the sub-module layout.
"""

from __future__ import annotations

from .errors import (
    ActivationManifestNotCompletedError,
    FitFailedError,
    InsufficientLabelDataError,
    LabelOutOfScopeError,
    LayerOutOfScopeError,
    ProbeError,
    ProbeManifestMissingError,
    ProbeManifestStaleError,
)
from .figures import render_probe_figures
from .fitter import FitResult, ProbeFitter, SklearnLogisticFitter
from .labels import LabelExtraction, extract_label
from .models import (
    PROBE_MANIFEST_SCHEMA_VERSION,
    SKLEARN_KWARGS_FIELDS,
    CellStatus,
    CompletionStatus,
    ProbeCellKey,
    ProbeCellRecord,
    ProbeManifest,
    SklearnKwargs,
    compute_probe_manifest_digest,
    compute_sklearn_kwargs_digest,
)
from .reader import ProbeArtifactReader
from .splits import SubFold, split_by_item_within_fold
from .trainer import (
    ProbeTrainer,
    TrainerInputs,
    TrainerOutcome,
    hash_preregistration,
    load_trainer_inputs,
    scan_resume_state,
)
from .writer import (
    append_predictions_long,
    finalize_label_predictions,
    write_coef_atomic,
    write_metrics_atomic,
    write_probe_manifest_atomic,
    write_summary,
)

__all__ = [
    "PROBE_MANIFEST_SCHEMA_VERSION",
    "SKLEARN_KWARGS_FIELDS",
    "ActivationManifestNotCompletedError",
    "CellStatus",
    "CompletionStatus",
    "FitFailedError",
    "FitResult",
    "InsufficientLabelDataError",
    "LabelExtraction",
    "LabelOutOfScopeError",
    "LayerOutOfScopeError",
    "ProbeArtifactReader",
    "ProbeCellKey",
    "ProbeCellRecord",
    "ProbeError",
    "ProbeFitter",
    "ProbeManifest",
    "ProbeManifestMissingError",
    "ProbeManifestStaleError",
    "ProbeTrainer",
    "SklearnKwargs",
    "SklearnLogisticFitter",
    "SubFold",
    "TrainerInputs",
    "TrainerOutcome",
    "append_predictions_long",
    "compute_probe_manifest_digest",
    "compute_sklearn_kwargs_digest",
    "extract_label",
    "finalize_label_predictions",
    "hash_preregistration",
    "load_trainer_inputs",
    "render_probe_figures",
    "scan_resume_state",
    "split_by_item_within_fold",
    "write_coef_atomic",
    "write_metrics_atomic",
    "write_probe_manifest_atomic",
    "write_summary",
]
