"""Read-only view over an existing probe-artifact tree (C07).

Used by :mod:`nl_ae.probes.figures` and any downstream report tooling. The
constructor validates the on-disk ``probe_manifest.json`` matches the fold the
caller asked for; per-cell artifact loads are lazy.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from nl_ae.pilot.models import ProbeLabel
from nl_ae.schema.paths import RunPaths, run_paths

from .errors import ProbeManifestMissingError, ProbeManifestStaleError
from .models import ProbeManifest

if TYPE_CHECKING:  # pragma: no cover
    import numpy as np
    import pandas as pd

Fold = Literal["pilot", "holdout"]


@dataclass(frozen=True)
class ProbeArtifactReader:
    paths: RunPaths
    fold: Fold
    manifest: ProbeManifest

    @classmethod
    def open(cls, run_dir: Path, fold: Fold) -> ProbeArtifactReader:
        paths = run_paths(run_dir.parent, run_dir.name)
        manifest_path = paths.fold_probes_dir(fold) / "probe_manifest.json"
        if not manifest_path.exists():
            raise ProbeManifestMissingError(
                f"probe_manifest.json not found: {manifest_path}"
            )
        manifest = ProbeManifest.model_validate_json(manifest_path.read_bytes())
        if manifest.fold != fold:
            raise ProbeManifestStaleError(
                f"manifest.fold={manifest.fold!r} but reader requested fold={fold!r}"
            )
        return cls(paths=paths, fold=fold, manifest=manifest)

    # --- internal paths ---------------------------------------------------

    def label_dir(self, label: ProbeLabel) -> Path:
        return self.paths.fold_probes_dir(self.fold) / label

    def cell_dir(self, label: ProbeLabel, layer: int) -> Path:
        return self.label_dir(label) / f"L{layer:02d}"

    # --- loaders ----------------------------------------------------------

    def load_coef(
        self, label: ProbeLabel, layer: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(coef, intercept)`` from the per-cell ``coef.npy``."""
        import numpy as np

        path = self.cell_dir(label, layer) / "coef.npy"
        with np.load(path.as_posix()) as data:
            return np.asarray(data["coef"]), np.asarray(data["intercept"])

    def load_metrics(self, label: ProbeLabel, layer: int) -> dict[str, Any]:
        path = self.cell_dir(label, layer) / "metrics.json"
        data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
        return data

    def load_summary(self, label: ProbeLabel) -> pd.DataFrame:
        import pyarrow.parquet as pq

        path = self.label_dir(label) / "summary.parquet"
        return pq.read_table(path.as_posix()).to_pandas()

    def load_predictions(self, label: ProbeLabel) -> pd.DataFrame:
        import pyarrow.parquet as pq

        path = self.label_dir(label) / "predictions.parquet"
        return pq.read_table(path.as_posix()).to_pandas()


__all__ = ["Fold", "ProbeArtifactReader"]
