"""Fold-aware reader over the activation cache (C06).

Opens a fold's per-layer parquet shards as a pyarrow Dataset; supports keyed
lookup, bulk ``load_all(layer)``, and resume-scan via ``completed_visit_keys``.
Refuses on any of: missing manifest, fold-vs-directory mismatch, cache-key
composition drift, or unfinalized completion status (unless ``allow_partial``).
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from nl_ae.schema.paths import RunPaths, run_paths

from .errors import (
    ActivationManifestMissingError,
    CacheKeyMismatchError,
)
from .models import (
    CACHE_KEY_COMPOSITION,
    ActivationManifest,
    compute_cache_key_composition_digest,
)

if TYPE_CHECKING:  # pragma: no cover
    import pyarrow as pa
    import pyarrow.dataset as pa_dataset


LOG = logging.getLogger(__name__)

Fold = Literal["pilot", "holdout"]
VisitKey = tuple[str, int, str]


@dataclass(frozen=True)
class ActivationCacheReader:
    """Lazy reader over ``runs/<run_id>/<fold>/activations/``.

    Construct with ``ActivationCacheReader.open(run_dir, fold)``. The constructor
    is hidden because the ``open`` factory validates the manifest before any
    parquet IO.
    """

    paths: RunPaths
    fold: Fold
    manifest: ActivationManifest

    # --- factories ------------------------------------------------------

    @classmethod
    def open(
        cls,
        run_dir: Path,
        fold: Fold,
        *,
        allow_partial: bool = False,
        expected_composition: tuple[str, ...] = CACHE_KEY_COMPOSITION,
    ) -> ActivationCacheReader:
        paths = run_paths(run_dir.parent, run_dir.name)
        cache_dir = paths.fold_activations_dir(fold)
        manifest_path = cache_dir / "activation_manifest.json"
        if not manifest_path.exists():
            raise ActivationManifestMissingError(
                f"activation_manifest.json not found: {manifest_path}"
            )
        manifest = ActivationManifest.model_validate_json(manifest_path.read_bytes())
        if manifest.fold != fold:
            raise CacheKeyMismatchError(
                f"manifest.fold={manifest.fold!r} but reader requested fold={fold!r}"
            )
        expected_digest = compute_cache_key_composition_digest(expected_composition)
        if expected_digest != manifest.cache_key_composition_digest:
            raise CacheKeyMismatchError(
                f"cache_key_composition_digest mismatch: "
                f"on-disk={manifest.cache_key_composition_digest} "
                f"expected={expected_digest}"
            )
        if not allow_partial and manifest.completion_status != "completed":
            raise CacheKeyMismatchError(
                f"cache completion_status={manifest.completion_status!r} "
                "(pass allow_partial=True to read an in-progress cache)"
            )
        return cls(paths=paths, fold=fold, manifest=manifest)

    # --- paths ----------------------------------------------------------

    @property
    def cache_dir(self) -> Path:
        return self.paths.fold_activations_dir(self.fold)

    def layer_dir(self, layer: int) -> Path:
        return self.cache_dir / f"L{layer:02d}"

    def layer_shard_paths(self, layer: int) -> list[Path]:
        """Final-only shard files for ``layer``, sorted by ``shard_index``."""
        layer_shards = next(
            (ls for ls in self.manifest.layer_shards if ls.layer == layer),
            None,
        )
        if layer_shards is None:
            return []
        return [self.cache_dir / s.relative_path for s in layer_shards.shards]

    # --- reading --------------------------------------------------------

    def iter_rows(
        self,
        layer: int,
        *,
        columns: tuple[str, ...] | None = None,
    ) -> Iterator[dict[str, object]]:
        """Stream one layer's rows as plain dicts (lazy, batched under the hood)."""
        import pyarrow.parquet as pq

        paths = self.layer_shard_paths(layer)
        for shard in paths:
            table = pq.read_table(shard.as_posix(), columns=list(columns) if columns else None)
            yield from table.to_pylist()

    def load_all(self, layer: int) -> pa.Table:
        """Concatenate every shard for ``layer`` into one pyarrow Table."""
        import pyarrow as pa
        import pyarrow.parquet as pq

        paths = self.layer_shard_paths(layer)
        if not paths:
            return pa.table({})
        tables = [pq.read_table(p.as_posix()) for p in paths]
        return pa.concat_tables(tables)

    def open_dataset(self, layer: int) -> pa_dataset.Dataset:
        """Return a pyarrow ``Dataset`` over a layer's shards (lazy, scan-friendly)."""
        import pyarrow.dataset as pa_dataset

        paths = self.layer_shard_paths(layer)
        if not paths:
            raise FileNotFoundError(
                f"no shards on disk for layer {layer} in {self.cache_dir}"
            )
        return pa_dataset.dataset([p.as_posix() for p in paths], format="parquet")

    def completed_visit_keys(self, *, scan_layer: int | None = None) -> frozenset[VisitKey]:
        """Return the ``(item_id, perm_id, template_id)`` set already cached.

        All layers carry identical visit sets (the writer rotates them in
        lockstep), so we scan the first manifest-declared layer by default. Use
        a different ``scan_layer`` only for assertions in tests.
        """
        layer = scan_layer if scan_layer is not None else self.manifest.layers[0]
        seen: set[VisitKey] = set()
        for row in self.iter_rows(
            layer,
            columns=("item_id", "permutation_id", "template_id"),
        ):
            key: VisitKey = (
                str(row["item_id"]),
                int(row["permutation_id"]),  # type: ignore[arg-type]
                str(row["template_id"]),
            )
            seen.add(key)
        return frozenset(seen)


__all__ = ["ActivationCacheReader", "VisitKey"]
