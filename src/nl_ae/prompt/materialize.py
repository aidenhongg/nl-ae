"""Rebuild ``runs/<run_id>/prompts/<prompt_hash>.txt`` sidecars without the model.

The pure replay loop here lets us recover from a Phase 1 run where
``save_rendered_prompts`` was disabled or where sidecars were lost: each row's
prompt is re-rendered deterministically from ``rows.jsonl`` + ``manifest.json``
and its SHA-256 is checked against the row's recorded ``prompt_hash`` before
being written.

Heavy imports (``nl_ae.config.loader``, the dataset loaders) live inside
:func:`load_materialize_inputs` so importing this module costs no more than
the rest of the prompt layer.
"""

from __future__ import annotations

import json
import logging
import os
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from nl_ae.data.canonical import CanonicalItem
from nl_ae.data.permute import PermutationMode, permutation_for
from nl_ae.schema.models import RunManifest, Sha256Hex
from nl_ae.schema.paths import RunPaths, run_paths
from nl_ae.schema.reader import load_manifest

from .errors import (
    ItemNotInLoaderError,
    ManifestNotCompletedError,
    PromptHashRecomputeMismatchError,
    SidecarCollisionError,
)
from .renderer import PromptRenderer

if TYPE_CHECKING:  # pragma: no cover
    from nl_ae.data.config import DatasetConfig


OnExisting = Literal["skip", "overwrite", "error"]


@dataclass(frozen=True)
class MaterializeInputs:
    """Validated, fully-resolved inputs the replay loop walks."""

    run_dir: Path
    paths: RunPaths
    run_manifest: RunManifest
    rows_path: Path
    item_index: dict[str, CanonicalItem]
    permutation_mode: PermutationMode
    chat_template_hash: Sha256Hex | None


@dataclass(frozen=True)
class MaterializeOutcome:
    rows_seen: int
    sidecars_written: int
    sidecars_existing: int
    wall_seconds: float
    status: Literal["completed", "failed", "aborted"]
    failure_reason: str | None


def load_materialize_inputs(
    run_dir: Path,
    *,
    hf_cache_dir: Path | None = None,
) -> MaterializeInputs:
    """Load manifest + rows, reparse the embedded config, and pre-index items.

    Lazy-imports ``config.loader`` and the dataset loaders so module-load
    stays free of ``transformers``/``datasets`` (matches the eval_cmd idiom).
    """
    paths = run_paths(run_dir.parent, run_dir.name)
    if not paths.manifest_json.exists():
        raise FileNotFoundError(f"manifest.json missing: {paths.manifest_json}")
    run_manifest = load_manifest(paths.manifest_json)
    if run_manifest.completion_status != "completed":
        raise ManifestNotCompletedError(
            f"Phase 1 manifest completion_status={run_manifest.completion_status!r}; "
            "materialize-prompts requires a completed run"
        )
    if run_manifest.config_yaml_text is None:
        raise ManifestNotCompletedError(
            "Phase 1 manifest is missing config_yaml_text; cannot reparse config"
        )
    if not paths.rows_jsonl.exists():
        raise FileNotFoundError(f"rows.jsonl missing: {paths.rows_jsonl}")

    from nl_ae.config.loader import load_config_from_text

    overrides_raw = run_manifest.cli_args.get("overrides")
    overrides: tuple[str, ...] = ()
    if isinstance(overrides_raw, str) and overrides_raw:
        overrides = tuple(overrides_raw.split(";"))
    cfg = load_config_from_text(run_manifest.config_yaml_text, overrides=overrides)

    wanted = _collect_wanted_item_ids(paths.rows_jsonl)
    item_index: dict[str, CanonicalItem] = {}
    for ds_name in cfg.dataset.enabled:
        ds_wanted = wanted.get(ds_name)
        if not ds_wanted:
            continue
        loader = _build_loader(
            ds_name, cfg.dataset, hf_cache_dir_override=hf_cache_dir
        )
        for item in loader.iter_items():
            if item.item_id in ds_wanted:
                item_index[item.item_id] = item

    missing = {iid for s in wanted.values() for iid in s} - item_index.keys()
    if missing:
        sample = sorted(missing)[:5]
        raise ItemNotInLoaderError(
            f"{len(missing)} item_id(s) referenced by rows.jsonl are not in any "
            f"configured dataset loader (sample: {sample})"
        )

    return MaterializeInputs(
        run_dir=paths.run_dir,
        paths=paths,
        run_manifest=run_manifest,
        rows_path=paths.rows_jsonl,
        item_index=item_index,
        permutation_mode=cfg.eval.plan.permutation_mode,
        chat_template_hash=run_manifest.chat_template_hash,
    )


def materialize_prompts(
    inputs: MaterializeInputs,
    *,
    renderer: PromptRenderer,
    on_existing: OnExisting = "skip",
    limit: int | None = None,
    logger: logging.Logger | None = None,
) -> MaterializeOutcome:
    """Walk ``rows.jsonl`` and write each missing prompt sidecar."""
    t0 = time.perf_counter()
    rows_seen = 0
    sidecars_written = 0
    sidecars_existing = 0
    prompts_dir = inputs.paths.prompts_dir
    prompts_dir.mkdir(parents=True, exist_ok=True)

    for row in _iter_rows_jsonl(inputs.rows_path):
        if limit is not None and rows_seen >= limit:
            break
        rows_seen += 1
        prompt_hash = str(row["prompt_hash"])
        sidecar = prompts_dir / f"{prompt_hash}.txt"
        if sidecar.exists():
            if on_existing == "error":
                raise SidecarCollisionError(
                    f"sidecar already exists: {sidecar} (use --on-existing skip|overwrite)"
                )
            if on_existing == "skip":
                sidecars_existing += 1
                continue
            # overwrite → fall through to re-render and replace.
        item_id = str(row["item_id"])
        item = inputs.item_index.get(item_id)
        if item is None:
            raise ItemNotInLoaderError(
                f"row references item_id={item_id!r} which is not in the loader index"
            )
        perm = permutation_for(
            item, int(row["permutation_id"]), mode=inputs.permutation_mode
        )
        final, computed_hash = renderer.render(perm, str(row["template_id"]))
        if computed_hash != prompt_hash:
            raise PromptHashRecomputeMismatchError(
                f"prompt hash mismatch for item_id={item_id!r} "
                f"template_id={row['template_id']!r} "
                f"permutation_id={row['permutation_id']!r}: "
                f"expected={prompt_hash} actual={computed_hash}"
            )
        _atomic_write(sidecar, final)
        sidecars_written += 1
        if logger is not None and rows_seen % 1000 == 0:
            logger.info(
                "progress %d (written=%d, existing=%d)",
                rows_seen,
                sidecars_written,
                sidecars_existing,
            )

    return MaterializeOutcome(
        rows_seen=rows_seen,
        sidecars_written=sidecars_written,
        sidecars_existing=sidecars_existing,
        wall_seconds=time.perf_counter() - t0,
        status="completed",
        failure_reason=None,
    )


# --- internals ---------------------------------------------------------


def _iter_rows_jsonl(rows_path: Path) -> Iterator[dict[str, object]]:
    with rows_path.open("rb") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            yield json.loads(line)


def _collect_wanted_item_ids(rows_path: Path) -> dict[str, set[str]]:
    wanted: dict[str, set[str]] = {}
    for row in _iter_rows_jsonl(rows_path):
        ds = str(row["dataset_name"])
        wanted.setdefault(ds, set()).add(str(row["item_id"]))
    return wanted


def _build_loader(
    ds_name: str,
    dataset_cfg: DatasetConfig,
    *,
    hf_cache_dir_override: Path | None,
) -> object:
    """Mirror eval_cmd's loader construction. Lazy-imports keep transformers out."""
    cache_dir = (
        hf_cache_dir_override.expanduser().resolve()
        if hf_cache_dir_override is not None
        else dataset_cfg.cache_dir.expanduser().resolve()
    )
    if ds_name == "mmlu":
        from nl_ae.data.mmlu_loader import MmluLoader

        return MmluLoader(
            cache_dir=cache_dir,
            split=dataset_cfg.mmlu_split,
            subjects=dataset_cfg.mmlu_subjects,
            offline=dataset_cfg.offline,
            hf_dataset_id=dataset_cfg.mmlu_hf_id,
            revision=dataset_cfg.mmlu_revision,
            limit=dataset_cfg.mmlu_limit,
        )
    if ds_name == "opinionqa":
        from nl_ae.data.opinionqa_loader import OpinionQaLoader

        return OpinionQaLoader(
            source_dir=dataset_cfg.opinionqa_source_dir,
            cache_dir=cache_dir,
            canonicalization=dataset_cfg.opinionqa_canonicalization,
            subset_path=dataset_cfg.opinionqa_subset_path,
            wave_filter=dataset_cfg.opinionqa_wave_filter,
            topic_filter=dataset_cfg.opinionqa_topic_filter,
            revision_tag=dataset_cfg.opinionqa_revision_tag,
            offline=dataset_cfg.offline,
            limit=dataset_cfg.opinionqa_limit,
        )
    raise ValueError(f"unknown dataset name in manifest: {ds_name!r}")


def _atomic_write(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


__all__ = [
    "MaterializeInputs",
    "MaterializeOutcome",
    "OnExisting",
    "load_materialize_inputs",
    "materialize_prompts",
]
