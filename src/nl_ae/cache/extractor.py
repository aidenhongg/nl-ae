"""Activation cache extractor: replay Phase 1 prompts forward-only.

Walks ``rows.jsonl`` in order, filters to the requested fold via
``PilotManifest.pilot_item_ids``, reads each row's prompt from the Phase 1
``prompts/<prompt_hash>.txt`` sidecar (validating the hash), and asks the
injected :class:`ActivationSource` for per-layer hidden states. Layers are
buffered and flushed in lockstep through :class:`ActivationCacheWriter`.

Decomposition-agnostic: the only contact with the model is the
:class:`ActivationSource` Protocol, so tests inject a fake source that returns
deterministic vectors without loading Qwen.
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol

from nl_ae.data.text_norm import nfc, sha256_hex_bytes
from nl_ae.pilot.manifest import load_pilot_manifest
from nl_ae.pilot.models import PilotManifest, Preregistration
from nl_ae.pilot.preregistration import require_preregistration
from nl_ae.schema.hashing import now_utc_iso
from nl_ae.schema.models import RunManifest, Sha256Hex
from nl_ae.schema.paths import RunPaths, run_paths
from nl_ae.schema.reader import load_manifest

from .errors import (
    PromptHashMismatchError,
    PromptSidecarMissingError,
)
from .models import (
    CACHE_KEY_COMPOSITION,
    ActivationManifest,
    LayerShardManifest,
    PositionPolicy,
    compute_cache_key_composition_digest,
    compute_prompt_hash_set_digest,
    compute_quantization_spec_digest,
)
from .reader import ActivationCacheReader
from .writer import ActivationCacheWriter

LOG = logging.getLogger(__name__)

Fold = Literal["pilot", "holdout"]
VisitKey = tuple[str, int, str]


class _HasTolist(Protocol):
    def tolist(self) -> list[float]: ...


class _ForwardOutputLike(Protocol):
    hidden_states: dict[int, _HasTolist] | None


class ActivationSource(Protocol):
    """The slice of :class:`Qwen25Wrapper` C06 depends on.

    Implementations must return per-layer post-block residuals at the
    last-prompt-token position. ``forward(prompt, capture_hiddens=True,
    record_layers_override=layers)`` must populate a
    ``hidden_states: dict[int, tensor-like]`` keyed by block index, where each
    tensor exposes ``.tolist()`` returning a length-``hidden_size`` float list.
    """

    @property
    def chat_template_hash(self) -> Sha256Hex: ...

    @property
    def n_layers(self) -> int: ...

    @property
    def hidden_size(self) -> int: ...

    def forward(
        self,
        prompt: str,
        *,
        capture_hiddens: bool = True,
        record_layers_override: Sequence[int] | None = None,
    ) -> _ForwardOutputLike: ...


# --- preflight bundle --------------------------------------------------


@dataclass(frozen=True)
class ExtractorInputs:
    """Validated, fully-resolved per-run inputs the extractor walks."""

    run_dir: Path
    paths: RunPaths
    run_manifest: RunManifest
    pilot_manifest: PilotManifest
    preregistration: Preregistration | None
    fold: Fold


@dataclass(frozen=True)
class ExtractorOutcome:
    rows_written: int
    rows_expected: int
    rows_skipped_resume: int
    status: Literal["completed", "failed", "aborted"]
    failure_reason: str | None


def load_extractor_inputs(run_dir: Path, fold: Fold) -> ExtractorInputs:
    """Load + cross-validate everything needed before the model is touched.

    Refuses on: missing Phase 1 manifest, missing pilot manifest, missing /
    unlocked / digest-mismatched preregistration (when ``fold == 'holdout'``).
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
            "extraction requires a completed run"
        )
    if not paths.rows_jsonl.exists():
        raise FileNotFoundError(f"rows.jsonl missing: {paths.rows_jsonl}")
    pilot_manifest = load_pilot_manifest(paths.pilot_manifest_json)
    preregistration: Preregistration | None = None
    if fold == "holdout":
        preregistration = require_preregistration(paths)
    return ExtractorInputs(
        run_dir=paths.run_dir,
        paths=paths,
        run_manifest=run_manifest,
        pilot_manifest=pilot_manifest,
        preregistration=preregistration,
        fold=fold,
    )


# --- streaming helpers --------------------------------------------------


def _iter_fold_rows(
    rows_path: Path,
    *,
    pilot_item_ids: frozenset[str],
    fold: Fold,
) -> Iterator[dict[str, object]]:
    with rows_path.open("rb") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            obj = json.loads(line)
            is_pilot = obj["item_id"] in pilot_item_ids
            if fold == "pilot" and not is_pilot:
                continue
            if fold == "holdout" and is_pilot:
                continue
            yield obj


def _read_prompt_sidecar(*, prompts_dir: Path, prompt_hash: str) -> str:
    sidecar = prompts_dir / f"{prompt_hash}.txt"
    if not sidecar.exists():
        raise PromptSidecarMissingError(
            f"prompt sidecar not found: {sidecar} — run "
            "`nlae materialize-prompts --run-dir <run_dir>` to rebuild sidecars "
            "without re-running the model"
        )
    text = sidecar.read_text(encoding="utf-8")
    normalized = nfc(text)
    actual = sha256_hex_bytes(normalized.encode("utf-8"))
    if actual != prompt_hash:
        raise PromptHashMismatchError(
            f"prompt sidecar {sidecar} hashes to {actual} but row.prompt_hash is "
            f"{prompt_hash}"
        )
    return normalized


def _resolve_model_commit(run_manifest: RunManifest) -> str:
    commit = run_manifest.model.hf_model_commit
    if commit is None or not commit.strip():
        return "unknown"
    return commit


def _hash_preregistration(prereg: Preregistration) -> Sha256Hex:
    payload = json.dumps(
        prereg.model_dump(mode="json"),
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def scan_resume_state(
    *, run_dir: Path, fold: Fold, layers: Sequence[int]
) -> tuple[tuple[LayerShardManifest, ...], frozenset[VisitKey]]:
    """Return ``(seed_shards, completed_visit_keys)`` from any prior cache state."""
    paths = run_paths(run_dir.parent, run_dir.name)
    cache_dir = paths.fold_activations_dir(fold)
    if not (cache_dir / "activation_manifest.json").exists():
        return (), frozenset()
    reader = ActivationCacheReader.open(run_dir, fold, allow_partial=True)
    by_layer = {ls.layer: ls for ls in reader.manifest.layer_shards}
    seed = tuple(
        by_layer.get(layer, LayerShardManifest(layer=layer)) for layer in layers
    )
    completed = reader.completed_visit_keys()
    return seed, completed


# --- extractor ----------------------------------------------------------


class ActivationCacheExtractor:
    """Orchestrate one ``extract-activations`` run for a single fold."""

    def __init__(
        self,
        *,
        inputs: ExtractorInputs,
        source: ActivationSource,
        layers: Sequence[int],
        shard_rows: int = 50_000,
        position_policy: PositionPolicy = "last_prompt_token",
        limit: int | None = None,
        seed_shards: Sequence[LayerShardManifest] = (),
        completed_visit_keys: frozenset[VisitKey] = frozenset(),
    ) -> None:
        if not layers:
            raise ValueError("layers must be non-empty")
        layers_sorted = tuple(sorted({int(layer) for layer in layers}))
        if any(layer < 0 or layer >= source.n_layers for layer in layers_sorted):
            raise ValueError(
                f"layers {layers_sorted} contains values out of [0, {source.n_layers})"
            )
        if (
            inputs.run_manifest.chat_template_hash is not None
            and source.chat_template_hash != inputs.run_manifest.chat_template_hash
        ):
            raise RuntimeError(
                f"chat_template_hash mismatch: source={source.chat_template_hash} "
                f"manifest={inputs.run_manifest.chat_template_hash}"
            )
        self._inputs = inputs
        self._source = source
        self._layers = layers_sorted
        self._shard_rows = int(shard_rows)
        self._position_policy = position_policy
        self._limit = limit
        self._seed_shards = tuple(seed_shards)
        self._completed = frozenset(completed_visit_keys)

    # --- main loop -----------------------------------------------------

    def run(self) -> ExtractorOutcome:
        paths = self._inputs.paths
        pilot_ids = frozenset(self._inputs.pilot_manifest.pilot_item_ids)
        in_scope = list(
            _iter_fold_rows(paths.rows_jsonl, pilot_item_ids=pilot_ids, fold=self._inputs.fold)
        )
        if self._limit is not None:
            in_scope = in_scope[: self._limit]
        rows_expected = len(in_scope)
        prompt_hash_set_digest = compute_prompt_hash_set_digest(
            str(row["prompt_hash"]) for row in in_scope
        )

        manifest = self._build_initial_manifest(
            rows_expected=rows_expected,
            prompt_hash_set_digest=prompt_hash_set_digest,
        )
        cache_dir = paths.fold_activations_dir(self._inputs.fold)
        rows_skipped = 0
        status: Literal["completed", "failed", "aborted"] = "completed"
        failure_reason: str | None = None

        with ActivationCacheWriter(
            cache_dir=cache_dir,
            manifest=manifest,
            seed_shards=self._seed_shards,
        ) as writer:
            try:
                for row in in_scope:
                    key: VisitKey = (
                        str(row["item_id"]),
                        int(row["permutation_id"]),  # type: ignore[arg-type]
                        str(row["template_id"]),
                    )
                    if key in self._completed:
                        rows_skipped += 1
                        continue
                    prompt_text = _read_prompt_sidecar(
                        prompts_dir=paths.prompts_dir,
                        prompt_hash=str(row["prompt_hash"]),
                    )
                    per_layer = self._capture(prompt_text)
                    writer.write_visit(
                        item_id=key[0],
                        permutation_id=key[1],
                        template_id=key[2],
                        prompt_hash=str(row["prompt_hash"]),
                        per_layer_vectors=per_layer,
                    )
            except KeyboardInterrupt:
                status = "aborted"
                failure_reason = "KeyboardInterrupt"
                writer.finalize(status="aborted", failure_reason=failure_reason)
                raise
            except Exception as exc:
                status = "failed"
                failure_reason = repr(exc)
                writer.finalize(status="failed", failure_reason=failure_reason)
                raise
            writer.finalize(status="completed")
            return ExtractorOutcome(
                rows_written=writer.rows_written,
                rows_expected=rows_expected,
                rows_skipped_resume=rows_skipped,
                status=status,
                failure_reason=failure_reason,
            )

    # --- helpers --------------------------------------------------------

    def _capture(self, prompt_text: str) -> dict[int, list[float]]:
        output = self._source.forward(
            prompt_text,
            capture_hiddens=True,
            record_layers_override=self._layers,
        )
        hiddens = getattr(output, "hidden_states", None) or {}
        missing = [layer for layer in self._layers if layer not in hiddens]
        if missing:
            raise RuntimeError(
                f"ActivationSource.forward returned no vectors for layers {missing}"
            )
        out: dict[int, list[float]] = {}
        for layer in self._layers:
            vec = list(hiddens[layer].tolist())
            if len(vec) != self._source.hidden_size:
                raise ValueError(
                    f"layer {layer} vector dim {len(vec)} != source.hidden_size "
                    f"{self._source.hidden_size}"
                )
            out[layer] = [float(v) for v in vec]
        return out

    def _build_initial_manifest(
        self,
        *,
        rows_expected: int,
        prompt_hash_set_digest: Sha256Hex,
    ) -> ActivationManifest:
        run_manifest = self._inputs.run_manifest
        quantization = run_manifest.model.quantization
        if quantization is None:
            raise RuntimeError(
                "run manifest is missing model.quantization; cannot construct cache key"
            )
        composition = CACHE_KEY_COMPOSITION
        seed_layer_shards = (
            self._seed_shards
            if self._seed_shards
            else tuple(LayerShardManifest(layer=layer) for layer in self._layers)
        )
        # Writer keeps all layers in lockstep; any layer's row count is the visit total.
        rows_already = seed_layer_shards[0].rows
        return ActivationManifest(
            run_id=run_manifest.run_id,
            fold=self._inputs.fold,
            layers=self._layers,
            position_policy=self._position_policy,
            activation_dtype="fp16",
            activation_dim=self._source.hidden_size,
            model_commit=_resolve_model_commit(run_manifest),
            quantization_spec_digest=compute_quantization_spec_digest(quantization),
            quantization_kind=quantization.kind,
            chat_template_hash=(
                run_manifest.chat_template_hash
                if run_manifest.chat_template_hash is not None
                else self._source.chat_template_hash
            ),
            shard_rows=self._shard_rows,
            layer_shards=seed_layer_shards,
            rows_written=rows_already,
            rows_expected=rows_expected,
            prompt_hash_set_digest=prompt_hash_set_digest,
            cache_key_composition=composition,
            cache_key_composition_digest=compute_cache_key_composition_digest(composition),
            pilot_manifest_digest=self._inputs.pilot_manifest.pilot_manifest_digest,
            preregistration_digest=(
                _hash_preregistration(self._inputs.preregistration)
                if self._inputs.preregistration is not None
                else None
            ),
            completion_status="in_progress",
            failure_reason=None,
            started_at=now_utc_iso(),
            ended_at=None,
        )


__all__ = [
    "ActivationCacheExtractor",
    "ActivationSource",
    "ExtractorInputs",
    "ExtractorOutcome",
    "Fold",
    "VisitKey",
    "load_extractor_inputs",
    "scan_resume_state",
]
