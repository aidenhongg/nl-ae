"""``nlae extract-activations`` — forward-only Phase 2 activation cache.

Reconstructs ``ModelConfig`` from the Phase 1 manifest (no Phase 1 YAML
required), loads the Qwen wrapper with the requested layer set, and walks the
fold-filtered ``rows.jsonl`` through :class:`ActivationCacheExtractor`. Prompts
are replayed from the ``prompts/<prompt_hash>.txt`` sidecars; the SHA-256 of
each sidecar's content must match the row's recorded ``prompt_hash`` or the
extraction refuses.
"""

from __future__ import annotations

import logging
import shutil
import sys
from pathlib import Path

import click

LOG = logging.getLogger("nl_ae.cli.extract_activations")


def _parse_layers(layers_arg: str | None, *, n_layers: int) -> tuple[int, ...]:
    """Accept ``"0,1,2"``, ``"0-27"``, or comma-separated mixed; default = all layers."""
    if layers_arg is None or layers_arg.strip() == "":
        return tuple(range(n_layers))
    out: set[int] = set()
    for part in layers_arg.split(","):
        token = part.strip()
        if not token:
            continue
        if "-" in token:
            lo_s, hi_s = token.split("-", 1)
            lo, hi = int(lo_s), int(hi_s)
            if lo > hi:
                raise click.UsageError(f"--layers range {token!r} has lo > hi")
            out.update(range(lo, hi + 1))
        else:
            out.add(int(token))
    if not out:
        raise click.UsageError("--layers parsed to an empty set")
    if min(out) < 0 or max(out) >= n_layers:
        raise click.UsageError(
            f"--layers contains values out of [0, {n_layers}): {sorted(out)}"
        )
    return tuple(sorted(out))


@click.command(
    "extract-activations",
    help=(
        "Replay a Phase 1 run forward-only and cache per-layer activations under "
        "<run_dir>/<fold>/activations/. Refuses --fold holdout without a locked "
        "preregistration."
    ),
)
@click.option(
    "--run-dir",
    "run_dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    required=True,
    help="Path to a completed Phase 1 run dir with rows.jsonl + pilot_manifest.json.",
)
@click.option(
    "--fold",
    type=click.Choice(["pilot", "holdout"], case_sensitive=False),
    required=True,
    help="pilot operates on PilotManifest.pilot_item_ids; holdout on the complement.",
)
@click.option(
    "--layers",
    "layers_arg",
    type=str,
    default=None,
    help='Comma-list / range over [0, n_layers). Examples: "0,5,20", "0-27". Default: all.',
)
@click.option(
    "--shard-rows",
    type=click.IntRange(min=1),
    default=50_000,
    show_default=True,
    help="Rows per parquet shard per layer.",
)
@click.option(
    "--limit",
    type=click.IntRange(min=1),
    default=None,
    help="Debug only: subsample the first N rows of the fold.",
)
@click.option(
    "--on-existing",
    type=click.Choice(["resume", "overwrite", "error"], case_sensitive=False),
    default="resume",
    show_default=True,
    help="resume from a prior cache, overwrite it, or refuse if present.",
)
@click.option(
    "--device",
    "device_map",
    type=click.Choice(["auto", "cuda:0", "cpu", "balanced"], case_sensitive=False),
    default="cuda:0",
    show_default=True,
    help="Wrapper device_map.",
)
@click.option(
    "--cache-dir",
    "model_cache_dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
    help="Override HF model cache dir (defaults to $HF_HOME).",
)
@click.option(
    "--attn-impl",
    type=click.Choice(["sdpa", "eager", "flash_attention_2"], case_sensitive=False),
    default="sdpa",
    show_default=True,
    help="Wrapper attn_implementation.",
)
def extract_activations_cmd(
    run_dir: Path,
    fold: str,
    layers_arg: str | None,
    shard_rows: int,
    limit: int | None,
    on_existing: str,
    device_map: str,
    model_cache_dir: Path | None,
    attn_impl: str,
) -> None:
    # Heavy ML imports live inside the handler so `nlae --help` stays light.
    from nl_ae.cache.errors import (
        CacheError,
        CacheKeyMismatchError,
        CacheLockError,
        PromptHashMismatchError,
        PromptSidecarMissingError,
    )
    from nl_ae.cache.extractor import (
        ActivationCacheExtractor,
        load_extractor_inputs,
        scan_resume_state,
    )
    from nl_ae.inference.loader import HookSpec, ModelConfig
    from nl_ae.inference.wrapper import Qwen25Wrapper
    from nl_ae.pilot.errors import (
        PilotDigestMismatchError,
        PilotManifestMissingError,
        PreregistrationInvalidError,
        PreregistrationMissingError,
        PreregistrationParseError,
        PreregistrationUnlockedError,
    )

    fold_lc = fold.lower()
    try:
        inputs = load_extractor_inputs(run_dir, fold_lc)  # type: ignore[arg-type]
    except PilotManifestMissingError as exc:
        click.echo(f"pilot manifest missing: {exc}", err=True)
        sys.exit(2)
    except (PreregistrationMissingError, PreregistrationParseError) as exc:
        click.echo(f"preregistration missing/invalid: {exc}", err=True)
        sys.exit(2)
    except (PreregistrationInvalidError, PreregistrationUnlockedError) as exc:
        click.echo(f"preregistration not locked or invalid: {exc}", err=True)
        sys.exit(3)
    except PilotDigestMismatchError as exc:
        click.echo(f"pilot digest mismatch: {exc}", err=True)
        sys.exit(3)
    except (FileNotFoundError, RuntimeError) as exc:
        click.echo(f"preflight failed: {exc}", err=True)
        sys.exit(2)

    cache_dir = inputs.paths.fold_activations_dir(inputs.fold)
    cache_existed = (cache_dir / "activation_manifest.json").exists()
    on_existing_lc = on_existing.lower()
    if cache_existed:
        if on_existing_lc == "error":
            click.echo(
                f"refusing to extract: cache exists at {cache_dir}. "
                "Use --on-existing resume|overwrite.",
                err=True,
            )
            sys.exit(3)
        if on_existing_lc == "overwrite":
            click.echo(f"--on-existing overwrite: deleting {cache_dir}")
            shutil.rmtree(cache_dir)
            cache_existed = False

    # Build ModelConfig from the recorded Phase 1 fingerprint. No Phase 1 YAML
    # required; the user only supplies device/cache overrides.
    fp = inputs.run_manifest.model
    if fp.quantization is None:
        click.echo(
            "Phase 1 manifest lacks a quantization spec; cannot reconstruct ModelConfig",
            err=True,
        )
        sys.exit(2)
    # First-pass layer parse against a placeholder count; we re-validate post-load.
    placeholder_n_layers = 64
    layers = _parse_layers(layers_arg, n_layers=placeholder_n_layers)

    model_config = ModelConfig(
        hf_model_id=fp.hf_model_id,
        hf_revision=fp.hf_model_commit,
        cache_dir=model_cache_dir,
        quantization=fp.quantization,
        hook_spec=HookSpec(
            record_layers=layers,
            position_policy="last_prompt_token",
            storage_dtype="fp16",
        ),
        attn_implementation=attn_impl.lower(),  # type: ignore[arg-type]
        device_map=device_map.lower(),  # type: ignore[arg-type]
    )

    click.echo(f"loading {fp.hf_model_id} (commit={fp.hf_model_commit or '<unpinned>'}) ...")
    try:
        with Qwen25Wrapper(
            config=model_config,
            pinned_chat_template_hash=inputs.run_manifest.chat_template_hash,
        ) as engine:
            # Re-parse layers against the actual layer count for a real bounds check.
            layers = _parse_layers(layers_arg, n_layers=engine.n_layers)
            click.echo(
                f"engine ready: n_layers={engine.n_layers} hidden_size={engine.hidden_size} "
                f"chat_template_hash={engine.chat_template_hash}"
            )

            seed_shards, completed = scan_resume_state(
                run_dir=inputs.run_dir, fold=inputs.fold, layers=layers
            )
            if completed:
                click.echo(
                    f"resume: {len(completed):,} visit(s) already cached; will be skipped"
                )

            extractor = ActivationCacheExtractor(
                inputs=inputs,
                source=engine,
                layers=layers,
                shard_rows=shard_rows,
                limit=limit,
                seed_shards=seed_shards,
                completed_visit_keys=completed,
            )
            outcome = extractor.run()
    except CacheLockError as exc:
        click.echo(str(exc), err=True)
        sys.exit(3)
    except (PromptHashMismatchError, PromptSidecarMissingError) as exc:
        click.echo(f"prompt sidecar drift: {exc}", err=True)
        sys.exit(4)
    except CacheKeyMismatchError as exc:
        click.echo(f"cache key drift: {exc}", err=True)
        sys.exit(4)
    except CacheError as exc:
        click.echo(f"cache error: {exc}", err=True)
        sys.exit(4)

    click.echo(
        f"fold={inputs.fold} status={outcome.status} "
        f"rows_written={outcome.rows_written:,}/{outcome.rows_expected:,} "
        f"resumed={outcome.rows_skipped_resume:,}"
    )
    if outcome.failure_reason:
        click.echo(f"  failure_reason: {outcome.failure_reason}", err=True)
        sys.exit(5)


__all__ = ["extract_activations_cmd"]
