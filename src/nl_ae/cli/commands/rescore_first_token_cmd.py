"""``nlae rescore-first-token`` — forward-only, in-place first-token rescore.

Heavy imports (``torch``/``transformers`` via :class:`Qwen25Wrapper`, the
config loader, dataset loaders) live inside the handler so ``nlae --help``
stays fast (matches ``materialize_prompts_cmd.py``). The engine is pinned to
the run's recorded model commit so the recomputed logits match the original
distribution.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import click


@click.command(
    "rescore-first-token",
    help=(
        "Re-derive the corrupted first-token fields (first_token_letter, "
        "letter_softmax, total_letter_mass, agreement_flag) of a completed "
        "run in place, with one forward pass per row. Prompts, free-text "
        "generation, the activation cache and the pilot fold are preserved. "
        "Run --dry-run first."
    ),
)
@click.option(
    "--run-dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    required=True,
    help="Path to the completed Phase 1 run dir.",
)
@click.option(
    "--limit",
    type=click.IntRange(min=1),
    default=None,
    help="Debug only: process the first N rows.",
)
@click.option(
    "--hf-cache-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
    help="Override HF cache dir for the model, tokenizer and dataset loaders.",
)
@click.option(
    "--dry-run",
    is_flag=True,
    default=False,
    help="Compute and report the pre/post deltas without writing anything.",
)
def rescore_first_token_cmd(
    run_dir: Path,
    limit: int | None,
    hf_cache_dir: Path | None,
    dry_run: bool,
) -> None:
    from nl_ae.config.loader import load_config_from_text
    from nl_ae.inference.extractor import RegexLadderExtractor
    from nl_ae.inference.rescore import rescore_first_token
    from nl_ae.inference.wrapper import ChatTemplateMismatch, InferenceError, Qwen25Wrapper
    from nl_ae.prompt.chat_adapter import make_chat_adapter
    from nl_ae.prompt.errors import (
        ItemNotInLoaderError,
        ManifestNotCompletedError,
        PromptHashRecomputeMismatchError,
        TemplateContentHashMismatchError,
    )
    from nl_ae.prompt.materialize import load_materialize_inputs
    from nl_ae.prompt.renderer import PromptRenderer
    from nl_ae.prompt.template_registry import TemplateRegistry
    from nl_ae.runtime.identity import git_sha_and_dirty

    try:
        inputs = load_materialize_inputs(run_dir, hf_cache_dir=hf_cache_dir)
    except ManifestNotCompletedError as exc:
        click.echo(f"manifest not completed: {exc}", err=True)
        sys.exit(2)
    except ItemNotInLoaderError as exc:
        click.echo(f"item index incomplete: {exc}", err=True)
        sys.exit(4)
    except (FileNotFoundError, RuntimeError) as exc:
        click.echo(f"preflight failed: {exc}", err=True)
        sys.exit(2)

    manifest = inputs.run_manifest
    if manifest.config_yaml_text is None:
        click.echo("manifest has no config_yaml_text; cannot rebuild engine", err=True)
        sys.exit(2)

    overrides_raw = manifest.cli_args.get("overrides")
    overrides = (
        tuple(overrides_raw.split(";"))
        if isinstance(overrides_raw, str) and overrides_raw
        else ()
    )
    cfg = load_config_from_text(manifest.config_yaml_text, overrides=overrides)

    if manifest.model.hf_model_commit is None:
        click.echo(
            "warning: manifest has no hf_model_commit; engine cannot be pinned "
            "to the original revision; recomputed logits may differ",
            err=True,
        )
    if manifest.chat_template_hash is None:
        click.echo(
            "warning: manifest has no chat_template_hash; relying on per-row "
            "prompt-hash check",
            err=True,
        )

    model_cfg = cfg.model.model_copy(
        update={
            "hf_revision": manifest.model.hf_model_commit,
            **({"cache_dir": hf_cache_dir} if hf_cache_dir is not None else {}),
        }
    )

    try:
        engine = Qwen25Wrapper(
            config=model_cfg,
            extractor=RegexLadderExtractor(),
            pinned_chat_template_hash=manifest.chat_template_hash,
        )
    except ChatTemplateMismatch as exc:
        click.echo(f"chat_template_hash mismatch: {exc}", err=True)
        sys.exit(4)
    except InferenceError as exc:
        click.echo(f"engine load failed: {exc}", err=True)
        sys.exit(2)

    try:
        try:
            registry = TemplateRegistry.from_records(manifest.prompt_templates)
        except TemplateContentHashMismatchError as exc:
            click.echo(f"template content hash drift: {exc}", err=True)
            sys.exit(4)
        renderer = PromptRenderer(
            registry,
            chat_adapter=make_chat_adapter(
                engine.tokenizer,
                identity_hash=engine.chat_template_hash,
                enabled=cfg.dataset.chat_enabled,
            ),
        )

        git_sha, git_dirty = git_sha_and_dirty()
        try:
            outcome = rescore_first_token(
                inputs,
                renderer=renderer,
                engine=engine,
                git_sha=git_sha,
                git_dirty=git_dirty,
                variant_policy="auto",
                dry_run=dry_run,
                limit=limit,
                logger=logging.getLogger("nl_ae.inference.rescore"),
            )
        except PromptHashRecomputeMismatchError as exc:
            click.echo(f"prompt hash mismatch (nothing written): {exc}", err=True)
            sys.exit(4)
        except ItemNotInLoaderError as exc:
            click.echo(f"item index incomplete (nothing written): {exc}", err=True)
            sys.exit(4)
    finally:
        engine.close()

    mode = "dry-run" if outcome.dry_run else "rescored"
    click.echo(
        f"run_id={manifest.run_id} {mode} rows={outcome.rows_seen} "
        f"changed={outcome.rows_changed} wall={outcome.wall_seconds:.1f}s"
    )
    click.echo(f"  first_token_letter before: {_fmt(outcome.first_token_before)}")
    click.echo(f"  first_token_letter after : {_fmt(outcome.first_token_after)}")
    if not outcome.dry_run:
        click.echo(
            f"  rows.jsonl sha256 {outcome.old_rows_sha256} -> "
            f"{outcome.new_rows_sha256}"
        )
        click.echo(f"  wrote {outcome.rescore_manifest_ref}")


def _fmt(hist: dict[str, int]) -> str:
    return "{" + ", ".join(f"{k}:{v}" for k, v in sorted(hist.items())) + "}"


__all__ = ["rescore_first_token_cmd"]
