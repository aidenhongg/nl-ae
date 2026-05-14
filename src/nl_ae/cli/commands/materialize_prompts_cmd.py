"""``nlae materialize-prompts`` — rebuild missing prompt sidecars from a run.

Heavy imports (``transformers.AutoTokenizer``, dataset loaders) live inside
the handler body so ``nlae --help`` stays fast (matches ``extract_activations_cmd.py``).
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import click


@click.command(
    "materialize-prompts",
    help=(
        "Rebuild runs/<run_id>/prompts/<prompt_hash>.txt sidecars from "
        "rows.jsonl + manifest.json without loading the model. Use this to "
        "recover from a Phase 1 run where save_rendered_prompts was disabled "
        "or where sidecars were lost."
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
    "--on-existing",
    type=click.Choice(["skip", "overwrite", "error"], case_sensitive=False),
    default="skip",
    show_default=True,
    help="Skip pre-existing sidecars (default), overwrite them, or refuse on collision.",
)
@click.option(
    "--hf-cache-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
    help="Override HF cache dir for tokenizer and dataset loaders. Defaults to $HF_HOME.",
)
def materialize_prompts_cmd(
    run_dir: Path,
    limit: int | None,
    on_existing: str,
    hf_cache_dir: Path | None,
) -> None:
    # Heavy imports stay inside the handler so `nlae --help` stays light.
    from nl_ae.data.text_norm import nfc, sha256_hex_bytes
    from nl_ae.prompt.chat_adapter import make_chat_adapter
    from nl_ae.prompt.errors import (
        ItemNotInLoaderError,
        ManifestNotCompletedError,
        PromptHashRecomputeMismatchError,
        SidecarCollisionError,
        TemplateContentHashMismatchError,
    )
    from nl_ae.prompt.materialize import (
        load_materialize_inputs,
        materialize_prompts,
    )
    from nl_ae.prompt.renderer import PromptRenderer
    from nl_ae.prompt.template_registry import TemplateRegistry

    on_existing_lc = on_existing.lower()

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

    from transformers import AutoTokenizer  # type: ignore[import-not-found]

    fp = inputs.run_manifest.model
    try:
        tok = AutoTokenizer.from_pretrained(
            fp.hf_model_id,
            revision=fp.hf_model_commit,
            cache_dir=str(hf_cache_dir) if hf_cache_dir is not None else None,
            trust_remote_code=False,
        )
    except Exception as exc:
        click.echo(f"tokenizer load failed: {exc}", err=True)
        sys.exit(2)

    live_template = getattr(tok, "chat_template", "") or ""
    live_hash = sha256_hex_bytes(nfc(live_template).encode("utf-8"))
    if inputs.chat_template_hash is None:
        click.echo(
            "warning: manifest has no chat_template_hash; relying on per-row "
            "prompt-hash check",
            err=True,
        )
    elif live_hash != inputs.chat_template_hash:
        click.echo(
            f"chat_template_hash mismatch: live={live_hash} "
            f"manifest={inputs.chat_template_hash}",
            err=True,
        )
        sys.exit(4)

    try:
        registry = TemplateRegistry.from_records(inputs.run_manifest.prompt_templates)
    except TemplateContentHashMismatchError as exc:
        click.echo(f"template content hash drift: {exc}", err=True)
        sys.exit(4)

    renderer = PromptRenderer(
        registry,
        chat_adapter=make_chat_adapter(tok, identity_hash=live_hash),
    )

    try:
        outcome = materialize_prompts(
            inputs,
            renderer=renderer,
            on_existing=on_existing_lc,  # type: ignore[arg-type]
            limit=limit,
            logger=logging.getLogger("nl_ae.prompt.materialize"),
        )
    except SidecarCollisionError as exc:
        click.echo(f"sidecar collision: {exc}", err=True)
        sys.exit(3)
    except PromptHashRecomputeMismatchError as exc:
        click.echo(f"prompt hash mismatch: {exc}", err=True)
        sys.exit(4)
    except ItemNotInLoaderError as exc:
        click.echo(f"item index incomplete: {exc}", err=True)
        sys.exit(4)

    click.echo(
        f"run_id={inputs.run_manifest.run_id} status={outcome.status} "
        f"rows={outcome.rows_seen} written={outcome.sidecars_written} "
        f"existing={outcome.sidecars_existing} wall={outcome.wall_seconds:.1f}s"
    )


__all__ = ["materialize_prompts_cmd"]
