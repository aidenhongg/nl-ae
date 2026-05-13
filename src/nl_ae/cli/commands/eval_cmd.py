"""``nlae eval`` — run the full Phase 1 evaluation.

Heavy imports (``torch``, ``transformers``) are deferred to the handler body
so ``nlae --help`` stays fast.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import click


@click.command("eval", help="Run the Phase 1 evaluation against the configured dataset(s).")
@click.option(
    "--config",
    "config_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
)
@click.option(
    "--set",
    "overrides",
    multiple=True,
    help="Hydra-style override key.sub=value (YAML-parsed value).",
)
@click.option(
    "--run-id",
    "run_id_override",
    type=str,
    default=None,
    help="Override the auto-minted run_id (rare; usually only to resume).",
)
def eval_cmd(
    config_path: Path,
    overrides: tuple[str, ...],
    run_id_override: str | None,
) -> None:
    # All heavy imports live here so `nlae --help` and `nlae version` stay light.
    from nl_ae.config.digest import compute_config_digest  # noqa: PLC0415
    from nl_ae.config.loader import load_config  # noqa: PLC0415
    from nl_ae.data.mmlu_loader import MmluLoader  # noqa: PLC0415
    from nl_ae.data.opinionqa_loader import OpinionQaLoader  # noqa: PLC0415
    from nl_ae.eval.runner import (  # noqa: PLC0415
        EvalRunner,
        ManifestBuilder,
    )
    from nl_ae.inference.extractor import RegexLadderExtractor  # noqa: PLC0415
    from nl_ae.inference.wrapper import Qwen25Wrapper  # noqa: PLC0415
    from nl_ae.prompt.letter_tokens import build_letter_token_table  # noqa: PLC0415
    from nl_ae.prompt.renderer import (  # noqa: PLC0415
        NullChatTemplateAdapter,
        PromptRenderer,
    )
    from nl_ae.prompt.template_registry import TemplateRegistry  # noqa: PLC0415
    from nl_ae.runtime.identity import (  # noqa: PLC0415
        gather_environment_fingerprint,
        mint_run_id,
    )
    from nl_ae.runtime.logging import setup_logging  # noqa: PLC0415
    from nl_ae.runtime.seeds import apply_seeds  # noqa: PLC0415
    from nl_ae.schema.paths import run_paths  # noqa: PLC0415

    cfg = load_config(config_path, overrides=overrides)
    digest = compute_config_digest(cfg)

    # 1. Mint or take the run identity.
    if run_id_override is not None:
        run_id = run_id_override
        from nl_ae.runtime.identity import git_sha_and_dirty  # noqa: PLC0415

        git_sha, git_dirty = git_sha_and_dirty()
        from nl_ae.schema.hashing import now_utc_iso  # noqa: PLC0415

        started_at = now_utc_iso()
    else:
        run_id, started_at, git_sha, git_dirty = mint_run_id(
            slug=cfg.run_identity.slug or None
        )

    output_root = cfg.output.output_dir.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    paths = run_paths(output_root, run_id)
    paths.run_dir.mkdir(parents=True, exist_ok=True)
    paths.logs_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logging(
        run_log_dir=paths.logs_dir,
        console_level=cfg.logging.console_level,
        file_level=cfg.logging.file_level,
        per_module_levels={k: v for k, v in cfg.logging.per_module_levels.items()},
        jsonl_destination=cfg.logging.jsonl_destination,
    )
    logger.info("run_id=%s config_digest=%s", run_id, digest)

    applied_seeds = apply_seeds(cfg.seeds, target_device=cfg.model.device_map)
    logger.info("applied seeds; deterministic_algorithms_actual=%s",
                applied_seeds.deterministic_algorithms_actual)

    env_fp = gather_environment_fingerprint()
    extractor = RegexLadderExtractor(allow_majority_vote=False)

    pinned_hash: str | None = None
    pinned_path = (
        cfg.dataset.pinned_chat_template_hash_path.expanduser().resolve()
        if cfg.dataset.pinned_chat_template_hash_path
        else None
    )
    if pinned_path and pinned_path.exists():
        pinned_hash = pinned_path.read_text(encoding="utf-8").strip()

    logger.info("loading model %s ...", cfg.model.hf_model_id)
    with Qwen25Wrapper(
        config=cfg.model, extractor=extractor, pinned_chat_template_hash=pinned_hash
    ) as engine:
        tokenizer = engine.tokenizer
        chat_template_hash = engine.chat_template_hash

        registry = TemplateRegistry(cfg.dataset.templates_dir.expanduser().resolve())
        registry.load()
        chat_adapter = _make_chat_adapter(
            tokenizer, identity_hash=chat_template_hash, enabled=cfg.dataset.chat_enabled
        )
        if chat_adapter is None:
            chat_adapter = NullChatTemplateAdapter(identity_hash=chat_template_hash)
        renderer = PromptRenderer(registry, chat_adapter=chat_adapter)

        # Dataset loaders.
        item_sources = []
        for ds_name in cfg.dataset.enabled:
            if ds_name == "mmlu":
                item_sources.append(
                    MmluLoader(
                        cache_dir=cfg.dataset.cache_dir.expanduser().resolve(),
                        split=cfg.dataset.mmlu_split,
                        subjects=cfg.dataset.mmlu_subjects,
                        offline=cfg.dataset.offline,
                        hf_dataset_id=cfg.dataset.mmlu_hf_id,
                        revision=cfg.dataset.mmlu_revision,
                        limit=cfg.dataset.mmlu_limit,
                    )
                )
            elif ds_name == "opinionqa":
                item_sources.append(
                    OpinionQaLoader(
                        source_dir=cfg.dataset.opinionqa_source_dir,
                        cache_dir=cfg.dataset.cache_dir.expanduser().resolve(),
                        canonicalization=cfg.dataset.opinionqa_canonicalization,
                        subset_path=cfg.dataset.opinionqa_subset_path,
                        wave_filter=cfg.dataset.opinionqa_wave_filter,
                        topic_filter=cfg.dataset.opinionqa_topic_filter,
                        revision_tag=cfg.dataset.opinionqa_revision_tag,
                        offline=cfg.dataset.offline,
                        limit=cfg.dataset.opinionqa_limit,
                    )
                )
        if not item_sources:
            click.echo("no datasets enabled", err=True)
            sys.exit(2)

        # Letter-token table — built over the union of all letters in scope (A..Z).
        from nl_ae.data.permute import LETTERS_26  # noqa: PLC0415

        letter_table = build_letter_token_table(
            tokenizer, LETTERS_26, variants=cfg.dataset.letter_token_variants
        )

        dataset_fingerprints = [src.emit_dataset_fingerprint() for src in item_sources]
        template_records = renderer.emit_template_records()
        seeds_dict = {
            "root": applied_seeds.root,
            "numpy": applied_seeds.numpy,
            "torch": applied_seeds.torch,
            "python": applied_seeds.python,
            "cuda": applied_seeds.cuda,
            "free_gen": applied_seeds.free_gen,
            "pythonhashseed": applied_seeds.pythonhashseed,
        }

        config_yaml_text = config_path.read_text(encoding="utf-8")
        builder = ManifestBuilder(
            run_id=run_id,
            git_sha=git_sha,
            git_dirty=git_dirty,
            config_digest=digest,
            config_yaml_text=config_yaml_text,
            cli_args={
                "config_path": str(config_path),
                "overrides": ";".join(overrides) if overrides else None,
                "run_id_override": run_id_override,
            },
            env=env_fp,
            model=engine.fingerprint(),
            datasets=dataset_fingerprints,
            prompt_templates=template_records,
            letter_token_table=letter_table,
            seeds=seeds_dict,
            chat_template_hash=chat_template_hash,
            extractor=extractor.record(),
            deterministic_algorithms_actual=applied_seeds.deterministic_algorithms_actual,
            applied_seeds_notes=applied_seeds.notes,
            kv_shared=False,
            notes=cfg.run_identity.notes or None,
        )
        runner = EvalRunner(
            config=cfg.eval,
            run_dir=paths.run_dir,
            item_sources=item_sources,
            renderer=renderer,
            engine=engine,
            manifest_builder=builder,
            letter_variants=cfg.dataset.letter_token_variants,
            logger=logging.getLogger("nl_ae.eval"),
        )
        outcome = runner.run()
        # ``started_at`` is now baked into the manifest; surface it for the user too.
        click.echo(
            f"run {outcome.run_id} {outcome.status} "
            f"rows={outcome.rows_written}/{outcome.rows_expected} "
            f"wall={outcome.wall_seconds:.1f}s started={started_at}"
        )


def _make_chat_adapter(tokenizer: object, *, identity_hash: str, enabled: bool):  # noqa: ANN202
    if not enabled or not hasattr(tokenizer, "apply_chat_template"):
        return None

    class HFChatAdapter:
        def __init__(self) -> None:
            self._tok = tokenizer
            self._identity = identity_hash

        def apply(
            self, *, system: str | None, user: str, add_generation_prompt: bool = True
        ) -> str:
            messages: list[dict[str, str]] = []
            if system:
                messages.append({"role": "system", "content": system})
            messages.append({"role": "user", "content": user})
            rendered = self._tok.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
            )
            return str(rendered)

        @property
        def identity(self) -> str:
            return self._identity

    return HFChatAdapter()


__all__ = ["eval_cmd"]
