"""``nlae info`` — print the resolved config + env fingerprint. No GPU access."""

from __future__ import annotations

import json
from pathlib import Path

import click

from nl_ae.config.digest import compute_config_digest
from nl_ae.config.loader import load_config
from nl_ae.runtime.identity import gather_environment_fingerprint


@click.command("info", help="Print resolved config + environment fingerprint.")
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
def info_cmd(config_path: Path, overrides: tuple[str, ...]) -> None:
    cfg = load_config(config_path, overrides=overrides)
    digest = compute_config_digest(cfg)
    env = gather_environment_fingerprint()
    payload = {
        "config_path": str(config_path),
        "config_digest": digest,
        "config_schema_version": cfg.config_schema_version,
        "env": env.model_dump(),
        "dataset": {
            "enabled": list(cfg.dataset.enabled),
            "templates_in_use": list(cfg.dataset.templates_in_use),
            "mmlu_subjects": (
                list(cfg.dataset.mmlu_subjects)
                if cfg.dataset.mmlu_subjects is not None
                else None
            ),
        },
        "model": {
            "hf_model_id": cfg.model.hf_model_id,
            "quantization": cfg.model.quantization.model_dump(),
            "device_map": cfg.model.device_map,
        },
        "eval": {
            "templates": list(cfg.eval.plan.template_ids),
            "permutations_per_item": cfg.eval.plan.permutations_per_item,
            "decode_strategy": cfg.eval.decoding.decode_strategy,
        },
    }
    click.echo(json.dumps(payload, indent=2, ensure_ascii=False))


__all__ = ["info_cmd"]
