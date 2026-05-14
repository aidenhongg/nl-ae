"""Click application — deliberately no heavy imports at module level.

Subcommand handlers ``import torch``/``transformers`` only when needed so
``nlae --help`` and ``nlae version`` finish in <100 ms on a GPU-less box.
"""

from __future__ import annotations

import click

from .commands import (
    aggregate_cmd,
    eval_cmd,
    extract_activations_cmd,
    info_cmd,
    materialize_prompts_cmd,
    pilot_init_cmd,
    preregistration_lock_cmd,
    version_cmd,
)


@click.group(help="nl-ae — minimal recreation of Wang et al. 2024.")
@click.version_option(package_name="nl-ae", prog_name="nlae")
def cli() -> None:  # pragma: no cover (Click intercepts)
    pass


cli.add_command(eval_cmd.eval_cmd)
cli.add_command(aggregate_cmd.aggregate_cmd)
cli.add_command(info_cmd.info_cmd)
cli.add_command(version_cmd.version_cmd)
cli.add_command(pilot_init_cmd.pilot_init_cmd)
cli.add_command(preregistration_lock_cmd.preregistration_lock_cmd)
cli.add_command(extract_activations_cmd.extract_activations_cmd)
cli.add_command(materialize_prompts_cmd.materialize_prompts_cmd)


def main() -> None:
    cli(standalone_mode=True)


__all__ = ["cli", "main"]
