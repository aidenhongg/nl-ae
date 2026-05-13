"""``nlae aggregate`` — render aggregates + figures + summary for a run."""

from __future__ import annotations

from pathlib import Path

import click


@click.command("aggregate", help="Aggregate a finished run and render figures.")
@click.option(
    "--run-dir",
    "run_dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    required=True,
)
@click.option(
    "--include-partial",
    is_flag=True,
    default=False,
    help="Aggregate a still-running JSONL (rows.jsonl.partial) — uses strict=False.",
)
def aggregate_cmd(run_dir: Path, include_partial: bool) -> None:
    # Heavy imports (pandas, matplotlib) live here.
    from nl_ae.report.summary import render_run  # noqa: PLC0415

    artifacts = render_run(run_dir, include_partial=include_partial)
    click.echo(f"summary: {artifacts.summary_md}")
    for fig in artifacts.figure_paths:
        click.echo(f"figure:  {fig}")


__all__ = ["aggregate_cmd"]
