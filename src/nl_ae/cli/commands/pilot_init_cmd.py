"""``nlae pilot-init`` — assign the deterministic ~5% pilot fold for a finished run.

No model load, no GPU; reads ``rows.jsonl`` + ``manifest.json`` once and writes
``pilot_manifest.json`` atomically. Idempotent on identical inputs.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import click

LOG = logging.getLogger("nl_ae.cli.pilot_init")


@click.command("pilot-init", help="Assign a deterministic stratified pilot fold for a finished run.")
@click.option(
    "--run-dir",
    "run_dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    required=True,
    help="Path to runs/<run_id>/ with a completed Phase 1 manifest.",
)
@click.option(
    "--frac",
    type=click.FloatRange(min=0.0, max=1.0, min_open=True, max_open=True),
    default=0.05,
    show_default=True,
    help="Per-stratum pilot fraction.",
)
@click.option(
    "--seed",
    type=click.IntRange(min=0, max=(1 << 32) - 1),
    default=0,
    show_default=True,
    help="Pilot assignment root seed (usually matches RunConfig.seeds.root).",
)
@click.option(
    "--stratify-by",
    "stratify_by_csv",
    default="subject,wave,topic,dataset_name",
    show_default=True,
    help="Comma-separated stratification chain; first non-null field per item wins.",
)
@click.option(
    "--min-per-stratum",
    type=click.IntRange(min=1),
    default=2,
    show_default=True,
    help="Floor on pilot items per stratum (activates only when stratum has >= 4 items).",
)
@click.option(
    "--on-existing",
    type=click.Choice(["resume", "overwrite", "error"], case_sensitive=False),
    default="resume",
    show_default=True,
    help="resume: no-op if digest matches, error if drift. overwrite: replace. error: refuse.",
)
def pilot_init_cmd(
    run_dir: Path,
    frac: float,
    seed: int,
    stratify_by_csv: str,
    min_per_stratum: int,
    on_existing: str,
) -> None:
    # Heavy-ish imports (yaml chain, pydantic-validated model loading) deferred.
    from nl_ae.pilot.errors import PilotFoldMismatchError  # noqa: PLC0415
    from nl_ae.pilot.manifest import (  # noqa: PLC0415
        assign_and_write_pilot_manifest,
        iter_item_summaries,
    )
    from nl_ae.schema.paths import run_paths  # noqa: PLC0415
    from nl_ae.schema.reader import load_manifest  # noqa: PLC0415

    paths = run_paths(run_dir.parent, run_dir.name)
    if not paths.manifest_json.exists():
        click.echo(f"manifest.json missing: {paths.manifest_json}", err=True)
        sys.exit(2)
    manifest = load_manifest(paths.manifest_json)
    if manifest.completion_status != "completed":
        click.echo(
            f"run status is {manifest.completion_status!r}; pilot-init requires 'completed'",
            err=True,
        )
        sys.exit(2)
    if not paths.rows_jsonl.exists():
        click.echo(f"rows.jsonl missing: {paths.rows_jsonl}", err=True)
        sys.exit(2)

    stratify_by = tuple(s.strip() for s in stratify_by_csv.split(",") if s.strip())
    if not stratify_by:
        click.echo("--stratify-by produced an empty chain", err=True)
        sys.exit(2)

    click.echo(f"reading rows from {paths.rows_jsonl} ...")
    items = list(iter_item_summaries(paths.rows_jsonl))
    click.echo(f"distinct items: {len(items):,}")

    try:
        pilot_manifest, was_written = assign_and_write_pilot_manifest(
            run_id=manifest.run_id,
            items=items,
            seed=seed,
            frac=frac,
            stratify_by=stratify_by,
            min_per_stratum=min_per_stratum,
            out_path=paths.pilot_manifest_json,
            on_existing=on_existing.lower(),
        )
    except PilotFoldMismatchError as exc:
        click.echo(f"pilot fold drift: {exc}", err=True)
        sys.exit(3)
    except FileExistsError as exc:
        click.echo(str(exc), err=True)
        click.echo(
            "pass --on-existing overwrite to replace, or --on-existing resume "
            "for idempotent re-runs",
            err=True,
        )
        sys.exit(3)

    click.echo(
        f"{'wrote' if was_written else 'reused'} {paths.pilot_manifest_json}"
    )
    click.echo(
        f"  digest: {pilot_manifest.pilot_manifest_digest}\n"
        f"  pilot: {pilot_manifest.n_pilot:,}   holdout: {pilot_manifest.n_holdout:,}   "
        f"total: {pilot_manifest.n_total:,}"
    )
    click.echo(f"  strata: {len(pilot_manifest.strata)}")
    # Brief per-source-field rollup.
    by_src: dict[str, list[int]] = {}
    for s in pilot_manifest.strata:
        by_src.setdefault(s.source_field, []).append(s.n_pilot)
    for src, counts in sorted(by_src.items()):
        click.echo(
            f"    {src}: {len(counts)} strata, "
            f"pilot_total={sum(counts):,}, "
            f"min={min(counts)}, max={max(counts)}"
        )


__all__ = ["pilot_init_cmd"]
