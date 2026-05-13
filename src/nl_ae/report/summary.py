"""Markdown summary + ``render_run`` orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from textwrap import dedent

from nl_ae.schema.models import RunManifest
from nl_ae.schema.paths import run_paths
from nl_ae.schema.reader import load_manifest

from .aggregate import AggregateBundle, aggregate_run
from .figures import render_all_figures


@dataclass(frozen=True)
class ReportArtifacts:
    run_dir: Path
    summary_md: Path
    figure_paths: tuple[Path, ...]
    bundle: AggregateBundle
    manifest: RunManifest


def render_summary_md(
    bundle: AggregateBundle,
    manifest: RunManifest,
    out_path: Path,
    *,
    figure_paths: list[Path],
) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    top1 = bundle.top1_disagreement
    lines: list[str] = []
    lines.append(f"# Run {manifest.run_id}")
    lines.append("")
    lines.append(dedent(f"""\
        - Started: {manifest.started_at}
        - Ended: {manifest.ended_at or '—'}
        - Status: {manifest.completion_status}
        - Rows: {manifest.rows_written} / {manifest.rows_expected or '?'}
        - Model: {manifest.model.hf_model_id} ({manifest.model.quantization.kind if manifest.model.quantization else 'unknown'})
        - Config digest: `{manifest.config_digest}`
    """))
    lines.append("## Top-1 disagreement")
    if top1 is None or len(top1) == 0:
        lines.append("_no data_")
    else:
        lines.append("| dataset | template | n | disagreement | 95% CI |")
        lines.append("|---|---|---:|---:|---|")
        for r in top1.itertuples():
            lines.append(
                f"| {r.dataset_name} | {r.template_id} | {r.n} | "
                f"{r.disagreement:.3f} | [{r.ci_lo:.3f}, {r.ci_hi:.3f}] |"
            )
    lines.append("")
    lines.append("## Figures")
    for path in figure_paths:
        rel = path.name
        lines.append(f"- ![{rel}]({rel})")
    body = "\n".join(lines) + "\n"
    out_path.write_text(body, encoding="utf-8")
    return out_path


def render_run(
    run_dir: Path,
    *,
    include_partial: bool = False,
) -> ReportArtifacts:
    paths = run_paths(run_dir.parent, run_dir.name)
    bundle = aggregate_run(
        run_dir, include_partial=include_partial, write_parquet=True
    )
    figs = render_all_figures(bundle, paths.figures_dir)
    manifest = load_manifest(paths.manifest_json)
    summary = render_summary_md(bundle, manifest, paths.summary_md, figure_paths=figs)
    return ReportArtifacts(
        run_dir=run_dir,
        summary_md=summary,
        figure_paths=tuple(figs),
        bundle=bundle,
        manifest=manifest,
    )


__all__ = ["ReportArtifacts", "render_run", "render_summary_md"]
