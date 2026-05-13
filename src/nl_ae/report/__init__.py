"""C04 — aggregator + figures + summary."""

from .aggregate import AggregateBundle, aggregate_run
from .figures import render_all_figures
from .summary import ReportArtifacts, render_run, render_summary_md

__all__ = [
    "AggregateBundle",
    "ReportArtifacts",
    "aggregate_run",
    "render_all_figures",
    "render_run",
    "render_summary_md",
]
