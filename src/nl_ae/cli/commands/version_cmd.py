"""``nlae version`` — zero-cost: only ``importlib.metadata``."""

from __future__ import annotations

import sys
from importlib.metadata import PackageNotFoundError, version

import click


@click.command("version", help="Print nl-ae version.")
def version_cmd() -> None:
    try:
        v = version("nl-ae")
    except PackageNotFoundError:
        v = "0.0.0+unknown"
    click.echo(f"nl-ae {v} (python {sys.version_info.major}.{sys.version_info.minor})")


__all__ = ["version_cmd"]
