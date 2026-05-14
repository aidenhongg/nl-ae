"""``nlae preregistration-lock`` — fill ``locked_at`` + ``locked_at_git_sha`` atomically.

Refuses on a dirty git tree unless ``--allow-dirty``. Verifies the
preregistration's pilot digest matches the on-disk pilot manifest before
locking.
"""

from __future__ import annotations

import sys
from pathlib import Path

import click


@click.command(
    "preregistration-lock",
    help="Lock the preregistration by stamping locked_at + locked_at_git_sha.",
)
@click.option(
    "--run-dir",
    "run_dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    required=True,
)
@click.option(
    "--allow-dirty",
    is_flag=True,
    default=False,
    help="Lock even if `git status --porcelain` is non-empty (advisory escape hatch).",
)
def preregistration_lock_cmd(run_dir: Path, allow_dirty: bool) -> None:
    from nl_ae.pilot.errors import (  # noqa: PLC0415
        GitTreeDirtyError,
        PilotDigestMismatchError,
        PreregistrationInvalidError,
        PreregistrationMissingError,
        PreregistrationParseError,
        PreregistrationUnlockedError,
    )
    from nl_ae.pilot.preregistration import lock_preregistration  # noqa: PLC0415

    try:
        locked = lock_preregistration(run_dir, allow_dirty=allow_dirty)
    except PreregistrationMissingError as exc:
        click.echo(f"preregistration missing: {exc}", err=True)
        sys.exit(2)
    except (PreregistrationParseError, PreregistrationInvalidError) as exc:
        click.echo(f"preregistration invalid: {exc}", err=True)
        sys.exit(2)
    except PreregistrationUnlockedError as exc:
        click.echo(f"already locked or unlock state inconsistent: {exc}", err=True)
        sys.exit(3)
    except PilotDigestMismatchError as exc:
        click.echo(f"pilot digest mismatch: {exc}", err=True)
        sys.exit(3)
    except GitTreeDirtyError as exc:
        click.echo(f"git tree dirty: {exc}", err=True)
        sys.exit(4)

    click.echo(f"locked at {locked.locked_at}  git_sha={locked.locked_at_git_sha}")


__all__ = ["preregistration_lock_cmd"]
