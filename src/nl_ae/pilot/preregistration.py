"""``preregistration.md`` parser + gate + locker (C09).

The file format is a YAML frontmatter block (``---`` fenced) followed by a
freeform Markdown body. Only the frontmatter is machine-checked; the body is
the human narrative.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import yaml
from pydantic import ValidationError

from nl_ae.schema.hashing import now_utc_iso
from nl_ae.schema.paths import RunPaths

from .errors import (
    GitTreeDirtyError,
    PilotDigestMismatchError,
    PreregistrationInvalidError,
    PreregistrationMissingError,
    PreregistrationParseError,
    PreregistrationUnlockedError,
)
from .manifest import load_pilot_manifest
from .models import Preregistration

_FRONTMATTER_RE = re.compile(r"\A---\r?\n(?P<fm>.*?)\r?\n---\r?\n(?P<body>.*)\Z", re.DOTALL)


def parse_preregistration_text(text: str) -> tuple[Preregistration, str]:
    """Split + validate a preregistration document. Returns ``(preregistration, body)``."""
    match = _FRONTMATTER_RE.match(text)
    if match is None:
        raise PreregistrationParseError(
            "preregistration.md must begin with a YAML frontmatter block fenced by `---`"
        )
    fm_raw = match.group("fm")
    body = match.group("body")
    try:
        data = yaml.safe_load(fm_raw)
    except yaml.YAMLError as exc:
        raise PreregistrationParseError(f"frontmatter YAML invalid: {exc}") from exc
    if not isinstance(data, dict):
        raise PreregistrationParseError(
            f"frontmatter must parse to a mapping; got {type(data).__name__}"
        )
    try:
        prereg = Preregistration.model_validate(data)
    except ValidationError as exc:
        raise PreregistrationInvalidError(str(exc)) from exc
    return prereg, body


def load_preregistration(run_dir: Path | RunPaths) -> tuple[Preregistration, str, Path]:
    """Read + parse ``preregistration.md`` from a run dir. Returns ``(prereg, body, path)``."""
    path = _preregistration_path(run_dir)
    if not path.exists():
        raise PreregistrationMissingError(f"preregistration.md not found: {path}")
    text = path.read_text(encoding="utf-8")
    prereg, body = parse_preregistration_text(text)
    return prereg, body, path


def require_preregistration(run_dir: Path | RunPaths) -> Preregistration:
    """Hard gate: every confirmatory ``--fold holdout`` command must call this.

    Raises:
      ``PreregistrationMissingError`` — file does not exist.
      ``PreregistrationParseError`` — frontmatter unparseable.
      ``PreregistrationInvalidError`` — schema validation failed (unknown
            field, bad enum, etc.).
      ``PreregistrationUnlockedError`` — frontmatter parses but ``locked_at``
            / ``locked_at_git_sha`` are absent.
      ``PilotDigestMismatchError`` — declared digest disagrees with on-disk
            ``pilot_manifest.json``.

    On success, returns a fully validated, locked ``Preregistration``.
    """
    rd = run_dir.run_dir if isinstance(run_dir, RunPaths) else run_dir
    prereg, _body, _path = load_preregistration(rd)
    if not prereg.is_locked:
        raise PreregistrationUnlockedError(
            f"preregistration.md is not locked yet (run nlae preregistration-lock): {rd}"
        )
    if prereg.holdout_vs_full != "holdout-only":
        raise PreregistrationInvalidError(
            f"holdout_vs_full={prereg.holdout_vs_full!r}; only 'holdout-only' is honored "
            "by current policy"
        )
    pilot_manifest_path = _pilot_manifest_path(rd)
    pilot_manifest = load_pilot_manifest(pilot_manifest_path)
    if prereg.pilot_manifest_digest != pilot_manifest.pilot_manifest_digest:
        raise PilotDigestMismatchError(
            "preregistration.pilot_manifest_digest does not match on-disk pilot manifest; "
            f"prereg={prereg.pilot_manifest_digest!r} "
            f"on-disk={pilot_manifest.pilot_manifest_digest!r}"
        )
    return prereg


def lock_preregistration(
    run_dir: Path | RunPaths,
    *,
    allow_dirty: bool = False,
) -> Preregistration:
    """Fill ``locked_at`` + ``locked_at_git_sha`` atomically in ``preregistration.md``.

    Refuses if the file is already locked. Refuses if the working tree is dirty
    unless ``allow_dirty=True``. Verifies pilot digest match before writing.
    The body of the file is preserved verbatim; only the frontmatter values are
    mutated.
    """
    rd = run_dir.run_dir if isinstance(run_dir, RunPaths) else run_dir
    prereg, body, path = load_preregistration(rd)
    if prereg.is_locked:
        raise PreregistrationUnlockedError(
            f"preregistration.md is already locked at {prereg.locked_at}; "
            "edit the file by hand if you really need to relock"
        )
    pilot_manifest = load_pilot_manifest(_pilot_manifest_path(rd))
    if prereg.pilot_manifest_digest != pilot_manifest.pilot_manifest_digest:
        raise PilotDigestMismatchError(
            "cannot lock: preregistration.pilot_manifest_digest disagrees with "
            f"on-disk pilot manifest ({prereg.pilot_manifest_digest!r} vs "
            f"{pilot_manifest.pilot_manifest_digest!r})"
        )

    git_sha, dirty = _git_sha_and_dirty(rd)
    if dirty and not allow_dirty:
        raise GitTreeDirtyError(
            "git tree is dirty; pass --allow-dirty to lock anyway "
            "(records the SHA as of HEAD, not the dirty state)"
        )
    if len(git_sha) != 40 or any(c not in "0123456789abcdef" for c in git_sha.lower()):
        raise GitTreeDirtyError(
            f"could not determine current git SHA in run_dir {rd}; got {git_sha!r}"
        )

    locked = prereg.model_copy(
        update={
            "locked_at": now_utc_iso(),
            "locked_at_git_sha": git_sha.lower(),
        }
    )
    # Re-validate (defense-in-depth: model_copy bypasses validators).
    locked = Preregistration.model_validate(locked.model_dump())
    _write_preregistration_atomic(path, locked, body)
    return locked


def _preregistration_path(run_dir: Path | RunPaths) -> Path:
    if isinstance(run_dir, RunPaths):
        return run_dir.run_dir / "preregistration.md"
    return run_dir / "preregistration.md"


def _pilot_manifest_path(run_dir: Path | RunPaths) -> Path:
    if isinstance(run_dir, RunPaths):
        return run_dir.run_dir / "pilot_manifest.json"
    return run_dir / "pilot_manifest.json"


def _git_sha_and_dirty(repo_dir: Path) -> tuple[str, bool]:
    """Return ``(40-hex sha, dirty?)``. Uses ``core.quotepath=false`` for Windows paths.

    Falls back to ``("0" * 40, False)`` only when git is unreachable AND the
    caller never validates. The caller is expected to validate the SHA format.
    """
    try:
        sha = (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=str(repo_dir),
                stderr=subprocess.DEVNULL,
            )
            .decode("ascii")
            .strip()
        )
        # Pin core.quotepath=false so non-ASCII paths don't confuse the porcelain check.
        status = subprocess.check_output(
            ["git", "-c", "core.quotepath=false", "status", "--porcelain"],
            cwd=str(repo_dir),
            stderr=subprocess.DEVNULL,
        ).decode("utf-8")
    except (FileNotFoundError, subprocess.CalledProcessError, OSError):
        return "0" * 40, False
    return sha, bool(status.strip())


def _write_preregistration_atomic(path: Path, prereg: Preregistration, body: str) -> None:
    """Render ``preregistration.md`` with refreshed frontmatter; preserve body verbatim."""
    fm_payload = prereg.model_dump(mode="json")
    fm_text = yaml.safe_dump(
        fm_payload, sort_keys=False, default_flow_style=False, allow_unicode=True
    )
    rendered = "---\n" + fm_text + "---\n" + (body if body.startswith("\n") else "\n" + body)
    payload = rendered.encode("utf-8")
    import os
    import tempfile

    fd, tmp_name = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(payload)
            f.flush()
            try:
                os.fsync(f.fileno())
            except OSError:
                pass
        os.replace(tmp_name, path)
    except BaseException:
        try:
            Path(tmp_name).unlink(missing_ok=True)
        finally:
            raise


__all__ = [
    "load_preregistration",
    "lock_preregistration",
    "parse_preregistration_text",
    "require_preregistration",
]
