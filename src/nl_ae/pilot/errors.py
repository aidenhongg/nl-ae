"""Typed exceptions for the pilot fold + preregistration discipline (C09)."""

from __future__ import annotations


class PilotError(RuntimeError):
    """Base for all pilot/preregistration errors."""


class PilotManifestMissingError(PilotError):
    """``pilot_manifest.json`` does not exist for the run."""


class PilotFoldMismatchError(PilotError):
    """On-disk pilot manifest disagrees with a re-derived assignment."""


class PreregistrationMissingError(PilotError):
    """``preregistration.md`` does not exist for the run."""


class PreregistrationParseError(PilotError):
    """``preregistration.md`` lacks a parseable YAML frontmatter block."""


class PreregistrationInvalidError(PilotError):
    """``preregistration.md`` frontmatter fails Pydantic validation."""


class PreregistrationUnlockedError(PilotError):
    """``preregistration.md`` frontmatter lacks ``locked_at`` / ``locked_at_git_sha``."""


class PilotDigestMismatchError(PilotError):
    """``Preregistration.pilot_manifest_digest`` does not match the on-disk pilot manifest."""


class GitTreeDirtyError(PilotError):
    """``git status --porcelain`` is non-empty at lock time (advisory)."""


__all__ = [
    "GitTreeDirtyError",
    "PilotDigestMismatchError",
    "PilotError",
    "PilotFoldMismatchError",
    "PilotManifestMissingError",
    "PreregistrationInvalidError",
    "PreregistrationMissingError",
    "PreregistrationParseError",
    "PreregistrationUnlockedError",
]
