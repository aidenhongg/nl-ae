"""Typed exceptions for the activation cache (C06)."""

from __future__ import annotations


class CacheError(RuntimeError):
    """Base for every activation-cache failure."""


class ActivationManifestMissingError(CacheError):
    """``activation_manifest.json`` does not exist for the requested fold."""


class CacheKeyMismatchError(CacheError):
    """Cache-key composition (or its digest) disagrees between reader and on-disk manifest."""


class PromptHashMismatchError(CacheError):
    """A replayed prompt's SHA-256 does not match the row's recorded ``prompt_hash``."""


class PromptSidecarMissingError(CacheError):
    """The Phase 1 ``prompts/<prompt_hash>.txt`` sidecar is absent for a row in scope."""


class DuplicateActivationRowError(CacheError):
    """Two rows in the same fold share ``(layer, item_id, permutation_id, template_id)``."""


class CacheLockError(CacheError):
    """``<fold>/activations/.run.lock`` is present (another extractor is open)."""


class ActivationCacheStateError(CacheError):
    """Cache state on disk is inconsistent (e.g., partial manifest + no shards)."""


__all__ = [
    "ActivationCacheStateError",
    "ActivationManifestMissingError",
    "CacheError",
    "CacheKeyMismatchError",
    "CacheLockError",
    "DuplicateActivationRowError",
    "PromptHashMismatchError",
    "PromptSidecarMissingError",
]
