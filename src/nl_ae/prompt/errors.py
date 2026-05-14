"""Typed prompt-layer exceptions raised by the materialize-prompts pipeline.

All concrete errors inherit from :class:`MaterializeError` so callers can catch
the whole family with one ``except`` clause; the CLI handler then maps each
subclass to its documented exit code.
"""

from __future__ import annotations


class MaterializeError(RuntimeError):
    """Base class for every materialize-prompts failure."""


class ManifestNotCompletedError(MaterializeError):
    """``RunManifest.completion_status`` is not ``"completed"``."""


class ChatTemplateHashMismatchError(MaterializeError):
    """The live tokenizer's chat-template hash diverged from the manifest."""


class TemplateContentHashMismatchError(MaterializeError):
    """A manifest template record's recomputed content hash doesn't match."""


class PromptHashRecomputeMismatchError(MaterializeError):
    """Replaying a row produced a prompt whose hash differs from the row's."""


class ItemNotInLoaderError(MaterializeError):
    """``rows.jsonl`` references an item_id absent from the dataset loader."""


class SidecarCollisionError(MaterializeError):
    """A sidecar already exists and ``--on-existing=error`` was requested."""


__all__ = [
    "ChatTemplateHashMismatchError",
    "ItemNotInLoaderError",
    "ManifestNotCompletedError",
    "MaterializeError",
    "PromptHashRecomputeMismatchError",
    "SidecarCollisionError",
    "TemplateContentHashMismatchError",
]
