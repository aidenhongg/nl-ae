"""Typed exceptions for per-layer linear probes (C07)."""

from __future__ import annotations


class ProbeError(RuntimeError):
    """Base for every probe-trainer failure."""


class ProbeManifestMissingError(ProbeError):
    """``probe_manifest.json`` does not exist for the requested fold."""


class ProbeManifestStaleError(ProbeError):
    """Resume refused: on-disk manifest digest disagrees with the requested inputs."""


class LabelOutOfScopeError(ProbeError):
    """Requested label set is not a subset of ``CANDIDATE_LABELS`` (or holdout drift)."""


class LayerOutOfScopeError(ProbeError):
    """Requested layer set is not a subset of the activation cache's layers."""


class ActivationManifestNotCompletedError(ProbeError):
    """The activation cache exists but its ``completion_status != 'completed'``."""


class FitFailedError(ProbeError):
    """The probe fit itself raised; recorded as a ``status='failed'`` cell."""


class InsufficientLabelDataError(ProbeError):
    """Train sub-fold has too few samples or too few distinct classes to fit."""


__all__ = [
    "ActivationManifestNotCompletedError",
    "FitFailedError",
    "InsufficientLabelDataError",
    "LabelOutOfScopeError",
    "LayerOutOfScopeError",
    "ProbeError",
    "ProbeManifestMissingError",
    "ProbeManifestStaleError",
]
