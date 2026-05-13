"""Typed config-layer exceptions."""

from __future__ import annotations


class ConfigError(Exception):
    """Base class for every config-layer error."""


class ConfigParseError(ConfigError):
    """YAML could not be parsed."""


class ConfigOverrideError(ConfigError):
    """``--set`` override could not be applied to the loaded payload."""


class EnvInterpolationError(ConfigError):
    """``${ENV:NAME}`` referenced an undefined variable (without a default)."""


__all__ = [
    "ConfigError",
    "ConfigOverrideError",
    "ConfigParseError",
    "EnvInterpolationError",
]
