"""YAML + ``${ENV:NAME}`` interpolation + Hydra-style ``--set`` overrides.

Loud failures: unknown YAML keys, undefined env vars (without default), type
mismatches across overrides, pydantic validation errors with multi-line
pretty-printing.
"""

from __future__ import annotations

import os
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from .errors import ConfigOverrideError, ConfigParseError, EnvInterpolationError
from .schema import RunConfig

_ENV_PATTERN = re.compile(r"\$\{ENV:([A-Z_][A-Z0-9_]*)(?::([^}]*))?\}")


def load_config(
    config_path: Path,
    *,
    overrides: Sequence[str] = (),
    cli_named: Mapping[str, object] | None = None,
    env: Mapping[str, str] | None = None,
    extra_strict: bool = True,
) -> RunConfig:
    if not config_path.is_file():
        raise ConfigParseError(f"config file not found: {config_path}")
    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        raise ConfigParseError(f"YAML parse error in {config_path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ConfigParseError(f"top-level config must be a mapping; got {type(raw).__name__}")

    resolved_env = dict(env) if env is not None else dict(os.environ)
    interpolated = interpolate_env(raw, env=resolved_env)
    merged = apply_overrides(interpolated, overrides, cli_named=cli_named)
    try:
        return RunConfig.model_validate(merged)
    except ValidationError as exc:
        raise ConfigParseError(pretty_validation_error(exc, config_path=config_path)) from exc


def parse_override(pair: str) -> tuple[tuple[str, ...], Any]:
    if "=" not in pair:
        raise ConfigOverrideError(f"override missing '=': {pair!r}")
    key_str, value_str = pair.split("=", 1)
    key_path = tuple(part for part in key_str.split(".") if part)
    if not key_path:
        raise ConfigOverrideError(f"override has empty key path: {pair!r}")
    try:
        value = yaml.safe_load(value_str)
    except yaml.YAMLError as exc:
        raise ConfigOverrideError(f"could not parse value for {key_str!r}: {exc}") from exc
    return key_path, value


def apply_overrides(
    base: Mapping[str, Any],
    overrides: Sequence[str],
    *,
    cli_named: Mapping[str, object] | None = None,
) -> dict[str, Any]:
    result: dict[str, Any] = _deepcopy_mapping(base)
    for raw in overrides:
        path, value = parse_override(raw)
        _set_path(result, path, value)
    if cli_named:
        for key, value in cli_named.items():
            _set_path(result, tuple(key.split(".")), value)
    return result


def interpolate_env(
    obj: Any,
    *,
    env: Mapping[str, str],
    allow_undefined: bool = False,
) -> Any:
    if isinstance(obj, dict):
        return {k: interpolate_env(v, env=env, allow_undefined=allow_undefined) for k, v in obj.items()}
    if isinstance(obj, list):
        return [interpolate_env(v, env=env, allow_undefined=allow_undefined) for v in obj]
    if isinstance(obj, str):
        return _interpolate_string(obj, env=env, allow_undefined=allow_undefined)
    return obj


def _interpolate_string(s: str, *, env: Mapping[str, str], allow_undefined: bool) -> str:
    def _resolve(match: re.Match[str]) -> str:
        name = match.group(1)
        default = match.group(2)
        if name in env:
            return env[name]
        if default is not None:
            return default
        if allow_undefined:
            return match.group(0)
        raise EnvInterpolationError(f"env var ${{{name}}} is not set and has no default")

    return _ENV_PATTERN.sub(_resolve, s)


def pretty_validation_error(exc: ValidationError, *, config_path: Path) -> str:
    lines = [f"config validation failed for {config_path}:"]
    for err in exc.errors():
        loc = ".".join(str(p) for p in err.get("loc", ()))
        msg = err.get("msg", "")
        ctx_val = err.get("input")
        ctx_repr = repr(ctx_val)
        if len(ctx_repr) > 80:
            ctx_repr = ctx_repr[:77] + "..."
        lines.append(f"  - {loc or '<root>'} | {msg} | got {ctx_repr}")
    return "\n".join(lines)


# --- internals ---------------------------------------------------------


def _deepcopy_mapping(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _deepcopy_mapping(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_deepcopy_mapping(v) for v in obj]
    return obj


def _set_path(target: dict[str, Any], path: tuple[str, ...], value: Any) -> None:
    cursor: Any = target
    for part in path[:-1]:
        if not isinstance(cursor, dict):
            raise ConfigOverrideError(
                f"cannot apply override at path {'.'.join(path)!r}: encountered non-dict ancestor"
            )
        nxt = cursor.get(part)
        if not isinstance(nxt, dict):
            nxt = {}
            cursor[part] = nxt
        cursor = nxt
    if not isinstance(cursor, dict):
        raise ConfigOverrideError(
            f"cannot apply override at path {'.'.join(path)!r}: leaf parent is not a dict"
        )
    cursor[path[-1]] = value


__all__ = [
    "apply_overrides",
    "interpolate_env",
    "load_config",
    "parse_override",
    "pretty_validation_error",
]
