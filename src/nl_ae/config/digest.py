"""Canonical SHA-256 ``config_digest`` (CI.19 — REQUIRED on RunManifest)."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Final

from nl_ae.schema.models import Sha256Hex

from .schema import RunConfig

CONFIG_DIGEST_EXCLUSIONS: Final[frozenset[str]] = frozenset(
    {
        "run_identity.run_id",
        "run_identity.slug",
        "run_identity.notes",
        "run_identity.tags",
        "output.figure_dpi",
        "output.derive_parquet_on_finalize",
        "logging.console_level",
        "logging.file_level",
        "logging.per_module_levels",
        "logging.jsonl_destination",
    }
)


def _drop_path(obj: Any, path: tuple[str, ...]) -> Any:
    if not path:
        return obj
    head, *rest = path
    if isinstance(obj, dict) and head in obj:
        if not rest:
            obj = {k: v for k, v in obj.items() if k != head}
        else:
            obj[head] = _drop_path(obj[head], tuple(rest))
    return obj


def _coerce(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _coerce(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_coerce(v) for v in value]
    if hasattr(value, "as_posix"):  # Path
        return value.as_posix()
    if hasattr(value, "model_dump"):
        return _coerce(value.model_dump(mode="json"))
    return value


def canonical_payload(cfg: RunConfig) -> dict[str, Any]:
    raw = cfg.model_dump(mode="json")
    raw = _coerce(raw)
    for excl in CONFIG_DIGEST_EXCLUSIONS:
        raw = _drop_path(raw, tuple(excl.split(".")))
    return raw


def canonical_dump_bytes(cfg: RunConfig) -> bytes:
    payload = canonical_payload(cfg)
    return json.dumps(
        payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")


def compute_config_digest(cfg: RunConfig) -> Sha256Hex:
    return hashlib.sha256(canonical_dump_bytes(cfg)).hexdigest()


__all__ = [
    "CONFIG_DIGEST_EXCLUSIONS",
    "canonical_dump_bytes",
    "canonical_payload",
    "compute_config_digest",
]
