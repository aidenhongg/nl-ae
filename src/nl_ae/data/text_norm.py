"""Text normalization + hashing primitives.

Every text hash in nl-ae is computed over **NFC-normalized, BOM-stripped,
LF-newlined UTF-8 bytes**. Keep these helpers centralized so callers cannot
drift.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata

from nl_ae.schema.models import ItemIdStr, Sha256Hex

_BOM = "﻿"
_NON_NEWLINE_CR = re.compile(r"\r\n?")
_SAFE_ID_CHARS = re.compile(r"[^A-Za-z0-9._:\-]")


def nfc(s: str) -> str:
    return unicodedata.normalize("NFC", s)


def strip_bom(s: str) -> str:
    return s.lstrip(_BOM)


def normalize_newlines(s: str, *, target: str = "\n") -> str:
    if target == "\n":
        return _NON_NEWLINE_CR.sub("\n", s)
    return _NON_NEWLINE_CR.sub(target, s)


def sha256_hex(text: str) -> Sha256Hex:
    return hashlib.sha256(nfc(text).encode("utf-8")).hexdigest()


def sha256_hex_bytes(b: bytes) -> Sha256Hex:
    return hashlib.sha256(b).hexdigest()


def safe_id_component(s: str, *, max_len: int = 64) -> str:
    """Coerce free text into the ``ItemIdStr`` charset."""
    cleaned = _SAFE_ID_CHARS.sub("-", s).strip("-")
    if not cleaned:
        cleaned = "x"
    return cleaned[:max_len]


def derive_item_id(*, prefix: str, payload: str, short_n: int = 12) -> ItemIdStr:
    """``<prefix>/q-<sha256(nfc(payload))[:short_n]>``."""
    digest = sha256_hex(payload)
    return f"{prefix}/q-{digest[:short_n]}"


__all__ = [
    "derive_item_id",
    "nfc",
    "normalize_newlines",
    "safe_id_component",
    "sha256_hex",
    "sha256_hex_bytes",
    "strip_bom",
]
