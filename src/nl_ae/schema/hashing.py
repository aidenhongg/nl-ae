"""SHA-256 hashing conventions used across the codebase.

Every text hash is computed over **UTF-8 NFC-normalized** bytes (newlines
already normalized by the caller). File hashes stream by chunk.
"""

from __future__ import annotations

import hashlib
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

from .models import Sha256Hex


def _sha256_hex(b: bytes) -> Sha256Hex:
    return hashlib.sha256(b).hexdigest()


def hash_prompt_text(text: str) -> Sha256Hex:
    """SHA-256 over NFC-normalized UTF-8 bytes."""
    nfc = unicodedata.normalize("NFC", text)
    return _sha256_hex(nfc.encode("utf-8"))


def hash_file(path: Path, *, chunk: int = 1 << 20) -> Sha256Hex:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            block = f.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def hash_json_bytes(b: bytes) -> Sha256Hex:
    return _sha256_hex(b)


def now_utc_iso() -> str:
    """RFC3339-compact UTC timestamp ending in ``Z`` (matches ``IsoUtcStr``)."""
    now = datetime.now(timezone.utc).replace(microsecond=0)
    return now.strftime("%Y-%m-%dT%H:%M:%SZ")


def make_run_id(started_at: str, git_sha: str, slug: str | None = None) -> str:
    """``<UTC-ISO8601-compact>-<git_sha7>[-<slug>]`` (lexicographic sort by start)."""
    compact = started_at.replace("-", "").replace(":", "").replace("Z", "Z")
    short = git_sha[:7]
    parts = [compact, short]
    if slug:
        parts.append(slug)
    return "-".join(parts)


__all__ = [
    "hash_file",
    "hash_json_bytes",
    "hash_prompt_text",
    "make_run_id",
    "now_utc_iso",
]
