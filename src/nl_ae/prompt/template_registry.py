"""Template files live as ``*.txt`` in a single directory.

Each template hashes over **strip_bom → utf8_decode → \\n-normalize → NFC →
sha256(utf-8 bytes)** (C02 D3.3). The hash is recorded on each row and on
the manifest so cross-run joins can filter by template content.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, StringConstraints

from nl_ae.data.text_norm import nfc, normalize_newlines, sha256_hex_bytes, strip_bom
from nl_ae.schema.models import Sha256Hex

TemplateIdStr = Annotated[
    str, StringConstraints(min_length=1, max_length=64, pattern=r"^[a-z0-9_\-\.]+$")
]
TemplateRole = Literal["system", "user", "assistant", "composite"]
ALLOWED_SLOTS: frozenset[str] = frozenset({"question", "choice_block", "system", "letter_set"})

_SLOT_RE = re.compile(r"\{([a-z_]+)\}")


class TemplateRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    template_id: TemplateIdStr
    template_content_hash: Sha256Hex
    template_text: str
    role: TemplateRole
    slots: frozenset[str]
    source_path: Path


def _detect_role(stem: str, body: str) -> TemplateRole:
    head = body.lstrip().lower()
    if stem.endswith(("_system", "-system")) or head.startswith(("system:", "# system")):
        return "system"
    if "{system}" in body:
        return "composite"
    return "user"


def _normalize_template_bytes(path: Path) -> tuple[str, Sha256Hex]:
    raw = path.read_bytes()
    text = raw.decode("utf-8")
    text = strip_bom(text)
    text = normalize_newlines(text, target="\n")
    text = nfc(text)
    return text, sha256_hex_bytes(text.encode("utf-8"))


class TemplateRegistry:
    """Lazy, dict-like view of ``templates_dir/*.txt``."""

    def __init__(self, templates_dir: Path) -> None:
        if not templates_dir.is_dir():
            raise FileNotFoundError(f"templates_dir does not exist or is not a dir: {templates_dir}")
        self._templates_dir = templates_dir
        self._records: dict[str, TemplateRecord] = {}
        self._loaded = False

    def load(self) -> None:
        if self._loaded:
            return
        for path in sorted(self._templates_dir.glob("*.txt")):
            stem = path.stem
            text, content_hash = _normalize_template_bytes(path)
            slots = frozenset(_SLOT_RE.findall(text))
            unknown = slots - ALLOWED_SLOTS
            if unknown:
                raise ValueError(
                    f"template {path.name} declares unknown slots: {sorted(unknown)} — "
                    f"allowed: {sorted(ALLOWED_SLOTS)}"
                )
            role = _detect_role(stem, text)
            self._records[stem] = TemplateRecord(
                template_id=stem,
                template_content_hash=content_hash,
                template_text=text,
                role=role,
                slots=slots,
                source_path=path,
            )
        self._loaded = True

    def __getitem__(self, template_id: str) -> TemplateRecord:
        self.load()
        if template_id not in self._records:
            raise KeyError(
                f"template {template_id!r} not found in {self._templates_dir} "
                f"(available: {sorted(self._records)})"
            )
        return self._records[template_id]

    def __contains__(self, template_id: object) -> bool:
        self.load()
        return isinstance(template_id, str) and template_id in self._records

    def ids(self) -> tuple[str, ...]:
        self.load()
        return tuple(sorted(self._records))

    def all_records(self) -> tuple[TemplateRecord, ...]:
        self.load()
        return tuple(self._records[k] for k in sorted(self._records))


__all__ = [
    "ALLOWED_SLOTS",
    "TemplateIdStr",
    "TemplateRecord",
    "TemplateRegistry",
    "TemplateRole",
]
