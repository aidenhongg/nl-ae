"""Prompt rendering — safe-format slot walker; never ``str.format``.

The final prompt is NFC-normalized, ``\\n``-newlined, and SHA-256-hashed in
the same byte form the scorer consumes.
"""

from __future__ import annotations

from typing import Literal, Protocol

from nl_ae.data.canonical import PermutedItem
from nl_ae.data.text_norm import nfc, sha256_hex_bytes
from nl_ae.schema.models import PromptTemplateRecord, Sha256Hex

from .template_registry import TemplateRecord, TemplateRegistry


class ChatTemplateAdapter(Protocol):
    """Wraps a user-message string into the model's chat-formatted prompt."""

    def apply(
        self, *, system: str | None, user: str, add_generation_prompt: bool = True
    ) -> str: ...

    @property
    def identity(self) -> Sha256Hex: ...


class NullChatTemplateAdapter:
    """No-op adapter for flat-template runs and tests."""

    def __init__(self, identity_hash: Sha256Hex = "0" * 64) -> None:
        self._identity = identity_hash

    def apply(
        self, *, system: str | None, user: str, add_generation_prompt: bool = True
    ) -> str:
        if system:
            return f"{system}\n\n{user}"
        return user

    @property
    def identity(self) -> Sha256Hex:
        return self._identity


ChoiceBlockStyle = Literal["A. text", "(A) text", "A) text"]
ChoiceSeparator = Literal["\n", "\n\n"]
TrailingPolicy = Literal["answer_colon", "answer_letter_only", "none"]


def _format_choice(style: ChoiceBlockStyle, letter: str, text: str) -> str:
    if style == "A. text":
        return f"{letter}. {text}"
    if style == "(A) text":
        return f"({letter}) {text}"
    if style == "A) text":
        return f"{letter}) {text}"
    raise ValueError(f"unknown choice_block_style: {style!r}")


def _format_trailing(policy: TrailingPolicy) -> str:
    if policy == "answer_colon":
        return "\nAnswer:"
    if policy == "answer_letter_only":
        return "\nAnswer (letter only):"
    if policy == "none":
        return ""
    raise ValueError(f"unknown trailing policy: {policy!r}")


def _safe_substitute(template_text: str, slots: dict[str, str]) -> str:
    """Walk ``{slot}`` occurrences without invoking ``str.format``.

    Any reference to an unknown slot is left as-is so the registry validator
    catches it earlier; this function only substitutes what it knows.
    """
    out_parts: list[str] = []
    i = 0
    n = len(template_text)
    while i < n:
        ch = template_text[i]
        if ch == "{":
            close = template_text.find("}", i + 1)
            if close == -1:
                out_parts.append(template_text[i:])
                break
            key = template_text[i + 1 : close]
            if key in slots:
                out_parts.append(slots[key])
            else:
                out_parts.append(template_text[i : close + 1])
            i = close + 1
        else:
            out_parts.append(ch)
            i += 1
    return "".join(out_parts)


class PromptRenderer:
    def __init__(
        self,
        registry: TemplateRegistry,
        *,
        chat_adapter: ChatTemplateAdapter | None = None,
        choice_block_style: ChoiceBlockStyle = "A. text",
        choice_separator: ChoiceSeparator = "\n",
        trailing: TrailingPolicy = "answer_colon",
        default_system: str | None = None,
    ) -> None:
        self._registry = registry
        self._chat_adapter = chat_adapter
        self._style = choice_block_style
        self._separator = choice_separator
        self._trailing = trailing
        self._default_system = default_system

    def render(self, perm_item: PermutedItem, template_id: str) -> tuple[str, Sha256Hex]:
        rec = self._registry[template_id]
        body = self._render_user_text(perm_item, rec)
        if self._chat_adapter is not None and rec.role != "system":
            final = self._chat_adapter.apply(
                system=self._default_system, user=body, add_generation_prompt=True
            )
        else:
            final = body
        final = nfc(final)
        return final, sha256_hex_bytes(final.encode("utf-8"))

    def render_components(
        self, perm_item: PermutedItem, template_id: str
    ) -> tuple[str | None, str, str, Sha256Hex]:
        """Return ``(system, user, final, prompt_hash)``."""
        rec = self._registry[template_id]
        body = self._render_user_text(perm_item, rec)
        if self._chat_adapter is not None and rec.role != "system":
            final = self._chat_adapter.apply(
                system=self._default_system, user=body, add_generation_prompt=True
            )
        else:
            final = body
        final = nfc(final)
        return self._default_system, body, final, sha256_hex_bytes(final.encode("utf-8"))

    def emit_template_records(self) -> list[PromptTemplateRecord]:
        return [
            PromptTemplateRecord(
                template_id=rec.template_id,
                template_content_hash=rec.template_content_hash,
                template_text=rec.template_text,
                role=rec.role,
            )
            for rec in self._registry.all_records()
        ]

    # ----------------------------------------------------------------

    def _render_user_text(self, perm_item: PermutedItem, rec: TemplateRecord) -> str:
        choice_block = self._separator.join(
            _format_choice(self._style, letter, text)
            for letter, text in zip(perm_item.letters, perm_item.choices_in_order, strict=True)
        )
        letter_set = ",".join(perm_item.letters)
        slots = {
            "question": perm_item.base.question,
            "choice_block": choice_block,
            "letter_set": letter_set,
            "system": self._default_system or "",
        }
        rendered = _safe_substitute(rec.template_text, slots)
        rendered = rendered.rstrip()
        rendered += _format_trailing(self._trailing)
        return rendered


__all__ = [
    "ChatTemplateAdapter",
    "ChoiceBlockStyle",
    "ChoiceSeparator",
    "NullChatTemplateAdapter",
    "PromptRenderer",
    "TrailingPolicy",
]
