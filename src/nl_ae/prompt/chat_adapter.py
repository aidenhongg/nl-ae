"""HuggingFace chat-template adapter, shared by ``nlae eval`` and ``nlae materialize-prompts``.

Wraps ``tokenizer.apply_chat_template`` so the renderer can stay decoupled from
HuggingFace specifics. The :func:`make_chat_adapter` factory falls back to
:class:`NullChatTemplateAdapter` when chat formatting is disabled or the
tokenizer doesn't speak the chat-template protocol.
"""

from __future__ import annotations

from typing import Any

from nl_ae.schema.models import Sha256Hex

from .renderer import ChatTemplateAdapter, NullChatTemplateAdapter


class HFChatAdapter:
    """Adapter that delegates to ``tokenizer.apply_chat_template``."""

    def __init__(self, tokenizer: Any, *, identity_hash: Sha256Hex) -> None:
        self._tokenizer = tokenizer
        self._identity = identity_hash

    def apply(
        self,
        *,
        system: str | None,
        user: str,
        add_generation_prompt: bool = True,
    ) -> str:
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": user})
        rendered = self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )
        return str(rendered)

    @property
    def identity(self) -> Sha256Hex:
        return self._identity


def make_chat_adapter(
    tokenizer: Any,
    *,
    identity_hash: Sha256Hex,
    enabled: bool = True,
) -> ChatTemplateAdapter:
    """Return an :class:`HFChatAdapter` when chat is enabled and supported.

    Falls back to :class:`NullChatTemplateAdapter` (passthrough with the same
    ``identity_hash``) when ``enabled`` is False or the tokenizer lacks
    ``apply_chat_template``.
    """
    if not enabled or not hasattr(tokenizer, "apply_chat_template"):
        return NullChatTemplateAdapter(identity_hash=identity_hash)
    return HFChatAdapter(tokenizer, identity_hash=identity_hash)


__all__ = ["HFChatAdapter", "make_chat_adapter"]
