"""C02 — prompt rendering and letter-token identity."""

from .identity import PromptIdentity
from .letter_tokens import (
    LetterVariant,
    LetterVariantPolicy,
    TokenizerLike,
    assert_single_token_per_letter_per_variant,
    build_letter_token_table,
    resolve_letter_variant,
    select_canonical_variant,
)
from .materialize import MaterializeOutcome, materialize_prompts
from .renderer import ChatTemplateAdapter, NullChatTemplateAdapter, PromptRenderer
from .template_registry import TemplateRecord, TemplateRegistry

__all__ = [
    "ChatTemplateAdapter",
    "LetterVariant",
    "LetterVariantPolicy",
    "MaterializeOutcome",
    "NullChatTemplateAdapter",
    "PromptIdentity",
    "PromptRenderer",
    "TemplateRecord",
    "TemplateRegistry",
    "TokenizerLike",
    "assert_single_token_per_letter_per_variant",
    "build_letter_token_table",
    "materialize_prompts",
    "resolve_letter_variant",
    "select_canonical_variant",
]
