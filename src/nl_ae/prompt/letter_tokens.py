"""Letter-token resolution.

Per CI.01 / CI.09: each letter gets one entry per variant. The scorer
filters to whichever variant matches the rendered prompt's tail (typically
``leading_space`` after ``"\\nAnswer: "`` or ``bare`` after no trailing space).
"""

from __future__ import annotations

import logging
from typing import Literal, Protocol

from nl_ae.inference.scoring import resolve_letter_variant
from nl_ae.schema.models import LetterStr, LetterTokenEntry

LOG = logging.getLogger(__name__)

LetterVariant = Literal["bare", "leading_space", "newline_prefixed"]
# Policy accepted by ``EvalConfig.letter_variant`` / ``score_and_generate``:
# ``"auto"`` resolves the concrete variant per-prompt via
# :func:`resolve_letter_variant`; the others pin it explicitly.
LetterVariantPolicy = Literal["auto", "bare", "leading_space", "newline_prefixed"]
_VARIANT_PREFIX: dict[LetterVariant, str] = {
    "bare": "",
    "leading_space": " ",
    "newline_prefixed": "\n",
}


class TokenizerLike(Protocol):
    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]: ...

    def decode(self, ids: list[int]) -> str: ...

    @property
    def name_or_path(self) -> str: ...


def build_letter_token_table(
    tokenizer: TokenizerLike,
    letters: tuple[LetterStr, ...],
    *,
    variants: tuple[LetterVariant, ...] = ("bare", "leading_space"),
) -> list[LetterTokenEntry]:
    table: list[LetterTokenEntry] = []
    for letter in letters:
        for variant in variants:
            prefix = _VARIANT_PREFIX[variant]
            payload = f"{prefix}{letter}"
            ids = tokenizer.encode(payload, add_special_tokens=False)
            if not ids:
                LOG.warning("tokenizer returned empty ids for variant=%s letter=%s", variant, letter)
                continue
            if len(ids) > 1:
                # Some tokenizers split " A" as [" ", "A"]; pick the trailing letter token.
                if variant == "bare":
                    token_id = ids[0]
                else:
                    token_id = ids[-1]
                LOG.debug(
                    "multi-token letter %s variant=%s ids=%s; using id=%d",
                    letter,
                    variant,
                    ids,
                    token_id,
                )
            else:
                token_id = ids[0]
            token_str = tokenizer.decode([token_id])
            table.append(
                LetterTokenEntry(
                    letter=letter,
                    token_id=token_id,
                    variant=variant,
                    token_str=token_str,
                )
            )
    return table


def assert_single_token_per_letter_per_variant(table: list[LetterTokenEntry]) -> None:
    """Raise if any (letter, variant) pair is missing or duplicated."""
    seen: dict[tuple[str, str], int] = {}
    for entry in table:
        key = (entry.letter, entry.variant)
        if key in seen:
            raise ValueError(f"duplicate letter_token entry for {key}: ids {seen[key]} and {entry.token_id}")
        seen[key] = entry.token_id


def select_canonical_variant(
    table: list[LetterTokenEntry], *, chosen: LetterVariant
) -> list[LetterTokenEntry]:
    """Return only the entries matching ``chosen`` variant, preserving order."""
    return [e for e in table if e.variant == chosen]


__all__ = [
    "LetterVariant",
    "LetterVariantPolicy",
    "TokenizerLike",
    "assert_single_token_per_letter_per_variant",
    "build_letter_token_table",
    "resolve_letter_variant",
    "select_canonical_variant",
]
