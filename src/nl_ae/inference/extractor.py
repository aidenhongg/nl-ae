"""Free-text → letter extractor.

``RegexLadderExtractor v1.0.0`` is the default. Rules fire in order; first
match wins. Letters outside ``allowed_letters`` are remapped to
``out_of_range``.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from typing import Annotated, Final, Protocol

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

from nl_ae.schema.models import (
    ExtractorMatchRule,
    ExtractorRecord,
    LetterStr,
    Sha256Hex,
)

ExtractorIdStr = Annotated[str, StringConstraints(min_length=1, max_length=64)]


class ExtractorOutcome(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    letter: LetterStr | None
    match_rule: ExtractorMatchRule
    matched_span: tuple[Annotated[int, Field(ge=0)], Annotated[int, Field(ge=0)]] | None
    extractor_id: ExtractorIdStr
    extractor_version: Annotated[str, StringConstraints(pattern=r"^\d+\.\d+\.\d+$")]
    rules_content_hash: Sha256Hex


class AnswerExtractorProtocol(Protocol):
    @property
    def extractor_id(self) -> str: ...

    @property
    def extractor_version(self) -> str: ...

    @property
    def rules_content_hash(self) -> str: ...

    def extract(
        self, free_text: str, *, allowed_letters: frozenset[LetterStr]
    ) -> ExtractorOutcome: ...

    def record(self) -> ExtractorRecord: ...


_RULES_V1: tuple[tuple[ExtractorMatchRule, str], ...] = (
    ("header_marker", r"(?i)\b(?:the\s+)?answer\s*(?:is|[:=])\s*\(?([A-Za-z])\)?"),
    ("my_answer_is", r"(?i)\bmy\s+answer\s+is\s*[:\-]?\s*\(?([A-Za-z])\)?"),
    ("the_answer_is", r"(?i)\bthe\s+(?:correct\s+)?answer\s+is\s*[:\-]?\s*\(?([A-Za-z])\)?"),
    ("boxed", r"\\boxed\{\s*([A-Za-z])\s*\}"),
    ("letter_only", r"^\s*\(?([A-Za-z])\)?[\.\)\:]?\s*$"),
    ("trailing_letter", r"\b([A-Z])\b\s*[.!?]?\s*$"),
    ("leading_letter", r"^\s*\(?([A-Z])\)?[\.\)\:]"),
)
_MAJORITY_VOTE_MULTIPLIER: Final[float] = 1.5


def _rules_hash(rules: tuple[tuple[ExtractorMatchRule, str], ...]) -> Sha256Hex:
    payload = json.dumps([list(r) for r in rules], sort_keys=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


class RegexLadderExtractor:
    EXTRACTOR_ID: Final[str] = "regex_ladder"
    EXTRACTOR_VERSION: Final[str] = "1.0.0"

    def __init__(self, *, allow_majority_vote: bool = False) -> None:
        self._allow_majority_vote = allow_majority_vote
        self._rules = _RULES_V1
        self._compiled: tuple[tuple[ExtractorMatchRule, re.Pattern[str]], ...] = tuple(
            (name, re.compile(pat, flags=re.MULTILINE)) for name, pat in self._rules
        )
        self._rules_hash = _rules_hash(self._rules)

    @property
    def extractor_id(self) -> str:
        return self.EXTRACTOR_ID

    @property
    def extractor_version(self) -> str:
        return self.EXTRACTOR_VERSION

    @property
    def rules_content_hash(self) -> str:
        return self._rules_hash

    def record(self) -> ExtractorRecord:
        rules_in_order: tuple[str, ...] = tuple(name for name, _ in self._rules) + (
            ("majority_vote",) if self._allow_majority_vote else ()
        )
        return ExtractorRecord(
            extractor_id=self.EXTRACTOR_ID,
            extractor_version=self.EXTRACTOR_VERSION,
            rules_content_hash=self._rules_hash,
            rules_in_order=rules_in_order,
        )

    def extract(
        self, free_text: str, *, allowed_letters: frozenset[LetterStr]
    ) -> ExtractorOutcome:
        text = free_text.strip()
        if not text:
            return self._outcome(None, "unparseable", None)

        for name, pat in self._compiled:
            m = pat.search(text)
            if m:
                letter = m.group(1).upper()
                if letter in allowed_letters:
                    return self._outcome(letter, name, m.span(1))
                # Letter matched but is not in the allowed set: keep walking;
                # the regex ladder may find an in-range candidate.
                continue

        if self._allow_majority_vote:
            counts = Counter(
                m.group(1).upper()
                for m in re.finditer(r"\b([A-Z])\b", text)
                if m.group(1).upper() in allowed_letters
            )
            if counts:
                top_letter, top_count = counts.most_common(1)[0]
                runner_up = counts.most_common(2)[1][1] if len(counts) > 1 else 0
                if top_count >= max(1, runner_up * _MAJORITY_VOTE_MULTIPLIER):
                    return self._outcome(top_letter, "majority_vote", None)

        # See if any allowed-but-out-of-range letter was found at all.
        for _name, pat in self._compiled:
            m = pat.search(text)
            if m and m.group(1).upper().isalpha():
                return self._outcome(None, "out_of_range", m.span(1))

        return self._outcome(None, "unparseable", None)

    def _outcome(
        self,
        letter: LetterStr | None,
        rule: ExtractorMatchRule,
        span: tuple[int, int] | None,
    ) -> ExtractorOutcome:
        return ExtractorOutcome(
            letter=letter,
            match_rule=rule,
            matched_span=span,
            extractor_id=self.EXTRACTOR_ID,
            extractor_version=self.EXTRACTOR_VERSION,
            rules_content_hash=self._rules_hash,
        )


class ExtractorRegistry:
    def __init__(self) -> None:
        self._extractors: dict[str, AnswerExtractorProtocol] = {}

    def register(self, extractor: AnswerExtractorProtocol) -> None:
        self._extractors[extractor.extractor_id] = extractor

    def get(self, extractor_id: str) -> AnswerExtractorProtocol:
        if extractor_id not in self._extractors:
            raise KeyError(
                f"extractor {extractor_id!r} not registered (known: {sorted(self._extractors)})"
            )
        return self._extractors[extractor_id]

    def all_records(self) -> tuple[ExtractorRecord, ...]:
        return tuple(e.record() for e in self._extractors.values())


class ExtractorConfig(BaseModel):
    """Public config consumed by ``RunConfig``."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    extractor_id: ExtractorIdStr = "regex_ladder"
    allow_majority_vote: bool = False


def default_extractor() -> RegexLadderExtractor:
    return RegexLadderExtractor()


__all__ = [
    "AnswerExtractorProtocol",
    "ExtractorConfig",
    "ExtractorIdStr",
    "ExtractorOutcome",
    "ExtractorRegistry",
    "RegexLadderExtractor",
    "default_extractor",
]
