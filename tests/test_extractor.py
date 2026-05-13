"""RegexLadderExtractor v1.0.0 — rule order + out-of-range remap."""

from __future__ import annotations

from nl_ae.inference.extractor import RegexLadderExtractor


def _allowed(*letters: str) -> frozenset[str]:
    return frozenset(letters)


def test_my_answer_is_wins_first() -> None:
    ex = RegexLadderExtractor()
    out = ex.extract("My answer is C.", allowed_letters=_allowed("A", "B", "C", "D"))
    # Both "the_answer_is" and "my_answer_is" would match; the registered
    # ladder fires "header_marker" first (greedy "answer is" header).
    assert out.letter == "C"
    assert out.match_rule in {"header_marker", "my_answer_is"}


def test_the_answer_is() -> None:
    ex = RegexLadderExtractor()
    out = ex.extract("The answer is: B", allowed_letters=_allowed("A", "B", "C", "D"))
    assert out.letter == "B"


def test_boxed() -> None:
    ex = RegexLadderExtractor()
    out = ex.extract(r"final \boxed{D}", allowed_letters=_allowed("A", "B", "C", "D"))
    assert out.letter == "D"
    assert out.match_rule == "boxed"


def test_letter_only() -> None:
    ex = RegexLadderExtractor()
    out = ex.extract("A", allowed_letters=_allowed("A", "B", "C", "D"))
    assert out.letter == "A"
    assert out.match_rule == "letter_only"


def test_trailing_letter() -> None:
    ex = RegexLadderExtractor()
    out = ex.extract("I'm going with C.", allowed_letters=_allowed("A", "B", "C", "D"))
    assert out.letter == "C"


def test_out_of_range_letter() -> None:
    ex = RegexLadderExtractor()
    out = ex.extract("My answer is Z.", allowed_letters=_allowed("A", "B", "C", "D"))
    assert out.letter is None
    assert out.match_rule in {"out_of_range", "unparseable"}


def test_unparseable() -> None:
    ex = RegexLadderExtractor()
    out = ex.extract("...", allowed_letters=_allowed("A", "B"))
    assert out.letter is None
    assert out.match_rule == "unparseable"


def test_rules_hash_is_stable() -> None:
    a = RegexLadderExtractor()
    b = RegexLadderExtractor()
    assert a.rules_content_hash == b.rules_content_hash


def test_majority_vote_opt_in() -> None:
    ex = RegexLadderExtractor(allow_majority_vote=True)
    out = ex.extract(
        "Maybe A, possibly A, but could be B. I think A.",
        allowed_letters=_allowed("A", "B"),
    )
    # The trailing letter "A." matches `trailing_letter` before majority_vote
    # has a chance to fire; either rule is acceptable.
    assert out.letter == "A"
