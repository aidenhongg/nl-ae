"""Pure first-token scoring core + variant resolver.

Model-free: ``torch`` is a light import (no weights, no GPU, no network) gated
behind :func:`pytest.importorskip` so the suite still runs on a GPU-less box.
The resolver test needs no torch at all.

The headline test pins the production bug (``firsttoken-fix.md`` §1): a fp16
full-vocab softmax over a vocabulary with a dominating non-letter token flushes
every per-letter probability to *exactly* ``0.0``, and ``argmax`` of the
all-zero vector returns index 0 → ``first_token_letter == "A"`` for every row.
"""

from __future__ import annotations

import pytest

from nl_ae.inference.scoring import resolve_letter_variant, score_letters_from_logits

torch = pytest.importorskip("torch")

# Verbatim A,B,C,D first-token logits from rows.jsonl line 1 of the corrupt run
# runs/20260515T023617Z-7830851-mvp/ (B >> A >> C ~ D).
ROW1_LOGITS = [11.375, 17.921875, 5.234375, 4.7734375]
ROW1_IDS = [362, 425, 356, 422]  # leading_space token ids A,B,C,D


def _build_fp16_vocab(*, peak: float, vocab: int = 4096):
    """fp16 logits: a flat floor, one dominating non-letter token, the 4 letters.

    ``peak`` is the dominating non-letter logit. The plan's illustrative ``+30``
    only underflows A/C/D — the *winning* letter B (17.92) survives, so it does
    not reproduce the observed all-zero symptom. ``+40`` puts every letter logit
    > ~17 nats below the max, which is exactly the real situation (the true
    argmax in the run is the *bare* "B" token the scorer never reads), so even
    B flushes to 0.0 in fp16.
    """
    logits = torch.full((vocab,), -30.0, dtype=torch.float16)
    logits[7] = peak  # a non-letter token dominates the distribution.
    ids = torch.tensor(ROW1_IDS, dtype=torch.long)
    for tid, lg in zip(ids.tolist(), ROW1_LOGITS, strict=True):
        logits[tid] = lg
    return logits, ids


def test_fp16_full_vocab_underflow_regression() -> None:
    logits, ids = _build_fp16_vocab(peak=40.0)

    # 1. Pin the bug: the old in-place arithmetic underflows to all-zero and its
    #    argmax is index 0 ("A") regardless of the model's real prediction.
    old_probs = torch.softmax(logits, dim=0)[ids]
    assert bool((old_probs == 0).all())
    assert int(torch.argmax(old_probs).item()) == 0  # spurious "A"

    # 2. The fix: argmax tracks the logits (B, index 1), probabilities are tiny
    #    but strictly positive, the total mass is real (<<1, never a spurious 0).
    argmax_idx, probs, total, valid = score_letters_from_logits(
        logits, ids, scoring_math="full_vocab_softmax"
    )
    assert argmax_idx == 1  # "B" — the true max letter logit
    assert valid is True
    assert all(p is not None and p > 0.0 for p in probs)
    assert 0.0 < total <= 1.0
    # Monotone in the logits: B > A > C > D.
    assert probs[1] > probs[0] > probs[2] > probs[3]


def test_fp16_vs_fp32_parity_well_conditioned() -> None:
    """For a non-underflowing case, fp16-input and fp32-input agree to ~1e-6."""
    base = torch.tensor([2.0, 5.0, 1.0, 0.5, 3.0, 4.0, 0.0, 2.5])
    ids = torch.tensor([1, 5, 4, 3], dtype=torch.long)

    _, p16, t16, _ = score_letters_from_logits(
        base.to(torch.float16), ids, scoring_math="full_vocab_softmax"
    )
    _, p32, t32, _ = score_letters_from_logits(
        base.to(torch.float32), ids, scoring_math="full_vocab_softmax"
    )
    assert t16 == pytest.approx(t32, abs=1e-6)
    for a, b in zip(p16, p32, strict=True):
        assert a == pytest.approx(b, abs=1e-6)


def test_argmax_is_logits_not_probs() -> None:
    """argmax is the max *letter logit* even when full-vocab probs all underflow."""
    logits, ids = _build_fp16_vocab(peak=45.0)
    argmax_idx, probs, total, _ = score_letters_from_logits(
        logits, ids, scoring_math="full_vocab_softmax"
    )
    # probs are vanishingly small (fp16-underflow territory) yet argmax still
    # equals argmax(letter_logits) — it is never read off the probabilities.
    expected = int(torch.argmax(logits[ids]).item())
    assert argmax_idx == expected == 1
    assert all(p is not None and p >= 0.0 for p in probs)
    assert total >= 0.0


def test_renormalize_over_letters_sums_to_one() -> None:
    logits = torch.tensor(ROW1_LOGITS, dtype=torch.float16)
    ids = torch.arange(4, dtype=torch.long)
    argmax_idx, probs, total, valid = score_letters_from_logits(
        logits, ids, scoring_math="renormalize_over_letters"
    )
    assert valid is True
    assert argmax_idx == 1  # max logit = B
    assert sum(p for p in probs if p is not None) == pytest.approx(1.0, abs=1e-5)
    assert total == pytest.approx(1.0, abs=1e-5)


def test_argmax_logits_only_emits_no_probs() -> None:
    logits = torch.tensor(ROW1_LOGITS, dtype=torch.float32)
    ids = torch.arange(4, dtype=torch.long)
    argmax_idx, probs, total, valid = score_letters_from_logits(
        logits, ids, scoring_math="argmax_logits_only"
    )
    assert argmax_idx == 1
    assert probs == [None, None, None, None]
    assert total == 0.0
    assert valid is False


def test_argmax_logits_only_still_satisfies_resultrow_invariants() -> None:
    """Regression: the argmax_logits_only shape must keep ResultRow valid (UR6)."""
    from nl_ae.schema.hashing import now_utc_iso
    from nl_ae.schema.models import LetterSoftmaxEntry, ResultRow

    logits = torch.tensor(ROW1_LOGITS, dtype=torch.float32)
    ids = torch.arange(4, dtype=torch.long)
    argmax_idx, probs, total, valid = score_letters_from_logits(
        logits, ids, scoring_math="argmax_logits_only"
    )
    letters = "ABCD"
    softmax = [
        LetterSoftmaxEntry(
            letter=letters[i],
            token_id=ROW1_IDS[i],
            prob=probs[i],
            prob_valid=valid,
            logit=ROW1_LOGITS[i],
        )
        for i in range(4)
    ]
    row = ResultRow(
        run_id="r",
        item_id="mmlu/v1/x/q-1",
        dataset_name="mmlu",
        dataset_split="test",
        template_id="mcq_flat_v1",
        permutation_id=0,
        prompt_hash="a" * 64,
        gold_letter="A",
        first_token_letter=letters[argmax_idx],
        free_text_letter=None,
        free_text_raw="",
        agreement_flag=None,
        letter_softmax=softmax,
        n_options=4,
        free_text_seed=None,
        decode_strategy="greedy",
        created_at=now_utc_iso(),
        extractor_id="regex_ladder",
        extractor_match_rule="unparseable",
        first_token_scoring_math="argmax_logits_only",
        total_letter_mass=total,
    )
    assert row.first_token_letter == "B"


def test_empty_letter_ids_raises() -> None:
    logits = torch.zeros(8, dtype=torch.float32)
    with pytest.raises(ValueError, match="letter_ids is empty"):
        score_letters_from_logits(
            logits, torch.tensor([], dtype=torch.long), scoring_math="full_vocab_softmax"
        )


@pytest.mark.parametrize(
    ("prompt", "expected"),
    [
        ("Question?\n\nA. x\nB. y\nAnswer:", "bare"),  # non-chat answer_colon
        ("...<|im_start|>assistant\n", "bare"),  # chat tail
        ("Question?\n\nA. x\nB. y\nAnswer: ", "leading_space"),  # trailing space
        ("trailing newline is consumed\n", "bare"),  # newline_prefixed unreachable
    ],
)
def test_resolve_letter_variant(prompt: str, expected: str) -> None:
    assert resolve_letter_variant(prompt) == expected
