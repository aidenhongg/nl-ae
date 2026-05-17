"""Pure first-token scoring core + prompt-tail letter-variant resolver.

Extracted from :meth:`Qwen25Wrapper.score_first_token` so the numerics are
unit-testable with a tiny synthetic vocab (no weights, no GPU). ``torch`` is
imported lazily inside the function; the module itself imports nothing from the
``nl_ae.prompt`` layer at runtime (only under ``TYPE_CHECKING``) so
``letter_tokens`` can re-export :func:`resolve_letter_variant` without a cycle.

Two defects motivated this module (see ``firsttoken-fix.md``):

* **fp16 full-vocab softmax underflow.** ``torch.softmax`` returns the input's
  dtype; over Qwen2.5's ~152k-token vocabulary the legitimate per-letter
  probabilities are tiny and flush to *exactly* ``0.0`` in fp16, which also
  made ``argmax`` of the all-zero vector return index 0 (always ``"A"``).
  :func:`score_letters_from_logits` computes in fp32 via the log-domain and
  takes ``argmax`` from the *logits*, never the probabilities.
* **letter-variant / position mismatch.** :func:`resolve_letter_variant`
  picks the variant the model actually emits as its first token from the
  rendered prompt's tail.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from nl_ae.schema.models import FirstTokenScoringMath

if TYPE_CHECKING:  # pragma: no cover - typing only; avoids a prompt->inference cycle.
    import torch

    from nl_ae.prompt.letter_tokens import LetterVariant


def score_letters_from_logits(
    logits: torch.Tensor,
    letter_ids: torch.Tensor,
    *,
    scoring_math: FirstTokenScoringMath,
) -> tuple[int, list[float | None], float, bool]:
    """Score the candidate letters from a next-token ``logits`` vector.

    Returns ``(argmax_local_idx, per_letter_prob, total_letter_mass,
    probs_valid)`` where ``argmax_local_idx`` indexes into ``letter_ids``.

    * ``argmax`` is **always** ``argmax(logits[letter_ids])`` — monotone in the
      logits and never re-coupled to softmax numerics, so ``first_token_letter``
      stays correct even when the true full-vocab letter mass is genuinely
      ~1e-9 or a future numeric regression reappears.
    * ``full_vocab_softmax``: ``log_softmax(logits.float())`` then ``.exp()`` —
      identical to a well-conditioned ``softmax`` but the log-domain path keeps
      small probabilities accurate instead of flushing them to zero. The total
      mass may be ``<< 1`` (that is the real measurement) but is never a
      spurious ``0.0``.
    * ``renormalize_over_letters``: fp32 softmax over the letter logits only;
      total mass is ``1.0`` by construction.
    * ``argmax_logits_only``: ``per_letter_prob`` is ``[None] * k``,
      ``total_letter_mass`` is ``0.0``, ``probs_valid`` is ``False`` (so
      ``ResultRow._check_invariants`` still holds — UR6).

    Probabilities are returned as Python ``float`` (float64), clamped to
    ``[0.0, 1.0]``.
    """
    import torch

    if letter_ids.numel() == 0:
        raise ValueError("letter_ids is empty")

    letter_logits = logits[letter_ids]
    argmax_idx = int(torch.argmax(letter_logits).item())
    k = int(letter_ids.numel())

    if scoring_math == "argmax_logits_only":
        return argmax_idx, [None] * k, 0.0, False

    if scoring_math == "renormalize_over_letters":
        probs = torch.softmax(letter_logits.float(), dim=0)
        per_letter = [_clamp01(float(p)) for p in probs.tolist()]
        return argmax_idx, list(per_letter), _clamp01(float(sum(per_letter))), True

    # full_vocab_softmax: log-domain in fp32 keeps tiny letter masses accurate.
    log_probs = torch.log_softmax(logits.float(), dim=0)
    probs = log_probs[letter_ids].exp()
    per_letter = [_clamp01(float(p)) for p in probs.tolist()]
    total_mass = _clamp01(float(probs.sum().item()))
    return argmax_idx, list(per_letter), total_mass, True


def resolve_letter_variant(prompt: str) -> LetterVariant:
    """Resolve which letter-token variant the model emits as its *next* token.

    The scored token is the model's first generated token after ``prompt``:

    * a trailing space (``"...Answer: "`` — non-chat ``trailing="answer_colon"``)
      means the next token is the space-prefixed letter → ``"leading_space"``;
    * otherwise (``"...Answer:"``, or a chat tail ``"...assistant\\n"``) the
      next token is the **bare** letter → ``"bare"``.

    ``"newline_prefixed"`` is intentionally unreachable here: a trailing ``\\n``
    is already consumed as a prompt token, so the next token is the bare
    letter. It stays defined for non-chat completeness.
    """
    if prompt.endswith(" "):
        return "leading_space"
    return "bare"


def _clamp01(x: float) -> float:
    return min(max(x, 0.0), 1.0)


__all__ = ["resolve_letter_variant", "score_letters_from_logits"]
