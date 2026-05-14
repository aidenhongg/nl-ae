"""Label extraction from Phase 1 rows for the six probe targets (C07).

Pure NumPy: no sklearn, no torch. Each label declares its own filtering rule
(see :func:`extract_label`); binary labels return ``y`` as a string array with
``classes = ("0", "1")``, multi-class labels return ``y`` as the sorted unique
values seen in the (fold-filtered) row set.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from nl_ae.pilot.models import ProbeLabel
from nl_ae.schema.models import ResultRow

if TYPE_CHECKING:  # pragma: no cover
    import numpy as np


@dataclass(frozen=True)
class LabelExtraction:
    """Per-label extraction output for one fold's row set.

    ``valid_mask`` is over the input row order (length ``N``); ``y`` is over
    the valid subset (length ``N_valid``). Binary labels carry the fixed pair
    ``("0", "1")`` regardless of which class actually appears.
    """

    valid_mask: np.ndarray
    y: np.ndarray
    classes: tuple[str, ...]
    n_dropped: int
    is_binary: bool


def extract_label(rows: Sequence[ResultRow], label: ProbeLabel) -> LabelExtraction:
    """Apply the per-label filter + target encoding to ``rows``.

    Filter rules:

    * ``disagreement_flag``: keep rows with both ``first_token_letter`` and
      ``free_text_letter`` non-null. ``y = "1"`` when the two disagree, else
      ``"0"``. Binary.
    * ``first_token_correct``: keep rows with both ``gold_letter`` and
      ``first_token_letter`` non-null (MMLU-only in practice). ``y = "1"`` iff
      ``first_token_letter == gold_letter``. Binary.
    * ``free_text_correct``: as above but with ``free_text_letter``. Binary.
    * ``first_token_letter``: keep rows with non-null ``first_token_letter``;
      ``y`` is the letter. Multi-class.
    * ``free_text_letter``: as above but with ``free_text_letter``. Multi-class.
    * ``extractor_match_rule``: keep every row (every row carries a rule);
      ``y`` is the rule string. Multi-class.
    """
    import numpy as np

    n = len(rows)
    mask = np.zeros(n, dtype=bool)
    values: list[str] = []
    binary = label in {"disagreement_flag", "first_token_correct", "free_text_correct"}

    if label == "disagreement_flag":
        for i, row in enumerate(rows):
            if row.first_token_letter is None or row.free_text_letter is None:
                continue
            mask[i] = True
            values.append("1" if row.first_token_letter != row.free_text_letter else "0")
    elif label == "first_token_correct":
        for i, row in enumerate(rows):
            if row.gold_letter is None or row.first_token_letter is None:
                continue
            mask[i] = True
            values.append("1" if row.first_token_letter == row.gold_letter else "0")
    elif label == "free_text_correct":
        for i, row in enumerate(rows):
            if row.gold_letter is None or row.free_text_letter is None:
                continue
            mask[i] = True
            values.append("1" if row.free_text_letter == row.gold_letter else "0")
    elif label == "first_token_letter":
        for i, row in enumerate(rows):
            if row.first_token_letter is None:
                continue
            mask[i] = True
            values.append(row.first_token_letter)
    elif label == "free_text_letter":
        for i, row in enumerate(rows):
            if row.free_text_letter is None:
                continue
            mask[i] = True
            values.append(row.free_text_letter)
    elif label == "extractor_match_rule":
        for i, row in enumerate(rows):
            mask[i] = True
            values.append(row.extractor_match_rule)
    else:  # pragma: no cover — exhaustive vs. ProbeLabel literal
        raise ValueError(f"unknown probe label: {label!r}")

    y = np.asarray(values, dtype=object)
    if binary:
        classes: tuple[str, ...] = ("0", "1")
    else:
        classes = tuple(sorted({str(v) for v in values}))
    n_dropped = int(n - mask.sum())
    return LabelExtraction(
        valid_mask=mask,
        y=y,
        classes=classes,
        n_dropped=n_dropped,
        is_binary=binary,
    )


__all__ = ["LabelExtraction", "extract_label"]
