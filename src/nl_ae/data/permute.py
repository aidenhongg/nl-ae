"""Letter sets + deterministic permutations.

The seeded permutation seed derives from ``sha256(item_id|permutation_id)``
alone (UR-R2.7): there is no run-level permutation seed in ``SeedConfig``.
"""

from __future__ import annotations

import hashlib
import random
from typing import Literal

from nl_ae.schema.models import LetterStr

from .canonical import CanonicalItem, PermutedItem

LETTERS_26: tuple[LetterStr, ...] = tuple(chr(c) for c in range(ord("A"), ord("Z") + 1))


def letters_for(n: int) -> tuple[LetterStr, ...]:
    if n < 2 or n > 26:
        raise ValueError(f"n_options must be in [2, 26], got {n}")
    return LETTERS_26[:n]


PermutationMode = Literal["identity", "seeded", "enumerated"]


def _permutation_seed(item_id: str, permutation_id: int) -> int:
    payload = f"{item_id}|{permutation_id}".encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], "big")


def permutation_for(
    item: CanonicalItem,
    permutation_id: int,
    *,
    mode: PermutationMode = "seeded",
) -> PermutedItem:
    n = item.n_options
    if permutation_id < 0:
        raise ValueError(f"permutation_id must be >= 0, got {permutation_id}")
    if mode == "identity":
        perm = tuple(range(n))
    elif mode == "seeded":
        rng = random.Random(_permutation_seed(item.item_id, permutation_id))
        perm_list = list(range(n))
        rng.shuffle(perm_list)
        perm = tuple(perm_list)
    elif mode == "enumerated":
        # Lexicographic permutation index; capped by factorial(n).
        from math import factorial  # noqa: PLC0415

        total = factorial(n)
        if permutation_id >= total:
            raise ValueError(
                f"permutation_id {permutation_id} >= n_options! ({total}) for enumerated mode"
            )
        avail = list(range(n))
        perm_list = []
        idx = permutation_id
        for k in range(n, 0, -1):
            f = factorial(k - 1)
            pos, idx = divmod(idx, f)
            perm_list.append(avail.pop(pos))
        perm = tuple(perm_list)
    else:  # pragma: no cover
        raise ValueError(f"unknown permutation mode: {mode!r}")
    return PermutedItem(
        base=item,
        permutation_id=permutation_id,
        perm=perm,
        letters=letters_for(n),
    )


def all_permutations(item: CanonicalItem, *, max_k: int = 24) -> list[PermutedItem]:
    """Enumerate up to ``max_k`` permutations in ``enumerated`` order."""
    from math import factorial  # noqa: PLC0415

    total = min(factorial(item.n_options), max_k)
    return [permutation_for(item, i, mode="enumerated") for i in range(total)]


__all__ = [
    "LETTERS_26",
    "PermutationMode",
    "all_permutations",
    "letters_for",
    "permutation_for",
]
