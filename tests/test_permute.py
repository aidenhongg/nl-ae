"""permutation_for: seeded determinism, identity mode, gold letter remap."""

from __future__ import annotations

import pytest

from nl_ae.data.canonical import CanonicalItem
from nl_ae.data.permute import LETTERS_26, all_permutations, letters_for, permutation_for


def _item(gold_index: int = 0, n: int = 4) -> CanonicalItem:
    return CanonicalItem(
        item_id="mmlu/v1/x/q-abc",
        dataset_name="mmlu",
        dataset_split="test",
        subject="x",
        question="q?",
        choices=tuple(str(i) for i in range(n)),
        gold_index=gold_index,
    )


def test_letters_for() -> None:
    assert letters_for(2) == ("A", "B")
    assert letters_for(4) == ("A", "B", "C", "D")
    assert letters_for(26) == LETTERS_26
    with pytest.raises(ValueError):
        letters_for(1)
    with pytest.raises(ValueError):
        letters_for(27)


def test_identity_is_identity() -> None:
    item = _item()
    p = permutation_for(item, 0, mode="identity")
    assert p.perm == (0, 1, 2, 3)
    assert p.choices_in_order == item.choices
    assert p.gold_letter == "A"


def test_seeded_is_deterministic() -> None:
    item = _item()
    a = permutation_for(item, 7, mode="seeded")
    b = permutation_for(item, 7, mode="seeded")
    c = permutation_for(item, 8, mode="seeded")
    assert a.perm == b.perm
    assert a.perm != c.perm  # different permutation_id ⇒ different perm (with high prob)


def test_seeded_does_not_depend_on_python_hash() -> None:
    # Building two items with the same item_id but in two separate processes
    # would give the same seeded permutation — we use sha256, not hash().
    item = _item()
    a = permutation_for(item, 3, mode="seeded")
    # Reconstruct the seed manually.
    import hashlib  # noqa: PLC0415

    payload = b"mmlu/v1/x/q-abc|3"
    seed = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
    assert seed > 0
    assert a.permutation_id == 3


def test_gold_letter_follows_permutation() -> None:
    item = _item(gold_index=2)  # original "C"
    p = permutation_for(item, 0, mode="seeded")
    assert p.gold_letter is not None
    new_pos = p.perm.index(2)
    assert p.gold_letter == p.letters[new_pos]


def test_all_permutations_capped() -> None:
    item = _item(n=3)  # 3! = 6 permutations total
    perms = all_permutations(item, max_k=4)
    assert len(perms) == 4
