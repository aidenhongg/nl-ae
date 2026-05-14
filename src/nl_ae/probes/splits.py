"""Deterministic per-item sub-fold assignment within one fold (C07).

``split_by_item_within_fold`` buckets each ``item_id`` into ``"train"`` /
``"val"`` / ``"test"`` via :func:`nl_ae.runtime.seeds.derive_child_seed`. All
permutations of an item land in the same sub-fold (cross-permutation leakage
is impossible), and the assignment is bit-identical across machines and
re-runs given the same ``(probe_seed, item_id set, frac)``.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Literal

from nl_ae.runtime.seeds import derive_child_seed

SubFold = Literal["train", "val", "test"]

_BUCKETS = 1_000_000


def split_by_item_within_fold(
    item_ids: Iterable[str],
    *,
    probe_seed: int,
    frac: tuple[float, float, float] = (0.70, 0.15, 0.15),
) -> dict[str, SubFold]:
    """Return ``{item_id: sub_fold}`` for every item in ``item_ids``.

    Buckets are computed via ``derive_child_seed(probe_seed, item_id.encode())
    % 1_000_000``; cut points are ``round(frac[i] * 1_000_000)``. No global RNG
    state is touched. Operates within a single fold; never references the
    other fold.
    """
    train_frac, val_frac, test_frac = frac
    if train_frac <= 0.0 or val_frac <= 0.0 or test_frac <= 0.0:
        raise ValueError(f"frac entries must be > 0; got {frac}")
    total = train_frac + val_frac + test_frac
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"frac must sum to 1.0; got {total}")
    train_cut = round(train_frac * _BUCKETS)
    val_cut = round((train_frac + val_frac) * _BUCKETS)

    out: dict[str, SubFold] = {}
    for item_id in item_ids:
        bucket = derive_child_seed(probe_seed, item_id.encode("utf-8")) % _BUCKETS
        if bucket < train_cut:
            out[item_id] = "train"
        elif bucket < val_cut:
            out[item_id] = "val"
        else:
            out[item_id] = "test"
    return out


__all__ = ["SubFold", "split_by_item_within_fold"]
