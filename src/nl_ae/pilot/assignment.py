"""Deterministic, stratified pilot-fold assignment.

``assign_pilot_fold`` is a pure function — same ``(seed, items, frac,
stratify_by, min_per_stratum)`` produces a bit-identical fold across machines.
SHA-256 bucketing per stratum; never Python ``hash()``.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

from nl_ae.runtime.seeds import derive_child_seed

from .models import StratumRecord

Fold = Literal["pilot", "holdout"]

# Bucket modulus: 1_000_000 gives 6 significant digits of selection granularity
# which is plenty for fractions down to ~1e-5.
_BUCKET_MOD: int = 1_000_000


@dataclass(frozen=True)
class StratumItem:
    """Lightweight view of one canonical item — only what stratification needs.

    ``metadata`` is preserved as a free-form mapping so callers that have richer
    item state (e.g., OpinionQA's ``wave``) can pass it through. ``pilot-init``
    reading ``rows.jsonl`` populates metadata={} since ``ResultRow`` does not
    carry per-item metadata; that path stratifies via ``subject`` ∪
    ``dataset_name`` fallback. See ``L2-architecture.md §"Stratum resolution"``.
    """

    item_id: str
    dataset_name: str
    subject: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


def _lookup_field(item: StratumItem, name: str) -> str | None:
    """First non-null hit across named attrs ∪ ``metadata`` dict."""
    direct = getattr(item, name, None)
    if direct is not None and not isinstance(direct, Mapping):
        s = str(direct).strip()
        if s:
            return s
    md = getattr(item, "metadata", None)
    if isinstance(md, Mapping):
        v = md.get(name)
        if v is not None:
            s = str(v).strip()
            if s:
                return s
    return None


def _resolve_stratum(
    item: StratumItem, stratify_by: Sequence[str]
) -> tuple[str, str]:
    """Walk ``stratify_by`` and return ``(stratum_key, source_field)``."""
    for name in stratify_by:
        v = _lookup_field(item, name)
        if v is not None:
            return v, name
    raise ValueError(
        f"no stratification field resolved for item_id={item.item_id!r}; "
        f"tried {tuple(stratify_by)!r}"
    )


def _bucket(seed: int, item_id: str) -> int:
    """Deterministic per-item bucket in ``[0, _BUCKET_MOD)``."""
    return derive_child_seed(seed, item_id.encode("utf-8")) % _BUCKET_MOD


def _min_pilot_count(n_total: int, frac: float, min_per_stratum: int) -> int:
    """Per-plan: floor activates only when ``n_total >= 4``.

    For smaller strata, the target is ``max(1, ceil(frac * n_total))``.
    """
    if n_total >= 4:
        return min_per_stratum
    return max(1, math.ceil(frac * n_total))


@dataclass(frozen=True)
class PilotFoldResult:
    """Output of ``assign_pilot_fold``: per-item fold + per-stratum records."""

    fold: dict[str, Fold]
    strata: tuple[StratumRecord, ...]
    pilot_item_ids: tuple[str, ...]

    @property
    def n_pilot(self) -> int:
        return len(self.pilot_item_ids)

    @property
    def n_holdout(self) -> int:
        return len(self.fold) - self.n_pilot

    @property
    def n_total(self) -> int:
        return len(self.fold)


def assign_pilot_fold(
    items: Iterable[StratumItem],
    *,
    seed: int,
    frac: float = 0.05,
    stratify_by: Sequence[str] = ("subject", "wave", "topic", "dataset_name"),
    min_per_stratum: int = 2,
) -> PilotFoldResult:
    """Assign each item to ``"pilot"`` or ``"holdout"`` deterministically.

    Algorithm (per ``modules/09-pilot-preregistration.md``):

    1. For each item, compute a SHA-256-derived bucket in ``[0, 1_000_000)``.
    2. Resolve the item's stratum by walking ``stratify_by`` and taking the
       first non-null field (named attribute, then ``metadata`` dict).
    3. For each stratum, items with ``bucket < frac * 1_000_000`` are
       *naturally* pilot. If too few qualify, promote the next lowest-bucket
       items until the per-stratum floor is met:

       - ``n_total >= 4`` → floor = ``min_per_stratum``.
       - ``n_total <  4`` → floor = ``max(1, ceil(frac * n_total))``.

       Floor is capped at ``n_total`` (a stratum with one item gets one pilot).

    Same inputs → bit-identical output across machines / Python versions /
    re-runs. Deduplication of input item_ids is enforced.
    """
    if not (0.0 < frac < 1.0):
        raise ValueError(f"frac must lie in (0, 1); got {frac!r}")
    if min_per_stratum < 1:
        raise ValueError(f"min_per_stratum must be >= 1; got {min_per_stratum!r}")
    if not stratify_by:
        raise ValueError("stratify_by must be non-empty")

    materialized: list[StratumItem] = []
    seen: set[str] = set()
    for it in items:
        if it.item_id in seen:
            raise ValueError(f"duplicate item_id in pilot input: {it.item_id!r}")
        seen.add(it.item_id)
        materialized.append(it)
    if not materialized:
        raise ValueError("assign_pilot_fold received zero items")

    by_stratum: dict[tuple[str, str], list[str]] = defaultdict(list)
    buckets: dict[str, int] = {}
    for it in materialized:
        key, source = _resolve_stratum(it, stratify_by)
        by_stratum[(key, source)].append(it.item_id)
        buckets[it.item_id] = _bucket(seed, it.item_id)

    threshold = int(round(frac * _BUCKET_MOD))
    fold: dict[str, Fold] = {}
    records: list[StratumRecord] = []
    pilot_item_ids: list[str] = []

    # Sort strata by (source_field, key) for deterministic record ordering.
    for (key, source_field), ids in sorted(by_stratum.items(), key=lambda kv: kv[0]):
        n_total = len(ids)
        floor = min(_min_pilot_count(n_total, frac, min_per_stratum), n_total)
        # By construction _BUCKET_MOD > threshold > 0; tie-break on item_id
        # (lex sort) so equal buckets land in a stable order.
        ids_by_bucket = sorted(ids, key=lambda iid: (buckets[iid], iid))
        natural = [iid for iid in ids if buckets[iid] < threshold]
        if len(natural) >= floor:
            pilot_in_stratum = set(natural)
        else:
            # Promote next lowest-bucket items above the natural threshold to
            # hit the floor. Lowest-bucket overall ⊇ natural pilot.
            pilot_in_stratum = set(ids_by_bucket[:floor])
        for iid in ids:
            fold[iid] = "pilot" if iid in pilot_in_stratum else "holdout"
        pilot_item_ids.extend(pilot_in_stratum)
        records.append(
            StratumRecord(
                key=key,
                source_field=source_field,
                n_total=n_total,
                n_pilot=len(pilot_in_stratum),
                n_holdout=n_total - len(pilot_in_stratum),
            )
        )

    return PilotFoldResult(
        fold=fold,
        strata=tuple(records),
        pilot_item_ids=tuple(sorted(pilot_item_ids)),
    )


__all__ = [
    "Fold",
    "PilotFoldResult",
    "StratumItem",
    "assign_pilot_fold",
]
