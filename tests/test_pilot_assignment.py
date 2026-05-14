"""assign_pilot_fold determinism, stratification, and floor behaviour (C09)."""

from __future__ import annotations

import pytest

from nl_ae.pilot.assignment import (
    StratumItem,
    _resolve_stratum,
    assign_pilot_fold,
)


def _mmlu(item_id: str, subject: str) -> StratumItem:
    return StratumItem(item_id=item_id, dataset_name="mmlu", subject=subject)


def _oqa(item_id: str, *, wave: str | None = None, topic: str | None = None) -> StratumItem:
    md: dict[str, str] = {}
    if wave is not None:
        md["wave"] = wave
    if topic is not None:
        md["topic"] = topic
    return StratumItem(
        item_id=item_id, dataset_name="opinionqa", subject=None, metadata=md
    )


def test_assign_pilot_fold_is_deterministic() -> None:
    items = [_mmlu(f"mmlu/v1/algebra/q-{i:04d}", "algebra") for i in range(100)]
    a = assign_pilot_fold(items, seed=0, frac=0.05)
    b = assign_pilot_fold(items, seed=0, frac=0.05)
    assert a.fold == b.fold
    assert a.pilot_item_ids == b.pilot_item_ids
    assert a.strata == b.strata


def test_assign_pilot_fold_different_seed_yields_different_fold() -> None:
    items = [_mmlu(f"mmlu/v1/algebra/q-{i:04d}", "algebra") for i in range(100)]
    a = assign_pilot_fold(items, seed=0, frac=0.05)
    b = assign_pilot_fold(items, seed=42, frac=0.05)
    assert a.pilot_item_ids != b.pilot_item_ids


def test_assign_pilot_fold_stratifies_by_subject() -> None:
    items = [_mmlu(f"mmlu/v1/{subj}/q-{i:04d}", subj) for subj in ("algebra", "anatomy") for i in range(100)]
    result = assign_pilot_fold(items, seed=0, frac=0.05)
    assert len(result.strata) == 2
    by_subj = {s.key: s for s in result.strata}
    assert by_subj["algebra"].n_total == 100
    assert by_subj["anatomy"].n_total == 100
    # Each stratum should get roughly frac * n items in pilot (with min floor).
    assert by_subj["algebra"].n_pilot >= 2
    assert by_subj["anatomy"].n_pilot >= 2
    for s in result.strata:
        assert s.source_field == "subject"


def test_floor_activates_for_small_strata_geq_4() -> None:
    # Stratum of 4 items, frac=0.01 → natural pilot ≈ 0, floor → 2.
    items = [_mmlu(f"mmlu/v1/algebra/q-{i:04d}", "algebra") for i in range(4)]
    result = assign_pilot_fold(items, seed=0, frac=0.01, min_per_stratum=2)
    assert result.strata[0].n_pilot == 2
    assert result.strata[0].n_holdout == 2


def test_smaller_stratum_uses_per_n_floor() -> None:
    # Stratum of 2: floor = max(1, ceil(0.05 * 2)) = 1.
    items = [_mmlu(f"mmlu/v1/algebra/q-{i:04d}", "algebra") for i in range(2)]
    result = assign_pilot_fold(items, seed=0, frac=0.05)
    assert result.strata[0].n_total == 2
    assert result.strata[0].n_pilot == 1
    assert result.strata[0].n_holdout == 1


def test_singleton_stratum_falls_back_to_pilot() -> None:
    items = [_mmlu("mmlu/v1/algebra/q-0001", "algebra")]
    result = assign_pilot_fold(items, seed=0, frac=0.05)
    assert result.strata[0].n_total == 1
    assert result.strata[0].n_pilot == 1
    assert result.strata[0].n_holdout == 0


def test_large_stratum_gets_natural_fraction_with_floor() -> None:
    items = [_mmlu(f"mmlu/v1/algebra/q-{i:04d}", "algebra") for i in range(200)]
    result = assign_pilot_fold(items, seed=0, frac=0.05)
    stratum = result.strata[0]
    # Natural 5% of 200 is ~10 items; floor is 2, doesn't bind. Allow generous
    # spread for bucket variance.
    assert 5 <= stratum.n_pilot <= 20


def test_pilot_holdout_disjoint_and_exhaustive() -> None:
    items = [_mmlu(f"mmlu/v1/algebra/q-{i:04d}", "algebra") for i in range(50)]
    items += [_mmlu(f"mmlu/v1/anatomy/q-{i:04d}", "anatomy") for i in range(50)]
    result = assign_pilot_fold(items, seed=0, frac=0.05)
    pilot = set(iid for iid, f in result.fold.items() if f == "pilot")
    holdout = set(iid for iid, f in result.fold.items() if f == "holdout")
    assert pilot.isdisjoint(holdout)
    assert pilot | holdout == {it.item_id for it in items}
    assert set(result.pilot_item_ids) == pilot


def test_assign_pilot_fold_rejects_duplicate_item_ids() -> None:
    items = [_mmlu("mmlu/v1/algebra/q-0001", "algebra")] * 2
    with pytest.raises(ValueError, match="duplicate item_id"):
        assign_pilot_fold(items, seed=0, frac=0.05)


def test_assign_pilot_fold_rejects_empty_input() -> None:
    with pytest.raises(ValueError, match="zero items"):
        assign_pilot_fold([], seed=0, frac=0.05)


def test_resolve_stratum_walks_chain_to_metadata() -> None:
    item = _oqa("opinionqa/v1/atp/q-0001", wave="atp", topic="health")
    key, src = _resolve_stratum(item, ("subject", "wave", "topic", "dataset_name"))
    assert (key, src) == ("atp", "wave")


def test_resolve_stratum_falls_back_to_dataset_name() -> None:
    # No subject, no metadata — only the dataset_name remains.
    item = StratumItem(item_id="x/v1/q-0001", dataset_name="oddset", subject=None)
    key, src = _resolve_stratum(item, ("subject", "wave", "topic", "dataset_name"))
    assert (key, src) == ("oddset", "dataset_name")


def test_resolve_stratum_raises_when_no_field_matches() -> None:
    item = StratumItem(item_id="x/v1/q-0001", dataset_name="ds", subject=None)
    with pytest.raises(ValueError, match="no stratification field"):
        _resolve_stratum(item, ("subject", "nonexistent"))
