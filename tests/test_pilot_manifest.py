"""PilotManifest writer + reader + digest stability (C09)."""

from __future__ import annotations

from pathlib import Path

import pytest

from nl_ae.pilot.assignment import StratumItem, assign_pilot_fold
from nl_ae.pilot.errors import (
    PilotFoldMismatchError,
    PilotManifestMissingError,
)
from nl_ae.pilot.manifest import (
    assign_and_write_pilot_manifest,
    build_pilot_manifest,
    compute_pilot_manifest_digest,
    iter_item_summaries,
    load_pilot_manifest,
    write_pilot_manifest,
)


def _mmlu(item_id: str, subject: str) -> StratumItem:
    return StratumItem(item_id=item_id, dataset_name="mmlu", subject=subject)


def _make_items(n_per_subj: int = 40) -> list[StratumItem]:
    return [
        _mmlu(f"mmlu/v1/{subj}/q-{i:04d}", subj)
        for subj in ("algebra", "anatomy", "astronomy")
        for i in range(n_per_subj)
    ]


def test_build_pilot_manifest_round_trip() -> None:
    items = _make_items()
    result = assign_pilot_fold(items, seed=0, frac=0.05)
    manifest = build_pilot_manifest(
        run_id="20260101T000000Z-deadbee-test",
        seed=0,
        frac=0.05,
        stratify_by=("subject", "dataset_name"),
        min_per_stratum=2,
        result=result,
    )
    assert manifest.n_total == 120
    assert manifest.n_pilot + manifest.n_holdout == 120
    # Reading back from JSON round-trips the digest.
    raw = manifest.model_dump_json()
    from nl_ae.pilot.models import PilotManifest

    rebuilt = PilotManifest.model_validate_json(raw)
    assert rebuilt == manifest


def test_pilot_manifest_digest_is_stable_across_timestamps() -> None:
    """The digest excludes created_at and completion_status."""
    items = _make_items()
    result = assign_pilot_fold(items, seed=0, frac=0.05)
    digest_a = compute_pilot_manifest_digest(
        run_id="r-1",
        seed=0,
        frac=0.05,
        stratify_by=("subject", "dataset_name"),
        min_per_stratum=2,
        strata=result.strata,
        pilot_item_ids=result.pilot_item_ids,
        n_pilot=result.n_pilot,
        n_holdout=result.n_holdout,
        n_total=result.n_total,
    )
    digest_b = compute_pilot_manifest_digest(
        run_id="r-1",
        seed=0,
        frac=0.05,
        stratify_by=("subject", "dataset_name"),
        min_per_stratum=2,
        strata=result.strata,
        pilot_item_ids=result.pilot_item_ids,
        n_pilot=result.n_pilot,
        n_holdout=result.n_holdout,
        n_total=result.n_total,
    )
    assert digest_a == digest_b


def test_pilot_manifest_digest_changes_with_seed() -> None:
    items = _make_items()
    r0 = assign_pilot_fold(items, seed=0, frac=0.05)
    r1 = assign_pilot_fold(items, seed=1, frac=0.05)
    d0 = compute_pilot_manifest_digest(
        run_id="r",
        seed=0,
        frac=0.05,
        stratify_by=("subject", "dataset_name"),
        min_per_stratum=2,
        strata=r0.strata,
        pilot_item_ids=r0.pilot_item_ids,
        n_pilot=r0.n_pilot,
        n_holdout=r0.n_holdout,
        n_total=r0.n_total,
    )
    d1 = compute_pilot_manifest_digest(
        run_id="r",
        seed=1,
        frac=0.05,
        stratify_by=("subject", "dataset_name"),
        min_per_stratum=2,
        strata=r1.strata,
        pilot_item_ids=r1.pilot_item_ids,
        n_pilot=r1.n_pilot,
        n_holdout=r1.n_holdout,
        n_total=r1.n_total,
    )
    assert d0 != d1


def test_write_pilot_manifest_creates_sidecar(tmp_path: Path) -> None:
    items = _make_items()
    result = assign_pilot_fold(items, seed=0, frac=0.05)
    manifest = build_pilot_manifest(
        run_id="r",
        seed=0,
        frac=0.05,
        stratify_by=("subject", "dataset_name"),
        min_per_stratum=2,
        result=result,
    )
    out = tmp_path / "pilot_manifest.json"
    raw_digest = write_pilot_manifest(out, manifest)
    assert out.exists()
    sidecar = tmp_path / "pilot_manifest.json.sha256"
    assert sidecar.exists()
    assert sidecar.read_text().strip() == raw_digest
    loaded = load_pilot_manifest(out)
    assert loaded.pilot_manifest_digest == manifest.pilot_manifest_digest


def test_load_pilot_manifest_missing_raises(tmp_path: Path) -> None:
    with pytest.raises(PilotManifestMissingError):
        load_pilot_manifest(tmp_path / "nope.json")


def test_assign_and_write_resume_is_noop_on_match(tmp_path: Path) -> None:
    items = _make_items()
    out = tmp_path / "pilot_manifest.json"
    m1, was_written_1 = assign_and_write_pilot_manifest(
        run_id="r",
        items=items,
        seed=0,
        frac=0.05,
        stratify_by=("subject", "dataset_name"),
        min_per_stratum=2,
        out_path=out,
        on_existing="resume",
    )
    assert was_written_1 is True
    m2, was_written_2 = assign_and_write_pilot_manifest(
        run_id="r",
        items=items,
        seed=0,
        frac=0.05,
        stratify_by=("subject", "dataset_name"),
        min_per_stratum=2,
        out_path=out,
        on_existing="resume",
    )
    assert was_written_2 is False
    assert m1.pilot_manifest_digest == m2.pilot_manifest_digest


def test_assign_and_write_resume_raises_on_drift(tmp_path: Path) -> None:
    items = _make_items()
    out = tmp_path / "pilot_manifest.json"
    assign_and_write_pilot_manifest(
        run_id="r",
        items=items,
        seed=0,
        frac=0.05,
        stratify_by=("subject", "dataset_name"),
        min_per_stratum=2,
        out_path=out,
        on_existing="resume",
    )
    # Same items but different seed → different fold → digest mismatch.
    with pytest.raises(PilotFoldMismatchError):
        assign_and_write_pilot_manifest(
            run_id="r",
            items=items,
            seed=99,
            frac=0.05,
            stratify_by=("subject", "dataset_name"),
            min_per_stratum=2,
            out_path=out,
            on_existing="resume",
        )


def test_assign_and_write_error_mode_refuses_overwrite(tmp_path: Path) -> None:
    items = _make_items()
    out = tmp_path / "pilot_manifest.json"
    assign_and_write_pilot_manifest(
        run_id="r",
        items=items,
        seed=0,
        frac=0.05,
        stratify_by=("subject", "dataset_name"),
        min_per_stratum=2,
        out_path=out,
        on_existing="resume",
    )
    with pytest.raises(FileExistsError):
        assign_and_write_pilot_manifest(
            run_id="r",
            items=items,
            seed=0,
            frac=0.05,
            stratify_by=("subject", "dataset_name"),
            min_per_stratum=2,
            out_path=out,
            on_existing="error",
        )


def test_iter_item_summaries_dedupes_across_permutations(tmp_path: Path) -> None:
    rows_path = tmp_path / "rows.jsonl"
    # Three permutations for two distinct items.
    lines = []
    for iid in ("mmlu/v1/algebra/q-0001", "mmlu/v1/algebra/q-0002"):
        for perm in range(3):
            lines.append(
                f'{{"item_id":"{iid}","dataset_name":"mmlu","subject":"algebra",'
                f'"permutation_id":{perm}}}'
            )
    rows_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    items = list(iter_item_summaries(rows_path))
    assert len(items) == 2
    assert {it.item_id for it in items} == {
        "mmlu/v1/algebra/q-0001",
        "mmlu/v1/algebra/q-0002",
    }
