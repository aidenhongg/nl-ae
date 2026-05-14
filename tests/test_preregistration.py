"""Preregistration parser + gate + locker (C09)."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from nl_ae.pilot.assignment import StratumItem, assign_pilot_fold
from nl_ae.pilot.errors import (
    PilotDigestMismatchError,
    PreregistrationInvalidError,
    PreregistrationMissingError,
    PreregistrationParseError,
    PreregistrationUnlockedError,
)
from nl_ae.pilot.manifest import (
    assign_and_write_pilot_manifest,
)
from nl_ae.pilot.models import Preregistration
from nl_ae.pilot.preregistration import (
    load_preregistration,
    parse_preregistration_text,
    require_preregistration,
)


def _write_pilot(tmp_path: Path) -> str:
    items = [
        StratumItem(item_id=f"mmlu/v1/algebra/q-{i:04d}", dataset_name="mmlu", subject="algebra")
        for i in range(40)
    ]
    manifest, _ = assign_and_write_pilot_manifest(
        run_id="20260101T000000Z-deadbee-test",
        items=items,
        seed=0,
        frac=0.05,
        stratify_by=("subject", "dataset_name"),
        min_per_stratum=2,
        out_path=tmp_path / "pilot_manifest.json",
        on_existing="resume",
    )
    return manifest.pilot_manifest_digest


def _prereg_text(*, digest: str, locked: bool = False) -> str:
    locked_at = '"2026-05-15T14:32:11Z"' if locked else "null"
    locked_sha = '"' + ("a" * 40) + '"' if locked else "null"
    return textwrap.dedent(
        f"""\
        ---
        schema_version: "1.0.0"
        run_id: "20260101T000000Z-deadbee-test"
        pilot_manifest_digest: "{digest}"
        locked_at: {locked_at}
        locked_at_git_sha: {locked_sha}
        holdout_vs_full: "holdout-only"
        labels:
          - disagreement_flag
        layers:
          - 20
        nla_scope:
          layer: 20
          fold: holdout
          limit: 2000
          decode_strategy: sampled
          temperature: 1.0
          max_new_tokens: 200
        primary_hypothesis: |
          Layer 20 linearly decodes disagreement above 0.65 test accuracy.
        secondary_hypotheses: []
        significance_threshold: 0.01
        multiple_comparison_correction: bonferroni
        n_comparisons: 1
        effect_size_metric: accuracy_minus_baseline
        effect_size_threshold: 0.10
        exploratory_allowed: true
        ---

        # Preregistration body — theoretical motivation goes here.
        """
    )


def test_parse_preregistration_text_round_trip() -> None:
    digest = "a" * 64
    text = _prereg_text(digest=digest, locked=True)
    prereg, body = parse_preregistration_text(text)
    assert isinstance(prereg, Preregistration)
    assert prereg.pilot_manifest_digest == digest
    assert prereg.is_locked
    assert "theoretical motivation" in body


def test_parse_preregistration_text_unlocked() -> None:
    text = _prereg_text(digest="b" * 64, locked=False)
    prereg, _ = parse_preregistration_text(text)
    assert not prereg.is_locked


def test_parse_rejects_missing_frontmatter() -> None:
    with pytest.raises(PreregistrationParseError):
        parse_preregistration_text("just markdown body\n")


def test_parse_rejects_unknown_field() -> None:
    digest = "a" * 64
    text = _prereg_text(digest=digest).replace(
        'pilot_manifest_digest', 'unknown_field_xyz'
    )
    with pytest.raises(PreregistrationInvalidError):
        parse_preregistration_text(text)


def test_parse_rejects_bad_layer() -> None:
    digest = "a" * 64
    text = _prereg_text(digest=digest).replace("  - 20", "  - 99")
    with pytest.raises(PreregistrationInvalidError):
        parse_preregistration_text(text)


def test_require_preregistration_missing(tmp_path: Path) -> None:
    with pytest.raises(PreregistrationMissingError):
        require_preregistration(tmp_path)


def test_require_preregistration_unlocked_raises(tmp_path: Path) -> None:
    digest = _write_pilot(tmp_path)
    (tmp_path / "preregistration.md").write_text(
        _prereg_text(digest=digest, locked=False), encoding="utf-8"
    )
    with pytest.raises(PreregistrationUnlockedError):
        require_preregistration(tmp_path)


def test_require_preregistration_digest_mismatch(tmp_path: Path) -> None:
    _write_pilot(tmp_path)
    bogus_digest = "f" * 64
    (tmp_path / "preregistration.md").write_text(
        _prereg_text(digest=bogus_digest, locked=True), encoding="utf-8"
    )
    with pytest.raises(PilotDigestMismatchError):
        require_preregistration(tmp_path)


def test_require_preregistration_ok(tmp_path: Path) -> None:
    digest = _write_pilot(tmp_path)
    (tmp_path / "preregistration.md").write_text(
        _prereg_text(digest=digest, locked=True), encoding="utf-8"
    )
    prereg = require_preregistration(tmp_path)
    assert prereg.is_locked
    assert prereg.pilot_manifest_digest == digest


def test_load_preregistration_returns_body(tmp_path: Path) -> None:
    digest = _write_pilot(tmp_path)
    (tmp_path / "preregistration.md").write_text(
        _prereg_text(digest=digest, locked=True), encoding="utf-8"
    )
    prereg, body, path = load_preregistration(tmp_path)
    assert prereg.is_locked
    assert "theoretical motivation" in body
    assert path == tmp_path / "preregistration.md"
