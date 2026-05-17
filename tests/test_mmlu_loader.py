"""MmluLoader — duplicate-question dedup + revision fingerprint.

Network-free: a fake ``datasets`` module is injected into ``sys.modules``
(same idiom as the materialize tests) so ``load_dataset`` returns canned rows,
including a repeated question that must be deduped keep-first.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

from nl_ae.data.mmlu_loader import MmluLoader


class _FakeInfo:
    version = "1.4.2"
    download_checksums = None


class _FakeDS:
    def __init__(self, rows: list[dict]) -> None:
        self._rows = rows
        self.info = _FakeInfo()

    def __iter__(self):
        return iter(self._rows)


def _install_fake_datasets(monkeypatch: pytest.MonkeyPatch, rows: list[dict]) -> None:
    fake = types.ModuleType("datasets")

    def load_dataset(hf_id, subject, *, split, cache_dir, revision):
        return _FakeDS(rows)

    fake.load_dataset = load_dataset  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "datasets", fake)


def _rows() -> list[dict]:
    return [
        {"question": "What is the Sun?", "choices": ["a", "b", "c", "d"], "answer": 0,
         "subject": "astronomy"},
        # Exact duplicate question in the same subject -> same item_id.
        {"question": "What is the Sun?", "choices": ["w", "x", "y", "z"], "answer": 2,
         "subject": "astronomy"},
        {"question": "Define a galaxy.", "choices": ["a", "b", "c", "d"], "answer": 1,
         "subject": "astronomy"},
        # Degenerate row (<2 choices) is skipped before id derivation.
        {"question": "broken", "choices": ["only"], "answer": 0, "subject": "astronomy"},
        # Same question text, different subject -> different prefix -> kept.
        {"question": "What is the Sun?", "choices": ["a", "b"], "answer": 0,
         "subject": "physics"},
    ]


def test_dedup_keeps_first_occurrence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_fake_datasets(monkeypatch, _rows())
    loader = MmluLoader(cache_dir=tmp_path, revision="rev-abc")
    items = list(loader.iter_items())

    ids = [it.item_id for it in items]
    assert len(ids) == len(set(ids)), "item_ids must be unique after dedup"
    # 1 astronomy "Sun" (first kept) + galaxy + physics "Sun" = 3; the broken
    # row and the duplicate astronomy "Sun" are dropped.
    assert len(items) == 3
    sun_astro = next(
        it for it in items if it.subject == "astronomy" and it.question == "What is the Sun?"
    )
    assert sun_astro.choices == ("a", "b", "c", "d")  # first occurrence, not ("w","x","y","z")
    assert sun_astro.gold_index == 0
    # Same question under a different subject keeps a distinct id.
    assert any(it.subject == "physics" for it in items)
    assert loader.item_count == 3


def test_limit_counts_unique_items(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_fake_datasets(monkeypatch, _rows())
    loader = MmluLoader(cache_dir=tmp_path, revision="rev-abc", limit=2)
    items = list(loader.iter_items())
    assert len(items) == 2  # limit is on unique items, dupes don't consume budget
    assert len({it.item_id for it in items}) == 2


def test_fingerprint_records_configured_revision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_fake_datasets(monkeypatch, _rows())
    loader = MmluLoader(cache_dir=tmp_path, revision="c0ffee-sha")
    fp = loader.emit_dataset_fingerprint()
    assert fp.commit_or_revision == "c0ffee-sha"  # pinned rev recorded, not "0.0.0"
    assert fp.item_count == 3


def test_fingerprint_falls_back_when_unpinned(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_fake_datasets(monkeypatch, _rows())
    loader = MmluLoader(cache_dir=tmp_path)  # revision=None
    fp = loader.emit_dataset_fingerprint()
    # Unpinned -> records the dataset-script version (the documented foot-gun).
    assert fp.commit_or_revision == "1.4.2"
