"""OpinionQA loader: variable-cardinality opinion-elicitation items.

Reads from ``tasksource/opinionqa`` (primary) or ``timchen0618/OpinionQA``
(fallback) via the HF ``datasets`` library, or from a vendored parquet
directory specified by ``source_dir``. Filters to the Wang et al. 2024
public-issues subset when ``subset_path`` is provided.

Items have **no gold answer** (CI.02 — ``gold_letter`` is nullable on the
result row).
"""

from __future__ import annotations

import logging
import os
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Literal

from nl_ae.schema.models import DatasetFingerprint

from .canonical import CanonicalItem
from .text_norm import derive_item_id, safe_id_component

LOG = logging.getLogger(__name__)

OpinionQaCanonicalization = Literal["keep_all", "drop_refusal", "fixed_4opt"]
_REFUSAL_DEFAULT = frozenset({"Refused", "Don't know", "Skipped"})


class OpinionQaLoader:
    def __init__(
        self,
        *,
        source_dir: Path | None,
        cache_dir: Path,
        canonicalization: OpinionQaCanonicalization = "keep_all",
        subset_path: Path | None = None,
        wave_filter: tuple[str, ...] | None = None,
        topic_filter: tuple[str, ...] | None = None,
        offline: bool = True,
        hf_dataset_id: str = "tasksource/opinionqa",
        hf_fallback_id: str = "timchen0618/OpinionQA",
        revision: str | None = None,
        revision_tag: str = "atp-v1",
        refusal_strings: Iterable[str] = _REFUSAL_DEFAULT,
        limit: int | None = None,
    ) -> None:
        self._source_dir = source_dir
        self._cache_dir = cache_dir
        self._canonicalization: OpinionQaCanonicalization = canonicalization
        self._subset_path = subset_path
        self._wave_filter = wave_filter
        self._topic_filter = topic_filter
        self._offline = offline
        self._hf_dataset_id = hf_dataset_id
        self._hf_fallback_id = hf_fallback_id
        self._configured_revision = revision
        self._revision_tag = revision_tag
        self._refusal_strings = frozenset(refusal_strings)
        self._limit = limit
        self._items_cached: list[CanonicalItem] | None = None
        self._resolved_source: str | None = None

    @property
    def revision(self) -> str:
        self._ensure_loaded()
        return self._resolved_source or self._revision_tag

    @property
    def item_count(self) -> int:
        self._ensure_loaded()
        assert self._items_cached is not None
        return len(self._items_cached)

    @property
    def item_id_scheme(self) -> str:
        return "opinionqa/v1/<wave>/q-<sha256(nfc(question))[:12]>"

    def iter_items(self) -> Iterator[CanonicalItem]:
        self._ensure_loaded()
        assert self._items_cached is not None
        yield from self._items_cached

    def emit_dataset_fingerprint(self) -> DatasetFingerprint:
        self._ensure_loaded()
        return DatasetFingerprint(
            name="opinionqa",
            hf_dataset_id=self._hf_dataset_id,
            split=self._revision_tag,
            commit_or_revision=self._resolved_source or self._revision_tag,
            item_count=self.item_count,
            item_id_scheme=self.item_id_scheme,
        )

    # ---------------------------------------------------------------

    def _ensure_loaded(self) -> None:
        if self._items_cached is not None:
            return
        if self._offline:
            os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

        rows = list(self._iter_raw())
        allowed_ids = self._load_subset_ids()
        items: list[CanonicalItem] = []
        for raw in rows:
            question = str(raw.get("question") or "").strip()
            choices_raw = tuple(str(c).strip() for c in (raw.get("choices") or ()))
            wave = str(raw.get("wave") or raw.get("survey") or "atp")
            topic = str(raw.get("topic") or raw.get("subject") or "")
            if self._wave_filter is not None and wave not in self._wave_filter:
                continue
            if self._topic_filter is not None and topic not in self._topic_filter:
                continue
            if not question or len(choices_raw) < 2:
                continue
            choices = self._canonicalize_choices(choices_raw)
            if choices is None or len(choices) < 2:
                continue
            prefix = f"opinionqa/v1/{safe_id_component(wave)}"
            item_id = derive_item_id(prefix=prefix, payload=question)
            if allowed_ids is not None and item_id not in allowed_ids:
                continue
            items.append(
                CanonicalItem(
                    item_id=item_id,
                    dataset_name="opinionqa",
                    dataset_split=self._revision_tag,
                    subject=topic or None,
                    question=question,
                    choices=choices,
                    gold_index=None,
                    metadata={"wave": wave},
                )
            )
            if self._limit is not None and len(items) >= self._limit:
                break
        self._items_cached = items

    def _canonicalize_choices(self, choices: tuple[str, ...]) -> tuple[str, ...] | None:
        if self._canonicalization == "keep_all":
            return choices
        if self._canonicalization == "drop_refusal":
            kept = tuple(c for c in choices if c not in self._refusal_strings)
            return kept
        if self._canonicalization == "fixed_4opt":
            return choices[:4] if len(choices) >= 4 else None
        raise ValueError(f"unknown canonicalization: {self._canonicalization!r}")

    def _load_subset_ids(self) -> set[str] | None:
        if self._subset_path is None:
            return None
        if not self._subset_path.exists():
            LOG.warning("OpinionQA subset_path does not exist: %s", self._subset_path)
            return None
        ids = {
            line.strip()
            for line in self._subset_path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        }
        return ids or None

    def _iter_raw(self) -> Iterator[dict[str, object]]:
        if self._source_dir is not None and self._source_dir.exists():
            yield from self._iter_from_parquet(self._source_dir)
            self._resolved_source = f"parquet:{self._source_dir}"
            return
        try:
            from datasets import load_dataset  # type: ignore[import-not-found]  # noqa: PLC0415
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "datasets library is not installed and no local source_dir provided"
            ) from exc
        last_exc: Exception | None = None
        for dataset_id in (self._hf_dataset_id, self._hf_fallback_id):
            try:
                ds = load_dataset(
                    dataset_id,
                    split="train",
                    cache_dir=str(self._cache_dir),
                    revision=self._configured_revision,
                )
                self._resolved_source = dataset_id
                for row in ds:
                    yield dict(row)
                return
            except (FileNotFoundError, ValueError, Exception) as exc:  # pragma: no cover
                last_exc = exc
                LOG.warning("OpinionQA load failed for %s: %r", dataset_id, exc)
        raise RuntimeError("could not load OpinionQA from any source") from last_exc

    def _iter_from_parquet(self, source_dir: Path) -> Iterator[dict[str, object]]:
        try:
            import pyarrow.parquet as pq  # noqa: PLC0415
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("pyarrow not installed; required for vendored OpinionQA") from exc
        files = sorted(source_dir.glob("*.parquet"))
        if not files:
            raise FileNotFoundError(f"no *.parquet under {source_dir}")
        for path in files:
            table = pq.read_table(path)
            for row in table.to_pylist():
                yield dict(row)


__all__ = ["OpinionQaCanonicalization", "OpinionQaLoader"]
