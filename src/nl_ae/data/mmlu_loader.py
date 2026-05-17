"""MMLU loader: streams canonical 4-option items via the ``datasets`` library.

``cais/mmlu``, full test split (14,042 items across 57 subjects) is the
default. The plan resolves U2.2 to "full test split" with an override
``mmlu_subjects: tuple[str, ...] | None``.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Iterator
from pathlib import Path
from typing import Literal

from nl_ae.schema.models import DatasetFingerprint

from .canonical import CanonicalItem
from .text_norm import derive_item_id, safe_id_component

LOG = logging.getLogger(__name__)


class MmluLoader:
    """Streaming loader for cais/mmlu.

    The loader is lazy: ``iter_items`` constructs items on demand, but
    ``item_count`` and ``revision`` force one ``load_dataset`` call so the
    eval orchestrator can compute ``expected_visits`` up-front (UR8).
    """

    def __init__(
        self,
        *,
        cache_dir: Path,
        split: Literal["test", "validation", "dev"] = "test",
        subjects: tuple[str, ...] | None = None,
        offline: bool = True,
        hf_dataset_id: str = "cais/mmlu",
        revision: str | None = None,
        limit: int | None = None,
    ) -> None:
        self._cache_dir = cache_dir
        self._split = split
        self._subjects = subjects
        self._offline = offline
        self._hf_dataset_id = hf_dataset_id
        self._configured_revision = revision
        self._limit = limit
        self._resolved_revision: str | None = None
        self._items_cached: list[CanonicalItem] | None = None

    @property
    def revision(self) -> str:
        self._ensure_loaded()
        return self._resolved_revision or "unknown"

    @property
    def item_count(self) -> int:
        self._ensure_loaded()
        assert self._items_cached is not None
        return len(self._items_cached)

    @property
    def item_id_scheme(self) -> str:
        return "mmlu/v1/<subject>/q-<sha256(nfc(question))[:12]>"

    def iter_items(self) -> Iterator[CanonicalItem]:
        self._ensure_loaded()
        assert self._items_cached is not None
        yield from self._items_cached

    def emit_dataset_fingerprint(self) -> DatasetFingerprint:
        self._ensure_loaded()
        return DatasetFingerprint(
            name="mmlu",
            hf_dataset_id=self._hf_dataset_id,
            split=self._split,
            commit_or_revision=self._resolved_revision,
            item_count=self.item_count,
            item_id_scheme=self.item_id_scheme,
        )

    # ---------------------------------------------------------------

    def _ensure_loaded(self) -> None:
        if self._items_cached is not None:
            return
        if self._offline:
            os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
        try:
            from datasets import load_dataset  # type: ignore[import-not-found]  # noqa: PLC0415
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "datasets library is not installed; `pip install nl-ae[model]`"
            ) from exc

        subjects = self._subjects
        if subjects is None:
            subjects = ("all",)
        items: list[CanonicalItem] = []
        revisions: set[str] = set()
        # MMLU contains rows whose question text repeats (sometimes with
        # different choices/answer). ``item_id`` hashes the question only, so
        # those collide; the activation cache enforces a hard
        # ``(item_id, perm, template)`` uniqueness invariant. Dedup keep-first
        # (deterministic given the pinned dataset revision) so rows.jsonl is
        # clean and the question is not double-weighted in aggregates.
        seen_ids: set[str] = set()
        dropped_dupes = 0
        for subject in subjects:
            LOG.info("loading MMLU subject=%s split=%s", subject, self._split)
            ds = load_dataset(
                self._hf_dataset_id,
                subject,
                split=self._split,
                cache_dir=str(self._cache_dir),
                revision=self._configured_revision,
            )
            ver = getattr(ds.info, "version", None) or getattr(ds.info, "download_checksums", None)
            if ver is not None:
                revisions.add(str(ver))
            for row in ds:
                question = str(row["question"]).strip()
                choices = tuple(str(c).strip() for c in row["choices"])
                if len(choices) < 2:
                    LOG.debug("skipping MMLU row with %d choices: %s", len(choices), question[:60])
                    continue
                answer = int(row["answer"])
                row_subject = str(row.get("subject") or subject)
                prefix = f"mmlu/v1/{safe_id_component(row_subject)}"
                item_id = derive_item_id(prefix=prefix, payload=question)
                if item_id in seen_ids:
                    dropped_dupes += 1
                    LOG.debug(
                        "skipping duplicate-question MMLU row item_id=%s subject=%s: %s",
                        item_id,
                        row_subject,
                        question[:60],
                    )
                    continue
                seen_ids.add(item_id)
                items.append(
                    CanonicalItem(
                        item_id=item_id,
                        dataset_name="mmlu",
                        dataset_split=self._split,
                        subject=row_subject,
                        question=question,
                        choices=choices,
                        gold_index=answer,
                        metadata={},
                    )
                )
                if self._limit is not None and len(items) >= self._limit:
                    break
            if self._limit is not None and len(items) >= self._limit:
                break

        if dropped_dupes:
            LOG.warning(
                "MMLU: dropped %d duplicate-question row(s) (keep-first dedup); "
                "%d unique items",
                dropped_dupes,
                len(items),
            )
        self._items_cached = items
        self._resolved_revision = (
            self._configured_revision
            or (sorted(revisions)[0] if revisions else None)
        )


__all__ = ["MmluLoader"]
