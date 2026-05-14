"""``pilot_manifest.json`` writer + reader.

Atomic file writes via tempfile + ``os.replace``; matching ``.sha256`` sidecar.
The ``pilot_manifest_digest`` field is computed over the canonical JSON encoding
of the assignment fields *excluding* ``created_at``, ``completion_status``, and
``pilot_manifest_digest`` itself — so two runs that produce identical fold
assignments produce bit-identical digests regardless of when they ran.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from nl_ae.schema.hashing import now_utc_iso
from nl_ae.schema.models import Sha256Hex

from .assignment import PilotFoldResult, StratumItem, assign_pilot_fold
from .errors import (
    PilotFoldMismatchError,
    PilotManifestMissingError,
)
from .models import (
    PILOT_MANIFEST_SCHEMA_VERSION,
    PilotManifest,
    StratumRecord,
)

_DIGEST_EXCLUDED_FIELDS: frozenset[str] = frozenset(
    {"created_at", "completion_status", "pilot_manifest_digest"}
)


def compute_pilot_manifest_digest(
    *,
    run_id: str,
    seed: int,
    frac: float,
    stratify_by: tuple[str, ...],
    min_per_stratum: int,
    strata: tuple[StratumRecord, ...],
    pilot_item_ids: tuple[str, ...],
    n_pilot: int,
    n_holdout: int,
    n_total: int,
    schema_version: str = PILOT_MANIFEST_SCHEMA_VERSION,
) -> Sha256Hex:
    """SHA-256 over canonical JSON of the assignment fields (deterministic order)."""
    payload: dict[str, Any] = {
        "schema_version": schema_version,
        "run_id": run_id,
        "seed": seed,
        "frac": frac,
        "stratify_by": list(stratify_by),
        "min_per_stratum": min_per_stratum,
        "strata": [s.model_dump(mode="json") for s in strata],
        "pilot_item_ids": list(pilot_item_ids),
        "n_pilot": n_pilot,
        "n_holdout": n_holdout,
        "n_total": n_total,
    }
    raw = json.dumps(
        payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def build_pilot_manifest(
    *,
    run_id: str,
    seed: int,
    frac: float,
    stratify_by: tuple[str, ...],
    min_per_stratum: int,
    result: PilotFoldResult,
) -> PilotManifest:
    """Wrap a ``PilotFoldResult`` into a ``PilotManifest`` (digest computed)."""
    digest = compute_pilot_manifest_digest(
        run_id=run_id,
        seed=seed,
        frac=frac,
        stratify_by=tuple(stratify_by),
        min_per_stratum=min_per_stratum,
        strata=result.strata,
        pilot_item_ids=result.pilot_item_ids,
        n_pilot=result.n_pilot,
        n_holdout=result.n_holdout,
        n_total=result.n_total,
    )
    return PilotManifest(
        run_id=run_id,
        seed=seed,
        frac=frac,
        stratify_by=tuple(stratify_by),
        min_per_stratum=min_per_stratum,
        strata=result.strata,
        pilot_item_ids=result.pilot_item_ids,
        n_pilot=result.n_pilot,
        n_holdout=result.n_holdout,
        n_total=result.n_total,
        created_at=now_utc_iso(),
        completion_status="committed",
        pilot_manifest_digest=digest,
    )


def write_pilot_manifest(path: Path, manifest: PilotManifest) -> Sha256Hex:
    """Atomic write + ``.sha256`` sidecar; returns SHA-256 of the raw file bytes."""
    payload = manifest.model_dump_json(indent=2, exclude_none=False).encode("utf-8")
    raw_digest = hashlib.sha256(payload).hexdigest()
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(payload)
            f.flush()
            try:
                os.fsync(f.fileno())
            except OSError:
                pass
        os.replace(tmp_name, path)
    except BaseException:
        try:
            Path(tmp_name).unlink(missing_ok=True)
        finally:
            raise
    sidecar = path.with_suffix(path.suffix + ".sha256")
    sidecar.write_text(raw_digest + "\n", encoding="utf-8")
    return raw_digest


def load_pilot_manifest(path: Path) -> PilotManifest:
    """Load + validate a pilot manifest. Raises ``PilotManifestMissingError`` if absent."""
    if not path.exists():
        raise PilotManifestMissingError(f"pilot_manifest.json not found: {path}")
    return PilotManifest.model_validate_json(path.read_bytes())


def assign_and_write_pilot_manifest(
    *,
    run_id: str,
    items: Iterable[StratumItem],
    seed: int,
    frac: float,
    stratify_by: tuple[str, ...],
    min_per_stratum: int,
    out_path: Path,
    on_existing: str = "error",
) -> tuple[PilotManifest, bool]:
    """Run the assignment + write the manifest atomically. Idempotent on identical inputs.

    Returns ``(manifest, was_written)``. If ``on_existing="resume"`` and an
    on-disk manifest exists, the function returns it unchanged when the
    re-derived digest matches; raises ``PilotFoldMismatchError`` on drift.
    ``on_existing="error"`` refuses to overwrite. ``on_existing="overwrite"``
    replaces unconditionally.
    """
    new_result = assign_pilot_fold(
        items,
        seed=seed,
        frac=frac,
        stratify_by=stratify_by,
        min_per_stratum=min_per_stratum,
    )
    new_manifest = build_pilot_manifest(
        run_id=run_id,
        seed=seed,
        frac=frac,
        stratify_by=stratify_by,
        min_per_stratum=min_per_stratum,
        result=new_result,
    )

    if out_path.exists():
        existing = load_pilot_manifest(out_path)
        if on_existing == "resume":
            if existing.pilot_manifest_digest != new_manifest.pilot_manifest_digest:
                raise PilotFoldMismatchError(
                    "existing pilot manifest disagrees with re-derived assignment; "
                    f"on-disk={existing.pilot_manifest_digest!r} "
                    f"derived={new_manifest.pilot_manifest_digest!r}"
                )
            return existing, False
        if on_existing == "error":
            raise FileExistsError(
                f"refusing to overwrite existing pilot manifest: {out_path}"
            )
        if on_existing != "overwrite":
            raise ValueError(
                f"on_existing must be one of resume|error|overwrite; got {on_existing!r}"
            )

    write_pilot_manifest(out_path, new_manifest)
    return new_manifest, True


def iter_item_summaries(rows_path: Path) -> Iterable[StratumItem]:
    """Yield one ``StratumItem`` per distinct ``item_id`` in ``rows.jsonl``.

    Reads JSON manually (skips Pydantic) for speed on 140k+ rows. ``metadata``
    is populated empty because ``ResultRow`` does not carry per-item metadata
    (see ``L2-architecture.md §"Stratum resolution"``); OpinionQA stratification
    falls back to ``subject`` (which the loader populates with ``topic``).
    """
    if not rows_path.exists():
        raise FileNotFoundError(f"rows.jsonl not found: {rows_path}")
    seen: set[str] = set()
    with rows_path.open("rb") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            obj = json.loads(line)
            iid = obj["item_id"]
            if iid in seen:
                continue
            seen.add(iid)
            yield StratumItem(
                item_id=iid,
                dataset_name=obj["dataset_name"],
                subject=obj.get("subject"),
                metadata={},
            )


__all__ = [
    "assign_and_write_pilot_manifest",
    "build_pilot_manifest",
    "compute_pilot_manifest_digest",
    "iter_item_summaries",
    "load_pilot_manifest",
    "write_pilot_manifest",
]
