"""Forward-only, in-place rescore of the broken first-token fields.

Recovers a Phase 1 run whose ``first_token_letter`` / ``letter_softmax`` /
``total_letter_mass`` were corrupted by the fp16 full-vocab-softmax underflow
*and* the letter-variant/position mismatch (``firsttoken-fix.md`` §1). Only
those fields (plus the derived ``agreement_flag`` and the re-measured
``wall_time_ms``) are recomputed; prompts, free-text generation, the C06
activation cache and the pilot fold are provably unaffected and preserved
verbatim.

The loop mirrors :mod:`nl_ae.prompt.materialize`: it reuses
:func:`nl_ae.prompt.materialize.load_materialize_inputs` for the manifest /
item-index / permutation-mode / chat-hash preflight (single-sourcing the
prompt-hash integrity gate), then per row re-renders the prompt, asserts the
recomputed ``prompt_hash`` matches, runs **one forward pass** through the
pinned engine, and rewrites ``rows.jsonl`` atomically. ``torch`` is never
imported here — the engine is injected behind :class:`FirstTokenScorer`, so
this module is unit-testable with a fake.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import time
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, cast

from nl_ae.data.permute import permutation_for
from nl_ae.prompt.errors import ItemNotInLoaderError, PromptHashRecomputeMismatchError
from nl_ae.prompt.letter_tokens import LetterVariantPolicy, resolve_letter_variant
from nl_ae.schema.hashing import hash_file, hash_json_bytes, now_utc_iso
from nl_ae.schema.models import LetterTokenEntry, ResultRow
from nl_ae.schema.reader import load_manifest
from nl_ae.schema.writer import derive_parquet_from_jsonl, write_manifest_atomic

if TYPE_CHECKING:  # pragma: no cover - typing only.
    from nl_ae.inference.outputs import FirstTokenScore
    from nl_ae.prompt.materialize import MaterializeInputs
    from nl_ae.prompt.renderer import PromptRenderer
    from nl_ae.schema.models import FirstTokenScoringMath


class FirstTokenScorer(Protocol):
    """The single engine method the forward-only rescore needs."""

    def score_first_token(
        self,
        prompt: str,
        *,
        letter_token_table: list[LetterTokenEntry],
        scoring_math: FirstTokenScoringMath,
    ) -> FirstTokenScore: ...


@dataclass(frozen=True)
class RescoreOutcome:
    rows_seen: int
    rows_changed: int
    dry_run: bool
    status: str
    wall_seconds: float
    first_token_before: dict[str, int]
    first_token_after: dict[str, int]
    per_scoring_math: dict[str, int]
    old_rows_sha256: str | None
    new_rows_sha256: str | None
    rescore_manifest_ref: str | None
    failure_reason: str | None = None
    notes: tuple[str, ...] = field(default_factory=tuple)


def rescore_first_token(
    inputs: MaterializeInputs,
    *,
    renderer: PromptRenderer,
    engine: FirstTokenScorer,
    git_sha: str,
    git_dirty: bool,
    variant_policy: LetterVariantPolicy = "auto",
    dry_run: bool = False,
    limit: int | None = None,
    logger: logging.Logger | None = None,
) -> RescoreOutcome:
    """Re-derive only the broken first-token fields for every row, in place.

    Raises :class:`PromptHashRecomputeMismatchError` (integrity gate) or
    :class:`ItemNotInLoaderError` *before* any replacement, so a failure leaves
    ``rows.jsonl`` byte-identical.
    """
    t0 = time.perf_counter()
    paths = inputs.paths
    manifest = inputs.run_manifest
    letter_index = _letter_token_index(manifest.letter_token_table)

    rows_path = paths.rows_jsonl
    old_sha = hash_file(rows_path) if rows_path.exists() else None
    tmp_path = rows_path.with_suffix(rows_path.suffix + ".tmp")

    rows_seen = 0
    rows_changed = 0
    before: Counter[str] = Counter()
    after: Counter[str] = Counter()
    per_math: Counter[str] = Counter()

    handle = None if dry_run else tmp_path.open("wb")
    try:
        for raw in _iter_rows_jsonl(rows_path):
            if limit is not None and rows_seen >= limit:
                break
            rows_seen += 1
            row, changed = _rescore_one(
                raw,
                inputs=inputs,
                renderer=renderer,
                engine=engine,
                letter_index=letter_index,
                variant_policy=variant_policy,
            )
            before[_letter_key(raw.get("first_token_letter"))] += 1
            after[_letter_key(row.first_token_letter)] += 1
            per_math[row.first_token_scoring_math] += 1
            if changed:
                rows_changed += 1
            if handle is not None:
                handle.write(row.model_dump_json(exclude_none=False).encode("utf-8") + b"\n")
            if logger is not None and rows_seen % 1000 == 0:
                logger.info("progress %d (changed=%d)", rows_seen, rows_changed)
    except BaseException:
        if handle is not None:
            handle.close()
            tmp_path.unlink(missing_ok=True)
        raise
    if handle is not None:
        handle.flush()
        with contextlib.suppress(OSError):
            os.fsync(handle.fileno())
        handle.close()

    notes: list[str] = []
    new_sha: str | None = None
    rescore_ref: str | None = None
    if not dry_run:
        os.replace(tmp_path, rows_path)
        new_sha = hash_file(rows_path)
        derive_parquet_from_jsonl(rows_path, paths.rows_parquet)
        rescore_ref = _write_rescore_manifest(
            paths.run_dir,
            run_id=manifest.run_id,
            git_sha=git_sha,
            git_dirty=git_dirty,
            old_rows_sha256=old_sha,
            new_rows_sha256=new_sha,
            rows_seen=rows_seen,
            rows_changed=rows_changed,
            first_token_before=dict(before),
            first_token_after=dict(after),
            per_scoring_math=dict(per_math),
            variant_policy=variant_policy,
        )
        notes.append(_stamp_manifest_notes(paths.manifest_json, rescore_ref=rescore_ref))

    return RescoreOutcome(
        rows_seen=rows_seen,
        rows_changed=rows_changed,
        dry_run=dry_run,
        status="completed",
        wall_seconds=time.perf_counter() - t0,
        first_token_before=dict(before),
        first_token_after=dict(after),
        per_scoring_math=dict(per_math),
        old_rows_sha256=old_sha,
        new_rows_sha256=new_sha,
        rescore_manifest_ref=rescore_ref,
        notes=tuple(notes),
    )


# --- internals ---------------------------------------------------------


def _rescore_one(
    raw: dict[str, object],
    *,
    inputs: MaterializeInputs,
    renderer: PromptRenderer,
    engine: FirstTokenScorer,
    letter_index: dict[tuple[str, str], LetterTokenEntry],
    variant_policy: LetterVariantPolicy,
) -> tuple[ResultRow, bool]:
    item_id = str(raw["item_id"])
    item = inputs.item_index.get(item_id)
    if item is None:
        raise ItemNotInLoaderError(
            f"row references item_id={item_id!r} which is not in the loader index"
        )
    perm = permutation_for(
        item, cast(int, raw["permutation_id"]), mode=inputs.permutation_mode
    )
    prompt, computed_hash = renderer.render(perm, str(raw["template_id"]))
    if computed_hash != raw["prompt_hash"]:
        raise PromptHashRecomputeMismatchError(
            f"prompt hash mismatch for item_id={item_id!r} "
            f"template_id={raw['template_id']!r} "
            f"permutation_id={raw['permutation_id']!r}: "
            f"expected={raw['prompt_hash']} actual={computed_hash}"
        )

    resolved = (
        resolve_letter_variant(prompt) if variant_policy == "auto" else variant_policy
    )
    letters = tuple(perm.letters)
    subset = [
        letter_index[(letter, resolved)]
        for letter in letters
        if (letter, resolved) in letter_index
    ]
    if len(subset) != len(letters):
        # Mirror score_and_generate's safety net: whichever variant is present.
        wanted = set(letters)
        subset = [e for e in inputs.run_manifest.letter_token_table if e.letter in wanted]

    scoring_math = cast("FirstTokenScoringMath", raw["first_token_scoring_math"])
    score = engine.score_first_token(
        prompt, letter_token_table=subset, scoring_math=scoring_math
    )

    free_letter = raw.get("free_text_letter")
    if score.argmax_letter is not None and free_letter is not None:
        agreement: bool | None = score.argmax_letter == free_letter
    else:
        agreement = None

    old_fingerprint = _score_fingerprint(raw)
    data = dict(raw)
    data["first_token_letter"] = score.argmax_letter
    data["letter_softmax"] = [ls.model_dump() for ls in score.per_letter]
    data["total_letter_mass"] = score.total_letter_mass
    data["agreement_flag"] = agreement
    data["wall_time_ms"] = score.wall_time_ms
    # first_token_scoring_math is re-stamped unchanged (same value).
    row = ResultRow.model_validate(data)
    return row, _score_fingerprint(data) != old_fingerprint


def _score_fingerprint(d: dict[str, object]) -> tuple[object, ...]:
    """The subset of fields the rescore is allowed to change."""
    softmax = cast("list[dict[str, object]]", d.get("letter_softmax") or [])
    probs = tuple((e.get("prob"), e.get("prob_valid")) for e in softmax)
    return (d.get("first_token_letter"), d.get("total_letter_mass"), probs)


def _letter_token_index(
    table: list[LetterTokenEntry],
) -> dict[tuple[str, str], LetterTokenEntry]:
    return {(e.letter, e.variant): e for e in table}


def _letter_key(value: object) -> str:
    return str(value) if value is not None else "<none>"


def _iter_rows_jsonl(rows_path: Path) -> Iterator[dict[str, object]]:
    if not rows_path.exists():
        return
    with rows_path.open("rb") as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue
            yield json.loads(stripped)


def _write_rescore_manifest(
    run_dir: Path,
    *,
    run_id: str,
    git_sha: str,
    git_dirty: bool,
    old_rows_sha256: str | None,
    new_rows_sha256: str | None,
    rows_seen: int,
    rows_changed: int,
    first_token_before: dict[str, int],
    first_token_after: dict[str, int],
    per_scoring_math: dict[str, int],
    variant_policy: str,
) -> str:
    """Write ``rescore_manifest.json`` (+ ``.sha256``) — audit-only sidecar.

    Deliberately *not* a ``RunManifest``/schema model: this carries remediation
    provenance without a SemVer schema bump (``firsttoken-fix.md`` §4.3/§5).
    """
    payload = {
        "kind": "rescore-first-token",
        "run_id": run_id,
        "rescored_at": now_utc_iso(),
        "git_sha": git_sha,
        "git_dirty": git_dirty,
        "variant_policy": variant_policy,
        "rows_seen": rows_seen,
        "rows_changed": rows_changed,
        "old_rows_jsonl_sha256": old_rows_sha256,
        "new_rows_jsonl_sha256": new_rows_sha256,
        "first_token_letter_before": first_token_before,
        "first_token_letter_after": first_token_after,
        "per_scoring_math": per_scoring_math,
        "wall_time_ms_note": (
            "wall_time_ms was overwritten with the forward-only time "
            "(faster than the original score+generate)"
        ),
    }
    out = run_dir / "rescore_manifest.json"
    body = json.dumps(payload, indent=2, sort_keys=True).encode("utf-8") + b"\n"
    out.write_bytes(body)
    (run_dir / "rescore_manifest.json.sha256").write_text(
        hash_json_bytes(body) + "\n", encoding="utf-8"
    )
    return "rescore_manifest.json"


def _stamp_manifest_notes(manifest_json: Path, *, rescore_ref: str) -> str:
    """Append a one-line audit stamp to ``manifest.notes`` (no schema change)."""
    manifest = load_manifest(manifest_json)
    stamp = (
        f"[rescore-first-token {now_utc_iso()}] first-token fields re-derived "
        f"forward-only; see {rescore_ref}"
    )
    existing = manifest.notes
    new_notes = stamp if not existing else f"{existing}\n{stamp}"
    # Only ``notes`` changes; run_id / seeds / config_digest / completion_status
    # are left exactly as the original run wrote them.
    write_manifest_atomic(manifest_json, manifest.model_copy(update={"notes": new_notes}))
    return stamp


__all__ = [
    "FirstTokenScorer",
    "RescoreOutcome",
    "rescore_first_token",
]
