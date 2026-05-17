"""``rescore_first_token`` — pure forward-only rescore loop.

Model-free: a duck-typed fake engine returns a canned ``FirstTokenScore`` (same
fake idiom as ``test_wrapper_layer_indexing`` / the materialize tests) and the
*real* ``PromptRenderer`` reproduces each row's ``prompt_hash`` so the
integrity gate is exercised for real. No ``torch``/``transformers``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from nl_ae.data.canonical import CanonicalItem
from nl_ae.data.permute import permutation_for
from nl_ae.inference.outputs import FirstTokenScore, LetterScore
from nl_ae.inference.rescore import rescore_first_token
from nl_ae.prompt.errors import PromptHashRecomputeMismatchError
from nl_ae.prompt.materialize import MaterializeInputs
from nl_ae.prompt.renderer import NullChatTemplateAdapter, PromptRenderer
from nl_ae.prompt.template_registry import TemplateRegistry
from nl_ae.schema.hashing import hash_file, now_utc_iso
from nl_ae.schema.models import (
    DatasetFingerprint,
    EnvFingerprint,
    LetterSoftmaxEntry,
    LetterTokenEntry,
    ModelFingerprint,
    PromptTemplateRecord,
    QuantizationSpec,
    ResultRow,
    RunManifest,
)
from nl_ae.schema.paths import run_paths
from nl_ae.schema.reader import load_manifest, load_rows
from nl_ae.schema.writer import write_manifest_atomic

TEMPLATE_ID = "mcq_flat_v1"
RUN_ID = "20260101T000000Z-deadbee-test"
RESCORE_FIELDS = {
    "first_token_letter",
    "letter_softmax",
    "total_letter_mass",
    "agreement_flag",
    "wall_time_ms",
}


# --- fakes --------------------------------------------------------------


class FakeEngine:
    """Returns a non-degenerate score so corrupt all-zero rows visibly change.

    ``argmax`` is the *second* letter in the passed (resolved-variant) subset
    → "B" for an A,B,C,D item; ``token_id`` is read back from the subset so the
    test also proves the resolved-variant entries reach the engine.
    """

    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[int, ...]]] = []

    def score_first_token(
        self,
        prompt: str,
        *,
        letter_token_table: list[LetterTokenEntry],
        scoring_math: str,
    ) -> FirstTokenScore:
        self.calls.append((scoring_math, tuple(e.token_id for e in letter_token_table)))
        k = len(letter_token_table)
        probs = [0.1] * k
        probs[1] = 0.6  # the winner ("B")
        per_letter = [
            LetterScore(
                letter=e.letter,
                token_id=e.token_id,
                prob=probs[i],
                prob_valid=True,
                logit=float(probs[i]),
            )
            for i, e in enumerate(letter_token_table)
        ]
        return FirstTokenScore(
            argmax_letter=letter_token_table[1].letter,
            per_letter=per_letter,
            scoring_math=scoring_math,  # type: ignore[arg-type]
            total_letter_mass=0.6,
            prompt_token_count=7,
            wall_time_ms=12.5,
            engine_call_id=len(self.calls),
        )


# --- builders -----------------------------------------------------------


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _renderer() -> PromptRenderer:
    registry = TemplateRegistry(_repo_root() / "templates")
    registry.load()
    return PromptRenderer(
        registry, chat_adapter=NullChatTemplateAdapter(identity_hash="0" * 64)
    )


def _items(n: int = 4) -> list[CanonicalItem]:
    return [
        CanonicalItem(
            item_id=f"mmlu/v1/test/q-{i:02d}",
            dataset_name="mmlu",
            dataset_split="test",
            subject="test",
            question=f"What is {i} + {i}?",
            choices=("zero", "one", "two", "three"),
            gold_index=2,
        )
        for i in range(n)
    ]


def _corrupt_row(item: CanonicalItem, *, prompt_hash: str, gold: str | None) -> dict:
    """A row with the exact production-bug shape: first="A", every prob 0.0."""
    return ResultRow(
        run_id=RUN_ID,
        item_id=item.item_id,
        dataset_name=item.dataset_name,
        dataset_split=item.dataset_split,
        subject=item.subject,
        template_id=TEMPLATE_ID,
        permutation_id=0,
        prompt_hash=prompt_hash,
        rendered_prompt_ref=f"prompts/{prompt_hash}.txt",
        gold_letter=gold,
        first_token_letter="A",  # spurious tie-break
        free_text_letter="B",
        free_text_raw="B",
        free_text_truncated=False,
        agreement_flag=False,  # A != B (consistent for the seed row)
        letter_softmax=[
            LetterSoftmaxEntry(letter=L, token_id=10 + i, prob=0.0, prob_valid=True, logit=g)
            for i, (L, g) in enumerate(zip("ABCD", [11.3, 17.9, 5.2, 4.7], strict=True))
        ],
        n_options=4,
        free_text_seed=None,
        decode_strategy="greedy",
        activation_ref=None,
        wall_time_ms=580.0,
        created_at=now_utc_iso(),
        extractor_id="regex_ladder",
        extractor_match_rule="letter_only",
        first_token_scoring_math="full_vocab_softmax",
        total_letter_mass=0.0,
    ).model_dump(mode="json")


def _manifest(*, notes: str | None = None) -> RunManifest:
    return RunManifest(
        run_id=RUN_ID,
        git_sha="deadbeefcafe",
        git_dirty=False,
        started_at=now_utc_iso(),
        ended_at=now_utc_iso(),
        completion_status="completed",
        env=EnvFingerprint(
            os_name="Linux",
            os_version="x",
            python_version="3.11",
            cuda_version=None,
            torch_version=None,
            transformers_version=None,
            bitsandbytes_version=None,
            accelerate_version=None,
            gpu_name=None,
            gpu_vram_mb=None,
        ),
        model=ModelFingerprint(
            hf_model_id="Qwen/Qwen2.5-7B-Instruct",
            hf_model_commit="a09a35458c702b33eeacc393d103063234e8bc28",
            hf_tokenizer_id="Qwen/Qwen2.5-7B-Instruct",
            hf_tokenizer_commit=None,
            quantization=QuantizationSpec(kind="fp16"),
        ),
        datasets=[
            DatasetFingerprint(
                name="mmlu",
                hf_dataset_id="cais/mmlu",
                split="test",
                commit_or_revision=None,
                item_count=4,
                item_id_scheme="x",
            )
        ],
        prompt_templates=[
            PromptTemplateRecord(
                template_id=TEMPLATE_ID,
                template_content_hash="a" * 64,
                template_text="Q: {question}\n\n{choice_block}\n",
                role="user",
            )
        ],
        letter_token_table=[
            LetterTokenEntry(letter=L, token_id=10 + i, variant="bare", token_str=L)
            for i, L in enumerate("ABCD")
        ],
        seeds={"root": 0},
        cli_args={"overrides": None},
        config_digest="c" * 64,
        config_yaml_text=None,
        chat_template_hash="0" * 64,
        notes=notes,
    )


def _setup(
    tmp_path: Path, *, bad_hash_on: int | None = None
) -> tuple[MaterializeInputs, list[CanonicalItem], list[dict]]:
    """Write a 4-row corrupt run; optionally inject a wrong hash on one row."""
    paths = run_paths(tmp_path, RUN_ID)
    paths.run_dir.mkdir(parents=True, exist_ok=True)
    renderer = _renderer()
    items = _items(4)

    rows: list[dict] = []
    for idx, item in enumerate(items):
        perm = permutation_for(item, 0, mode="seeded")
        _prompt, real_hash = renderer.render(perm, TEMPLATE_ID)
        ph = "f" * 64 if bad_hash_on == idx else real_hash
        rows.append(_corrupt_row(item, prompt_hash=ph, gold=perm.gold_letter))

    with paths.rows_jsonl.open("wb") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False).encode("utf-8") + b"\n")
    write_manifest_atomic(paths.manifest_json, _manifest())

    inputs = MaterializeInputs(
        run_dir=paths.run_dir,
        paths=paths,
        run_manifest=load_manifest(paths.manifest_json),
        rows_path=paths.rows_jsonl,
        item_index={it.item_id: it for it in items},
        permutation_mode="seeded",
        chat_template_hash="0" * 64,
    )
    return inputs, items, rows


# --- tests --------------------------------------------------------------


def test_happy_path_only_allowed_fields_change(tmp_path: Path) -> None:
    inputs, _, seed_rows = _setup(tmp_path)

    outcome = rescore_first_token(
        inputs,
        renderer=_renderer(),
        engine=FakeEngine(),
        git_sha="cafe1234",
        git_dirty=False,
    )

    assert outcome.status == "completed"
    assert outcome.rows_seen == 4
    assert outcome.rows_changed == 4
    assert outcome.first_token_before == {"A": 4}
    assert outcome.first_token_after == {"B": 4}
    assert outcome.per_scoring_math == {"full_vocab_softmax": 4}

    new_rows = [r.model_dump(mode="json") for r in load_rows(inputs.rows_path)]
    assert len(new_rows) == 4
    for old, new in zip(seed_rows, new_rows, strict=True):
        changed = {k for k in old if old[k] != new[k]}
        assert changed <= RESCORE_FIELDS, f"unexpected field churn: {changed}"
        # The corruption is actually undone:
        assert new["first_token_letter"] == "B"
        assert new["total_letter_mass"] == pytest.approx(0.6)
        assert all(e["prob"] > 0.0 for e in new["letter_softmax"])
        assert new["agreement_flag"] is True  # new "B" == free_text_letter "B"
        assert new["wall_time_ms"] == pytest.approx(12.5)
        # Preserved verbatim:
        assert new["free_text_raw"] == old["free_text_raw"]
        assert new["prompt_hash"] == old["prompt_hash"]
        assert new["gold_letter"] == old["gold_letter"]
        assert new["extractor_match_rule"] == old["extractor_match_rule"]
        assert new["created_at"] == old["created_at"]


def test_atomic_rewrite_and_rescore_manifest(tmp_path: Path) -> None:
    inputs, _i, _r = _setup(tmp_path)
    old_sha = hash_file(inputs.rows_path)

    outcome = rescore_first_token(
        inputs, renderer=_renderer(), engine=FakeEngine(), git_sha="abc", git_dirty=True
    )

    paths = inputs.paths
    assert not paths.rows_jsonl.with_suffix(".jsonl.tmp").exists()  # no temp left
    assert paths.rows_parquet.exists()
    new_sha = hash_file(paths.rows_jsonl)
    assert outcome.old_rows_sha256 == old_sha != new_sha
    assert outcome.new_rows_sha256 == new_sha

    rm = paths.run_dir / "rescore_manifest.json"
    rm_sha = paths.run_dir / "rescore_manifest.json.sha256"
    assert rm.exists() and rm_sha.exists()
    body = rm.read_bytes()
    from nl_ae.schema.hashing import hash_json_bytes

    assert rm_sha.read_text().strip() == hash_json_bytes(body)
    payload = json.loads(body)
    assert payload["old_rows_jsonl_sha256"] == old_sha
    assert payload["new_rows_jsonl_sha256"] == new_sha
    assert payload["rows_changed"] == 4
    assert payload["git_sha"] == "abc"
    assert payload["variant_policy"] == "auto"
    assert payload["first_token_letter_before"] == {"A": 4}


def test_manifest_notes_stamped_identity_untouched(tmp_path: Path) -> None:
    inputs, _i, _r = _setup(tmp_path)
    before = load_manifest(inputs.paths.manifest_json)

    rescore_first_token(
        inputs, renderer=_renderer(), engine=FakeEngine(), git_sha="x", git_dirty=False
    )

    after = load_manifest(inputs.paths.manifest_json)
    assert after.notes is not None and "rescore-first-token" in after.notes
    # The run identity is never mutated by a rescore.
    assert after.run_id == before.run_id
    assert after.config_digest == before.config_digest
    assert after.seeds == before.seeds
    assert after.completion_status == before.completion_status


def test_prompt_hash_mismatch_is_integrity_gate(tmp_path: Path) -> None:
    inputs, _i, _r = _setup(tmp_path, bad_hash_on=2)
    sha_before = hash_file(inputs.rows_path)

    with pytest.raises(PromptHashRecomputeMismatchError):
        rescore_first_token(
            inputs, renderer=_renderer(), engine=FakeEngine(), git_sha="x", git_dirty=False
        )

    # Nothing was written: rows.jsonl byte-identical, no temp, no sidecars.
    assert hash_file(inputs.rows_path) == sha_before
    assert not inputs.paths.rows_jsonl.with_suffix(".jsonl.tmp").exists()
    assert not (inputs.paths.run_dir / "rescore_manifest.json").exists()
    assert load_manifest(inputs.paths.manifest_json).notes is None


def test_dry_run_writes_nothing(tmp_path: Path) -> None:
    inputs, _i, _r = _setup(tmp_path)
    sha_before = hash_file(inputs.rows_path)

    outcome = rescore_first_token(
        inputs,
        renderer=_renderer(),
        engine=FakeEngine(),
        git_sha="x",
        git_dirty=False,
        dry_run=True,
    )

    assert outcome.dry_run is True
    assert outcome.rows_seen == 4
    assert outcome.rows_changed == 4
    assert outcome.first_token_after == {"B": 4}
    assert outcome.new_rows_sha256 is None
    assert outcome.rescore_manifest_ref is None
    # Disk is pristine.
    assert hash_file(inputs.rows_path) == sha_before
    assert not (inputs.paths.run_dir / "rescore_manifest.json").exists()
    assert not inputs.paths.rows_jsonl.with_suffix(".jsonl.tmp").exists()
    assert load_manifest(inputs.paths.manifest_json).notes is None


def test_limit_processes_prefix_only(tmp_path: Path) -> None:
    inputs, _i, _r = _setup(tmp_path)
    outcome = rescore_first_token(
        inputs,
        renderer=_renderer(),
        engine=FakeEngine(),
        git_sha="x",
        git_dirty=False,
        limit=2,
    )
    assert outcome.rows_seen == 2
    # The rewritten file holds only the processed prefix.
    assert sum(1 for _ in load_rows(inputs.rows_path)) == 2
