"""ActivationCacheExtractor — fold filter, prompt replay, resume, preregistration gate.

The wrapper / model is fully bypassed: a deterministic fake
:class:`ActivationSource` returns one tensor per requested layer keyed off the
prompt's SHA-256 (so test assertions can match exact bytes), and Phase 1
artifacts are minted in-process with the schema package.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from nl_ae.cache.errors import (
    PromptHashMismatchError,
    PromptSidecarMissingError,
)
from nl_ae.cache.extractor import (
    ActivationCacheExtractor,
    load_extractor_inputs,
    scan_resume_state,
)
from nl_ae.cache.reader import ActivationCacheReader
from nl_ae.pilot.errors import PreregistrationMissingError, PreregistrationUnlockedError
from nl_ae.pilot.manifest import write_pilot_manifest
from nl_ae.pilot.models import (
    NlaScopeSpec,
    PilotManifest,
    Preregistration,
    StratumRecord,
)
from nl_ae.schema.hashing import now_utc_iso
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
from nl_ae.schema.writer import ResultsWriter

DIM = 8
N_LAYERS = 4
CHAT_TEMPLATE_HASH = "c" * 64


@dataclass
class _ForwardOutput:
    hidden_states: dict[int, _FakeVec] | None


@dataclass(frozen=True)
class _FakeVec:
    values: tuple[float, ...]

    def tolist(self) -> list[float]:
        return list(self.values)


@dataclass
class FakeSource:
    """Deterministic stand-in for :class:`Qwen25Wrapper`.

    Per-layer output for a prompt is ``[float(sha256(prompt)[i:i+2]) + layer for i in 0..dim-1]``
    truncated/cast to floats — distinct per prompt and per layer.
    """

    hidden_size: int = DIM
    n_layers: int = N_LAYERS
    chat_template_hash: str = CHAT_TEMPLATE_HASH
    calls: list[tuple[str, tuple[int, ...]]] = field(default_factory=list)

    def forward(
        self,
        prompt: str,
        *,
        capture_hiddens: bool = True,
        record_layers_override: tuple[int, ...] | None = None,
    ) -> _ForwardOutput:
        layers = tuple(record_layers_override or ())
        self.calls.append((prompt, layers))
        if not capture_hiddens:
            return _ForwardOutput(hidden_states=None)
        digest = hashlib.sha256(prompt.encode("utf-8")).digest()
        base = [float(digest[i % len(digest)]) for i in range(self.hidden_size)]
        return _ForwardOutput(
            hidden_states={
                layer: _FakeVec(tuple(v + layer for v in base)) for layer in layers
            },
        )


# --- run-dir factory ----------------------------------------------------


def _hash_prompt(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _row(run_id: str, *, item_id: str, perm: int, prompt: str) -> tuple[ResultRow, str]:
    prompt_hash = _hash_prompt(prompt)
    return (
        ResultRow(
            run_id=run_id,
            item_id=item_id,
            dataset_name="mmlu",
            dataset_split="test",
            subject="algebra",
            template_id="mcq_flat_v1",
            permutation_id=perm,
            prompt_hash=prompt_hash,
            rendered_prompt_ref=f"prompts/{prompt_hash}.txt",
            gold_letter="A",
            first_token_letter="A",
            free_text_letter="A",
            free_text_raw="A.",
            agreement_flag=True,
            letter_softmax=[
                LetterSoftmaxEntry(
                    letter="A", token_id=1, prob=0.7, prob_valid=True, logit=0.0
                ),
                LetterSoftmaxEntry(
                    letter="B", token_id=2, prob=0.3, prob_valid=True, logit=0.0
                ),
            ],
            n_options=2,
            free_text_seed=None,
            decode_strategy="greedy",
            created_at=now_utc_iso(),
            extractor_id="regex_ladder",
            extractor_match_rule="my_answer_is",
            first_token_scoring_math="full_vocab_softmax",
            total_letter_mass=1.0,
        ),
        prompt,
    )


def _build_phase1(tmp_path: Path, *, pilot_items: list[str], holdout_items: list[str]) -> Path:
    run_id = "20260101T000000Z-deadbee-test"
    run_dir = tmp_path / run_id
    paths = run_paths(run_dir.parent, run_dir.name)
    manifest = RunManifest(
        run_id=run_id,
        git_sha="deadbeefcafe",
        git_dirty=False,
        started_at=now_utc_iso(),
        completion_status="in_progress",
        env=EnvFingerprint(
            os_name="Windows",
            os_version="11",
            python_version="3.13.0",
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
            hf_model_commit=None,
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
                item_count=len(pilot_items) + len(holdout_items),
                item_id_scheme="mmlu/v1/<subject>/q-<sha256>[:12]",
            )
        ],
        prompt_templates=[
            PromptTemplateRecord(
                template_id="mcq_flat_v1",
                template_content_hash="0" * 64,
                template_text="x",
                role="user",
            )
        ],
        letter_token_table=[
            LetterTokenEntry(letter="A", token_id=10, variant="bare", token_str="A"),
            LetterTokenEntry(letter="B", token_id=11, variant="bare", token_str="B"),
        ],
        seeds={"root": 0},
        cli_args={"config_path": "x"},
        config_digest="a" * 64,
        chat_template_hash=CHAT_TEMPLATE_HASH,
    )
    with ResultsWriter(run_dir, manifest, on_existing="error") as writer:
        for idx, iid in enumerate(pilot_items + holdout_items):
            row, prompt = _row(
                run_id, item_id=iid, perm=idx % 3, prompt=f"prompt for {iid}#{idx}"
            )
            writer.write_row(row)
            # Phase 1's renderer writes sidecars; mimic that.
            sidecar = paths.prompts_dir / f"{row.prompt_hash}.txt"
            sidecar.write_text(prompt, encoding="utf-8")
        writer.finalize(emit_parquet=False)

    # Pilot manifest.
    pilot_manifest = PilotManifest(
        run_id=run_id,
        seed=0,
        frac=0.5,
        stratify_by=("subject",),
        min_per_stratum=1,
        strata=(
            StratumRecord(
                key="algebra",
                source_field="subject",
                n_total=len(pilot_items) + len(holdout_items),
                n_pilot=len(pilot_items),
                n_holdout=len(holdout_items),
            ),
        ),
        pilot_item_ids=tuple(sorted(pilot_items)),
        n_pilot=len(pilot_items),
        n_holdout=len(holdout_items),
        n_total=len(pilot_items) + len(holdout_items),
        created_at=now_utc_iso(),
        completion_status="committed",
        pilot_manifest_digest="abc" + "0" * 61,
    )
    write_pilot_manifest(paths.pilot_manifest_json, pilot_manifest)
    return run_dir


def _write_preregistration(
    run_dir: Path,
    *,
    pilot_manifest_digest: str,
    locked: bool = True,
) -> None:
    prereg = Preregistration(
        run_id="20260101T000000Z-deadbee-test",
        pilot_manifest_digest=pilot_manifest_digest,
        holdout_vs_full="holdout-only",
        labels=("first_token_correct",),
        layers=(0,),
        nla_scope=NlaScopeSpec(
            layer=0,
            fold="holdout",
            limit=10,
            decode_strategy="greedy",
            temperature=0.7,
            max_new_tokens=64,
        ),
        primary_hypothesis="x",
        significance_threshold=0.05,
        multiple_comparison_correction="bonferroni",
        n_comparisons=1,
        effect_size_metric="accuracy_minus_baseline",
        effect_size_threshold=0.0,
        locked_at=now_utc_iso() if locked else None,
        locked_at_git_sha=("a" * 40) if locked else None,
    )
    import yaml

    fm = yaml.safe_dump(prereg.model_dump(mode="json"), sort_keys=False)
    (run_dir / "preregistration.md").write_text(
        "---\n" + fm + "---\n\n# preregistration body\n",
        encoding="utf-8",
    )


# --- tests --------------------------------------------------------------


def test_extractor_pilot_happy_path(tmp_path: Path) -> None:
    pilot_items = ["mmlu/v1/algebra/q-pilot01", "mmlu/v1/algebra/q-pilot02"]
    holdout_items = ["mmlu/v1/algebra/q-hold01", "mmlu/v1/algebra/q-hold02"]
    run_dir = _build_phase1(tmp_path, pilot_items=pilot_items, holdout_items=holdout_items)

    inputs = load_extractor_inputs(run_dir, "pilot")
    extractor = ActivationCacheExtractor(
        inputs=inputs,
        source=FakeSource(),
        layers=(0, 1, 2),
        shard_rows=100,
    )
    outcome = extractor.run()
    assert outcome.status == "completed"
    # Each pilot item appears once in rows.jsonl (see _build_phase1), so pilot
    # has exactly 2 rows.
    assert outcome.rows_written == 2
    assert outcome.rows_expected == 2

    reader = ActivationCacheReader.open(run_dir, "pilot")
    assert reader.manifest.layers == (0, 1, 2)
    assert reader.manifest.fold == "pilot"
    visit_keys = reader.completed_visit_keys()
    assert {k[0] for k in visit_keys} == set(pilot_items)
    # Holdout cache should not exist.
    assert not (run_dir / "holdout" / "activations" / "activation_manifest.json").exists()


def test_extractor_filters_to_fold(tmp_path: Path) -> None:
    pilot_items = ["mmlu/v1/algebra/q-pilot01"]
    holdout_items = ["mmlu/v1/algebra/q-hold01", "mmlu/v1/algebra/q-hold02"]
    run_dir = _build_phase1(tmp_path, pilot_items=pilot_items, holdout_items=holdout_items)

    # Pilot reads 1 row.
    inputs = load_extractor_inputs(run_dir, "pilot")
    ActivationCacheExtractor(
        inputs=inputs, source=FakeSource(), layers=(0,), shard_rows=100
    ).run()
    pilot_reader = ActivationCacheReader.open(run_dir, "pilot")
    assert pilot_reader.manifest.rows_written == 1

    # Holdout requires a locked preregistration → write one matching the pilot digest.
    from nl_ae.pilot.manifest import load_pilot_manifest

    pm = load_pilot_manifest(run_dir / "pilot_manifest.json")
    _write_preregistration(
        run_dir, pilot_manifest_digest=pm.pilot_manifest_digest, locked=True
    )
    holdout_inputs = load_extractor_inputs(run_dir, "holdout")
    ActivationCacheExtractor(
        inputs=holdout_inputs, source=FakeSource(), layers=(0,), shard_rows=100
    ).run()
    holdout_reader = ActivationCacheReader.open(run_dir, "holdout")
    assert holdout_reader.manifest.rows_written == 2
    assert holdout_reader.manifest.preregistration_digest is not None
    # No leakage between folds.
    pilot_keys = pilot_reader.completed_visit_keys()
    holdout_keys = holdout_reader.completed_visit_keys()
    assert pilot_keys.isdisjoint(holdout_keys)


def test_holdout_refuses_without_preregistration(tmp_path: Path) -> None:
    run_dir = _build_phase1(
        tmp_path,
        pilot_items=["mmlu/v1/algebra/q-p1"],
        holdout_items=["mmlu/v1/algebra/q-h1"],
    )
    with pytest.raises(PreregistrationMissingError):
        load_extractor_inputs(run_dir, "holdout")


def test_holdout_refuses_when_preregistration_unlocked(tmp_path: Path) -> None:
    run_dir = _build_phase1(
        tmp_path,
        pilot_items=["mmlu/v1/algebra/q-p1"],
        holdout_items=["mmlu/v1/algebra/q-h1"],
    )
    from nl_ae.pilot.manifest import load_pilot_manifest

    pm = load_pilot_manifest(run_dir / "pilot_manifest.json")
    _write_preregistration(
        run_dir, pilot_manifest_digest=pm.pilot_manifest_digest, locked=False
    )
    with pytest.raises(PreregistrationUnlockedError):
        load_extractor_inputs(run_dir, "holdout")


def test_extractor_refuses_missing_prompt_sidecar(tmp_path: Path) -> None:
    run_dir = _build_phase1(
        tmp_path,
        pilot_items=["mmlu/v1/algebra/q-p1"],
        holdout_items=["mmlu/v1/algebra/q-h1"],
    )
    # Wipe the prompts dir.
    prompts_dir = run_dir / "prompts"
    for p in prompts_dir.glob("*.txt"):
        p.unlink()

    inputs = load_extractor_inputs(run_dir, "pilot")
    extractor = ActivationCacheExtractor(
        inputs=inputs, source=FakeSource(), layers=(0,), shard_rows=100
    )
    with pytest.raises(PromptSidecarMissingError):
        extractor.run()


def test_extractor_refuses_tampered_prompt_sidecar(tmp_path: Path) -> None:
    run_dir = _build_phase1(
        tmp_path,
        pilot_items=["mmlu/v1/algebra/q-p1"],
        holdout_items=["mmlu/v1/algebra/q-h1"],
    )
    # Mutate every sidecar so the pilot row's hash check fails regardless of order.
    for sidecar in (run_dir / "prompts").glob("*.txt"):
        sidecar.write_text("tampered prompt body", encoding="utf-8")

    inputs = load_extractor_inputs(run_dir, "pilot")
    extractor = ActivationCacheExtractor(
        inputs=inputs, source=FakeSource(), layers=(0,), shard_rows=100
    )
    with pytest.raises(PromptHashMismatchError):
        extractor.run()


def test_extractor_resume_skips_completed_visits(tmp_path: Path) -> None:
    run_dir = _build_phase1(
        tmp_path,
        pilot_items=[f"mmlu/v1/algebra/q-p{i:02d}" for i in range(6)],
        holdout_items=["mmlu/v1/algebra/q-h1"],
    )
    inputs = load_extractor_inputs(run_dir, "pilot")
    # First pass: write only the first 3 rows by limit.
    source = FakeSource()
    ActivationCacheExtractor(
        inputs=inputs, source=source, layers=(0, 1), shard_rows=100, limit=3
    ).run()
    first_pass_calls = len(source.calls)

    # Second pass: resume — should skip the first 3 and only forward through the rest.
    seed_shards, completed = scan_resume_state(
        run_dir=run_dir, fold="pilot", layers=(0, 1)
    )
    assert len(completed) == 3
    source2 = FakeSource()
    outcome = ActivationCacheExtractor(
        inputs=inputs,
        source=source2,
        layers=(0, 1),
        shard_rows=100,
        seed_shards=seed_shards,
        completed_visit_keys=completed,
    ).run()
    assert outcome.rows_skipped_resume == 3
    # Second pass touched only 3 new rows.
    assert len(source2.calls) == 3
    assert first_pass_calls == 3

    reader = ActivationCacheReader.open(run_dir, "pilot")
    assert reader.manifest.rows_written == 6
    for ls in reader.manifest.layer_shards:
        assert ls.rows == 6


def test_extractor_refuses_chat_template_drift(tmp_path: Path) -> None:
    run_dir = _build_phase1(
        tmp_path,
        pilot_items=["mmlu/v1/algebra/q-p1"],
        holdout_items=["mmlu/v1/algebra/q-h1"],
    )
    inputs = load_extractor_inputs(run_dir, "pilot")
    drifted = FakeSource(chat_template_hash="f" * 64)
    with pytest.raises(RuntimeError, match="chat_template_hash mismatch"):
        ActivationCacheExtractor(
            inputs=inputs, source=drifted, layers=(0,), shard_rows=100
        )


def test_extractor_refuses_out_of_range_layers(tmp_path: Path) -> None:
    run_dir = _build_phase1(
        tmp_path,
        pilot_items=["mmlu/v1/algebra/q-p1"],
        holdout_items=["mmlu/v1/algebra/q-h1"],
    )
    inputs = load_extractor_inputs(run_dir, "pilot")
    with pytest.raises(ValueError, match="out of"):
        ActivationCacheExtractor(
            inputs=inputs, source=FakeSource(n_layers=4), layers=(5,), shard_rows=100
        )


def test_extractor_deterministic_under_replay(tmp_path: Path) -> None:
    """Same inputs → bit-identical activation vectors across runs."""
    run_dir = _build_phase1(
        tmp_path,
        pilot_items=["mmlu/v1/algebra/q-p1", "mmlu/v1/algebra/q-p2"],
        holdout_items=["mmlu/v1/algebra/q-h1"],
    )
    inputs = load_extractor_inputs(run_dir, "pilot")
    ActivationCacheExtractor(
        inputs=inputs, source=FakeSource(), layers=(0, 1, 2), shard_rows=100
    ).run()
    reader = ActivationCacheReader.open(run_dir, "pilot")
    vectors_a = sorted(
        ((r["item_id"], r["permutation_id"], tuple(r["activation"])) for r in reader.iter_rows(0)),
        key=lambda t: (t[0], t[1]),
    )

    # Re-extract into a separate fold dir by wiping pilot.
    import shutil

    shutil.rmtree(run_dir / "pilot")
    ActivationCacheExtractor(
        inputs=inputs, source=FakeSource(), layers=(0, 1, 2), shard_rows=100
    ).run()
    reader2 = ActivationCacheReader.open(run_dir, "pilot")
    vectors_b = sorted(
        ((r["item_id"], r["permutation_id"], tuple(r["activation"])) for r in reader2.iter_rows(0)),
        key=lambda t: (t[0], t[1]),
    )
    assert vectors_a == vectors_b
