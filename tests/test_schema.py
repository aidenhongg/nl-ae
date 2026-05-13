"""Schema 1.1.0 invariants — UR6 prob_valid, CI.02 nullable gold, CI.19 required digest."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from nl_ae.schema.hashing import now_utc_iso
from nl_ae.schema.models import (
    SCHEMA_VERSION,
    DatasetFingerprint,
    EnvFingerprint,
    LetterSoftmaxEntry,
    ModelFingerprint,
    PromptTemplateRecord,
    QuantizationSpec,
    LetterTokenEntry,
    ResultRow,
    RunManifest,
)


def _softmax(probs: list[float]) -> list[LetterSoftmaxEntry]:
    letters = "ABCDEFGHIJ"
    return [
        LetterSoftmaxEntry(
            letter=letters[i], token_id=100 + i, prob=p, prob_valid=True, logit=0.0
        )
        for i, p in enumerate(probs)
    ]


def _base_row(**over: object) -> ResultRow:
    fields: dict[str, object] = dict(
        run_id="20260101T000000Z-deadbee",
        item_id="mmlu/v1/abstract_algebra/q-abc123def456",
        dataset_name="mmlu",
        dataset_split="test",
        subject="abstract_algebra",
        template_id="mcq_flat_v1",
        permutation_id=0,
        prompt_hash="0" * 64,
        gold_letter="A",
        first_token_letter="A",
        free_text_letter="A",
        free_text_raw="My answer is A.",
        agreement_flag=True,
        letter_softmax=_softmax([0.6, 0.2, 0.1, 0.1]),
        n_options=4,
        free_text_seed=None,
        decode_strategy="greedy",
        created_at=now_utc_iso(),
        extractor_id="regex_ladder",
        extractor_match_rule="my_answer_is",
        first_token_scoring_math="full_vocab_softmax",
        total_letter_mass=1.0,
    )
    fields.update(over)
    return ResultRow.model_validate(fields)


def test_result_row_round_trip() -> None:
    row = _base_row()
    serialized = row.model_dump_json()
    restored = ResultRow.model_validate_json(serialized)
    assert restored == row
    assert restored.schema_version == SCHEMA_VERSION


def test_gold_letter_nullable_for_opinionqa() -> None:
    row = _base_row(
        dataset_name="opinionqa",
        subject=None,
        gold_letter=None,
        agreement_flag=None,
        first_token_letter="B",
        free_text_letter=None,
    )
    assert row.gold_letter is None
    assert row.agreement_flag is None


def test_argmax_logits_only_requires_prob_valid_false() -> None:
    bad_softmax = [
        LetterSoftmaxEntry(letter="A", token_id=1, prob=0.7, prob_valid=True, logit=1.0),
        LetterSoftmaxEntry(letter="B", token_id=2, prob=0.3, prob_valid=True, logit=0.5),
    ]
    with pytest.raises(ValidationError):
        _base_row(
            n_options=2,
            letter_softmax=bad_softmax,
            first_token_scoring_math="argmax_logits_only",
        )

    good_softmax = [
        LetterSoftmaxEntry(letter="A", token_id=1, prob=None, prob_valid=False, logit=1.0),
        LetterSoftmaxEntry(letter="B", token_id=2, prob=None, prob_valid=False, logit=0.5),
    ]
    row = _base_row(
        n_options=2,
        letter_softmax=good_softmax,
        first_token_scoring_math="argmax_logits_only",
        total_letter_mass=0.0,
    )
    assert all(not e.prob_valid for e in row.letter_softmax)


def test_full_vocab_softmax_requires_valid_probs() -> None:
    bad_softmax = [
        LetterSoftmaxEntry(letter="A", token_id=1, prob=None, prob_valid=False, logit=1.0),
        LetterSoftmaxEntry(letter="B", token_id=2, prob=None, prob_valid=False, logit=0.5),
    ]
    with pytest.raises(ValidationError):
        _base_row(
            n_options=2,
            letter_softmax=bad_softmax,
            first_token_scoring_math="full_vocab_softmax",
            total_letter_mass=0.0,
        )


def test_agreement_flag_consistency() -> None:
    with pytest.raises(ValidationError):
        _base_row(first_token_letter="A", free_text_letter="B", agreement_flag=True)


def test_letter_softmax_n_options_mismatch() -> None:
    with pytest.raises(ValidationError):
        _base_row(n_options=4, letter_softmax=_softmax([0.5, 0.5]))


def test_letter_softmax_duplicate_letters() -> None:
    bad = [
        LetterSoftmaxEntry(letter="A", token_id=1, prob=0.5, prob_valid=True, logit=0.0),
        LetterSoftmaxEntry(letter="A", token_id=2, prob=0.5, prob_valid=True, logit=0.0),
    ]
    with pytest.raises(ValidationError):
        _base_row(n_options=2, letter_softmax=bad)


def _minimal_manifest(**over: object) -> RunManifest:
    fields: dict[str, object] = dict(
        run_id="20260101T000000Z-deadbee",
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
                item_count=14042,
                item_id_scheme="mmlu/v1/<subject>/q-<sha256>[:12]",
            )
        ],
        prompt_templates=[
            PromptTemplateRecord(
                template_id="mcq_flat_v1",
                template_content_hash="0" * 64,
                template_text="ignored",
                role="user",
            )
        ],
        letter_token_table=[
            LetterTokenEntry(letter="A", token_id=10, variant="bare", token_str="A"),
            LetterTokenEntry(letter="B", token_id=11, variant="bare", token_str="B"),
        ],
        seeds={"root": 0},
        cli_args={"config_path": "examples/mvp.yaml"},
        config_digest="a" * 64,
    )
    fields.update(over)
    return RunManifest.model_validate(fields)


def test_manifest_requires_config_digest() -> None:
    base = _minimal_manifest().model_dump()
    base.pop("config_digest")
    with pytest.raises(ValidationError):
        RunManifest.model_validate(base)


def test_manifest_resumed_from_partial_default_false() -> None:
    m = _minimal_manifest()
    assert m.resumed_from_partial is False
