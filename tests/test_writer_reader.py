"""Writer + reader round-trip, partial-line tolerance, lock file."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

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
from nl_ae.schema.reader import ResultsReader, detect_status
from nl_ae.schema.writer import ResultsWriter


def _minimal_manifest(run_id: str) -> RunManifest:
    return RunManifest(
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
                item_count=1,
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
    )


def _row(run_id: str, idx: int) -> ResultRow:
    return ResultRow(
        run_id=run_id,
        item_id=f"mmlu/v1/abstract_algebra/q-{idx:012d}",
        dataset_name="mmlu",
        dataset_split="test",
        subject="abstract_algebra",
        template_id="mcq_flat_v1",
        permutation_id=0,
        prompt_hash="0" * 64,
        gold_letter="A",
        first_token_letter="A",
        free_text_letter="A",
        free_text_raw="Answer is A.",
        agreement_flag=True,
        letter_softmax=[
            LetterSoftmaxEntry(letter="A", token_id=1, prob=0.7, prob_valid=True, logit=0.0),
            LetterSoftmaxEntry(letter="B", token_id=2, prob=0.3, prob_valid=True, logit=0.0),
        ],
        n_options=2,
        free_text_seed=None,
        decode_strategy="greedy",
        created_at=now_utc_iso(),
        extractor_id="regex_ladder",
        extractor_match_rule="my_answer_is",
        first_token_scoring_math="full_vocab_softmax",
        total_letter_mass=1.0,
    )


def test_writer_reader_round_trip(tmp_path: Path) -> None:
    run_id = "20260101T000000Z-deadbee-test"
    run_dir = tmp_path / run_id
    manifest = _minimal_manifest(run_id)
    with ResultsWriter(run_dir, manifest, on_existing="error") as writer:
        for i in range(5):
            writer.write_row(_row(run_id, i))
        writer.finalize(emit_parquet=False)

    reader = ResultsReader(run_dir)
    rows = list(reader.iter_rows())
    assert len(rows) == 5
    keys = list(reader.iter_keys())
    assert len(keys) == 5
    assert reader.manifest.completion_status == "completed"
    assert reader.manifest.rows_written == 5
    assert detect_status(run_dir) == "completed"


def test_writer_clears_lock_after_finalize(tmp_path: Path) -> None:
    run_id = "20260101T000000Z-deadbee-test"
    run_dir = tmp_path / run_id
    manifest = _minimal_manifest(run_id)
    with ResultsWriter(run_dir, manifest, on_existing="error") as writer:
        writer.write_row(_row(run_id, 0))
        writer.finalize(emit_parquet=False)
    assert not (run_dir / "run.lock").exists()


def test_reader_tolerates_torn_last_line(tmp_path: Path) -> None:
    run_id = "20260101T000000Z-deadbee-test"
    run_dir = tmp_path / run_id
    manifest = _minimal_manifest(run_id)
    with ResultsWriter(run_dir, manifest, on_existing="error") as writer:
        writer.write_row(_row(run_id, 0))
        # Don't finalize.
    # Simulate a torn write into the .partial file.
    partial = run_dir / "rows.jsonl.partial"
    with partial.open("ab") as f:
        f.write(b'{"run_id": "trunc')  # no newline, malformed JSON
    reader = ResultsReader(run_dir, strict=False)
    rows = list(reader.iter_rows())
    assert len(rows) == 1


def test_writer_refuses_run_id_mismatch(tmp_path: Path) -> None:
    run_id = "20260101T000000Z-deadbee-test"
    manifest = _minimal_manifest(run_id)
    with pytest.raises(ValueError):
        ResultsWriter(tmp_path / "different-name", manifest)
