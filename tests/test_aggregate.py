"""Aggregator smoke test against a synthetic on-disk run."""

from __future__ import annotations

from pathlib import Path

import pytest

pd = pytest.importorskip("pandas", reason="pandas required for aggregator tests")

from nl_ae.schema.hashing import now_utc_iso  # noqa: E402
from nl_ae.schema.models import (  # noqa: E402
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
from nl_ae.schema.writer import ResultsWriter  # noqa: E402

from nl_ae.report.aggregate import aggregate_run  # noqa: E402


def _manifest(run_id: str) -> RunManifest:
    return RunManifest(
        run_id=run_id,
        git_sha="deadbeefcafe",
        git_dirty=False,
        started_at=now_utc_iso(),
        completion_status="in_progress",
        env=EnvFingerprint(
            os_name="Windows", os_version="11", python_version="3.13.0",
            cuda_version=None, torch_version=None, transformers_version=None,
            bitsandbytes_version=None, accelerate_version=None,
            gpu_name=None, gpu_vram_mb=None,
        ),
        model=ModelFingerprint(
            hf_model_id="Qwen/Qwen2.5-7B-Instruct", hf_model_commit=None,
            hf_tokenizer_id="Qwen/Qwen2.5-7B-Instruct", hf_tokenizer_commit=None,
            quantization=QuantizationSpec(kind="fp16"),
        ),
        datasets=[
            DatasetFingerprint(
                name="mmlu", hf_dataset_id="cais/mmlu", split="test",
                commit_or_revision=None, item_count=2,
                item_id_scheme="mmlu/v1/<subject>/q-<sha256>[:12]",
            )
        ],
        prompt_templates=[
            PromptTemplateRecord(
                template_id="mcq_flat_v1", template_content_hash="0" * 64,
                template_text="x", role="user",
            )
        ],
        letter_token_table=[
            LetterTokenEntry(letter="A", token_id=10, variant="bare", token_str="A"),
            LetterTokenEntry(letter="B", token_id=11, variant="bare", token_str="B"),
        ],
        seeds={"root": 0},
        cli_args={},
        config_digest="a" * 64,
    )


def _row(run_id: str, *, item: str, perm: int, ft: str, fg: str, gold: str, subj: str) -> ResultRow:
    return ResultRow(
        run_id=run_id,
        item_id=f"mmlu/v1/{subj}/q-{item}",
        dataset_name="mmlu",
        dataset_split="test",
        subject=subj,
        template_id="mcq_flat_v1",
        permutation_id=perm,
        prompt_hash="0" * 64,
        gold_letter=gold,
        first_token_letter=ft,
        free_text_letter=fg,
        free_text_raw=f"Answer: {fg}",
        agreement_flag=ft == fg,
        letter_softmax=[
            LetterSoftmaxEntry(letter="A", token_id=1, prob=0.6 if ft == "A" else 0.4,
                                prob_valid=True, logit=0.0),
            LetterSoftmaxEntry(letter="B", token_id=2, prob=0.4 if ft == "A" else 0.6,
                                prob_valid=True, logit=0.0),
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


def test_aggregate_run_smoke(tmp_path: Path) -> None:
    run_id = "20260101T000000Z-deadbee-agg"
    run_dir = tmp_path / run_id
    manifest = _manifest(run_id)
    rows = [
        _row(run_id, item="a1", perm=0, ft="A", fg="A", gold="A", subj="algebra"),
        _row(run_id, item="a1", perm=1, ft="A", fg="B", gold="A", subj="algebra"),
        _row(run_id, item="a2", perm=0, ft="B", fg="B", gold="A", subj="algebra"),
        _row(run_id, item="b1", perm=0, ft="A", fg="A", gold="B", subj="biology"),
    ]
    with ResultsWriter(run_dir, manifest, on_existing="error") as w:
        for r in rows:
            w.write_row(r)
        w.finalize(emit_parquet=False)

    bundle = aggregate_run(run_dir, write_parquet=False)
    assert len(bundle.rows) == 4
    assert len(bundle.top1_disagreement) >= 1
    assert {"dataset_name", "template_id", "disagreement"}.issubset(
        bundle.top1_disagreement.columns
    )
    assert len(bundle.per_subject_mmlu) >= 1
    assert "accuracy_first_token" in bundle.per_subject_mmlu.columns
