"""Pydantic v2 source-of-truth for every on-disk artifact.

Schema 1.1.0 — see .plans/conductor-plan/COHERENCE.md for the bundled CI batch
and .plans/conductor-plan/PLAN.md §7 for the URL Round 2 corrections.
"""

from __future__ import annotations

from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

SCHEMA_VERSION: Literal["1.1.0"] = "1.1.0"

LetterStr = Annotated[str, StringConstraints(min_length=1, max_length=1, pattern=r"^[A-Z]$")]
ItemIdStr = Annotated[
    str, StringConstraints(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9._:\-/]+$")
]
Sha256Hex = Annotated[
    str, StringConstraints(min_length=64, max_length=64, pattern=r"^[0-9a-f]{64}$")
]
IsoUtcStr = Annotated[
    str, StringConstraints(pattern=r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z$")
]


class LetterSoftmaxEntry(BaseModel):
    """One per-letter scoring record.

    `prob` is `None` and `prob_valid=False` only under
    `first_token_scoring_math == "argmax_logits_only"` (UR6: pydantic v2 cannot
    represent NaN under ``Field(ge=0, le=1)``).
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    letter: LetterStr
    token_id: Annotated[int, Field(ge=0)]
    prob: Annotated[float, Field(ge=0.0, le=1.0)] | None
    prob_valid: bool
    logit: float


ExtractorMatchRule = Literal[
    "header_marker",
    "my_answer_is",
    "the_answer_is",
    "boxed",
    "letter_only",
    "trailing_letter",
    "leading_letter",
    "majority_vote",
    "unparseable",
    "out_of_range",
]
FirstTokenScoringMath = Literal[
    "full_vocab_softmax", "renormalize_over_letters", "argmax_logits_only"
]
DecodeStrategy = Literal["greedy", "sampled"]


class ResultRow(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.1.0"] = SCHEMA_VERSION
    run_id: Annotated[str, StringConstraints(min_length=1, max_length=64)]
    item_id: ItemIdStr
    dataset_name: Annotated[str, StringConstraints(min_length=1, max_length=64)]
    dataset_split: Annotated[str, StringConstraints(min_length=1, max_length=32)]
    subject: str | None = None
    template_id: Annotated[str, StringConstraints(min_length=1, max_length=64)]
    permutation_id: Annotated[int, Field(ge=0)]
    prompt_hash: Sha256Hex
    rendered_prompt_ref: Annotated[str, StringConstraints(min_length=1, max_length=256)] | None = (
        None
    )
    gold_letter: LetterStr | None  # CI.02: OpinionQA has no ground truth.
    first_token_letter: LetterStr | None
    free_text_letter: LetterStr | None
    free_text_raw: str
    free_text_truncated: bool = False
    agreement_flag: bool | None
    letter_softmax: list[LetterSoftmaxEntry] = Field(min_length=2, max_length=26)
    n_options: Annotated[int, Field(ge=2, le=26)]
    free_text_seed: Annotated[int, Field(ge=0)] | None
    decode_strategy: DecodeStrategy
    activation_ref: Annotated[str, StringConstraints(max_length=256)] | None = None
    wall_time_ms: Annotated[float, Field(ge=0.0)] | None = None
    created_at: IsoUtcStr

    # CI.06 set (extractor + scoring math provenance).
    extractor_id: Annotated[str, StringConstraints(min_length=1, max_length=64)]
    extractor_match_rule: ExtractorMatchRule
    first_token_scoring_math: FirstTokenScoringMath
    total_letter_mass: Annotated[float, Field(ge=0.0, le=1.0)]

    @model_validator(mode="after")
    def _check_invariants(self) -> Self:
        letters = [e.letter for e in self.letter_softmax]
        if len(set(letters)) != len(letters):
            raise ValueError("letter_softmax must have unique letters")
        if len(self.letter_softmax) != self.n_options:
            raise ValueError(
                f"len(letter_softmax)={len(self.letter_softmax)} must equal n_options={self.n_options}"
            )
        if (
            self.agreement_flag is not None
            and self.first_token_letter is not None
            and self.free_text_letter is not None
        ):
            expected = self.first_token_letter == self.free_text_letter
            if self.agreement_flag != expected:
                raise ValueError(
                    "agreement_flag inconsistent with first_token_letter/free_text_letter"
                )
        if self.first_token_scoring_math == "argmax_logits_only":
            if any(e.prob_valid for e in self.letter_softmax):
                raise ValueError(
                    "argmax_logits_only must emit prob_valid=False for every letter (UR6)"
                )
        else:
            if any(not e.prob_valid or e.prob is None for e in self.letter_softmax):
                raise ValueError(
                    "non-argmax scoring math must emit a valid prob for every letter"
                )
        return self


class ResultRowKey(BaseModel):
    """CI.13 — key-only projection for resume + Phase 2 label indexing."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    run_id: Annotated[str, StringConstraints(min_length=1, max_length=64)]
    item_id: ItemIdStr
    permutation_id: Annotated[int, Field(ge=0)]
    template_id: Annotated[str, StringConstraints(min_length=1, max_length=64)]


class PromptTemplateRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    template_id: Annotated[str, StringConstraints(min_length=1, max_length=64)]
    template_content_hash: Sha256Hex  # CI.04 — renamed from template_text_hash.
    template_text: str
    role: Literal["system", "user", "assistant", "composite"]


class LetterTokenEntry(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    letter: LetterStr
    token_id: Annotated[int, Field(ge=0)]
    variant: Literal["bare", "leading_space", "newline_prefixed"]  # CI.01 / CI.09
    token_str: str


class QuantizationSpec(BaseModel):
    """CI.07 — structured replacement for the prior `quantization` literal."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    kind: Literal["int4-nf4", "int4-fp4", "int8", "fp16", "bf16", "cpu-fp32"] = "fp16"
    double_quant: bool = True
    compute_dtype: Literal["fp16", "bf16", "fp32"] = "fp16"
    bnb_4bit_quant_storage: Literal["uint8", "fp16"] = "uint8"
    cpu_offload_layers: Annotated[int, Field(ge=0)] = 0


class ExtractorRecord(BaseModel):
    """CI.08 — full extractor recipe on the manifest."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    extractor_id: Annotated[str, StringConstraints(min_length=1, max_length=64)]
    extractor_version: Annotated[str, StringConstraints(pattern=r"^\d+\.\d+\.\d+$")]
    rules_content_hash: Sha256Hex
    rules_in_order: tuple[str, ...]


class PermutationCoverage(BaseModel):
    """CI.12 — per-template visit coverage so the aggregator doesn't scan rows."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    template_id: Annotated[str, StringConstraints(min_length=1, max_length=64)]
    items_visited: Annotated[int, Field(ge=0)]
    permutations_per_item_mode: Annotated[int, Field(ge=1)]
    distinct_permutation_ids: tuple[int, ...]


class EnvFingerprint(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    os_name: str
    os_version: str
    python_version: str
    cuda_version: str | None
    torch_version: str | None
    transformers_version: str | None
    bitsandbytes_version: str | None
    accelerate_version: str | None
    gpu_name: str | None
    gpu_vram_mb: Annotated[int, Field(ge=0)] | None


class ModelFingerprint(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    hf_model_id: str
    hf_model_commit: str | None
    hf_tokenizer_id: str
    hf_tokenizer_commit: str | None
    quantization: QuantizationSpec | None  # CI.07: structured (was Literal[...]).


class DatasetFingerprint(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    name: str
    hf_dataset_id: str | None
    split: str
    commit_or_revision: str | None
    item_count: Annotated[int, Field(ge=0)]
    item_id_scheme: str


class RunManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.1.0"] = SCHEMA_VERSION
    run_id: Annotated[str, StringConstraints(min_length=1, max_length=64)]
    git_sha: Annotated[str, StringConstraints(min_length=7, max_length=40)]
    git_dirty: bool
    started_at: IsoUtcStr
    ended_at: IsoUtcStr | None = None
    completion_status: Literal["in_progress", "completed", "failed", "aborted"]
    failure_reason: str | None = None
    env: EnvFingerprint
    model: ModelFingerprint
    datasets: list[DatasetFingerprint] = Field(min_length=1)
    prompt_templates: list[PromptTemplateRecord] = Field(min_length=1)
    # CI.05: 26 letters × up to 3 variants.
    letter_token_table: list[LetterTokenEntry] = Field(min_length=2, max_length=78)
    seeds: dict[str, int]
    rows_written: Annotated[int, Field(ge=0)] = 0
    rows_expected: Annotated[int, Field(ge=0)] | None = None
    cli_args: dict[str, str | int | float | bool | None]
    notes: str | None = None

    # --- schema 1.1.0 additions ---
    chat_template_hash: Sha256Hex | None = None  # CI.03
    extractor: ExtractorRecord | None = None  # CI.08
    kv_shared: bool = False  # CI.10
    permutation_coverage: list[PermutationCoverage] | None = None  # CI.12
    applied_seeds_notes: tuple[str, ...] = ()  # CI.17
    deterministic_algorithms_actual: Literal[
        "off", "warn_only", "strict", "unavailable"
    ] = "warn_only"  # CI.18
    config_digest: Sha256Hex  # CI.19 (REQUIRED)
    config_yaml_text: str | None = None  # CI.20
    resumed_from_partial: bool = False  # UR-R2.6
    activation_manifest_ref: str | None = None  # CI.A


__all__ = [
    "DecodeStrategy",
    "ExtractorMatchRule",
    "FirstTokenScoringMath",
    "SCHEMA_VERSION",
    "DatasetFingerprint",
    "EnvFingerprint",
    "ExtractorRecord",
    "IsoUtcStr",
    "ItemIdStr",
    "LetterSoftmaxEntry",
    "LetterStr",
    "LetterTokenEntry",
    "ModelFingerprint",
    "PermutationCoverage",
    "PromptTemplateRecord",
    "QuantizationSpec",
    "ResultRow",
    "ResultRowKey",
    "RunManifest",
    "Sha256Hex",
]
