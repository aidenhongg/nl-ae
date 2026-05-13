"""Return values from the model wrapper.

Tensor-bearing outputs use frozen dataclasses (avoid pydantic
``arbitrary_types_allowed`` for hot-path types). Aggregated outputs that
flow into ``ResultRow`` use pydantic with their own validation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

from nl_ae.schema.models import ExtractorMatchRule, FirstTokenScoringMath, LetterStr

if TYPE_CHECKING:
    import torch


@dataclass(frozen=True)
class ForwardOutput:
    last_token_logits: "torch.Tensor"
    prompt_token_count: int
    hidden_states: dict[int, "torch.Tensor"] | None
    position_policy: str | None
    wall_time_ms: float
    engine_call_id: int


@dataclass(frozen=True)
class GenerateOutput:
    text: str
    generated_token_ids: list[int]
    truncated: bool
    finish_reason: Literal["eos", "stop_string", "max_new_tokens", "error"]
    prompt_token_count: int
    new_token_count: int
    wall_time_ms: float
    decode_strategy: Literal["greedy", "sampled"]
    seed: int | None
    generator_id: str
    engine_call_id: int


class LetterScore(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    letter: LetterStr
    token_id: Annotated[int, Field(ge=0)]
    prob: Annotated[float, Field(ge=0.0, le=1.0)] | None
    prob_valid: bool
    logit: float


class FirstTokenScore(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    argmax_letter: LetterStr | None
    per_letter: list[LetterScore] = Field(min_length=2, max_length=26)
    scoring_math: FirstTokenScoringMath
    total_letter_mass: Annotated[float, Field(ge=0.0, le=1.0)]
    prompt_token_count: Annotated[int, Field(ge=1)]
    wall_time_ms: Annotated[float, Field(ge=0.0)]
    scorer_id: Annotated[str, StringConstraints(min_length=1, max_length=64)] = "first_token_v1"
    engine_call_id: int


@dataclass(frozen=True)
class FreeGenResult:
    text: str
    truncated: bool
    finish_reason: Literal["eos", "stop_string", "max_new_tokens", "error"]
    extracted_letter: LetterStr | None
    match_rule: ExtractorMatchRule
    extractor_id: str
    seed: int | None
    decode_strategy: Literal["greedy", "sampled"]
    wall_time_ms: float
    new_token_count: int
    generator_id: str
    engine_call_id: int


class ScoringOutputs(BaseModel):
    """Final aggregated payload handed to the eval orchestrator."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    first_token_letter: LetterStr | None
    letter_softmax: list[LetterScore] = Field(min_length=2, max_length=26)
    free_text_raw: str
    free_text_truncated: bool
    free_text_letter: LetterStr | None
    extractor_id: Annotated[str, StringConstraints(min_length=1, max_length=64)]
    agreement_flag: bool | None
    decode_strategy: Literal["greedy", "sampled"]
    decode_seed: Annotated[int, Field(ge=0)] | None
    scoring_math: FirstTokenScoringMath
    total_letter_mass: Annotated[float, Field(ge=0.0, le=1.0)]
    extractor_match_rule: ExtractorMatchRule
    forward_wall_time_ms: Annotated[float, Field(ge=0.0)]
    generate_wall_time_ms: Annotated[float, Field(ge=0.0)]
    total_wall_time_ms: Annotated[float, Field(ge=0.0)]
    provenance: dict[str, str] = Field(default_factory=dict)
    engine_call_id: int


__all__ = [
    "FirstTokenScore",
    "ForwardOutput",
    "FreeGenResult",
    "GenerateOutput",
    "LetterScore",
    "ScoringOutputs",
]
