"""Dataset config consumed by RunConfig.

Owned by C02. Imported by ``nl_ae.config.schema.RunConfig`` per CI.14.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints


class DatasetConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    # Which datasets are in scope for this run.
    enabled: tuple[Literal["mmlu", "opinionqa"], ...] = Field(min_length=1)

    # MMLU.
    mmlu_split: Literal["test", "validation", "dev"] = "test"
    mmlu_subjects: tuple[str, ...] | None = None  # None => all 57 subjects.
    mmlu_hf_id: Annotated[str, StringConstraints(min_length=1)] = "cais/mmlu"
    mmlu_revision: str | None = None
    mmlu_limit: Annotated[int, Field(ge=1)] | None = None

    # OpinionQA.
    opinionqa_source_dir: Path | None = None
    opinionqa_canonicalization: Literal["keep_all", "drop_refusal", "fixed_4opt"] = "keep_all"
    opinionqa_revision_tag: Annotated[str, StringConstraints(min_length=1)] = "atp-v1"
    opinionqa_subset_path: Path | None = None  # data/opinionqa_wang2024_414.txt
    opinionqa_wave_filter: tuple[str, ...] | None = None
    opinionqa_topic_filter: tuple[str, ...] | None = None
    opinionqa_limit: Annotated[int, Field(ge=1)] | None = None

    # Shared HF cache root.
    cache_dir: Path
    offline: bool = True

    # Templates + chat-template pinning (C02 §3).
    templates_dir: Path
    templates_in_use: tuple[
        Annotated[str, StringConstraints(min_length=1, max_length=64, pattern=r"^[a-z0-9_\-\.]+$")],
        ...,
    ] = Field(min_length=1)
    chat_enabled: bool = True
    pinned_chat_template_hash_path: Path

    # Letter-token variants to enumerate (C02 D3.6).
    letter_token_variants: tuple[
        Literal["bare", "leading_space", "newline_prefixed"], ...
    ] = ("bare", "leading_space")


__all__ = ["DatasetConfig"]
