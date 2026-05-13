"""Model-load config (``ModelConfig`` + ``HookSpec``) consumed by ``RunConfig``."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

from nl_ae.schema.models import QuantizationSpec


class HookSpec(BaseModel):
    """Phase 2 / 3 hook configuration.

    ``record_layers=None`` (the MVP default) means no hooks are registered;
    Phase 2 sets a tuple of layer indices to capture per ``position_policy``.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)
    record_layers: tuple[Annotated[int, Field(ge=0)], ...] | None = None
    position_policy: Literal[
        "last_prompt_token", "all_prompt_tokens", "answer_letter_position"
    ] = "last_prompt_token"
    storage_dtype: Literal["fp16", "bf16", "fp32"] = "fp16"


class ModelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    hf_model_id: Annotated[str, StringConstraints(min_length=1)] = "Qwen/Qwen2.5-7B-Instruct"
    hf_revision: str | None = None
    cache_dir: Path | None = None

    quantization: QuantizationSpec = QuantizationSpec()
    hook_spec: HookSpec = HookSpec()

    attn_implementation: Literal["sdpa", "eager", "flash_attention_2"] = "sdpa"
    device_map: Literal["auto", "cuda:0", "cpu", "balanced"] = "cuda:0"
    max_memory_per_gpu_mb: Annotated[int, Field(ge=1024)] | None = None

    default_max_new_tokens: Annotated[int, Field(ge=1, le=512)] = 64
    default_stop_strings: tuple[str, ...] = ("\n\n",)

    pinned_chat_template_hash_path: Path | None = None


__all__ = ["HookSpec", "ModelConfig"]
