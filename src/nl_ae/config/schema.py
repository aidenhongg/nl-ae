"""``RunConfig`` — frozen, validated, every-knob-visible composition.

Sub-configs owned by other components are imported and composed; this module
only defines the wrapper plus the four module-owned sub-configs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

from nl_ae.data.config import DatasetConfig
from nl_ae.eval.runner import EvalConfig
from nl_ae.inference.decoding import DecodingConfig
from nl_ae.inference.extractor import ExtractorConfig
from nl_ae.inference.loader import ModelConfig
from nl_ae.schema.models import Sha256Hex  # noqa: F401  (re-exported via schema package)

RunIdStr = Annotated[
    str, StringConstraints(min_length=1, max_length=64, pattern=r"^[0-9A-Za-z._:\-]+$")
]
TagStr = Annotated[
    str, StringConstraints(min_length=1, max_length=32, pattern=r"^[A-Za-z0-9._\-]+$")
]


class OutputConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    output_dir: Path
    run_id_prefix: Annotated[
        str, StringConstraints(max_length=32, pattern=r"^[A-Za-z0-9._\-]*$")
    ] = ""
    derive_parquet_on_finalize: bool = True
    save_rendered_prompts: bool = True
    figure_dpi: Annotated[int, Field(ge=72, le=600)] = 144
    max_free_text_chars: Annotated[int, Field(ge=128, le=65_536)] = 2048


class SeedConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    root: Annotated[int, Field(ge=0, le=(1 << 32) - 1)] = 0
    numpy: Annotated[int, Field(ge=0, le=(1 << 32) - 1)] | None = None
    torch: Annotated[int, Field(ge=0, le=(1 << 32) - 1)] | None = None
    python: Annotated[int, Field(ge=0, le=(1 << 32) - 1)] | None = None
    cuda: Annotated[int, Field(ge=0, le=(1 << 32) - 1)] | None = None
    free_gen: Annotated[int, Field(ge=0, le=(1 << 32) - 1)] | None = None
    pythonhashseed: Annotated[int, Field(ge=0, le=(1 << 32) - 1)] | None = None
    deterministic_algorithms: Literal["off", "warn_only", "strict"] = "warn_only"
    cudnn_deterministic: bool = True
    cudnn_benchmark: bool = False


class RunIdentityConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    run_id: RunIdStr | None = None
    slug: Annotated[str, StringConstraints(max_length=32, pattern=r"^[A-Za-z0-9._\-]*$")] = ""
    tags: tuple[TagStr, ...] = ()
    notes: Annotated[str, StringConstraints(max_length=4096)] = ""
    expected_python_version: str | None = None
    expected_cuda_version: str | None = None


class LogConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    console_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    file_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "DEBUG"
    per_module_levels: dict[str, Literal["DEBUG", "INFO", "WARNING", "ERROR"]] = Field(
        default_factory=dict
    )
    jsonl_destination: Literal["run_dir", "stderr", "off"] = "run_dir"


class RunConfig(BaseModel):
    """The one and only top-level config."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    config_schema_version: Literal["1.0.0"] = "1.0.0"
    dataset: DatasetConfig
    model: ModelConfig
    decoding: DecodingConfig
    extractor: ExtractorConfig = ExtractorConfig()
    eval: EvalConfig
    output: OutputConfig
    seeds: SeedConfig = SeedConfig()
    run_identity: RunIdentityConfig = RunIdentityConfig()
    logging: LogConfig = LogConfig()

    @model_validator(mode="after")
    def _check_invariants(self) -> "RunConfig":
        # 1. record_layers consistency between eval and model.hook_spec.
        if self.eval.record_layers is not None:
            hook_layers = self.model.hook_spec.record_layers or ()
            if tuple(self.eval.record_layers) != tuple(hook_layers):
                raise ValueError(
                    "eval.record_layers and model.hook_spec.record_layers must match exactly"
                )
        # 2. Decoding seed / strategy coherence is enforced inside DecodingConfig already.
        # 3. Output dir must not be a tilde path; let CLI / loader expand it.
        if str(self.output.output_dir).startswith("~"):
            raise ValueError("output.output_dir starts with '~'; expand before loading")
        # 4. dataset.templates_in_use ⊆ eval.plan.template_ids.
        plan_ids = set(self.eval.plan.template_ids)
        used_ids = set(self.dataset.templates_in_use)
        unused_in_plan = used_ids - plan_ids
        if unused_in_plan:
            raise ValueError(
                f"dataset.templates_in_use has ids not present in eval.plan.template_ids: "
                f"{sorted(unused_in_plan)}"
            )
        return self


__all__ = [
    "LogConfig",
    "OutputConfig",
    "RunConfig",
    "RunIdentityConfig",
    "RunIdStr",
    "SeedConfig",
    "TagStr",
]
