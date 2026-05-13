"""Decoding policy + ``seed_all`` (UR-R2.2: call at top of ``score_and_generate``)."""

from __future__ import annotations

import os
import random
from typing import TYPE_CHECKING, Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

if TYPE_CHECKING:  # pragma: no cover
    pass


class DecodingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    strategy: Literal["greedy", "sampled"] = "greedy"
    max_new_tokens: Annotated[int, Field(ge=1, le=512)] = 64
    temperature: Annotated[float, Field(gt=0.0, le=2.0)] | None = None
    top_p: Annotated[float, Field(gt=0.0, le=1.0)] | None = None
    top_k: Annotated[int, Field(ge=0)] | None = None
    seed: Annotated[int, Field(ge=0)] | None = None
    stop_strings: tuple[str, ...] = ("\n\n",)
    repetition_penalty: Annotated[float, Field(ge=0.5, le=2.0)] = 1.0

    @model_validator(mode="after")
    def _check_decode(self) -> "DecodingConfig":
        if self.strategy == "greedy":
            if self.seed is not None:
                raise ValueError("greedy decoding must not carry a seed")
            if self.temperature is not None or self.top_p is not None or self.top_k is not None:
                raise ValueError(
                    "greedy decoding must not carry temperature/top_p/top_k"
                )
        else:  # sampled
            if self.seed is None:
                raise ValueError("sampled decoding requires an explicit seed")
            if self.temperature is None:
                raise ValueError("sampled decoding requires an explicit temperature")
        return self


def seed_all(seed: int) -> None:
    """Seed every PRNG we can reach.

    UR-R2.2: called at the top of ``score_and_generate`` so first-token
    scoring and free-gen share a deterministic state.
    """
    os.environ["PYTHONHASHSEED"] = str(seed & 0xFFFFFFFF)
    random.seed(seed)
    try:
        import numpy as np  # noqa: PLC0415

        np.random.seed(seed & 0xFFFFFFFF)
    except ImportError:
        pass
    try:
        import torch  # noqa: PLC0415

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def hf_generate_kwargs(cfg: DecodingConfig, *, tokenizer: Any) -> dict[str, Any]:
    """Translate ``DecodingConfig`` to ``transformers.generate`` kwargs."""
    kwargs: dict[str, Any] = {
        "max_new_tokens": cfg.max_new_tokens,
        "repetition_penalty": cfg.repetition_penalty,
        "pad_token_id": getattr(tokenizer, "pad_token_id", None)
        or getattr(tokenizer, "eos_token_id", None),
    }
    if cfg.strategy == "greedy":
        kwargs.update({"do_sample": False})
    else:
        kwargs.update(
            {
                "do_sample": True,
                "temperature": cfg.temperature,
            }
        )
        if cfg.top_p is not None:
            kwargs["top_p"] = cfg.top_p
        if cfg.top_k is not None:
            kwargs["top_k"] = cfg.top_k
    return kwargs


__all__ = ["DecodingConfig", "hf_generate_kwargs", "seed_all"]
