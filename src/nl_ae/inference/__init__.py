"""C03 — model wrapper, scoring, decoding, extractor."""

from .decoding import DecodingConfig, hf_generate_kwargs, seed_all
from .extractor import (
    AnswerExtractorProtocol,
    ExtractorConfig,
    ExtractorOutcome,
    ExtractorRegistry,
    RegexLadderExtractor,
    default_extractor,
)
from .loader import HookSpec, ModelConfig
from .outputs import (
    FirstTokenScore,
    ForwardOutput,
    FreeGenResult,
    GenerateOutput,
    LetterScore,
    ScoringOutputs,
)

__all__ = [
    "AnswerExtractorProtocol",
    "DecodingConfig",
    "ExtractorConfig",
    "ExtractorOutcome",
    "ExtractorRegistry",
    "FirstTokenScore",
    "ForwardOutput",
    "FreeGenResult",
    "GenerateOutput",
    "HookSpec",
    "LetterScore",
    "ModelConfig",
    "RegexLadderExtractor",
    "ScoringOutputs",
    "default_extractor",
    "hf_generate_kwargs",
    "seed_all",
]
