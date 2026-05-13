"""C02 — datasets and permutations."""

from .canonical import CanonicalItem, PermutedItem
from .config import DatasetConfig
from .mmlu_loader import MmluLoader
from .opinionqa_loader import OpinionQaLoader
from .permute import LETTERS_26, all_permutations, letters_for, permutation_for
from .text_norm import (
    derive_item_id,
    nfc,
    normalize_newlines,
    sha256_hex,
    sha256_hex_bytes,
    strip_bom,
)

__all__ = [
    "CanonicalItem",
    "DatasetConfig",
    "LETTERS_26",
    "MmluLoader",
    "OpinionQaLoader",
    "PermutedItem",
    "all_permutations",
    "derive_item_id",
    "letters_for",
    "nfc",
    "normalize_newlines",
    "permutation_for",
    "sha256_hex",
    "sha256_hex_bytes",
    "strip_bom",
]
