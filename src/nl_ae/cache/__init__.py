"""C06 — per-fold activation cache (extract + sharded parquet store).

Replays Phase 1 prompts forward-only, captures per-layer post-block residuals at
the last-prompt-token position, and writes them layer-major under
``runs/<run_id>/<fold>/activations/``. The cache is fold-aware: a pilot cache
and a holdout cache for the same run share every cache-key field except
``fold`` and the ``prompt_hash_set`` fingerprint.
"""

from .errors import (
    ActivationCacheStateError,
    ActivationManifestMissingError,
    CacheError,
    CacheKeyMismatchError,
    CacheLockError,
    DuplicateActivationRowError,
    PromptHashMismatchError,
    PromptSidecarMissingError,
)
from .extractor import (
    ActivationCacheExtractor,
    ActivationSource,
    ExtractorInputs,
    ExtractorOutcome,
    Fold,
    VisitKey,
    load_extractor_inputs,
    scan_resume_state,
)
from .models import (
    ACTIVATION_MANIFEST_SCHEMA_VERSION,
    ACTIVATION_PARQUET_COLUMNS,
    CACHE_KEY_COMPOSITION,
    ActivationCacheKey,
    ActivationDtype,
    ActivationManifest,
    LayerShardManifest,
    PositionPolicy,
    ShardRecord,
    compute_cache_key_composition_digest,
    compute_prompt_hash_set_digest,
    compute_quantization_spec_digest,
)
from .reader import ActivationCacheReader
from .writer import ActivationCacheWriter, write_activation_manifest_atomic

__all__ = [
    "ACTIVATION_MANIFEST_SCHEMA_VERSION",
    "ACTIVATION_PARQUET_COLUMNS",
    "CACHE_KEY_COMPOSITION",
    "ActivationCacheExtractor",
    "ActivationCacheKey",
    "ActivationCacheReader",
    "ActivationCacheStateError",
    "ActivationCacheWriter",
    "ActivationDtype",
    "ActivationManifest",
    "ActivationManifestMissingError",
    "ActivationSource",
    "CacheError",
    "CacheKeyMismatchError",
    "CacheLockError",
    "DuplicateActivationRowError",
    "ExtractorInputs",
    "ExtractorOutcome",
    "Fold",
    "LayerShardManifest",
    "PositionPolicy",
    "PromptHashMismatchError",
    "PromptSidecarMissingError",
    "ShardRecord",
    "VisitKey",
    "compute_cache_key_composition_digest",
    "compute_prompt_hash_set_digest",
    "compute_quantization_spec_digest",
    "load_extractor_inputs",
    "scan_resume_state",
    "write_activation_manifest_atomic",
]
