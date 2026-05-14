"""C09 — pilot fold assignment + preregistration gate.

The discipline layer for Phase 2/3. Owns deterministic pilot/holdout fold
assignment, the ``preregistration.md`` schema, and the hard gate that keeps
confirmatory commands from running before a hypothesis is locked.
"""

from .assignment import (
    Fold,
    PilotFoldResult,
    StratumItem,
    assign_pilot_fold,
)
from .errors import (
    GitTreeDirtyError,
    PilotDigestMismatchError,
    PilotError,
    PilotFoldMismatchError,
    PilotManifestMissingError,
    PreregistrationInvalidError,
    PreregistrationMissingError,
    PreregistrationParseError,
    PreregistrationUnlockedError,
)
from .manifest import (
    assign_and_write_pilot_manifest,
    build_pilot_manifest,
    compute_pilot_manifest_digest,
    iter_item_summaries,
    load_pilot_manifest,
    write_pilot_manifest,
)
from .models import (
    CANDIDATE_LABELS,
    PILOT_MANIFEST_SCHEMA_VERSION,
    PREREGISTRATION_SCHEMA_VERSION,
    NlaScopeSpec,
    PilotManifest,
    Preregistration,
    ProbeLabel,
    StratumRecord,
)
from .preregistration import (
    load_preregistration,
    lock_preregistration,
    parse_preregistration_text,
    require_preregistration,
)

__all__ = [
    "CANDIDATE_LABELS",
    "PILOT_MANIFEST_SCHEMA_VERSION",
    "PREREGISTRATION_SCHEMA_VERSION",
    "Fold",
    "GitTreeDirtyError",
    "NlaScopeSpec",
    "PilotDigestMismatchError",
    "PilotError",
    "PilotFoldMismatchError",
    "PilotFoldResult",
    "PilotManifest",
    "PilotManifestMissingError",
    "Preregistration",
    "PreregistrationInvalidError",
    "PreregistrationMissingError",
    "PreregistrationParseError",
    "PreregistrationUnlockedError",
    "ProbeLabel",
    "StratumItem",
    "StratumRecord",
    "assign_and_write_pilot_manifest",
    "assign_pilot_fold",
    "build_pilot_manifest",
    "compute_pilot_manifest_digest",
    "iter_item_summaries",
    "load_pilot_manifest",
    "load_preregistration",
    "lock_preregistration",
    "parse_preregistration_text",
    "require_preregistration",
    "write_pilot_manifest",
]
