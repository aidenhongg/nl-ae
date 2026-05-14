"""Pydantic v2 models for ``pilot_manifest.json`` and ``preregistration.md``.

Both schemas are SemVer'd at 1.0.0; ``extra="forbid"`` so typo'd fields raise
loudly instead of silently corrupting the discipline gate.
"""

from __future__ import annotations

from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

from nl_ae.schema.models import IsoUtcStr, ItemIdStr, Sha256Hex

PILOT_MANIFEST_SCHEMA_VERSION: Literal["1.0.0"] = "1.0.0"
PREREGISTRATION_SCHEMA_VERSION: Literal["1.0.0"] = "1.0.0"

CANDIDATE_LABELS = (
    "disagreement_flag",
    "first_token_correct",
    "free_text_correct",
    "first_token_letter",
    "free_text_letter",
    "extractor_match_rule",
)
ProbeLabel = Literal[
    "disagreement_flag",
    "first_token_correct",
    "free_text_correct",
    "first_token_letter",
    "free_text_letter",
    "extractor_match_rule",
]

# Qwen2.5-7B has 28 decoder layers (L0..L27).
LayerIndex = Annotated[int, Field(ge=0, lt=28)]
GitSha40 = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{40}$")]
SeedInt = Annotated[int, Field(ge=0, le=(1 << 32) - 1)]
PilotFracFloat = Annotated[float, Field(gt=0.0, lt=1.0)]


class StratumRecord(BaseModel):
    """One stratum's identity + per-fold counts."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    key: Annotated[str, StringConstraints(min_length=1, max_length=128)]
    source_field: Annotated[str, StringConstraints(min_length=1, max_length=32)]
    n_total: Annotated[int, Field(ge=1)]
    n_pilot: Annotated[int, Field(ge=0)]
    n_holdout: Annotated[int, Field(ge=0)]

    @model_validator(mode="after")
    def _sums_match(self) -> Self:
        if self.n_pilot + self.n_holdout != self.n_total:
            raise ValueError(
                f"stratum {self.key!r}: n_pilot+n_holdout={self.n_pilot + self.n_holdout} "
                f"!= n_total={self.n_total}"
            )
        return self


class PilotManifest(BaseModel):
    """Pilot fold assignment for one Phase 1 run.

    ``pilot_manifest_digest`` is the SHA-256 over the canonical (sorted-keys,
    no-whitespace) JSON encoding of every field except ``created_at``,
    ``completion_status``, and ``pilot_manifest_digest`` itself. Two runs that
    produce identical assignments produce bit-identical digests.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["1.0.0"] = PILOT_MANIFEST_SCHEMA_VERSION
    run_id: Annotated[str, StringConstraints(min_length=1, max_length=64)]
    seed: SeedInt
    frac: PilotFracFloat
    stratify_by: tuple[Annotated[str, StringConstraints(min_length=1, max_length=32)], ...] = (
        Field(min_length=1)
    )
    min_per_stratum: Annotated[int, Field(ge=1)]
    strata: tuple[StratumRecord, ...] = Field(min_length=1)
    pilot_item_ids: tuple[ItemIdStr, ...]
    n_pilot: Annotated[int, Field(ge=0)]
    n_holdout: Annotated[int, Field(ge=0)]
    n_total: Annotated[int, Field(ge=1)]
    created_at: IsoUtcStr
    completion_status: Literal["in_progress", "committed"] = "committed"
    pilot_manifest_digest: Sha256Hex

    @model_validator(mode="after")
    def _check(self) -> Self:
        if len(self.pilot_item_ids) != self.n_pilot:
            raise ValueError(
                f"len(pilot_item_ids)={len(self.pilot_item_ids)} != n_pilot={self.n_pilot}"
            )
        if self.n_pilot + self.n_holdout != self.n_total:
            raise ValueError(
                f"n_pilot+n_holdout={self.n_pilot + self.n_holdout} != n_total={self.n_total}"
            )
        sorted_unique = tuple(sorted(set(self.pilot_item_ids)))
        if sorted_unique != self.pilot_item_ids:
            raise ValueError("pilot_item_ids must be sorted and deduplicated")
        stratum_pilot_sum = sum(s.n_pilot for s in self.strata)
        stratum_total_sum = sum(s.n_total for s in self.strata)
        if stratum_pilot_sum != self.n_pilot:
            raise ValueError(
                f"sum(strata.n_pilot)={stratum_pilot_sum} != n_pilot={self.n_pilot}"
            )
        if stratum_total_sum != self.n_total:
            raise ValueError(
                f"sum(strata.n_total)={stratum_total_sum} != n_total={self.n_total}"
            )
        return self


class NlaScopeSpec(BaseModel):
    """Confirmatory NLA scope declared in ``preregistration.md``."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    layer: LayerIndex
    fold: Literal["pilot", "holdout"] = "holdout"
    limit: Annotated[int, Field(ge=1)] | None
    decode_strategy: Literal["sampled", "greedy"]
    temperature: Annotated[float, Field(gt=0.0, le=2.0)]
    max_new_tokens: Annotated[int, Field(ge=1, le=2048)]


class Preregistration(BaseModel):
    """Parsed YAML frontmatter of ``preregistration.md``.

    The body of the file (post-frontmatter Markdown) is preserved verbatim
    by the writer but is not part of this model — only the frontmatter is
    machine-checked. Reviewers / future-you read the body for the theoretical
    motivation behind every preregistered hypothesis.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["1.0.0"] = PREREGISTRATION_SCHEMA_VERSION
    run_id: Annotated[str, StringConstraints(min_length=1, max_length=64)]
    pilot_manifest_digest: Sha256Hex
    locked_at: IsoUtcStr | None = None
    locked_at_git_sha: GitSha40 | None = None
    holdout_vs_full: Literal["holdout-only", "full-dataset"]
    labels: tuple[ProbeLabel, ...] = Field(min_length=1)
    layers: tuple[LayerIndex, ...] = Field(min_length=1)
    nla_scope: NlaScopeSpec
    primary_hypothesis: Annotated[str, StringConstraints(min_length=1)]
    secondary_hypotheses: tuple[str, ...] = ()
    significance_threshold: Annotated[float, Field(gt=0.0, lt=1.0)]
    multiple_comparison_correction: Literal["bonferroni", "holm", "fdr_bh", "none"]
    n_comparisons: Annotated[int, Field(ge=1)]
    effect_size_metric: Literal["accuracy_minus_baseline", "roc_auc", "macro_f1"]
    effect_size_threshold: Annotated[float, Field(ge=0.0)]
    exploratory_allowed: bool = True

    @model_validator(mode="after")
    def _check(self) -> Self:
        if len(set(self.labels)) != len(self.labels):
            raise ValueError("preregistration.labels must be unique")
        if len(set(self.layers)) != len(self.layers):
            raise ValueError("preregistration.layers must be unique")
        # Lock invariants: either both set or both null.
        if (self.locked_at is None) != (self.locked_at_git_sha is None):
            raise ValueError(
                "locked_at and locked_at_git_sha must be set together (both null = unlocked)"
            )
        return self

    @property
    def is_locked(self) -> bool:
        return self.locked_at is not None and self.locked_at_git_sha is not None


__all__ = [
    "CANDIDATE_LABELS",
    "PILOT_MANIFEST_SCHEMA_VERSION",
    "PREREGISTRATION_SCHEMA_VERSION",
    "LayerIndex",
    "NlaScopeSpec",
    "PilotManifest",
    "Preregistration",
    "ProbeLabel",
    "StratumRecord",
]
