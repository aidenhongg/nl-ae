"""Single-pass eval orchestrator + ``ManifestBuilder``.

The runner is decomposition-agnostic: it accepts protocols rather than
concrete loaders, so unit tests can inject fakes that don't load Qwen.

Key fixes baked in from the plan:
- UR-R2.1: ``_seed_for`` uses ``derive_child_seed`` (SHA-256), never Python ``hash``.
- UR8: ``expected_visits`` is metadata-driven, not generator-tee.
- UR-R2.6: ``RunManifest.resumed_from_partial`` is True iff the run started
  with a non-empty rows_jsonl_partial.
- CI.11/CI.19: ``config_digest`` is required and checked on resume.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Iterable, Iterator
from contextlib import AbstractContextManager
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import Annotated, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

from nl_ae.data.canonical import CanonicalItem
from nl_ae.data.permute import PermutationMode, permutation_for
from nl_ae.inference.decoding import DecodingConfig
from nl_ae.inference.outputs import LetterScore, ScoringOutputs
from nl_ae.prompt.letter_tokens import LetterVariant, TokenizerLike, build_letter_token_table
from nl_ae.prompt.renderer import PromptRenderer
from nl_ae.runtime.seeds import derive_child_seed
from nl_ae.schema.hashing import now_utc_iso
from nl_ae.schema.models import (
    DatasetFingerprint,
    EnvFingerprint,
    ExtractorRecord,
    FirstTokenScoringMath,
    LetterSoftmaxEntry,
    LetterTokenEntry,
    ModelFingerprint,
    PermutationCoverage,
    PromptTemplateRecord,
    ResultRow,
    RunManifest,
    Sha256Hex,
)
from nl_ae.schema.paths import run_paths
from nl_ae.schema.writer import ResultsWriter

from .plan import EvalVisit, RunPlan
from .resume import ResumeMismatchError, ResumeState, scan_resume_state

LOG = logging.getLogger(__name__)


# --- protocols ----------------------------------------------------------


class ScoringEngine(Protocol):
    """Anything that produces ``ScoringOutputs`` from a prompt + letter table."""

    @property
    def tokenizer(self) -> TokenizerLike: ...

    @property
    def chat_template_hash(self) -> Sha256Hex: ...

    def fingerprint(self) -> ModelFingerprint: ...

    def score_and_generate(
        self,
        prompt: str,
        *,
        letter_token_table: list[LetterTokenEntry],
        letter_set: tuple[str, ...],
        decoding: DecodingConfig,
        scoring_math: FirstTokenScoringMath = ...,
        variant: LetterVariant = ...,
        max_free_text_chars: int = ...,
    ) -> ScoringOutputs: ...

    def try_clear_cache(self) -> None: ...


class CanonicalItemSource(Protocol):
    def iter_items(self) -> Iterator[CanonicalItem]: ...

    @property
    def item_count(self) -> int: ...

    def emit_dataset_fingerprint(self) -> DatasetFingerprint: ...


# --- configs ------------------------------------------------------------


class RunDecodingPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    decode_strategy: Literal["greedy", "sampled"] = "greedy"
    base_seed: Annotated[int, Field(ge=0)] | None = None
    max_new_tokens: Annotated[int, Field(ge=1, le=512)] = 64
    scoring_math: FirstTokenScoringMath = "full_vocab_softmax"
    temperature: Annotated[float, Field(gt=0.0, le=2.0)] | None = None
    top_p: Annotated[float, Field(gt=0.0, le=1.0)] | None = None
    top_k: Annotated[int, Field(ge=0)] | None = None


class EvalConfig(BaseModel):
    """C04's top-level eval config (formerly ``EvalRunnerConfig``; CI.16)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    plan: RunPlan
    decoding: RunDecodingPolicy = RunDecodingPolicy()
    batch_size: Literal[1] = 1
    log_every: Annotated[int, Field(ge=1)] = 50
    stdout_eta_every: Annotated[int, Field(ge=1)] = 10
    record_layers: tuple[Annotated[int, Field(ge=0)], ...] | None = None
    fail_fast: bool = True
    save_rendered_prompts: bool = True
    on_existing_dir: Literal["resume", "overwrite", "error"] = "resume"
    letter_variant: LetterVariant = "leading_space"
    max_free_text_chars: Annotated[int, Field(ge=128, le=65_536)] = 2048


# Legacy alias requested by CI.16.
EvalRunnerConfig = EvalConfig


# --- outcome ------------------------------------------------------------


@dataclass(frozen=True)
class RunOutcome:
    run_id: str
    run_dir: Path
    rows_written: int
    rows_expected: int
    status: Literal["completed", "failed", "aborted"]
    failure_reason: str | None
    resumed: bool
    wall_seconds: float
    permutation_coverage: tuple[PermutationCoverage, ...]


# --- manifest builder ---------------------------------------------------


class ManifestBuilder:
    """Assemble the initial ``RunManifest`` from component fingerprints."""

    def __init__(
        self,
        *,
        run_id: str,
        git_sha: str,
        git_dirty: bool,
        config_digest: Sha256Hex,
        config_yaml_text: str | None,
        cli_args: dict[str, str | int | float | bool | None],
        env: EnvFingerprint,
        model: ModelFingerprint,
        datasets: list[DatasetFingerprint],
        prompt_templates: list[PromptTemplateRecord],
        letter_token_table: list[LetterTokenEntry],
        seeds: dict[str, int],
        chat_template_hash: Sha256Hex | None,
        extractor: ExtractorRecord | None,
        deterministic_algorithms_actual: Literal[
            "off", "warn_only", "strict", "unavailable"
        ],
        applied_seeds_notes: tuple[str, ...] = (),
        kv_shared: bool = False,
        notes: str | None = None,
        rows_expected: int | None = None,
        resumed_from_partial: bool = False,
    ) -> None:
        self._run_id = run_id
        self._git_sha = git_sha
        self._git_dirty = git_dirty
        self._config_digest = config_digest
        self._config_yaml_text = config_yaml_text
        self._cli_args = cli_args
        self._env = env
        self._model = model
        self._datasets = datasets
        self._prompt_templates = prompt_templates
        self._letter_token_table = letter_token_table
        self._seeds = seeds
        self._chat_template_hash = chat_template_hash
        self._extractor = extractor
        self._deterministic_actual = deterministic_algorithms_actual
        self._applied_seeds_notes = applied_seeds_notes
        self._kv_shared = kv_shared
        self._notes = notes
        self._rows_expected = rows_expected
        self._resumed_from_partial = resumed_from_partial

    def config_digest(self) -> Sha256Hex:
        return self._config_digest

    def build(self) -> RunManifest:
        return RunManifest(
            run_id=self._run_id,
            git_sha=self._git_sha,
            git_dirty=self._git_dirty,
            started_at=now_utc_iso(),
            completion_status="in_progress",
            env=self._env,
            model=self._model,
            datasets=self._datasets,
            prompt_templates=self._prompt_templates,
            letter_token_table=self._letter_token_table,
            seeds=self._seeds,
            rows_written=0,
            rows_expected=self._rows_expected,
            cli_args=self._cli_args,
            notes=self._notes,
            chat_template_hash=self._chat_template_hash,
            extractor=self._extractor,
            kv_shared=self._kv_shared,
            applied_seeds_notes=self._applied_seeds_notes,
            deterministic_algorithms_actual=self._deterministic_actual,
            config_digest=self._config_digest,
            config_yaml_text=self._config_yaml_text,
            resumed_from_partial=self._resumed_from_partial,
        )


# --- seed derivation ----------------------------------------------------


def _seed_for(
    visit: EvalVisit, *, run_id: str, base_seed: int | None, decode_strategy: str
) -> int | None:
    """UR-R2.1: SHA-256-derived per-row seed. Never Python ``hash()``."""
    if decode_strategy != "sampled" or base_seed is None:
        return None
    tag = f"{run_id}|{visit.item.item_id}|{visit.permutation_id}|{visit.template_id}".encode(
        "utf-8"
    )
    return derive_child_seed(base_seed, tag)


# --- runner -------------------------------------------------------------


class EvalRunner(AbstractContextManager["EvalRunner"]):
    def __init__(
        self,
        *,
        config: EvalConfig,
        run_dir: Path,
        item_sources: Iterable[CanonicalItemSource],
        renderer: PromptRenderer,
        engine: ScoringEngine,
        manifest_builder: ManifestBuilder,
        letter_variants: tuple[LetterVariant, ...] = ("bare", "leading_space"),
        logger: logging.Logger | None = None,
    ) -> None:
        self._config = config
        self._run_dir = run_dir
        self._item_sources = tuple(item_sources)
        if not self._item_sources:
            raise ValueError("item_sources is empty")
        self._renderer = renderer
        self._engine = engine
        self._manifest_builder = manifest_builder
        self._letter_variants = letter_variants
        self._logger = logger or LOG
        self._writer: ResultsWriter | None = None
        self._coverage: dict[str, dict[str, set[int]]] = {}
        self._visits_completed = 0
        self._expected_visits = self._compute_expected_visits()

    # --- properties -----------------------------------------------------

    @property
    def run_dir(self) -> Path:
        return self._run_dir

    @property
    def expected_visits(self) -> int:
        return self._expected_visits

    @property
    def visits_completed(self) -> int:
        return self._visits_completed

    # --- context manager ------------------------------------------------

    def __enter__(self) -> EvalRunner:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        if self._writer is not None and exc is not None:
            try:
                self._writer.finalize(status="failed", failure_reason=repr(exc), emit_parquet=False)
            except Exception:  # pragma: no cover
                pass

    # --- expected_visits (UR8) ----------------------------------------

    def _compute_expected_visits(self) -> int:
        total_items = sum(src.item_count for src in self._item_sources)
        return (
            total_items
            * len(self._config.plan.permutation_ids)
            * len(self._config.plan.template_ids)
        )

    # --- main loop ------------------------------------------------------

    def run(self) -> RunOutcome:
        plan = self._config.plan
        manifest = self._manifest_builder.build()
        paths = run_paths(self._run_dir.parent, self._run_dir.name)
        resume_state: ResumeState = scan_resume_state(
            self._run_dir,
            expected_run_id=manifest.run_id,
            expected_config_digest=manifest.config_digest,
        )
        resumed = not resume_state.is_fresh
        if resumed:
            # Surface UR-R2.6.
            manifest = manifest.model_copy(update={"resumed_from_partial": True})

        manifest = manifest.model_copy(update={"rows_expected": self._expected_visits})

        on_existing = self._config.on_existing_dir
        if resumed and on_existing == "error":
            raise ResumeMismatchError(
                f"run dir {self._run_dir} contains prior rows; "
                "use on_existing_dir='resume' or 'overwrite'"
            )

        t_start = time.perf_counter()
        status: Literal["completed", "failed", "aborted"] = "completed"
        failure_reason: str | None = None

        with ResultsWriter(
            self._run_dir,
            manifest,
            on_existing=on_existing,
            fsync_every=32,
        ) as writer:
            self._writer = writer
            try:
                for visit in self._iter_visits():
                    if resume_state.is_completed(visit):
                        continue
                    row = self._process_visit(visit, manifest_run_id=manifest.run_id)
                    writer.write_row(row)
                    self._record_coverage(visit)
                    self._visits_completed += 1
                    if self._visits_completed % self._config.log_every == 0:
                        self._logger.info(
                            "progress %d / %d (%.1f%%)",
                            self._visits_completed,
                            self._expected_visits,
                            100.0 * self._visits_completed / max(self._expected_visits, 1),
                        )
            except KeyboardInterrupt:
                status, failure_reason = "aborted", "KeyboardInterrupt"
            except Exception as exc:
                status, failure_reason = "failed", repr(exc)
                if self._config.fail_fast:
                    writer.update_manifest(
                        completion_status="failed",
                        failure_reason=failure_reason,
                        permutation_coverage=self._coverage_payload(),
                    )
                    writer.finalize(
                        status="failed", failure_reason=failure_reason, emit_parquet=False
                    )
                    self._writer = None
                    raise
                self._logger.exception("eval failure (continuing): %r", exc)

            writer.update_manifest(
                permutation_coverage=self._coverage_payload(),
            )
            writer.finalize(
                status=status,
                failure_reason=failure_reason,
                emit_parquet=status == "completed",
            )
            self._writer = None

        wall = time.perf_counter() - t_start
        return RunOutcome(
            run_id=manifest.run_id,
            run_dir=self._run_dir,
            rows_written=self._visits_completed,
            rows_expected=self._expected_visits,
            status=status,
            failure_reason=failure_reason,
            resumed=resumed,
            wall_seconds=wall,
            permutation_coverage=tuple(self._coverage_payload()),
        )

    # --- inner helpers --------------------------------------------------

    def _iter_visits(self) -> Iterator[EvalVisit]:
        plan = self._config.plan
        for src in self._item_sources:
            yield from plan.iter_visits(src.iter_items())

    def _process_visit(self, visit: EvalVisit, *, manifest_run_id: str) -> ResultRow:
        perm = permutation_for(
            visit.item, visit.permutation_id, mode=self._config.plan.permutation_mode
        )
        prompt, prompt_hash = self._renderer.render(perm, visit.template_id)
        letter_table = build_letter_token_table(
            self._engine.tokenizer,
            perm.letters,
            variants=self._letter_variants,
        )

        seed = _seed_for(
            visit,
            run_id=manifest_run_id,
            base_seed=self._config.decoding.base_seed,
            decode_strategy=self._config.decoding.decode_strategy,
        )
        decoding = DecodingConfig(
            strategy=self._config.decoding.decode_strategy,
            max_new_tokens=self._config.decoding.max_new_tokens,
            seed=seed,
            temperature=(
                self._config.decoding.temperature
                if self._config.decoding.decode_strategy == "sampled"
                else None
            ),
            top_p=(
                self._config.decoding.top_p
                if self._config.decoding.decode_strategy == "sampled"
                else None
            ),
            top_k=(
                self._config.decoding.top_k
                if self._config.decoding.decode_strategy == "sampled"
                else None
            ),
        )

        outputs = self._engine.score_and_generate(
            prompt,
            letter_token_table=letter_table,
            letter_set=tuple(perm.letters),
            decoding=decoding,
            scoring_math=self._config.decoding.scoring_math,
            variant=self._config.letter_variant,
            max_free_text_chars=self._config.max_free_text_chars,
        )

        rendered_ref: str | None = None
        if self._config.save_rendered_prompts and self._writer is not None:
            sidecar = self._writer.paths.prompts_dir / f"{prompt_hash}.txt"
            if not sidecar.exists():
                sidecar.parent.mkdir(parents=True, exist_ok=True)
                sidecar.write_text(prompt, encoding="utf-8")
            rendered_ref = f"prompts/{prompt_hash}.txt"

        softmax_rows = [
            LetterSoftmaxEntry(
                letter=ls.letter,
                token_id=ls.token_id,
                prob=ls.prob,
                prob_valid=ls.prob_valid,
                logit=ls.logit,
            )
            for ls in outputs.letter_softmax
        ]

        return ResultRow(
            run_id=manifest_run_id,
            item_id=visit.item.item_id,
            dataset_name=visit.item.dataset_name,
            dataset_split=visit.item.dataset_split,
            subject=visit.item.subject,
            template_id=visit.template_id,
            permutation_id=visit.permutation_id,
            prompt_hash=prompt_hash,
            rendered_prompt_ref=rendered_ref,
            gold_letter=perm.gold_letter,
            first_token_letter=outputs.first_token_letter,
            free_text_letter=outputs.free_text_letter,
            free_text_raw=outputs.free_text_raw,
            free_text_truncated=outputs.free_text_truncated,
            agreement_flag=outputs.agreement_flag,
            letter_softmax=softmax_rows,
            n_options=perm.n_options,
            free_text_seed=outputs.decode_seed,
            decode_strategy=outputs.decode_strategy,
            activation_ref=None,
            wall_time_ms=outputs.total_wall_time_ms,
            created_at=now_utc_iso(),
            extractor_id=outputs.extractor_id,
            extractor_match_rule=outputs.extractor_match_rule,
            first_token_scoring_math=outputs.scoring_math,
            total_letter_mass=outputs.total_letter_mass,
        )

    def _record_coverage(self, visit: EvalVisit) -> None:
        per_template = self._coverage.setdefault(visit.template_id, {})
        item_perms = per_template.setdefault(visit.item.item_id, set())
        item_perms.add(visit.permutation_id)

    def _coverage_payload(self) -> list[PermutationCoverage]:
        out: list[PermutationCoverage] = []
        for template_id in sorted(self._coverage):
            items = self._coverage[template_id]
            distinct_perms: set[int] = set()
            for perms in items.values():
                distinct_perms.update(perms)
            out.append(
                PermutationCoverage(
                    template_id=template_id,
                    items_visited=len(items),
                    permutations_per_item_mode=self._config.plan.permutations_per_item,
                    distinct_permutation_ids=tuple(sorted(distinct_perms)),
                )
            )
        return out


# Re-export PermutationMode for callers building RunPlan from CLI strings.
_ = PermutationMode

__all__ = [
    "EvalConfig",
    "EvalRunner",
    "EvalRunnerConfig",
    "ManifestBuilder",
    "RunDecodingPolicy",
    "RunOutcome",
    "ScoringEngine",
    "CanonicalItemSource",
]
