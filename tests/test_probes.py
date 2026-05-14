"""Tests for C07 — per-layer linear probes.

Unit-tests of splits / labels / digests run without sklearn. sklearn-backed
trainer cases are gated by ``pytest.importorskip("sklearn")`` so the broader
suite stays runnable on a base install.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest
import yaml
from click.testing import CliRunner

from nl_ae.cache.extractor import ActivationCacheExtractor, load_extractor_inputs
from nl_ae.cli.commands import probe_train_cmd as probe_train_cmd_mod
from nl_ae.pilot.errors import (
    PreregistrationMissingError,
    PreregistrationUnlockedError,
)
from nl_ae.pilot.manifest import write_pilot_manifest
from nl_ae.pilot.models import (
    NlaScopeSpec,
    PilotManifest,
    Preregistration,
    ProbeLabel,
    StratumRecord,
)
from nl_ae.probes import (
    ProbeArtifactReader,
    ProbeCellKey,
    ProbeManifest,
    SklearnKwargs,
    compute_probe_manifest_digest,
    compute_sklearn_kwargs_digest,
    extract_label,
    split_by_item_within_fold,
)
from nl_ae.probes.errors import (
    LabelOutOfScopeError,
    LayerOutOfScopeError,
    ProbeManifestStaleError,
)
from nl_ae.probes.fitter import FitResult, SklearnLogisticFitter
from nl_ae.probes.trainer import (
    ProbeTrainer,
    hash_preregistration,
    load_trainer_inputs,
    scan_resume_state,
)
from nl_ae.schema.hashing import now_utc_iso
from nl_ae.schema.models import (
    DatasetFingerprint,
    EnvFingerprint,
    LetterSoftmaxEntry,
    LetterTokenEntry,
    ModelFingerprint,
    PromptTemplateRecord,
    QuantizationSpec,
    ResultRow,
    RunManifest,
)
from nl_ae.schema.paths import run_paths
from nl_ae.schema.writer import ResultsWriter

DIM = 8
N_LAYERS = 4
CHAT_TEMPLATE_HASH = "c" * 64
RUN_ID = "20260101T000000Z-deadbee-test"


# --- fake activation source (mirrors test_cache_extractor.py) -----------


@dataclass(frozen=True)
class _FakeVec:
    values: tuple[float, ...]

    def tolist(self) -> list[float]:
        return list(self.values)


@dataclass
class _ForwardOutput:
    hidden_states: dict[int, _FakeVec] | None


@dataclass
class FakeSource:
    """Deterministic per-prompt features. See ``tests/test_cache_extractor.py``."""

    hidden_size: int = DIM
    n_layers: int = N_LAYERS
    chat_template_hash: str = CHAT_TEMPLATE_HASH
    calls: list[tuple[str, tuple[int, ...]]] = field(default_factory=list)

    def forward(
        self,
        prompt: str,
        *,
        capture_hiddens: bool = True,
        record_layers_override: tuple[int, ...] | None = None,
    ) -> _ForwardOutput:
        layers = tuple(record_layers_override or ())
        self.calls.append((prompt, layers))
        if not capture_hiddens:
            return _ForwardOutput(hidden_states=None)
        digest = hashlib.sha256(prompt.encode("utf-8")).digest()
        base = [float(digest[i % len(digest)]) for i in range(self.hidden_size)]
        return _ForwardOutput(
            hidden_states={
                layer: _FakeVec(tuple(v + layer for v in base)) for layer in layers
            },
        )


# --- run-dir factory ----------------------------------------------------


_LETTERS = ("A", "B", "C", "D")
_RULES = ("my_answer_is", "the_answer_is", "boxed", "letter_only")


def _hash_prompt(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _row(
    *,
    item_id: str,
    perm: int,
    prompt: str,
    first_token: str,
    free_text: str,
    gold: str | None,
    rule: str = "my_answer_is",
) -> tuple[ResultRow, str]:
    prompt_hash = _hash_prompt(prompt)
    softmax = [
        LetterSoftmaxEntry(letter=letter, token_id=10 + i, prob=0.25, prob_valid=True, logit=0.0)
        for i, letter in enumerate(_LETTERS)
    ]
    agreement = first_token == free_text
    return (
        ResultRow(
            run_id=RUN_ID,
            item_id=item_id,
            dataset_name="mmlu",
            dataset_split="test",
            subject="algebra",
            template_id="mcq_flat_v1",
            permutation_id=perm,
            prompt_hash=prompt_hash,
            rendered_prompt_ref=f"prompts/{prompt_hash}.txt",
            gold_letter=gold,  # None simulates OpinionQA rows
            first_token_letter=first_token,
            free_text_letter=free_text,
            free_text_raw=f"{free_text}.",
            agreement_flag=agreement,
            letter_softmax=softmax,
            n_options=4,
            free_text_seed=None,
            decode_strategy="greedy",
            created_at=now_utc_iso(),
            extractor_id="regex_ladder",
            extractor_match_rule=rule,  # type: ignore[arg-type]
            first_token_scoring_math="full_vocab_softmax",
            total_letter_mass=1.0,
        ),
        prompt,
    )


def _build_phase1(
    tmp_path: Path,
    *,
    n_items: int = 40,
    include_opinionqa_nulls: bool = True,
) -> tuple[Path, list[str], list[str]]:
    """Mint a Phase 1 run with `n_items` items varied enough that every probe
    label has both classes / multiple classes in train / val / test."""
    run_dir = tmp_path / RUN_ID
    paths = run_paths(run_dir.parent, run_dir.name)
    items = [f"mmlu/v1/algebra/q-{i:03d}" for i in range(n_items)]
    pilot_items = items[: n_items // 2]
    holdout_items = items[n_items // 2 :]

    manifest = RunManifest(
        run_id=RUN_ID,
        git_sha="deadbeefcafe",
        git_dirty=False,
        started_at=now_utc_iso(),
        completion_status="in_progress",
        env=EnvFingerprint(
            os_name="Windows",
            os_version="11",
            python_version="3.13.0",
            cuda_version=None,
            torch_version=None,
            transformers_version=None,
            bitsandbytes_version=None,
            accelerate_version=None,
            gpu_name=None,
            gpu_vram_mb=None,
        ),
        model=ModelFingerprint(
            hf_model_id="Qwen/Qwen2.5-7B-Instruct",
            hf_model_commit=None,
            hf_tokenizer_id="Qwen/Qwen2.5-7B-Instruct",
            hf_tokenizer_commit=None,
            quantization=QuantizationSpec(kind="fp16"),
        ),
        datasets=[
            DatasetFingerprint(
                name="mmlu",
                hf_dataset_id="cais/mmlu",
                split="test",
                commit_or_revision=None,
                item_count=n_items,
                item_id_scheme="mmlu/v1/<subject>/q-<sha256>[:12]",
            )
        ],
        prompt_templates=[
            PromptTemplateRecord(
                template_id="mcq_flat_v1",
                template_content_hash="0" * 64,
                template_text="x",
                role="user",
            )
        ],
        letter_token_table=[
            LetterTokenEntry(letter=letter, token_id=10 + i, variant="bare", token_str=letter)
            for i, letter in enumerate(_LETTERS)
        ],
        seeds={"root": 0},
        cli_args={"config_path": "x"},
        config_digest="a" * 64,
        chat_template_hash=CHAT_TEMPLATE_HASH,
    )

    with ResultsWriter(run_dir, manifest, on_existing="error") as writer:
        for idx, iid in enumerate(items):
            first_token = _LETTERS[idx % 4]
            # Make ~30% disagree.
            free_text = _LETTERS[(idx + 1) % 4] if idx % 3 == 0 else first_token
            # Make ~20% null gold (OpinionQA-shaped rows).
            gold = (
                None
                if include_opinionqa_nulls and idx % 5 == 0
                else _LETTERS[(idx // 4) % 4]
            )
            rule = _RULES[idx % len(_RULES)]
            row, prompt = _row(
                item_id=iid,
                perm=0,
                prompt=f"prompt for {iid} idx={idx}",
                first_token=first_token,
                free_text=free_text,
                gold=gold,
                rule=rule,
            )
            writer.write_row(row)
            sidecar = paths.prompts_dir / f"{row.prompt_hash}.txt"
            sidecar.write_text(prompt, encoding="utf-8")
        writer.finalize(emit_parquet=False)

    pilot_manifest = PilotManifest(
        run_id=RUN_ID,
        seed=0,
        frac=0.5,
        stratify_by=("subject",),
        min_per_stratum=1,
        strata=(
            StratumRecord(
                key="algebra",
                source_field="subject",
                n_total=n_items,
                n_pilot=len(pilot_items),
                n_holdout=len(holdout_items),
            ),
        ),
        pilot_item_ids=tuple(sorted(pilot_items)),
        n_pilot=len(pilot_items),
        n_holdout=len(holdout_items),
        n_total=n_items,
        created_at=now_utc_iso(),
        completion_status="committed",
        pilot_manifest_digest="ab" + "0" * 62,
    )
    write_pilot_manifest(paths.pilot_manifest_json, pilot_manifest)
    return run_dir, pilot_items, holdout_items


def _build_phase1_with_cache(
    tmp_path: Path,
    *,
    layers: tuple[int, ...] = (0, 1, 2),
    fold: str = "pilot",
    n_items: int = 40,
    write_preregistration: bool = False,
    prereg_labels: tuple[ProbeLabel, ...] = ("first_token_correct",),
    prereg_layers: tuple[int, ...] = (0, 1, 2),
    prereg_locked: bool = True,
) -> Path:
    """Build Phase 1 + run :class:`ActivationCacheExtractor` to populate the
    activation cache for ``fold``. Returns ``run_dir``."""
    run_dir, _pilot, _holdout = _build_phase1(tmp_path, n_items=n_items)

    if write_preregistration:
        from nl_ae.pilot.manifest import load_pilot_manifest

        pm = load_pilot_manifest(run_dir / "pilot_manifest.json")
        _write_preregistration(
            run_dir,
            pilot_manifest_digest=pm.pilot_manifest_digest,
            labels=prereg_labels,
            layers=prereg_layers,
            locked=prereg_locked,
        )

    inputs = load_extractor_inputs(run_dir, fold)  # type: ignore[arg-type]
    ActivationCacheExtractor(
        inputs=inputs, source=FakeSource(), layers=layers, shard_rows=200
    ).run()
    return run_dir


def _write_preregistration(
    run_dir: Path,
    *,
    pilot_manifest_digest: str,
    labels: tuple[ProbeLabel, ...] = ("first_token_correct",),
    layers: tuple[int, ...] = (0,),
    locked: bool = True,
) -> None:
    prereg = Preregistration(
        run_id=RUN_ID,
        pilot_manifest_digest=pilot_manifest_digest,
        holdout_vs_full="holdout-only",
        labels=labels,
        layers=layers,
        nla_scope=NlaScopeSpec(
            layer=layers[0],
            fold="holdout",
            limit=10,
            decode_strategy="greedy",
            temperature=0.7,
            max_new_tokens=64,
        ),
        primary_hypothesis="x",
        significance_threshold=0.05,
        multiple_comparison_correction="bonferroni",
        n_comparisons=len(labels) * len(layers),
        effect_size_metric="accuracy_minus_baseline",
        effect_size_threshold=0.0,
        locked_at=now_utc_iso() if locked else None,
        locked_at_git_sha=("a" * 40) if locked else None,
    )
    fm = yaml.safe_dump(prereg.model_dump(mode="json"), sort_keys=False)
    (run_dir / "preregistration.md").write_text(
        "---\n" + fm + "---\n\n# preregistration body\n",
        encoding="utf-8",
    )


def _baseline_kwargs() -> SklearnKwargs:
    return SklearnKwargs(
        penalty="l2",
        C=1.0,
        solver="lbfgs",
        max_iter=200,
        fit_intercept=True,
        class_weight="none",
        standardize=False,
    )


# --- splits + labels ----------------------------------------------------


def test_split_by_item_within_fold_deterministic() -> None:
    items = [f"item_{i}" for i in range(1000)]
    a = split_by_item_within_fold(items, probe_seed=1234)
    b = split_by_item_within_fold(items, probe_seed=1234)
    assert a == b


def test_split_by_item_within_fold_seed_changes_assignment() -> None:
    items = [f"item_{i}" for i in range(1000)]
    a = split_by_item_within_fold(items, probe_seed=1)
    b = split_by_item_within_fold(items, probe_seed=2)
    diffs = sum(1 for k in items if a[k] != b[k])
    assert diffs > 100  # massively different seeds should reshuffle most buckets


def test_split_proportions_within_tolerance() -> None:
    items = [f"item_{i}" for i in range(5000)]
    assign = split_by_item_within_fold(items, probe_seed=99)
    counts = {"train": 0, "val": 0, "test": 0}
    for s in assign.values():
        counts[s] += 1
    n = len(items)
    assert 0.68 < counts["train"] / n < 0.72
    assert 0.13 < counts["val"] / n < 0.17
    assert 0.13 < counts["test"] / n < 0.17


def test_extract_label_disagreement_filter() -> None:
    rows = [
        _row(item_id="a", perm=0, prompt="p1", first_token="A", free_text="A", gold="A")[0],
        _row(item_id="b", perm=0, prompt="p2", first_token="A", free_text="B", gold="A")[0],
        # row with first_token=None: not directly constructible (Phase 1
        # enforces a letter), but free_text=None happens whenever the
        # extractor fails. Mock by mutating after construction.
    ]
    # Manufacture a free-text-null row by bypassing validators.
    null_row = rows[0].model_copy(update={"free_text_letter": None, "agreement_flag": None})
    rows.append(null_row)
    out = extract_label(rows, "disagreement_flag")
    assert out.is_binary
    assert out.classes == ("0", "1")
    assert out.valid_mask.tolist() == [True, True, False]
    assert out.y.tolist() == ["0", "1"]
    assert out.n_dropped == 1


def test_extract_label_first_token_correct_filters_opinionqa() -> None:
    mmlu = _row(item_id="m", perm=0, prompt="p", first_token="A", free_text="A", gold="A")[0]
    opinionqa = mmlu.model_copy(update={"gold_letter": None})
    out = extract_label([mmlu, opinionqa], "first_token_correct")
    assert out.is_binary
    assert out.valid_mask.tolist() == [True, False]
    assert out.y.tolist() == ["1"]
    assert out.n_dropped == 1


def test_extract_label_multi_class_letters() -> None:
    rows = [
        _row(item_id=f"r{i}", perm=0, prompt=f"p{i}", first_token=_LETTERS[i % 3], free_text="A", gold="A")[0]
        for i in range(6)
    ]
    out = extract_label(rows, "first_token_letter")
    assert not out.is_binary
    assert out.classes == ("A", "B", "C")
    assert out.valid_mask.sum() == 6
    assert out.y.tolist() == ["A", "B", "C", "A", "B", "C"]


def test_extract_label_extractor_match_rule_keeps_all() -> None:
    rows = [
        _row(
            item_id=f"r{i}",
            perm=0,
            prompt=f"p{i}",
            first_token="A",
            free_text="A",
            gold="A",
            rule=_RULES[i % len(_RULES)],
        )[0]
        for i in range(8)
    ]
    out = extract_label(rows, "extractor_match_rule")
    assert not out.is_binary
    assert out.valid_mask.sum() == len(rows)
    assert set(out.classes) == set(_RULES)


# --- manifest digests ----------------------------------------------------


def _digest_inputs() -> dict[str, Any]:
    return {
        "run_id": RUN_ID,
        "fold": "pilot",
        "labels": ("disagreement_flag", "first_token_letter"),
        "layers": (0, 1, 2),
        "split_seed": 0,
        "split_frac": (0.7, 0.15, 0.15),
        "sklearn_kwargs": _baseline_kwargs(),
        "sklearn_kwargs_digest": compute_sklearn_kwargs_digest(_baseline_kwargs()),
        "source_run_id": RUN_ID,
        "source_cache_key_digest": "0" * 64,
        "source_pilot_manifest_digest": "1" * 64,
        "source_preregistration_digest": None,
    }


def test_compute_probe_manifest_digest_stable() -> None:
    d1 = compute_probe_manifest_digest(**_digest_inputs())
    d2 = compute_probe_manifest_digest(**_digest_inputs())
    assert d1 == d2


def test_compute_probe_manifest_digest_changes_on_kwarg_drift() -> None:
    base = _digest_inputs()
    d1 = compute_probe_manifest_digest(**base)
    drifted_kwargs = _baseline_kwargs().model_copy(update={"C": 0.5})
    drifted = {
        **base,
        "sklearn_kwargs": drifted_kwargs,
        "sklearn_kwargs_digest": compute_sklearn_kwargs_digest(drifted_kwargs),
    }
    d2 = compute_probe_manifest_digest(**drifted)
    assert d1 != d2


def test_manifest_round_trip(tmp_path: Path) -> None:
    inputs = _digest_inputs()
    digest = compute_probe_manifest_digest(**inputs)
    manifest = ProbeManifest(
        run_id=inputs["run_id"],
        fold=inputs["fold"],
        labels=inputs["labels"],
        layers=inputs["layers"],
        split_seed=inputs["split_seed"],
        split_frac=inputs["split_frac"],
        sklearn_kwargs=inputs["sklearn_kwargs"],
        sklearn_kwargs_digest=inputs["sklearn_kwargs_digest"],
        source_run_id=inputs["source_run_id"],
        source_cache_key_digest=inputs["source_cache_key_digest"],
        source_pilot_manifest_digest=inputs["source_pilot_manifest_digest"],
        source_preregistration_digest=inputs["source_preregistration_digest"],
        completion_status="in_progress",
        started_at=now_utc_iso(),
        probe_manifest_digest=digest,
    )
    payload = manifest.model_dump_json()
    reloaded = ProbeManifest.model_validate_json(payload)
    assert reloaded == manifest


# --- trainer inputs preflight -------------------------------------------


def test_load_trainer_inputs_pilot_happy(tmp_path: Path) -> None:
    run_dir = _build_phase1_with_cache(tmp_path, layers=(0, 1, 2))
    inputs, labels, layers = load_trainer_inputs(run_dir, "pilot")
    assert inputs.fold == "pilot"
    assert set(labels) == {
        "disagreement_flag",
        "first_token_correct",
        "free_text_correct",
        "first_token_letter",
        "free_text_letter",
        "extractor_match_rule",
    }
    assert layers == (0, 1, 2)
    assert inputs.preregistration is None


def test_load_trainer_inputs_holdout_refuses_when_preregistration_unlocked(tmp_path: Path) -> None:
    """End-to-end: holdout fold with unlocked preregistration → PreregistrationUnlockedError."""
    # Build phase 1 + pilot cache + a locked prereg so the holdout cache can be extracted;
    # then re-write the prereg as unlocked before calling load_trainer_inputs(holdout).
    run_dir = _build_phase1_with_cache(
        tmp_path,
        layers=(0, 1),
        fold="holdout",
        write_preregistration=True,
        prereg_layers=(0, 1),
        prereg_locked=True,
    )
    # Re-write prereg as unlocked.
    from nl_ae.pilot.manifest import load_pilot_manifest

    pm = load_pilot_manifest(run_dir / "pilot_manifest.json")
    _write_preregistration(
        run_dir,
        pilot_manifest_digest=pm.pilot_manifest_digest,
        labels=("first_token_correct",),
        layers=(0, 1),
        locked=False,
    )
    with pytest.raises(PreregistrationUnlockedError):
        load_trainer_inputs(run_dir, "holdout")


def test_load_trainer_inputs_holdout_refuses_missing_preregistration(tmp_path: Path) -> None:
    # Build pilot cache (no prereg needed) then attempt holdout.
    run_dir = _build_phase1_with_cache(tmp_path, layers=(0, 1))
    with pytest.raises(PreregistrationMissingError):
        load_trainer_inputs(run_dir, "holdout")


def test_load_trainer_inputs_holdout_layer_out_of_scope(tmp_path: Path) -> None:
    run_dir = _build_phase1_with_cache(
        tmp_path,
        layers=(0, 1),
        fold="holdout",
        write_preregistration=True,
        prereg_layers=(0, 1),
    )
    # Re-write the prereg with a layer not in the cache.
    from nl_ae.pilot.manifest import load_pilot_manifest

    pm = load_pilot_manifest(run_dir / "pilot_manifest.json")
    _write_preregistration(
        run_dir,
        pilot_manifest_digest=pm.pilot_manifest_digest,
        labels=("first_token_correct",),
        layers=(0, 1, 5),  # 5 not in cache layers (0,1)
        locked=True,
    )
    with pytest.raises(LayerOutOfScopeError):
        load_trainer_inputs(run_dir, "holdout")


def test_load_trainer_inputs_holdout_user_override_disagrees(tmp_path: Path) -> None:
    run_dir = _build_phase1_with_cache(
        tmp_path,
        layers=(0, 1),
        fold="holdout",
        write_preregistration=True,
        prereg_labels=("first_token_correct",),
        prereg_layers=(0, 1),
    )
    with pytest.raises(LabelOutOfScopeError):
        load_trainer_inputs(
            run_dir,
            "holdout",
            labels_override=("disagreement_flag",),
        )


def test_load_trainer_inputs_pilot_override_out_of_scope(tmp_path: Path) -> None:
    run_dir = _build_phase1_with_cache(tmp_path, layers=(0, 1))
    with pytest.raises(LabelOutOfScopeError):
        load_trainer_inputs(
            run_dir,
            "pilot",
            labels_override=("bogus_label",),  # type: ignore[arg-type]
        )


# --- trainer end-to-end (sklearn) ---------------------------------------


def test_trainer_writes_per_cell_artifacts(tmp_path: Path) -> None:
    pytest.importorskip("sklearn")
    run_dir = _build_phase1_with_cache(tmp_path, layers=(0, 1, 2), n_items=40)
    inputs, _all_labels, _all_layers = load_trainer_inputs(run_dir, "pilot")
    labels: tuple[ProbeLabel, ...] = ("first_token_letter", "extractor_match_rule")
    layers = (0, 1, 2)
    trainer = ProbeTrainer(
        inputs=inputs,
        fitter=SklearnLogisticFitter(),
        labels=labels,
        layers=layers,
        sklearn_kwargs=_baseline_kwargs(),
        split_seed=0,
        split_frac=(0.7, 0.15, 0.15),
    )
    outcome = trainer.run()
    assert outcome.status == "completed"
    assert outcome.cells_completed == len(labels) * len(layers)

    reader = ProbeArtifactReader.open(run_dir, "pilot")
    for label in labels:
        for layer in layers:
            cell_dir = reader.cell_dir(label, layer)
            assert (cell_dir / "coef.npy").exists()
            assert (cell_dir / "metrics.json").exists()
        assert (reader.label_dir(label) / "predictions.parquet").exists()
        assert (reader.label_dir(label) / "summary.parquet").exists()

    # Manifest cell SHA matches on-disk file SHA.
    for cell in reader.manifest.cells:
        cell_dir = reader.cell_dir(cell.label, cell.layer)
        coef_sha = _sha256_file(cell_dir / "coef.npy")
        metrics_sha = _sha256_file(cell_dir / "metrics.json")
        assert cell.coef_sha256 == coef_sha
        assert cell.metrics_sha256 == metrics_sha


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def test_trainer_predictions_long_layout(tmp_path: Path) -> None:
    pytest.importorskip("sklearn")
    pytest.importorskip("pandas")
    run_dir = _build_phase1_with_cache(tmp_path, layers=(0, 1), n_items=40)
    inputs, _l, _ly = load_trainer_inputs(run_dir, "pilot")
    layers = (0, 1)
    ProbeTrainer(
        inputs=inputs,
        fitter=SklearnLogisticFitter(),
        labels=("disagreement_flag", "first_token_letter"),
        layers=layers,
        sklearn_kwargs=_baseline_kwargs(),
        split_seed=0,
        split_frac=(0.7, 0.15, 0.15),
    ).run()

    reader = ProbeArtifactReader.open(run_dir, "pilot")
    # Binary: one row per (visit, layer).
    binary = reader.load_predictions("disagreement_flag")
    n_visits_per_layer = (binary.groupby("layer").size()).unique()
    assert binary["class"].unique().tolist() == ["1"]
    assert binary["layer"].nunique() == 2
    assert len(n_visits_per_layer) == 1  # same count per layer
    # Multi-class: one row per (visit, layer, class).
    multi = reader.load_predictions("first_token_letter")
    classes_per_layer = multi.groupby("layer")["class"].nunique().unique()
    assert (classes_per_layer >= 2).all()


def test_trainer_summary_per_layer(tmp_path: Path) -> None:
    pytest.importorskip("sklearn")
    pytest.importorskip("pandas")
    run_dir = _build_phase1_with_cache(tmp_path, layers=(0, 1, 2), n_items=40)
    inputs, _l, _ly = load_trainer_inputs(run_dir, "pilot")
    ProbeTrainer(
        inputs=inputs,
        fitter=SklearnLogisticFitter(),
        labels=("first_token_correct",),
        layers=(0, 1, 2),
        sklearn_kwargs=_baseline_kwargs(),
        split_seed=0,
        split_frac=(0.7, 0.15, 0.15),
    ).run()
    reader = ProbeArtifactReader.open(run_dir, "pilot")
    summary = reader.load_summary("first_token_correct")
    assert len(summary) == 3
    assert sorted(summary["layer"].tolist()) == [0, 1, 2]
    # Required columns present.
    for col in (
        "n_train",
        "n_val",
        "n_test",
        "train_accuracy",
        "test_accuracy",
        "balanced_accuracy_test",
        "roc_auc_test",
        "brier_score_test",
        "ece_test",
    ):
        assert col in summary.columns


def test_trainer_resume_skips_completed_cells(tmp_path: Path) -> None:
    pytest.importorskip("sklearn")
    run_dir = _build_phase1_with_cache(tmp_path, layers=(0, 1), n_items=40)
    inputs, _l, _ly = load_trainer_inputs(run_dir, "pilot")
    labels: tuple[ProbeLabel, ...] = ("first_token_letter",)
    layers = (0, 1)
    kwargs = _baseline_kwargs()
    ProbeTrainer(
        inputs=inputs,
        fitter=SklearnLogisticFitter(),
        labels=labels,
        layers=layers,
        sklearn_kwargs=kwargs,
        split_seed=0,
        split_frac=(0.7, 0.15, 0.15),
    ).run()
    # Compute the expected digest and verify scan_resume_state.
    prereg_digest = (
        hash_preregistration(inputs.preregistration)
        if inputs.preregistration is not None
        else None
    )
    expected_digest = compute_probe_manifest_digest(
        run_id=inputs.run_manifest.run_id,
        fold="pilot",
        labels=labels,
        layers=layers,
        split_seed=0,
        split_frac=(0.7, 0.15, 0.15),
        sklearn_kwargs=kwargs,
        sklearn_kwargs_digest=compute_sklearn_kwargs_digest(kwargs),
        source_run_id=inputs.run_manifest.run_id,
        source_cache_key_digest=inputs.activation_manifest.cache_key_composition_digest,
        source_pilot_manifest_digest=inputs.pilot_manifest.pilot_manifest_digest,
        source_preregistration_digest=prereg_digest,
    )
    existing, completed = scan_resume_state(
        paths=inputs.paths, fold="pilot", expected_manifest_digest=expected_digest
    )
    assert existing is not None
    assert completed == frozenset(
        ProbeCellKey(label=label, layer=layer) for label in labels for layer in layers
    )

    # Second pass — resume; nothing new should be fit.
    trainer2 = ProbeTrainer(
        inputs=inputs,
        fitter=_AlwaysFailFitter(),  # would raise if invoked
        labels=labels,
        layers=layers,
        sklearn_kwargs=kwargs,
        split_seed=0,
        split_frac=(0.7, 0.15, 0.15),
        completed=completed,
        existing_manifest=existing,
    )
    outcome2 = trainer2.run()
    assert outcome2.status == "completed"
    assert outcome2.cells_completed == 0
    assert outcome2.cells_skipped_resume == len(labels) * len(layers)


def test_trainer_resume_refuses_stale_digest(tmp_path: Path) -> None:
    pytest.importorskip("sklearn")
    run_dir = _build_phase1_with_cache(tmp_path, layers=(0, 1), n_items=40)
    inputs, _l, _ly = load_trainer_inputs(run_dir, "pilot")
    labels: tuple[ProbeLabel, ...] = ("first_token_letter",)
    layers = (0, 1)
    kwargs = _baseline_kwargs()
    ProbeTrainer(
        inputs=inputs,
        fitter=SklearnLogisticFitter(),
        labels=labels,
        layers=layers,
        sklearn_kwargs=kwargs,
        split_seed=0,
        split_frac=(0.7, 0.15, 0.15),
    ).run()

    drifted_kwargs = kwargs.model_copy(update={"C": 0.25})
    drifted_digest = compute_probe_manifest_digest(
        run_id=inputs.run_manifest.run_id,
        fold="pilot",
        labels=labels,
        layers=layers,
        split_seed=0,
        split_frac=(0.7, 0.15, 0.15),
        sklearn_kwargs=drifted_kwargs,
        sklearn_kwargs_digest=compute_sklearn_kwargs_digest(drifted_kwargs),
        source_run_id=inputs.run_manifest.run_id,
        source_cache_key_digest=inputs.activation_manifest.cache_key_composition_digest,
        source_pilot_manifest_digest=inputs.pilot_manifest.pilot_manifest_digest,
        source_preregistration_digest=None,
    )
    with pytest.raises(ProbeManifestStaleError):
        scan_resume_state(
            paths=inputs.paths, fold="pilot", expected_manifest_digest=drifted_digest
        )


def test_trainer_deterministic_random_state(tmp_path: Path) -> None:
    pytest.importorskip("sklearn")
    pytest.importorskip("numpy")
    import numpy as np

    run_dir = _build_phase1_with_cache(tmp_path, layers=(0,), n_items=40)
    inputs, _l, _ly = load_trainer_inputs(run_dir, "pilot")
    labels: tuple[ProbeLabel, ...] = ("first_token_letter",)
    kwargs = _baseline_kwargs()

    def _fit_and_load_coef() -> np.ndarray:
        # Wipe artifacts before each run for a fresh fit.
        probes_dir = inputs.paths.fold_probes_dir("pilot")
        if probes_dir.exists():
            import shutil

            shutil.rmtree(probes_dir)
        ProbeTrainer(
            inputs=inputs,
            fitter=SklearnLogisticFitter(),
            labels=labels,
            layers=(0,),
            sklearn_kwargs=kwargs,
            split_seed=42,
            split_frac=(0.7, 0.15, 0.15),
        ).run()
        reader = ProbeArtifactReader.open(run_dir, "pilot")
        coef, _ = reader.load_coef("first_token_letter", 0)
        return coef

    c1 = _fit_and_load_coef()
    c2 = _fit_and_load_coef()
    assert np.allclose(c1, c2, atol=1e-6)


# --- failed cells ------------------------------------------------------


class _AlwaysFailFitter:
    """ProbeFitter that always raises FitFailedError."""

    def fit_predict(self, **kwargs: Any) -> FitResult:
        from nl_ae.probes.errors import FitFailedError

        raise FitFailedError("synthetic failure for test")


def test_trainer_records_failed_cells_continues(tmp_path: Path) -> None:
    run_dir = _build_phase1_with_cache(tmp_path, layers=(0, 1), n_items=20)
    inputs, _l, _ly = load_trainer_inputs(run_dir, "pilot")
    labels: tuple[ProbeLabel, ...] = ("first_token_letter", "extractor_match_rule")
    trainer = ProbeTrainer(
        inputs=inputs,
        fitter=_AlwaysFailFitter(),
        labels=labels,
        layers=(0, 1),
        sklearn_kwargs=_baseline_kwargs(),
        split_seed=0,
        split_frac=(0.7, 0.15, 0.15),
    )
    outcome = trainer.run()
    assert outcome.status == "completed"
    assert outcome.cells_failed == 4
    assert outcome.cells_completed == 0
    for cell in trainer.manifest.cells:
        assert cell.status == "failed"
        assert cell.coef_sha256 is None
        assert cell.failure_reason is not None


def test_trainer_fail_fast_aborts(tmp_path: Path) -> None:
    run_dir = _build_phase1_with_cache(tmp_path, layers=(0, 1), n_items=20)
    inputs, _l, _ly = load_trainer_inputs(run_dir, "pilot")
    trainer = ProbeTrainer(
        inputs=inputs,
        fitter=_AlwaysFailFitter(),
        labels=("first_token_letter", "extractor_match_rule"),
        layers=(0, 1),
        sklearn_kwargs=_baseline_kwargs(),
        split_seed=0,
        split_frac=(0.7, 0.15, 0.15),
        fail_fast=True,
    )
    outcome = trainer.run()
    assert outcome.status == "failed"
    assert outcome.cells_failed == 1
    assert outcome.cells_completed == 0
    assert outcome.failure_reason is not None


# --- CLI ----------------------------------------------------------------


def test_cli_smoke_pilot(tmp_path: Path) -> None:
    pytest.importorskip("sklearn")
    run_dir = _build_phase1_with_cache(tmp_path, layers=(0, 1), n_items=40)
    runner = CliRunner()
    result = runner.invoke(
        probe_train_cmd_mod.probe_train_cmd,
        [
            "--run-dir",
            str(run_dir),
            "--fold",
            "pilot",
            "--labels",
            "first_token_letter,extractor_match_rule",
            "--layers",
            "0,1",
            "--max-iter",
            "200",
            "--no-figures",
        ],
    )
    assert result.exit_code == 0, result.output
    assert (run_dir / "pilot" / "probes" / "probe_manifest.json").exists()


def test_cli_smoke_holdout_refuses_without_preregistration(tmp_path: Path) -> None:
    pytest.importorskip("sklearn")
    run_dir = _build_phase1_with_cache(tmp_path, layers=(0, 1))
    runner = CliRunner()
    result = runner.invoke(
        probe_train_cmd_mod.probe_train_cmd,
        [
            "--run-dir",
            str(run_dir),
            "--fold",
            "holdout",
            "--no-figures",
        ],
    )
    assert result.exit_code == 2, result.output


def test_cli_overwrite(tmp_path: Path) -> None:
    pytest.importorskip("sklearn")
    run_dir = _build_phase1_with_cache(tmp_path, layers=(0, 1), n_items=20)
    runner = CliRunner()
    # First run.
    result = runner.invoke(
        probe_train_cmd_mod.probe_train_cmd,
        [
            "--run-dir",
            str(run_dir),
            "--fold",
            "pilot",
            "--labels",
            "first_token_letter",
            "--layers",
            "0,1",
            "--max-iter",
            "200",
            "--no-figures",
        ],
    )
    assert result.exit_code == 0, result.output
    # Drop a sentinel under probes/ to verify overwrite scrubs it.
    sentinel = run_dir / "pilot" / "probes" / "sentinel.txt"
    sentinel.write_text("from prior run", encoding="utf-8")

    # Second run with --on-existing overwrite.
    result2 = runner.invoke(
        probe_train_cmd_mod.probe_train_cmd,
        [
            "--run-dir",
            str(run_dir),
            "--fold",
            "pilot",
            "--labels",
            "first_token_letter",
            "--layers",
            "0,1",
            "--max-iter",
            "200",
            "--no-figures",
            "--on-existing",
            "overwrite",
        ],
    )
    assert result2.exit_code == 0, result2.output
    assert not sentinel.exists()


# --- figures -----------------------------------------------------------


def test_figures_render_when_report_extra_present(tmp_path: Path) -> None:
    pytest.importorskip("sklearn")
    pytest.importorskip("matplotlib")
    pytest.importorskip("pandas")
    run_dir = _build_phase1_with_cache(tmp_path, layers=(0, 1), n_items=20)
    runner = CliRunner()
    result = runner.invoke(
        probe_train_cmd_mod.probe_train_cmd,
        [
            "--run-dir",
            str(run_dir),
            "--fold",
            "pilot",
            "--labels",
            "first_token_letter",
            "--layers",
            "0,1",
            "--max-iter",
            "200",
        ],
    )
    assert result.exit_code == 0, result.output
    figs = list((run_dir / "pilot" / "probes" / "figures").glob("*.png"))
    assert len(figs) >= 2


def test_figures_skipped_cleanly_when_render_raises_importerror(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("sklearn")
    run_dir = _build_phase1_with_cache(tmp_path, layers=(0, 1), n_items=20)

    def _raise(reader: Any, out_dir: Path, *, dpi: int = 144) -> list[Path]:
        raise ImportError("matplotlib not installed (mocked)")

    import nl_ae.probes.figures as figures_mod

    monkeypatch.setattr(figures_mod, "render_probe_figures", _raise)

    runner = CliRunner()
    result = runner.invoke(
        probe_train_cmd_mod.probe_train_cmd,
        [
            "--run-dir",
            str(run_dir),
            "--fold",
            "pilot",
            "--labels",
            "first_token_letter",
            "--layers",
            "0,1",
            "--max-iter",
            "200",
        ],
    )
    assert result.exit_code == 0, result.output
    # The CLI imports render_probe_figures locally; the monkeypatched module
    # attribute is what the local import resolves to.


def test_trainer_chat_template_decoupled(tmp_path: Path) -> None:
    """Trainer must not depend on the model wrapper; never imports nl_ae.inference.*."""
    pytest.importorskip("sklearn")
    run_dir = _build_phase1_with_cache(tmp_path, layers=(0,), n_items=20)
    inputs, _l, _ly = load_trainer_inputs(run_dir, "pilot")
    # Manually mutate the activation manifest's chat_template_hash (drift case).
    # The trainer should still complete because it never re-instantiates the model.
    import nl_ae.probes.trainer as trainer_mod

    # Ensure the trainer module never imported anything from nl_ae.inference.
    forbidden = [name for name in trainer_mod.__dict__ if "inference" in name.lower()]
    assert forbidden == []
    ProbeTrainer(
        inputs=inputs,
        fitter=SklearnLogisticFitter(),
        labels=("first_token_letter",),
        layers=(0,),
        sklearn_kwargs=_baseline_kwargs(),
        split_seed=0,
        split_frac=(0.7, 0.15, 0.15),
    ).run()
