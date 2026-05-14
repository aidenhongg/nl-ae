"""``nlae probe-train`` — fit per-``(label, layer)`` linear probes (C07).

Refuses ``--fold holdout`` without a locked preregistration. Pilot mode trains
the full candidate label pool over every cached layer by default; holdout mode
takes its labels/layers from ``preregistration.md`` verbatim.
"""

from __future__ import annotations

import logging
import shutil
import sys
from pathlib import Path

import click

from ._common import parse_layers_arg

LOG = logging.getLogger("nl_ae.cli.probe_train")


def _parse_labels_csv(csv_arg: str | None) -> tuple[str, ...] | None:
    if csv_arg is None or not csv_arg.strip():
        return None
    parts = [p.strip() for p in csv_arg.split(",") if p.strip()]
    if not parts:
        raise click.UsageError("--labels parsed to an empty set")
    return tuple(parts)


def _parse_split_frac(csv_arg: str) -> tuple[float, float, float]:
    parts = [p.strip() for p in csv_arg.split(",") if p.strip()]
    if len(parts) != 3:
        raise click.UsageError(
            f"--split-frac must be three comma-separated floats; got {csv_arg!r}"
        )
    try:
        values = tuple(float(p) for p in parts)
    except ValueError as exc:
        raise click.UsageError(f"--split-frac contains a non-float: {exc}") from exc
    if abs(sum(values) - 1.0) > 1e-6:
        raise click.UsageError(f"--split-frac must sum to 1.0; got {sum(values)}")
    return values  # type: ignore[return-value]


@click.command(
    "probe-train",
    help=(
        "Fit per-(label, layer) linear probes over the activation cache for one "
        "fold. Refuses --fold holdout without a locked preregistration."
    ),
)
@click.option(
    "--run-dir",
    "run_dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    required=True,
    help="Path to a Phase 1 run dir with rows.jsonl + pilot_manifest.json + activation cache.",
)
@click.option(
    "--fold",
    type=click.Choice(["pilot", "holdout"], case_sensitive=False),
    required=True,
    help="Pilot trains the candidate label pool; holdout obeys preregistration.",
)
@click.option(
    "--labels",
    "labels_arg",
    type=str,
    default=None,
    help=(
        "Comma-separated label subset. Pilot default: all six. Holdout: omit "
        "(the preregistration locks the label set)."
    ),
)
@click.option(
    "--layers",
    "layers_arg",
    type=str,
    default=None,
    help='Comma-list or range over [0, 28). Examples: "0,5,20", "0-27". Default: all cached.',
)
@click.option(
    "--split-seed",
    type=click.IntRange(min=0, max=(1 << 32) - 1),
    default=0,
    show_default=True,
    help="Per-item sub-fold seed (derive_child_seed). 0 by default.",
)
@click.option(
    "--split-frac",
    "split_frac_csv",
    type=str,
    default="0.70,0.15,0.15",
    show_default=True,
    help="train,val,test fractions; must sum to 1.0.",
)
@click.option("--penalty", type=click.Choice(["l2"]), default="l2", show_default=True)
@click.option(
    "--C",
    "C_val",
    type=click.FloatRange(min=0.0, min_open=True),
    default=1.0,
    show_default=True,
    help="Inverse regularization strength.",
)
@click.option(
    "--max-iter",
    type=click.IntRange(min=1),
    default=1000,
    show_default=True,
    help="lbfgs max iterations.",
)
@click.option(
    "--class-weight",
    type=click.Choice(["none", "balanced"], case_sensitive=False),
    default="none",
    show_default=True,
)
@click.option(
    "--standardize/--no-standardize",
    default=False,
    show_default=True,
    help="Fit-transform StandardScaler on train sub-fold; record (mean, std).",
)
@click.option(
    "--fit-intercept/--no-fit-intercept",
    default=True,
    show_default=True,
)
@click.option(
    "--figures/--no-figures",
    default=True,
    show_default=True,
    help="Render exploratory figures under <fold>/probes/figures/ (needs nl-ae[report]).",
)
@click.option(
    "--on-existing",
    type=click.Choice(["resume", "overwrite", "error"], case_sensitive=False),
    default="resume",
    show_default=True,
)
@click.option(
    "--fail-fast/--no-fail-fast",
    default=False,
    show_default=True,
    help="Abort on the first cell failure (default: log + continue).",
)
def probe_train_cmd(
    run_dir: Path,
    fold: str,
    labels_arg: str | None,
    layers_arg: str | None,
    split_seed: int,
    split_frac_csv: str,
    penalty: str,
    C_val: float,
    max_iter: int,
    class_weight: str,
    standardize: bool,
    fit_intercept: bool,
    figures: bool,
    on_existing: str,
    fail_fast: bool,
) -> None:
    # Heavy imports inside the handler so `nlae --help` stays light.
    from nl_ae.cache.errors import (
        ActivationManifestMissingError,
        CacheKeyMismatchError,
    )
    from nl_ae.pilot.errors import (
        PilotDigestMismatchError,
        PilotManifestMissingError,
        PreregistrationInvalidError,
        PreregistrationMissingError,
        PreregistrationParseError,
        PreregistrationUnlockedError,
    )
    from nl_ae.probes.errors import (
        ActivationManifestNotCompletedError,
        LabelOutOfScopeError,
        LayerOutOfScopeError,
        ProbeManifestStaleError,
    )
    from nl_ae.probes.fitter import SklearnLogisticFitter
    from nl_ae.probes.models import (
        SklearnKwargs,
        compute_probe_manifest_digest,
        compute_sklearn_kwargs_digest,
    )
    from nl_ae.probes.reader import ProbeArtifactReader
    from nl_ae.probes.trainer import (
        ProbeTrainer,
        hash_preregistration,
        load_trainer_inputs,
        scan_resume_state,
    )

    fold_lc = fold.lower()

    # 1. Preflight.
    try:
        labels_override = _parse_labels_csv(labels_arg)
        layers_override = (
            parse_layers_arg(layers_arg, n_layers=28) if layers_arg else None
        )
        # type-cast labels_override to ProbeLabel tuple at call site;
        # load_trainer_inputs validates against CANDIDATE_LABELS.
        inputs, labels, layers = load_trainer_inputs(
            run_dir,
            fold_lc,  # type: ignore[arg-type]
            labels_override=labels_override,  # type: ignore[arg-type]
            layers_override=layers_override,
        )
    except PilotManifestMissingError as exc:
        click.echo(f"pilot manifest missing: {exc}", err=True)
        sys.exit(2)
    except (PreregistrationMissingError, PreregistrationParseError) as exc:
        click.echo(f"preregistration missing/unparseable: {exc}", err=True)
        sys.exit(2)
    except (PreregistrationInvalidError, PreregistrationUnlockedError) as exc:
        click.echo(f"preregistration not locked / invalid: {exc}", err=True)
        sys.exit(3)
    except PilotDigestMismatchError as exc:
        click.echo(f"pilot digest mismatch: {exc}", err=True)
        sys.exit(3)
    except (LabelOutOfScopeError, LayerOutOfScopeError) as exc:
        click.echo(f"scope error: {exc}", err=True)
        sys.exit(3)
    except ActivationManifestMissingError as exc:
        click.echo(f"activation cache missing: {exc}", err=True)
        sys.exit(2)
    except (CacheKeyMismatchError, ActivationManifestNotCompletedError) as exc:
        click.echo(f"cache state error: {exc}", err=True)
        sys.exit(4)
    except (FileNotFoundError, RuntimeError) as exc:
        click.echo(f"preflight failed: {exc}", err=True)
        sys.exit(2)

    # 2. Existing-dir policy.
    probes_dir = inputs.paths.fold_probes_dir(inputs.fold)
    manifest_path = probes_dir / "probe_manifest.json"
    on_existing_lc = on_existing.lower()
    existing_manifest_on_disk = manifest_path.exists()
    if existing_manifest_on_disk and on_existing_lc == "error":
        click.echo(
            f"refusing to train: {manifest_path} exists. "
            "Use --on-existing resume|overwrite.",
            err=True,
        )
        sys.exit(3)
    if existing_manifest_on_disk and on_existing_lc == "overwrite":
        click.echo(f"--on-existing overwrite: deleting {probes_dir}")
        shutil.rmtree(probes_dir)
        existing_manifest_on_disk = False

    # 3. Compose sklearn kwargs + split frac.
    sklearn_kwargs = SklearnKwargs(
        penalty=penalty,  # type: ignore[arg-type]
        C=C_val,
        solver="lbfgs",
        max_iter=max_iter,
        fit_intercept=fit_intercept,
        class_weight=class_weight.lower(),  # type: ignore[arg-type]
        standardize=standardize,
    )
    split_frac = _parse_split_frac(split_frac_csv)

    # 4. Resume scan if a manifest is on disk.
    from nl_ae.probes.models import ProbeCellKey

    existing_manifest = None
    completed: frozenset[ProbeCellKey] = frozenset()
    if existing_manifest_on_disk:
        expected_digest = compute_probe_manifest_digest(
            run_id=inputs.run_manifest.run_id,
            fold=inputs.fold,
            labels=labels,
            layers=layers,
            split_seed=split_seed,
            split_frac=split_frac,
            sklearn_kwargs=sklearn_kwargs,
            sklearn_kwargs_digest=compute_sklearn_kwargs_digest(sklearn_kwargs),
            source_run_id=inputs.run_manifest.run_id,
            source_cache_key_digest=inputs.activation_manifest.cache_key_composition_digest,
            source_pilot_manifest_digest=inputs.pilot_manifest.pilot_manifest_digest,
            source_preregistration_digest=(
                hash_preregistration(inputs.preregistration)
                if inputs.preregistration is not None
                else None
            ),
        )
        try:
            existing_manifest, completed = scan_resume_state(
                paths=inputs.paths,
                fold=inputs.fold,
                expected_manifest_digest=expected_digest,
            )
        except ProbeManifestStaleError as exc:
            click.echo(f"resume refused (digest drift): {exc}", err=True)
            sys.exit(3)
        if completed:
            click.echo(f"resume: {len(completed)} cell(s) already completed; will be skipped")

    # 5. Run.
    trainer = ProbeTrainer(
        inputs=inputs,
        fitter=SklearnLogisticFitter(),
        labels=labels,
        layers=layers,
        sklearn_kwargs=sklearn_kwargs,
        split_seed=split_seed,
        split_frac=split_frac,
        completed=completed,
        existing_manifest=existing_manifest,
        fail_fast=fail_fast,
    )
    outcome = trainer.run()

    # 6. Figures (best-effort).
    if figures:
        try:
            reader = ProbeArtifactReader.open(run_dir, inputs.fold)
            from nl_ae.probes.figures import render_probe_figures

            figure_dir = probes_dir / "figures"
            rendered = render_probe_figures(reader, figure_dir)
            click.echo(f"figures: wrote {len(rendered)} file(s) under {figure_dir}")
        except ImportError as exc:
            click.echo(f"figures: skipped (matplotlib not installed): {exc}", err=True)

    # 7. Summary.
    click.echo(
        f"fold={inputs.fold} status={outcome.status} "
        f"cells={outcome.cells_completed}/{outcome.cells_expected} "
        f"resumed={outcome.cells_skipped_resume} failed={outcome.cells_failed}"
    )
    if outcome.status != "completed":
        click.echo(f"  failure_reason: {outcome.failure_reason}", err=True)
        sys.exit(5)


__all__ = ["probe_train_cmd"]
