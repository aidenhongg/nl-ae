"""Five MVP figures (C04 D4.7).

Matplotlib is optional. Each renderer imports ``matplotlib`` lazily so the
schema/eval modules stay light.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from .aggregate import AggregateBundle

LOG = logging.getLogger(__name__)


def _matplotlib() -> tuple:
    try:
        import matplotlib  # noqa: PLC0415

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # noqa: PLC0415

        return matplotlib, plt
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("matplotlib required for figures; install nl-ae[report]") from exc


def render_top1_disagreement_bar(bundle: "AggregateBundle", out_dir: Path) -> Path:
    _, plt = _matplotlib()
    df = bundle.top1_disagreement
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "top1_disagreement.png"
    if len(df) == 0:
        return out
    fig, ax = plt.subplots(figsize=(7, 4))
    labels = [f"{r.dataset_name} / {r.template_id}" for r in df.itertuples()]
    ax.bar(labels, df["disagreement"], yerr=[df["disagreement"] - df["ci_lo"], df["ci_hi"] - df["disagreement"]])
    ax.set_ylabel("Top-1 disagreement rate")
    ax.set_title("First-token vs free-gen disagreement")
    ax.set_ylim(0.0, 1.0)
    fig.autofmt_xdate(rotation=30)
    fig.tight_layout()
    fig.savefig(out, dpi=144)
    plt.close(fig)
    return out


def render_per_letter_confusion_heatmap(
    bundle: "AggregateBundle", out_dir: Path
) -> list[Path]:
    _, plt = _matplotlib()
    df = bundle.per_letter_confusion
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    if len(df) == 0:
        return paths
    for (ds, tid), group in df.groupby(["dataset_name", "template_id"]):
        pivot = (
            group.pivot_table(
                index="first_token_letter",
                columns="free_text_letter",
                values="count",
                fill_value=0,
            )
            .sort_index()
            .sort_index(axis=1)
        )
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(pivot.values, aspect="auto")
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns)
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index)
        ax.set_xlabel("free-gen letter")
        ax.set_ylabel("first-token letter")
        ax.set_title(f"Confusion — {ds} / {tid}")
        fig.colorbar(im, ax=ax)
        fig.tight_layout()
        out = out_dir / f"confusion_{ds}_{tid}.png"
        fig.savefig(out, dpi=144)
        plt.close(fig)
        paths.append(out)
    return paths


def render_per_position_bias_bar(bundle: "AggregateBundle", out_dir: Path) -> Path:
    _, plt = _matplotlib()
    df = bundle.per_position_bias
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "per_position_bias.png"
    if len(df) == 0:
        return out
    fig, ax = plt.subplots(figsize=(7, 4))
    for tid, group in df.groupby("template_id"):
        means = group.groupby("first_token_letter")["share"].mean().sort_index()
        ax.plot(means.index, means.values, marker="o", label=tid)
    ax.set_xlabel("argmax letter")
    ax.set_ylabel("mean share across permutations")
    ax.set_title("Per-position letter bias")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=144)
    plt.close(fig)
    return out


def render_per_subject_mmlu_bar(bundle: "AggregateBundle", out_dir: Path) -> Path:
    _, plt = _matplotlib()
    df = bundle.per_subject_mmlu
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "per_subject_mmlu.png"
    if len(df) == 0:
        return out
    df = df.sort_values("accuracy_first_token", ascending=False).head(40)
    fig, ax = plt.subplots(figsize=(10, max(4, 0.25 * len(df))))
    ax.barh(df["subject"], df["accuracy_first_token"], alpha=0.7, label="first-token")
    ax.barh(df["subject"], df["accuracy_free_text"], alpha=0.5, label="free-gen")
    ax.set_xlabel("accuracy")
    ax.set_title("Per-subject MMLU accuracy")
    ax.invert_yaxis()
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=144)
    plt.close(fig)
    return out


def render_calibration_curve(bundle: "AggregateBundle", out_dir: Path) -> Path | None:
    if bundle.calibration is None or len(bundle.calibration) == 0:
        return None
    _, plt = _matplotlib()
    df = bundle.calibration
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "calibration.png"
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "--", color="gray", label="perfect")
    ax.scatter(df["mean_prob"], df["empirical_accuracy"], s=df["n"].clip(upper=200))
    ax.set_xlabel("mean predicted prob")
    ax.set_ylabel("empirical accuracy")
    ax.set_title("Calibration — MMLU first-token")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=144)
    plt.close(fig)
    return out


def render_all_figures(bundle: "AggregateBundle", out_dir: Path) -> list[Path]:
    paths: list[Path] = []
    paths.append(render_top1_disagreement_bar(bundle, out_dir))
    paths.extend(render_per_letter_confusion_heatmap(bundle, out_dir))
    paths.append(render_per_position_bias_bar(bundle, out_dir))
    paths.append(render_per_subject_mmlu_bar(bundle, out_dir))
    cal_path = render_calibration_curve(bundle, out_dir)
    if cal_path is not None:
        paths.append(cal_path)
    return [p for p in paths if p is not None]


__all__ = [
    "render_all_figures",
    "render_calibration_curve",
    "render_per_letter_confusion_heatmap",
    "render_per_position_bias_bar",
    "render_per_subject_mmlu_bar",
    "render_top1_disagreement_bar",
]
