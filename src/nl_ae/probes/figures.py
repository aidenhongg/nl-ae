"""Exploratory probe figures (C07).

Three figure types per probe run:

* ``<label>_accuracy_by_layer.png`` — train/val/test accuracy vs layer.
* ``<label>_train_test_gap.png`` — ``(train - test)`` accuracy vs layer; the
  overfit smell pilot review is meant to surface.
* ``probe_accuracy_heatmap.png`` — one heatmap over ``(label, layer)`` test
  accuracy.

Matplotlib + pandas are lazy-imported; missing the ``[report]`` extra raises a
clean ``ImportError`` rather than dying at module load.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from .reader import ProbeArtifactReader


def _matplotlib() -> tuple:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "matplotlib is required for probe figures; install nl-ae[report]"
        ) from exc
    return matplotlib, plt


def render_probe_figures(
    reader: ProbeArtifactReader,
    out_dir: Path,
    *,
    dpi: int = 144,
) -> list[Path]:
    """Render every figure type into ``out_dir``. Returns the list of paths."""
    _, plt = _matplotlib()
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    heatmap_rows: list[tuple[str, int, float | None]] = []
    for label in reader.manifest.labels:
        try:
            summary = reader.load_summary(label)
        except FileNotFoundError:
            continue
        if len(summary) == 0:
            continue
        summary = summary.sort_values("layer")

        # Figure 1 — train/val/test accuracy by layer.
        fig, ax = plt.subplots(figsize=(7, 4))
        for col, marker, name in (
            ("train_accuracy", "o", "train"),
            ("val_accuracy", "s", "val"),
            ("test_accuracy", "^", "test"),
        ):
            if col in summary.columns:
                ax.plot(
                    summary["layer"],
                    summary[col],
                    marker=marker,
                    label=name,
                )
        ax.set_xlabel("layer")
        ax.set_ylabel("accuracy")
        ax.set_title(f"{label} — probe accuracy by layer")
        ax.set_ylim(0.0, 1.05)
        ax.legend()
        fig.tight_layout()
        path = out_dir / f"{label}_accuracy_by_layer.png"
        fig.savefig(path, dpi=dpi)
        plt.close(fig)
        paths.append(path)

        # Figure 2 — train minus test (overfit gap) by layer.
        fig, ax = plt.subplots(figsize=(7, 3))
        if "train_accuracy" in summary.columns and "test_accuracy" in summary.columns:
            gap = summary["train_accuracy"] - summary["test_accuracy"]
            ax.bar(summary["layer"], gap)
            ax.axhline(0.0, color="black", linewidth=0.5)
        ax.set_xlabel("layer")
        ax.set_ylabel("train - test accuracy")
        ax.set_title(f"{label} — overfit gap by layer")
        fig.tight_layout()
        path = out_dir / f"{label}_train_test_gap.png"
        fig.savefig(path, dpi=dpi)
        plt.close(fig)
        paths.append(path)

        for _, row in summary.iterrows():
            heatmap_rows.append((label, int(row["layer"]), row.get("test_accuracy")))

    # Figure 3 — cross-label heatmap of test accuracy.
    if heatmap_rows:
        import numpy as np

        labels_sorted = sorted({r[0] for r in heatmap_rows})
        layers_sorted = sorted({r[1] for r in heatmap_rows})
        grid = np.full((len(labels_sorted), len(layers_sorted)), np.nan, dtype=float)
        label_idx = {label: i for i, label in enumerate(labels_sorted)}
        layer_idx = {layer: i for i, layer in enumerate(layers_sorted)}
        for hl, lyr, acc in heatmap_rows:
            if acc is None:
                continue
            grid[label_idx[hl], layer_idx[lyr]] = float(acc)

        fig, ax = plt.subplots(figsize=(max(6.0, 0.4 * len(layers_sorted)), 1.0 + 0.5 * len(labels_sorted)))
        im = ax.imshow(grid, aspect="auto", vmin=0.0, vmax=1.0, cmap="viridis")
        ax.set_xticks(range(len(layers_sorted)))
        ax.set_xticklabels([f"L{layer:02d}" for layer in layers_sorted], rotation=45, ha="right")
        ax.set_yticks(range(len(labels_sorted)))
        ax.set_yticklabels(labels_sorted)
        ax.set_xlabel("layer")
        ax.set_title("Probe test accuracy by (label, layer)")
        fig.colorbar(im, ax=ax, fraction=0.025)
        fig.tight_layout()
        path = out_dir / "probe_accuracy_heatmap.png"
        fig.savefig(path, dpi=dpi)
        plt.close(fig)
        paths.append(path)

    return paths


__all__ = ["render_probe_figures"]
