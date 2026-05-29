"""Shared plotting style + helpers so every figure looks consistent.
Headless (Agg). One savefig path (figures/), one palette, one annotation idiom."""
from __future__ import annotations
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from . import data

DPI = 150

# Semantic colors (kept consistent across figures).
C_KNOW = "#2a9d8f"     # knowledge probe (teal)
C_PRED = "#e76f51"     # prediction probe (warm)
C_MODEL = "#264653"    # model first-token (dark slate)
C_KAPPA = "#6a4c93"    # KAPPA recreation (purple)
C_TRUE = "#2a9d8f"     # know-correct (teal)
C_FALSE = "#e63946"    # know-wrong (red)
ACC_CMAP = "viridis"   # first-token accuracy colormap (shared by graphs 3 & 4)
ACC_VMIN, ACC_VMAX = 0.54, 0.72   # shared accuracy color range (covers KAPPA + NLA methods)


def acc_mappable():
    """ScalarMappable for the shared first-token-accuracy colorbar."""
    import matplotlib as mpl
    return mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(ACC_VMIN, ACC_VMAX), cmap=ACC_CMAP)

# Distinct, legible colors for the NLA steering operators.
METHOD_COLORS = {
    "E0": "#8d99ae", "E1": "#457b9d", "E2": "#1d3557",
    "E4": "#2a9d8f", "T1": "#e76f51", "T2": "#9d0208",
}


def apply():
    plt.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": DPI,
        "font.size": 11, "axes.titlesize": 13, "axes.titleweight": "bold",
        "axes.labelsize": 11.5, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.6,
        "legend.frameon": False, "legend.fontsize": 9.5,
        "figure.facecolor": "white", "axes.facecolor": "white",
    })


def savefig(fig, name: str, caption: str | None = None, tight: bool = True) -> str:
    """Write figures/<name>.png; return the path. Adds a small source footer.
    tight=False for figures with manual gridspec marginals (tight_layout warns)."""
    data.FIGURES.mkdir(parents=True, exist_ok=True)
    foot = "NLA-final · example_level test (n=2615)" if caption is None else caption
    fig.text(0.995, 0.004, foot, ha="right", va="bottom", fontsize=7,
             color="#888", style="italic")
    if tight:
        fig.tight_layout(rect=(0, 0.015, 1, 1))
    path = data.FIGURES / f"{name}.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {path}")
    return str(path)


def bar_labels(ax, bars, fmt="{:.3f}", dy=0.0, fontsize=10, color="#222"):
    for b in bars:
        h = b.get_height()
        ax.text(b.get_x() + b.get_width() / 2, h + dy, fmt.format(h),
                ha="center", va="bottom", fontsize=fontsize, color=color, fontweight="bold")
