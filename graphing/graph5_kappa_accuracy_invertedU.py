"""Extra (E1) — KAPPA recreation: the first-token accuracy inverted-U vs alpha,
against the monotonically rising off-manifold error.

Companion to Graph 3 (which puts OME on y, ACC on color); here ACC is the star.
Left axis: first-token ACC (inverted-U, peak 0.715 @ alpha=10, collapse 0.600 @ 30).
Right axis: OME = 1-cos (monotone up). Shows the tradeoff KAPPA cannot escape.
"""
from __future__ import annotations
import numpy as np
from . import data, style

BASE_ACC = 0.6699  # exp04 alpha=0 base on its eval subset (MAIN-EXP S3 frontier table)


def main():
    style.apply()
    import matplotlib.pyplot as plt
    k = data.kappa_per_alpha()
    a, ome, acc = k["alpha"], k["ome"], k["acc"]
    have = ~np.isnan(acc)
    aa = np.concatenate([[0.0], a[have]]); ac = np.concatenate([[BASE_ACC], acc[have]])

    fig, ax = plt.subplots(figsize=(8.8, 5.8))
    ax.axhline(BASE_ACC, ls=":", lw=1.2, color="#999")
    ax.text(30, BASE_ACC - 0.004, f"base (no steering) {BASE_ACC:.3f}", ha="right", va="top",
            fontsize=8.5, color="#999")
    ax.plot(aa, ac, "-o", color=style.C_MODEL, lw=2.2, ms=9, zorder=5, label="first-token accuracy")
    # shade beneficial (ACC>base) region span
    ax.fill_between(aa, BASE_ACC, ac, where=ac >= BASE_ACC, color=style.C_KNOW, alpha=0.12)

    ipk = int(np.argmax(ac))
    ax.annotate(f"peak {ac[ipk]:.3f}\n@ α={aa[ipk]:.0f}", (aa[ipk], ac[ipk]),
                xytext=(aa[ipk] + 2.5, ac[ipk] + 0.012), fontsize=9.5, fontweight="bold",
                color=style.C_MODEL, arrowprops=dict(arrowstyle="->", color=style.C_MODEL))
    ax.annotate("collapse 0.600", (30, ac[-1]), xytext=(24, 0.625), fontsize=9,
                color="#c1121f", arrowprops=dict(arrowstyle="->", color="#c1121f"))

    ax.set_xlabel("steering strength  α   (KAPPA scaling factor)")
    ax.set_ylabel("first-token accuracy", color=style.C_MODEL)
    ax.set_ylim(0.58, 0.74); ax.set_xlim(-1.5, 31.5)
    ax.set_title("KAPPA accuracy is an inverted-U; off-manifold error only rises")

    ax2 = ax.twinx()
    ax2.plot(a, ome, "--s", color="#bb9", lw=1.6, ms=5, alpha=0.9, label="off-manifold error (OME)")
    ax2.set_ylabel("off-manifold error  OME = 1 − cos", color="#9a8")
    ax2.set_ylim(0.24, 0.78); ax2.grid(False)
    ax2.spines["top"].set_visible(False)

    l1, lb1 = ax.get_legend_handles_labels()
    l2, lb2 = ax2.get_legend_handles_labels()
    ax.legend(l1 + l2, lb1 + lb2, loc="lower center", fontsize=9)
    style.savefig(fig, "graph5_kappa_accuracy_invertedU",
                  caption="NLA-final · exp04 single-L20 ACC(α) + round-trip OME")


if __name__ == "__main__":
    main()
