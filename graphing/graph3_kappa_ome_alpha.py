"""Graph 3 — KAPPA recreation: off-manifold error vs. scaling factor alpha,
colored by first-token accuracy.

x = steering strength alpha (the closed-form KAPPA scaling factor)
y = off-manifold error OME = 1 - cos(h', AR(AV(h')))   [NLA round-trip; headline metric]
color = exp04 first-token accuracy at that alpha (where measured: alpha in {1,2,5,10,20,30})

Story: OME climbs monotonically with alpha (Spearman -0.99) — the edit is increasingly
off-manifold — yet first-token accuracy is an inverted-U (peak 0.715 @ alpha=10, collapse
0.600 @ alpha=30). KAPPA can only buy accuracy by paying off-manifold error.
"""
from __future__ import annotations
import numpy as np
from . import data, style


def main():
    style.apply()
    import matplotlib.pyplot as plt
    k = data.kappa_per_alpha()
    a, ome, acc = k["alpha"], k["ome"], k["acc"]
    yerr = np.vstack([k["cos_ci_hi"] - k["cos"], k["cos"] - k["cos_ci_lo"]])  # OME CI from cos CI

    fig, ax = plt.subplots(figsize=(8.8, 5.8))
    ax.plot(a, ome, "-", color="#bbb", lw=1.6, zorder=1)
    ax.errorbar(a, ome, yerr=yerr, fmt="none", ecolor="#ccc", elinewidth=1, capsize=2, zorder=2)

    have = ~np.isnan(acc)
    # alphas without an exp04 ACC: show as hollow gray markers
    ax.scatter(a[~have], ome[~have], s=85, facecolors="white", edgecolors="#aaa",
               linewidths=1.4, zorder=3, label="alpha not in exp04 ACC sweep")
    sc = ax.scatter(a[have], ome[have], c=acc[have], cmap=style.ACC_CMAP,
                    vmin=style.ACC_VMIN, vmax=style.ACC_VMAX, s=170, edgecolors="k",
                    linewidths=0.6, zorder=4)
    cb = fig.colorbar(sc, ax=ax, pad=0.015)
    cb.set_label("first-token accuracy (exp04)")

    floor = ome[a == 0][0]
    ax.axhline(floor, ls="--", lw=1, color="#2a9d8f")
    ax.text(30, floor - 0.012, f"on-manifold floor  OME={floor:.3f}  (alpha=0)",
            ha="right", va="top", fontsize=8.5, color="#2a9d8f")

    # annotate the inverted-U turning points
    def note(al, txt, dy):
        i = int(np.where(a == al)[0][0])
        ax.annotate(txt, (a[i], ome[i]), xytext=(a[i] - 3.2, ome[i] + dy),
                    fontsize=9, ha="center", color="#222",
                    arrowprops=dict(arrowstyle="->", color="#666", lw=1))
    note(10.0, "ACC peak 0.715\n(ratio 0.66)", 0.085)
    note(30.0, "ACC collapse\n0.600", 0.02)

    ax.set_xlabel("steering strength  α   (KAPPA scaling factor)")
    ax.set_ylabel("off-manifold error   OME = 1 − cos(h′, AR·AV(h′))")
    ax.set_title("KAPPA buys accuracy only by going off-manifold")
    ax.set_xlim(-1.5, 31.5); ax.set_ylim(0.24, 0.78)
    ax.legend(loc="upper left", fontsize=9)
    style.savefig(fig, "graph3_kappa_ome_alpha",
                  caption="NLA-final · KAPPA single-L20 round-trip, 1024-row subset (×11 α)")


if __name__ == "__main__":
    main()
