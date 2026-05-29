"""Extra (E2) — THE money graph: first-token accuracy vs. off-manifold error.

Plots the KAPPA single-L20 sweep as a frontier curve (alpha annotated) and overlays
every NLA language-steering trial as a point with bootstrap CIs, in the common,
NLA-independent off-manifold metric ratio = ‖Δh‖/‖h‖.

WIN region (the experiment's goal) = top-left: KAPPA's peak accuracy (0.715) at lower
off-manifold error than KAPPA's 0.66. It is empty. Every NLA edit lands at anchor-level
accuracy and >= KAPPA-peak off-manifold error -> NULL.
"""
from __future__ import annotations
import numpy as np
from . import data, style

BASE_ACC = 0.6699


def main():
    style.apply()
    import matplotlib.pyplot as plt
    rep = data.load_lang_report()

    # KAPPA frontier (ratio, acc) including the alpha=0 no-edit anchor at ratio 0.
    kf = rep["kappa_frontier"]
    al = sorted(float(k) for k in kf)
    kr = [0.0] + [kf[str(a)]["ratio"] for a in al]
    ka = [BASE_ACC] + [kf[str(a)]["acc"] for a in al]
    klab = [0.0] + al

    fig, ax = plt.subplots(figsize=(9.2, 6.2))

    # WIN region: ACC >= kappa peak, ratio < kappa peak ratio
    peak = rep["kappa_peak"]
    ax.axhspan(peak["acc"], 0.78, xmin=0, xmax=1, color="#2a9d8f", alpha=0.0)  # placeholder for layering
    ax.add_patch(plt.Rectangle((0, peak["acc"]), peak["ratio"], 0.78 - peak["acc"],
                               color="#2a9d8f", alpha=0.10, zorder=0))
    ax.text(peak["ratio"] / 2, (peak["acc"] + 0.75) / 2, "WIN region\n(empty)", ha="center",
            va="center", fontsize=10, color="#2a9d8f", fontweight="bold", alpha=0.8)

    ax.plot(kr, ka, "-", color=style.C_KAPPA, lw=2, zorder=3)
    ax.scatter(kr, ka, s=70, color=style.C_KAPPA, zorder=4, label="KAPPA single-L20 sweep")
    for r, a_, lb in zip(kr, ka, klab):
        if lb == 1.0:                      # overlaps α=0/α=2 in the low cluster
            continue
        off = (8, 7) if lb == 10.0 else (5, -13)
        ax.annotate(f"α={lb:.0f}", (r, a_), xytext=off, textcoords="offset points",
                    fontsize=8, color=style.C_KAPPA)
    ax.scatter([peak["ratio"]], [peak["acc"]], s=320, marker="*", color=style.C_KAPPA,
               edgecolor="k", linewidth=0.6, zorder=6)

    # The 4 surgical edits are nearly coincident at ratio~0.74 -> fan labels out to
    # distinct right-side slots with thin leader lines; T1/T2 are isolated.
    FAN = {"E0": (0.95, 0.602), "E1": (0.95, 0.624), "E2": (0.95, 0.647), "E4": (0.95, 0.670)}
    for m in rep["methods"]:
        op, c = m["op"], style.METHOD_COLORS[m["op"]]
        xerr = np.array([[m["mean_ratio"] - m["ratio_ci"][0]], [m["ratio_ci"][1] - m["mean_ratio"]]])
        yerr = np.array([[m["acc"] - m["acc_ci"][0]], [m["acc_ci"][1] - m["acc"]]])
        ax.errorbar(m["mean_ratio"], m["acc"], xerr=xerr, yerr=yerr, fmt="o", ms=10,
                    color=c, ecolor=c, elinewidth=1.1, alpha=0.95, capsize=3, zorder=5,
                    markeredgecolor="k", markeredgewidth=0.5)
        if op in FAN:
            ax.annotate(op, (m["mean_ratio"], m["acc"]), xytext=FAN[op],
                        fontsize=10, fontweight="bold", color=c, va="center",
                        arrowprops=dict(arrowstyle="-", color=c, lw=0.8, alpha=0.7))
        else:
            ax.annotate(op, (m["mean_ratio"], m["acc"]), xytext=(9, 7),
                        textcoords="offset points", fontsize=10, fontweight="bold", color=c)

    ax.axhline(peak["acc"], ls=":", lw=1, color="#bbb")
    ax.axvline(peak["ratio"], ls=":", lw=1, color="#bbb")
    ax.set_xlabel("off-manifold error   ratio = ‖Δh‖ / ‖h‖   (lower is better →)")
    ax.set_ylabel("first-token accuracy   (higher is better ↑)")
    ax.set_title("No NLA edit reaches KAPPA's accuracy at any off-manifold cost (NULL)")
    ax.set_xlim(-0.05, 2.15); ax.set_ylim(0.52, 0.75)
    ax.legend(loc="lower right", fontsize=9)
    style.savefig(fig, "graph6_pareto_frontier",
                  caption="NLA-final · KAPPA (1024 subset) vs. NLA methods (tiny256), ratio metric")


if __name__ == "__main__":
    main()
