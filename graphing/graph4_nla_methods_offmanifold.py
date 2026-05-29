"""Graph 4 — NLA language-steering trials: off-manifold error, colored by
first-token accuracy.

Each operator is a single replacement edit (ĥ_steer replaces the L20 residual), so
unlike KAPPA there is no alpha sweep and no round-trip OME was computed for the
methods (Phase-4 OME was skipped post-NULL). The available off-manifold error is the
NLA-independent ratio = ‖ĥ_steer − h_orig‖/‖h_orig‖.

  x = method trial (E0 anchor; E1/E2/E4 surgical edits; T1/T2 templates)
  y = off-manifold error (ratio), with bootstrap CI
  color = first-token accuracy (shared scale with Graph 3)

Story: every trial lands at anchor-level accuracy (best E4 0.648); none approaches
KAPPA's peak 0.715 (near-yellow on this scale). Templates push *further* off-manifold
(ratio ~1.08) and score *worse* — they inject a constant bias (see Graph 4b y-balance).
NULL: the AR→patch channel does not transmit the target answer.
"""
from __future__ import annotations
import numpy as np
from . import data, style

GROUP = {"E0": "anchor", "E1": "surgical edits", "E2": "surgical edits",
         "E4": "surgical edits", "T1": "templates", "T2": "templates"}


def main():
    style.apply()
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    rep = data.load_lang_report()
    methods = rep["methods"]
    ops = [m["op"] for m in methods]
    ratio = np.array([m["mean_ratio"] for m in methods])
    acc = np.array([m["acc"] for m in methods])
    rci = np.array([m["ratio_ci"] for m in methods])
    rerr = np.vstack([ratio - rci[:, 0], rci[:, 1] - ratio])
    x = np.arange(len(ops))

    sm = style.acc_mappable()
    colors = sm.to_rgba(acc)

    fig, ax = plt.subplots(figsize=(9.2, 5.9))
    bars = ax.bar(x, ratio, 0.66, color=colors, edgecolor="k", linewidth=0.6, zorder=3)
    ax.errorbar(x, ratio, yerr=rerr, fmt="none", ecolor="#333", elinewidth=1.1, capsize=3, zorder=4)

    # KAPPA references in ratio units
    peak_ratio = rep["kappa_peak"]["ratio"]; peak_acc = rep["kappa_peak"]["acc"]
    ax.axhline(peak_ratio, ls="--", lw=1.4, color=style.C_KAPPA)
    ax.text(0.015, 0.92, f"– – –  KAPPA peak: ratio {peak_ratio:.2f} → ACC {peak_acc:.3f}",
            transform=ax.transAxes, ha="left", va="center", fontsize=9.5,
            color=style.C_KAPPA, fontweight="bold")

    # per-bar ACC label
    for xi, b, a_ in zip(x, bars, acc):
        ax.text(xi, b.get_height() + rerr[1][int(xi)] + 0.03, f"ACC\n{a_:.3f}",
                ha="center", va="bottom", fontsize=9, fontweight="bold", color="#222")

    # group brackets under the axis
    seen = {}
    for xi, op in zip(x, ops):
        seen.setdefault(GROUP[op], []).append(xi)
    for g, xs in seen.items():
        ax.text(np.mean(xs), -0.16, g, ha="center", va="top", fontsize=9,
                color="#666", style="italic", transform=ax.get_xaxis_transform())

    cb = fig.colorbar(sm, ax=ax, pad=0.015)
    cb.set_label("first-token accuracy")
    cb.ax.axhline(peak_acc, color=style.C_KAPPA, lw=2)
    cb.ax.text(1.6, peak_acc, "KAPPA\npeak", transform=cb.ax.get_yaxis_transform(),
               va="center", fontsize=7.5, color=style.C_KAPPA)

    ax.set_xticks(x); ax.set_xticklabels(ops)
    ax.set_ylabel("off-manifold error   ratio = ‖ĥ_steer − h‖ / ‖h‖")
    ax.set_xlabel("")
    ax.set_title("NLA steering trials: off-manifold yet no accuracy gain (NULL)")
    ax.set_ylim(0, 1.32)
    ax.margins(x=0.04)
    style.savefig(fig, "graph4_nla_methods_offmanifold",
                  caption="NLA-final · language-steering lever test, tiny256 (n=256), normmatch")


if __name__ == "__main__":
    main()
