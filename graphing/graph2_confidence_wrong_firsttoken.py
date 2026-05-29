"""Graph 2 — Model confidence vs. knowledge-probe confidence on WRONG first-tokens.

Subset: example_level test rows where the model's first-token (readout) is wrong
(model_readout_correct == False; n=888). For each, plot:
  x = model_confidence   (max softmax over the 4 answer symbols — confidence in its
                          own, wrong, answer; wide spread 0.33..1.0)
  y = know_confidence    (knowledge-probe max softmax; saturated near 1.0)
  color = know_correct   (did the knowledge probe still pick the TRUE answer?)

Story: the knowledge probe is near-certain on almost every wrong-output row, and
*correct* on 74% of them — the model is often confidently wrong while the activation
confidently encodes the right answer. Marginals expose the asymmetry (model spread
vs. knowledge ceiling).
"""
from __future__ import annotations
import numpy as np
from . import data, style


def main():
    style.apply()
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    d = data.load_features("test")
    wrong = ~d["model_readout_correct"]
    mc = d["model_confidence"][wrong].astype(float)
    kc = d["know_confidence"][wrong].astype(float)
    kk = d["know_correct"][wrong].astype(bool)
    n = int(wrong.sum())
    frac_known = float(np.mean(kk))

    fig = plt.figure(figsize=(8.8, 7.2))
    gs = GridSpec(2, 2, width_ratios=(4.2, 1.1), height_ratios=(1.1, 4.2),
                  wspace=0.04, hspace=0.04)
    ax = fig.add_subplot(gs[1, 0])
    axt = fig.add_subplot(gs[0, 0], sharex=ax)
    axr = fig.add_subplot(gs[1, 1], sharey=ax)

    for m, col, lab in [(kk, style.C_TRUE, f"knowledge CORRECT  (n={int(kk.sum())}, {frac_known:.0%})"),
                        (~kk, style.C_FALSE, f"knowledge wrong  (n={int((~kk).sum())}, {1-frac_known:.0%})")]:
        ax.scatter(mc[m], kc[m], s=22, c=col, alpha=0.45, edgecolors="none", label=lab, zorder=3)

    # quadrant guide: model confidently wrong (>0.9) while knowledge confidently high (>0.9)
    ax.axvline(0.9, ls=":", lw=1, color="#ccc"); ax.axhline(0.9, ls=":", lw=1, color="#ccc")
    n_conf_wrong = int(np.sum((mc > 0.9) & (kc > 0.9) & kk))
    ax.annotate(f"confidently wrong, yet the\nknowledge probe is confident\n& CORRECT  (n={n_conf_wrong})",
                xy=(0.99, 0.992), xytext=(0.66, 0.62), fontsize=9, color="#1d3557",
                ha="center", arrowprops=dict(arrowstyle="->", color="#1d3557", lw=1.2))
    ax.text(0.235, 0.965, "← knowledge probe near-certain on nearly every wrong-output row",
            fontsize=8.5, color="#2a9d8f", va="center")

    ax.set_xlabel("model confidence in its (wrong) first-token  =  max softmax over {A,B,C,D}")
    ax.set_ylabel("knowledge-probe confidence")
    ax.set_xlim(0.2, 1.02); ax.set_ylim(0.45, 1.02)
    ax.legend(loc="lower left", fontsize=9)

    bins = np.linspace(0.2, 1.0, 33)
    axt.hist([mc[kk], mc[~kk]], bins=bins, stacked=True, color=[style.C_TRUE, style.C_FALSE], alpha=0.75)
    axt.set_ylabel("count", fontsize=9); axt.tick_params(labelbottom=False)
    axt.set_title("Model is confidently wrong while the activation still knows", pad=8)
    binsy = np.linspace(0.45, 1.0, 28)
    axr.hist([kc[kk], kc[~kk]], bins=binsy, stacked=True, orientation="horizontal",
             color=[style.C_TRUE, style.C_FALSE], alpha=0.75)
    axr.set_xlabel("count", fontsize=9); axr.tick_params(labelleft=False)

    style.savefig(fig, "graph2_confidence_wrong_firsttoken", tight=False,
                  caption=f"NLA-final · wrong first-token rows, example_level test (n={n})")


if __name__ == "__main__":
    main()
