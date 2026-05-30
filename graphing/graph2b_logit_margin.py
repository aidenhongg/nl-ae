"""Graph 2b — the PRE-SOFTMAX version of Graph 2.

Graph 2 plotted *confidence* (max softmax). The knowledge-probe axis there
saturates: 92.5% of wrong-first-token rows have know_confidence > 0.99, so the
softmax pins everything against 1.0 and hides structure. This figure plots the
**logit margin** instead (top class logit - runner-up), the pre-softmax quantity
that actually drives the softmax, on the same subset.

  subset : example_level test rows where the model's first token is WRONG (n=888)
  x = model logit margin  = top - runner-up over the 4 answer-symbol logits
                            (how strongly the model preferred its own wrong answer)
  y = knowledge-probe logit margin = top - runner-up over the 4 probe-class logits
                            (how *decisively* the activation encodes its pick)
  color = know_correct    (did the probe still pick the TRUE answer?)

What softmax hid and this shows: the probe's margin is ~2x larger when it is
RIGHT (median 35.6) than when it is wrong (16.8) -- the activation is most
decisive exactly when it is correct -- while the model still shows large margins
on its wrong answers (confidently wrong survives the pre-softmax view).

Probe logits are recomputed as W_know @ h + b_know on the RAW layer-20
activations (the validated convention: reproduces the stored know_confidence /
argmax exactly). Weights come from the documented exp04 probe mirror.
"""
from __future__ import annotations
import numpy as np
from . import data, style

# Provenance (documented project layout): the L20 knowledge probe + the raw
# activations it reads. Both are the same assets exp04/NLA-final already use.
KNOW_PROBE = data.ROOT.parent / "exp04" / "05_out_pulled" / "02_probes" / "example_level" / "know" / "layer20.npz"
H_ORIG = data.ROOT / "out" / "_ship" / "exp04" / "05_out_pulled" / "03_kappa" / "emb" / "h_layer20_orig.npy"


def _margin(logits: np.ndarray) -> np.ndarray:
    """top - runner-up along axis 1."""
    s = np.sort(logits, axis=1)
    return s[:, -1] - s[:, -2]


def compute():
    d = data.load_features("test")
    z = np.load(KNOW_PROBE); W, b = z["W"], z["b"]            # (4, d), (4,)
    h = np.load(H_ORIG)                                       # (6536, d)
    probe_logits = h[d["row_index"]] @ W.T + b               # (n_test, 4), raw-h convention
    model_logits = np.array([list(x) for x in d["logits_symbols"]], dtype=float)

    wrong = ~d["model_readout_correct"].astype(bool)
    return {
        "mm": _margin(model_logits[wrong]),
        "pm": _margin(probe_logits[wrong]),
        "kk": d["know_correct"][wrong].astype(bool),
        "n": int(wrong.sum()),
    }


def main():
    style.apply()
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    c = compute()
    mm, pm, kk, n = c["mm"], c["pm"], c["kk"], c["n"]
    frac_known = float(kk.mean())
    med_ok, med_no = float(np.median(pm[kk])), float(np.median(pm[~kk]))
    xmax = float(np.percentile(mm, 99.5)) * 1.04
    ymax = float(np.percentile(pm, 99.5)) * 1.04

    fig = plt.figure(figsize=(8.8, 7.2))
    gs = GridSpec(2, 2, width_ratios=(4.2, 1.1), height_ratios=(1.1, 4.2),
                  wspace=0.04, hspace=0.04)
    ax = fig.add_subplot(gs[1, 0])
    axt = fig.add_subplot(gs[0, 0], sharex=ax)
    axr = fig.add_subplot(gs[1, 1], sharey=ax)

    for mask, col, lab in [(kk, style.C_TRUE, f"knowledge CORRECT  (n={int(kk.sum())}, {frac_known:.0%})"),
                           (~kk, style.C_FALSE, f"knowledge wrong  (n={int((~kk).sum())}, {1 - frac_known:.0%})")]:
        ax.scatter(mm[mask], pm[mask], s=22, c=col, alpha=0.45, edgecolors="none", label=lab, zorder=3)

    # median probe-margin lines: the activation is markedly more decisive when right
    ax.axhline(med_ok, color=style.C_TRUE, lw=1.6, ls="--", zorder=2)
    ax.axhline(med_no, color=style.C_FALSE, lw=1.6, ls="--", zorder=2)
    ax.text(xmax * 0.02, med_ok, f"median when CORRECT = {med_ok:.1f}", color=style.C_TRUE,
            va="bottom", ha="left", fontsize=8.5, fontweight="bold")
    ax.text(xmax * 0.02, med_no, f"median when wrong = {med_no:.1f}", color=style.C_FALSE,
            va="top", ha="left", fontsize=8.5, fontweight="bold")

    ax.text(0.015, 0.985, "softmax pinned 92.5% of these rows at confidence > 0.99\n— the logit margin de-saturates them.",
            transform=ax.transAxes, va="top", ha="left", fontsize=8.5, color="#555")

    ax.set_xlabel("model logit margin for its (wrong) answer   (top − runner-up symbol logit)  →")
    ax.set_ylabel("knowledge-probe logit margin   (top − runner-up)  →")
    ax.set_xlim(0, xmax); ax.set_ylim(0, ymax)
    ax.legend(loc="upper right", fontsize=9)

    axt.hist([mm[kk], mm[~kk]], bins=np.linspace(0, xmax, 34), stacked=True,
             color=[style.C_TRUE, style.C_FALSE], alpha=0.75)
    axt.set_ylabel("count", fontsize=9); axt.tick_params(labelbottom=False)
    axt.set_title("Pre-softmax: the activation is most decisive exactly when it's right", pad=8)
    axr.hist([pm[kk], pm[~kk]], bins=np.linspace(0, ymax, 30), stacked=True, orientation="horizontal",
             color=[style.C_TRUE, style.C_FALSE], alpha=0.75)
    axr.set_xlabel("count", fontsize=9); axr.tick_params(labelleft=False)

    style.savefig(fig, "graph2b_logit_margin", tight=False,
                  caption=f"NLA-final · wrong first-token rows, example_level test (n={n}) · pre-softmax logit margins")


if __name__ == "__main__":
    main()
