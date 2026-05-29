"""Graph 1 — Knowledge probe vs. prediction probe vs. first-token accuracy.

Each is a "reader" of the same L20 activation, measured against two references:
  - ground truth (gt_symbol)         : who decodes the *correct* answer
  - the model's own readout (model_symbol): who decodes what the model *says*

Story: the knowledge probe reads the truth 85% of the time, but the model only
*says* it 66% of the time (a +19pt latent-knowledge gap); the prediction probe
mirrors the model (91% agreement), confirming the readout itself is decodable.
"""
from __future__ import annotations
import numpy as np
from . import data, style


def compute(d):
    gt, know, pred, model = d["gt_symbol"], d["know_argmax_symbol"], d["pred_argmax_symbol"], d["model_symbol"]
    return {
        "know": {"gt": float(np.mean(know == gt)), "model": float(np.mean(d["agree_know_model"]))},
        "pred": {"gt": float(np.mean(pred == gt)), "model": float(np.mean(d["pred_matches_model"]))},
        "ft":   {"gt": float(np.mean(d["model_readout_correct"])), "model": 1.0},
    }


def main():
    style.apply()
    import matplotlib.pyplot as plt
    d = data.load_features("test")
    n = len(d["gt_symbol"])
    s = compute(d)

    fig, ax = plt.subplots(figsize=(8.6, 5.6))
    chance = 0.25
    ax.axhline(chance, ls=":", lw=1.2, color="#bbb", zorder=0)
    ax.text(2.62, chance + 0.006, "chance (4-way)", fontsize=8, color="#999", ha="right")

    w = 0.34
    xpos = {"know": 0, "pred": 1, "ft": 2}
    colmap = {"know": style.C_KNOW, "pred": style.C_PRED, "ft": style.C_MODEL}
    solid, hatched = [], []
    for key, x in xpos.items():
        c = colmap[key]
        if key == "ft":
            b = ax.bar(x, s[key]["gt"], w * 1.15, color=c, zorder=3)
            solid += list(b)
            ax.annotate("first-token =\nthe model's own answer", (x, s[key]["gt"] + 0.075),
                        ha="center", va="bottom", fontsize=8.5, color="#555", style="italic")
        else:
            b1 = ax.bar(x - w / 2, s[key]["gt"], w, color=c, zorder=3)
            b2 = ax.bar(x + w / 2, s[key]["model"], w, color=c, alpha=0.45,
                        hatch="////", edgecolor="white", zorder=3)
            solid += list(b1); hatched += list(b2)
    style.bar_labels(ax, solid, dy=0.006)
    style.bar_labels(ax, hatched, dy=0.006, color="#555")

    # The knowledge gap: model says (first-token) -> activation knows (knowledge probe).
    kg, ft = s["know"]["gt"], s["ft"]["gt"]
    ax.annotate("", xy=(-w / 2, kg), xytext=(-w / 2, ft),
                arrowprops=dict(arrowstyle="<->", color="#c1121f", lw=1.6))
    ax.text(-w / 2 - 0.06, (kg + ft) / 2, f"+{kg - ft:.3f}\nlatent\nknowledge",
            ha="right", va="center", fontsize=9, color="#c1121f", fontweight="bold")

    ax.set_xticks(list(xpos.values()))
    ax.set_xticklabels(["Knowledge probe", "Prediction probe", "Model first-token"])
    ax.set_ylabel("accuracy")
    ax.set_ylim(0, 1.0)
    ax.set_title("The model knows more than it says (L20, example-level test)")

    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(facecolor="#777", label="vs. ground truth"),
                       Patch(facecolor="#777", alpha=0.45, hatch="////", label="vs. model's readout")],
              loc="upper right")
    ax.margins(x=0.08)
    style.savefig(fig, "graph1_probe_accuracy", caption=f"NLA-final · example_level test (n={n})")


if __name__ == "__main__":
    main()
