"""Extra (E3) — WHY the NLA steering is NULL: templates collapse to one letter.

For each trial, the distribution of the model's predicted symbol ŷ over the 256 rows.
Targets X are ~balanced across A/B/C/D (~25% each), so a method that genuinely steered
toward X would stay balanced. Surgical edits (E0/E1/E2/E4) do — they barely move ŷ.
Templates inject a CONSTANT bias: T1 -> B (36%), T2 -> D (41%), regardless of the
per-row target. That constant offset, not steering, is why ACC does not rise.
"""
from __future__ import annotations
import numpy as np
from . import data, style

SYM = ["A", "B", "C", "D"]
SYM_COLORS = {"A": "#4c72b0", "B": "#dd8452", "C": "#55a868", "D": "#c44e52"}


def main():
    style.apply()
    import matplotlib.pyplot as plt
    rep = data.load_lang_report()
    methods = rep["methods"]
    ops = [m["op"] for m in methods]
    n = methods[0]["n"]
    uniform = n / 4.0

    x = np.arange(len(ops)); w = 0.2
    fig, ax = plt.subplots(figsize=(9.6, 5.9))
    for j, s in enumerate(SYM):
        vals = [m["y_balance"][s] for m in methods]
        ax.bar(x + (j - 1.5) * w, vals, w, color=SYM_COLORS[s], label=f"ŷ = {s}",
               edgecolor="white", linewidth=0.4, zorder=3)

    ax.axhline(uniform, ls="--", lw=1.3, color="#444")
    ax.text(len(ops) - 0.5, uniform + 2, f"balanced target (≈{uniform:.0f}/letter)",
            ha="right", va="bottom", fontsize=9, color="#444")

    # call out the collapses (labels sit just above each spike; no title collision)
    for op, sym in [("T1", "B"), ("T2", "D")]:
        i = ops.index(op); v = methods[i]["y_balance"][sym]
        j = SYM.index(sym)
        ax.text(i + (j - 1.5) * w, v + 2.5, f"{op} → {sym}\n{v}/{n} = {v/n:.0%}",
                fontsize=9, fontweight="bold", color=SYM_COLORS[sym], ha="center", va="bottom")

    ax.set_xticks(x); ax.set_xticklabels(ops)
    ax.set_ylabel("count of predicted symbol ŷ  (of 256)")
    ax.set_title("Templates collapse to a fixed letter — the bias behind the NULL")
    ax.set_ylim(0, 124)
    ax.legend(ncol=4, loc="upper left", fontsize=9)
    style.savefig(fig, "graph7_method_letter_bias",
                  caption="NLA-final · predicted-symbol balance per trial, tiny256 (n=256)")


if __name__ == "__main__":
    main()
