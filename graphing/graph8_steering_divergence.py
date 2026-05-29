"""Extra (E4) — the KAPPA mechanism: steering briefly aligns the prediction with
the knowledge, then the off-manifold blow-up re-opens the gap.

y = mean KL(prediction-probe ‖ knowledge-probe) on the steered activation h'(α), test split.
At α=0 the two disagree (KL 8.37 — this IS the "knows but doesn't say" gap from Graph 1,
in divergence form). KAPPA collapses it to ~0.4 by α≈0.5 (prediction snapped onto the
knowledge), but flipping the discrete readout needs a bigger push (ACC peaks at α=10),
and by then the off-manifold collapse is already re-opening the gap; by α=30 it is back
near baseline. Mechanism + its limit in one curve.
"""
from __future__ import annotations
import numpy as np
from . import data, style


def main():
    style.apply()
    import matplotlib.pyplot as plt
    sd = data.load_steered_divergence()
    test = sd["split"] == "test"
    al = sd["alpha"][test]; kl = sd["kl_pred_know_steered"][test]
    alphas = np.array(data.ALPHAS)
    mean_kl = np.array([kl[al == a].mean() for a in alphas])

    fig, ax = plt.subplots(figsize=(8.8, 5.8))
    ax.plot(alphas, mean_kl, "-o", color=style.C_KAPPA, lw=2.2, ms=8, zorder=4)

    base = mean_kl[0]
    ax.axhline(base, ls=":", lw=1.2, color="#999")
    ax.annotate(f"baseline gap (α=0): KL={base:.2f}\n= the model's prediction ≠ its knowledge",
                (0, base), xytext=(3.0, base + 0.35), fontsize=9, color="#666",
                arrowprops=dict(arrowstyle="->", color="#999"))

    imin = int(np.argmin(mean_kl))
    ax.annotate(f"min KL={mean_kl[imin]:.2f} @ α={alphas[imin]:.1f}\nprediction snapped onto knowledge",
                (alphas[imin], mean_kl[imin]), xytext=(4.5, 2.4), fontsize=9, color=style.C_KNOW,
                fontweight="bold", arrowprops=dict(arrowstyle="->", color=style.C_KNOW))

    # where accuracy peaks (alpha=10) — already on the way back up
    ax.axvline(10, ls="--", lw=1.3, color=style.C_MODEL, alpha=0.7)
    ax.annotate("ACC peak (α=10):\ngap already re-opening", (10, mean_kl[alphas == 10][0]),
                xytext=(11.5, 3.2), fontsize=9, color=style.C_MODEL,
                arrowprops=dict(arrowstyle="->", color=style.C_MODEL))
    ax.annotate("off-manifold: gap back to baseline", (30, mean_kl[-1]),
                xytext=(17, 8.2), fontsize=9, color="#c1121f",
                arrowprops=dict(arrowstyle="->", color="#c1121f"))

    ax.set_xlabel("steering strength  α   (KAPPA scaling factor)")
    ax.set_ylabel("mean KL( prediction-probe ‖ knowledge-probe )  on h′(α)")
    ax.set_title("How KAPPA helps — and why it runs out: aligning prediction with knowledge")
    ax.set_xlim(-1.5, 31.5); ax.set_ylim(-0.4, 10.6)
    style.savefig(fig, "graph8_steering_divergence",
                  caption="NLA-final · F3 pred↔know divergence on steered h′(α), example_level test (n=2615)")


if __name__ == "__main__":
    main()
