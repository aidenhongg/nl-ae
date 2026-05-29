# NLA-final — Off-manifold steering error via Natural Language Autoencoders

**Run completed 2026-05-28.** Continuation of the exp04 KAPPA steering pilot. We probe whether a
**Natural Language Autoencoder** (NLA) round-trip — Activation Verbalizer (AV) → text →
Activation Reconstructor (AR) — gives a *directional* "naturalness" signal that degrades as the
KAPPA closed-form edit pushes a Qwen2.5-7B-Instruct **layer-20** activation off-manifold, and
whether that signal predicts exp04's downstream accuracy curve.

- **Model / layer:** `Qwen/Qwen2.5-7B-Instruct`, HF `hidden_states[20]`, d_model 3584.
- **NLA:** `kitft/nla-qwen2.5-7b-L20-{av,ar}` (open re-impl, served via sglang `input_embeds` for the
  AV; the AR is an in-process truncated-21-layer Qwen + learned `value_head`). injection_scale 150,
  mse_scale √3584. Round-trip metric = `cos(h, ĥ)` (and `mse = 2(1−cos)`).
- **α grid:** {0, 0.5, 1, 2, 3, 5, 7, 10, 15, 20, 30}; single-layer L20 edit (exactly reproduces the
  stored exp04 a2/a10 arrays — Phase A gate). Paired **1024-row** test subset for the dense sweep.

## Calibration gate (Phase C) — PASSED
FVE(orig) on the 1024-row subset: **mean cos = 0.726** (range 0.646–0.768), matching the AV card's
in-distribution 0.752. Verbalizations are coherent English describing the MCQ activation structure
(e.g. *"Structured answer format with numbered definitions about 'snake'"*). This validated the
serving path + conventions before the sweep.

## H1 — off-manifold error grows with steering strength ✅ STRONG
NLA round-trip fidelity falls **monotonically** with α. **Spearman(cos, α) = −0.991.**

| α | 0 | 0.5 | 1 | 2 | 3 | 5 | 7 | 10 | 15 | 20 | 30 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **mean cos** | .725 | .726 | .724 | .718 | .708 | .681 | .645 | .587 | .494 | .416 | **.307** |
| **OME = 1−cos** | .275 | .274 | .276 | .282 | .292 | .319 | .355 | .413 | .506 | .584 | **.693** |
| ratio ‖Δh‖/‖h‖ | 0 | .03 | .06 | .12 | .19 | .32 | .46 | .66 | 1.00 | 1.34 | 2.03 |

OME tracks the steering magnitude (ratio) tightly — the NLA's *directional* probe captures the same
off-manifold growth that exp04 characterized only by magnitude.

## H2 — does off-manifold error predict accuracy damage? ⚠️ NUANCED
exp04's single-layer-L20 accuracy is an **inverted-U** (steering *helps* before it hurts):
base 0.670 → peak **0.715 at α=10** (ratio 0.66) → collapse **0.600 at α=30** (ratio 2.0).
OME rises monotonically, so:
- **Spearman(OME, ACC-drop) = 0.086** (weak — rank orders diverge: OME monotone, ACC inverted-U).
- **Pearson = 0.588** (moderate — anchored by α=30: highest OME ↔ largest accuracy drop).

**Reading:** the NLA off-manifold signal reliably **flags the catastrophic high-α collapse** but does
*not* predict the beneficial low/mid-α regime — there is a "productively off-manifold" zone (α≈5–10)
where steering improves accuracy despite OME already at 0.32–0.41. So directional off-manifold-ness is
**necessary but not sufficient** for task degradation. This refines, rather than confirms, a simple
"off-manifold ⇒ bad" story.

## H3 — the signal is directional, not magnitude ✅ CONFIRMED
Rescaling the α=10 inputs by c (the AV unit-normalizes, so the injected vector is identical):
**cos = 0.5875 (c=0.5), 0.5873 (c=1), 0.5877 (c=2)** — invariant. Any FVE drop under steering is a
**direction** change, orthogonal to exp04's magnitude (ratio) story. (Built-in control.)

## H4 — verbalizations stay editable through moderate α ✅ (qualitative)
NL stays coherent and on-topic through α≤10, then degenerates by α=30. Example (row 6, a bird-species MCQ):
- **orig:** *"…definitions and questions about a bird species… 'What kind of bird?'"*
- **α=2:** *"…numbered questions… 'what animal is'…"*  (coherent, on-topic)
- **α=10:** *"…'Find the name of the following species'…"*  (coherent, drifting)
- **α=30:** *"…a textbook query about finding the minimum distance…"*  (**semantics lost**)

This pairs with the cos curve and is the decision-gate signal for the *next* stage (language-space
steering): a minimal text edit + AR reconstruction is viable where verbalizations remain coherent
(α≲10), not at the collapse point.

## Methodology note
The plan's fixed-denominator FVE = 1 − mean_mse/Var0 is **degenerate here**: Var0 (variance of the
unit-normalized original cohort) is only **0.107** — L20 last-token MCQ activations cluster tightly in
direction — while the round-trip MSE is ~0.55, so "FVE" goes strongly negative. The **mean cosine
(and OME = 1−cos) is the interpretable primary metric** (which plan §6 in fact designates as primary);
the fixed-Var0 FVE is reported only for reference. Spearman tests are unaffected (rank-invariant).

## Deliverables (mirrored to `s3://iaxphg9saj/nla/` and local `out/`)
- `fve/fve_by_alpha.json`, `fve/per_row.parquet` (11,264 rows), `fve/rescale_control.json`, `fve/analysis.json`
- `recon/h_recon_a{α}.npy` — 11 × (1024, 3584) reconstructed activations (for the language-steering stage)
- `nl/orig.parquet` (6,536), `nl/steered_a{α}.parquet` (11 × 1,024), `nl/headline_a{2,10,30}.parquet` (3 × 2,615)
- `report/calibration.json`, `report/fve_figures.png` (FVE vs α; FVE vs ratio; OME vs ACC-drop)

## Compute / cost
1× RTX A6000 48GB (SECURE) [+ an earlier A40 for Phase A–C]; ~27,700 AV round-trips at ~2.4 rows/s
(threaded sglang concurrency). **Total run cost ≈ $2.61** (well under the $5 ceiling). Standing storage
(shared 30GB volume `iaxphg9saj`) ≈ $2.10/mo.
