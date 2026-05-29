# Outlook (next stage) — Language-space steering to reduce off-manifold error

**Status:** DESIGN SKETCH + DECISION GATE. Not part of the committed Component-1/2 build; we choose the instantiation *after* seeing the NL data (per the brief: "depending on the NLAE data we collect").

## The idea

Raw KAPPA steering adds `Δh = P·r` directly to the residual; once `‖Δh‖/‖h‖ ≳ 1` the result is **off-manifold** and the model degrades (exp04). The NLA gives a route that is **on-manifold by construction**: the AR reconstructs an activation by running *real text* through the first ~20 layers of Qwen + a learned affine head — so **`AR(z)` is (an affine image of) a genuine model activation of some text**. Therefore:

```
h_orig ──AV──► z_orig (text)
                 │  minimal, targeted edit toward the "knowledge" content
                 ▼
              z_edit (text)
                 │
                 ▼ AR
              ĥ_steer  ←  on-manifold steered activation  (low off-manifold error)
```

If a *small* text edit suffices to move the prediction toward the model's own knowledge, we get KAPPA's intended effect (raise ACC/AGR) **without** the residual blow-up.

## Why Components 1 & 2 gate this
- **Component 1** tells us the off-manifold cost curve and whether `AR(z)` reconstructions are faithful (FVE(orig)). It also yields the baseline: round-trip `ĥ_orig = AR(AV(h_orig))` is the *no-edit* on-manifold anchor.
- **Component 2** tells us whether steered/edited descriptions stay **coherent and editable** (H4). If steered NL collapses to gibberish even at moderate α, text-space editing is the *better* path (we never verbalize the off-manifold vector; we edit the *original* text). If steered NL is coherent, the steered text itself is a target.

## Candidate edit operators (rank by what the data supports)
1. **Probe-guided textual assertion (preferred default).** From the knowledge probe `W_know`, identify the correct option per example (exp04 already has `know` argmax). Minimally edit `z_orig` to *assert that knowledge* (templated: "The correct answer is <X> because …"), then `ĥ = AR(z_edit)`. Pure text edit; AR guarantees on-manifold.
2. **Interpolated-text / contrastive.** If steered NL is coherent, take `z_steer(α*)` at the calibrated-good α and find the **minimal edit** from `z_orig` toward `z_steer` (sentence-level diff), reconstruct. Tests whether the NLA's own steered verbalization carries usable signal.
3. **Latent-then-verbalize hybrid.** Apply a *small* KAPPA edit (α in the on-manifold regime, ratio≲0.3), verbalize the result, reconstruct — using the NLA as an on-manifold "projection" of a mild latent nudge. Compare ĥ to the raw mild edit.

## Evaluation (the real test, needs GPU re-forward like exp04)
Patch `ĥ_steer` into Qwen at L20 (the exp04 `model_forward` residual hook supports capture==edit) and re-forward TEST to read symbols. Metrics mirror exp04: **ACC, AGR** before/after, plus:
- **Off-manifold error** of `ĥ_steer` (its `1−FVE` and `‖ĥ−h_orig‖/‖h_orig‖`) vs. the raw KAPPA edit at matched "intervention strength."
- **Win condition:** `ĥ_steer` raises ACC/AGR **at lower off-manifold error** than raw KAPPA, OR matches KAPPA's best (multi-6 @α2: ACC 0.670→0.738) while staying on-manifold. Use the example_level test split + the 512/full-test conventions from exp04 for comparability.

## Decision gate (after Components 1 & 2)
Proceed to build this stage **iff**:
- Calibration green and **FVE(orig) ≳ 0.6** (AR reconstructions are faithful enough to patch back), AND
- Steered/edited NL is **coherent** at the α we care about (Component-2 drift analysis), AND
- FVE(α) vs ACC(α) (H2) shows the NLA off-manifold proxy is meaningful (so "reduce off-manifold error" is a real lever).

If FVE(orig) is too low to patch faithfully, pivot: use the NLA purely as a **measurement/diagnostic** (Components 1–2 stand alone as a result: "NLA FVE quantifies steering off-manifoldness and predicts degradation"), and pursue language-space steering via prompt-level intervention instead of activation patching.

## Artifacts this stage would add
`nla/steered_lang/` : `z_edit` corpus, `h_recon_lang_a*.npy`, re-forward `acc/agr` json, comparison report vs raw KAPPA.
