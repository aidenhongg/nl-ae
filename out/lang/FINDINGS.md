# Language-space steering (MAIN-EXP) — FINDINGS

**Date:** 2026-05-29 · **Pod:** `n3rftfbvhkcrtq` (A100 80GB PCIe, SECURE, RO; ~0.65 GPU-h, ~$0.90) · **Status:** Phase-1 lever test COMPLETE → **NULL.** Phases 2–4 not run (§1b STOP gate fired).

## Verdict: NULL — the AR→patch channel does not transmit an answer

The MAIN-EXP thesis was: *edit the NLA verbalization of an L20 activation → reconstruct an on-manifold activation through the AR → patch it back → steer the MCQ answer at strictly lower off-manifold error than KAPPA.* The Phase-1 lever test (tiny256, n=256, the decisive go/no-go) **falsifies this at the channel-transmission step.** No edit operator moves the forced-choice readout toward the target `X = know_argmax_symbol`.

## The evidence (tiny256, normmatch unless noted; base ACC 0.6604, know-ceiling 0.8543)

| op | mechanism | ACC | success(ŷ==X) | ratio ‖Δh‖/‖h‖ | ŷ balance | ACC 95% CI |
|---|---|---:|---:|---:|---|---|
| **E0** | **no-edit anchor** | **0.633** | **0.625** | 0.74 | A64 B61 C69 D62 | [0.570, 0.692] |
| E1 | surgical letter-flip | 0.633 | 0.633 | 0.74 | A61 B61 C70 D64 | [0.570, 0.692] |
| E2 | append assertion | 0.645 | 0.645 | 0.74 | A65 B58 C67 D66 | [0.586, 0.703] |
| E4 | E1+E2 (best) | 0.648 | 0.652 | 0.74 | A61 B60 C68 D67 | [0.586, 0.707] |
| T1 | bare template "answer is (X)" | 0.578 | 0.570 | 1.09 | **A22 B91** C69 D74 | [0.516, 0.637] |
| T2 | rich template | 0.547 | 0.531 | 1.07 | A34 B51 C67 **D104** | [0.484, 0.605] |

**KAPPA single-L20 target to beat:** ACC **0.715** @ OME 0.413 / ratio 0.663. On-manifold floor OME ≈ 0.276.

Free-generation (independent confirmation): E0 anchor parses **100%** (gen_acc 0.633); T1 parses only **46%** (gen_success 0.546) — the template patch *disrupts* generation rather than steering it.

## Robustness post-mortem (out/lang/analyze_null.py, out/lang/null_diagnostics.json) — the NULL is not an artifact

Asked whether outliers or regex errors drove the result. They did not:
- **Off-manifold `ratio` is not outlier-driven.** Per-datapoint distributions are extremely tight: E0/E1/E2/E4 mean≈median≈**0.740**, std **0.022**, max 0.84, **0 outliers >2.0**; T1 1.089, T2 1.072 (same std). Mean≈median ⇒ no skew. Trimming the worst-10%-ratio rows does **not** raise success (E4 0.652→0.630).
- **The E1 regex is provably correct.** It fired on **92/256 (35.9%)** rows; on **all 92** the substitution landed exactly on X (92/92), with **0 option-letter clobbers**. T1/T2 are templates (no regex), well-formed.
- **A weak but REAL graded signal exists — the channel is not dead, just ~10× too weak.** argmax is coarse; the model's probability mass on the target, Δp(X) vs the E0 anchor, is the sensitive test: **E2 +0.030 (t≈7.3, p(X)↑ on 70% of rows), E4 +0.037 (t≈7.1, 73%)**, E1 +0.010 (only fires 36%). Templates push the *wrong* way (T1 −0.058, T2 −0.145). So the *append* edit transmits a small, highly-significant nudge toward X — but it is far too weak to matter, and (critically) it buys **no** off-manifold saving: E4 sits at ratio 0.74 (= the no-edit anchor) yet only ACC 0.648, **strictly dominated** by KAPPA's peak (ACC 0.715 at ratio 0.66) on the load-bearing NLA-independent metric. There is no (ACC, ratio) point anywhere near the empty top-left the WIN required.

## Why this is a real NULL, not a bug or small-sample fluke

1. **The plumbing is verified.** The parity gate (no-op-identity + locality, vs HF `output_hidden_states`) passed on the real Qwen for every eval. The replacement `edit_fn` patches L20 last-token faithfully.
2. **The channel itself works** — the E0 anchor (patch the AR round-trip of the *unedited* verbalization) reproduces base behaviour: ACC 0.633 ≈ base 0.66, AGR 0.625 ≈ base 0.649. So the AR reconstruction is faithful enough to patch (this *clears* the §8 "too lossy" core risk). The failure is specific to the **edit**, not the channel.
3. **No operator steers.** Every operator's success(ŷ==X) sits within noise of the 0.625 anchor; the best (E4, 0.652) is +0.027 (7/256 rows). A working channel would push success toward the 0.854 know-ceiling. E4's 95% CI [0.586, 0.707] **overlaps the anchor and its upper bound is below KAPPA's 0.715** — n=256 already excludes a win.
4. **Two distinct failure modes, both fatal:**
   - **Surgical edits (E1) are too subtle:** flipping one asserted letter inside a ~200-char meta-description leaves the reconstruction essentially unchanged (E1 ratio 0.74 = E0's 0.74) → readout unchanged.
   - **Template edits (T1/T2) move ĥ a lot but in the wrong direction:** ratio ~1.1, and the ŷ balance collapses to a *constant* letter (T1→B, T2→D) **regardless of X** — a fixed off-target bias from the generic template's reconstruction, the opposite of steering.
   There is no operating point where an edit both moves ĥ meaningfully **and** moves it toward the answer-readout direction.

## Interpretation

The NLA verbalization is **descriptive, not directive.** It captures *what an activation is about* (topic, format, "this looks like an answer task about a beloved character…") — a third-person meta-description — not the first-person answer-commitment state that drives the L20→logit readout. Reconstructing an answer-edited description therefore yields an on-manifold activation whose *semantics* shifted but whose *answer-readout direction* shifted only **faintly**: the append edits carry a real, highly-significant but tiny nudge toward X (Δp(X) ≈ +0.03–0.04, t≈7) — the channel has very low *gain* for answer-steering — while template edits push the wrong way. The round-trip is faithful for content (Component-1 result) but the behavioural lever KAPPA exploits is **present at ~1/10th the strength needed and at no off-manifold discount**. This is §8 risk #2 ("meta-description, not 1st-person") confirmed empirically — a low-gain channel, not a broken one — and it is why the edit-ladder is dominated by KAPPA rather than a Pareto win.

A corollary: **the deferred SLM-rewrite fallbacks (L1/L2) cannot rescue this.** They are *better text edits*, but the failure is not edit-quality — it is that the channel does not expose the answer direction to *any* text edit. T1 (a maximal, explicit assertion) already fails; an SLM minimal-rewrite sits between E1 and T1, both of which fail.

## What stands, and the honest scope

- **Component 1 (off-manifold-FVE, COMPLETE)** and **Component 2 (NL collection)** are unaffected — they are the standalone deliverables. This NULL is a *negative result about steering*, not about the NLA's reconstruction fidelity.
- **Caveat (n):** established at n=256 across all 6 distinct-mechanism *implemented* operators + free-gen. E3/E5/E6/E7 are hedge-strip modifiers / combinations of already-failed operators (logically covered). A sweep512/full confirmation would only tighten CIs that already exclude a win.
- **Caveat (untested operator):** a Phase-0 design artifact (`donor_set.json`, `template_candidates.json`) holds a **"donor-patching"** mechanism — patch the AR-reconstruction of a *real* high-confidence X-committed verbalization (pred==know==X) instead of a synthetic template or self-edit. It was **not implemented in `lang_steer.py` nor tested.** Assessment: it is a wholesale *replacement* like T1/T2 (donors are tied to *other* questions, so it injects the donor question's semantics) and the channel is demonstrably low-gain — so it is *expected* to behave like T1/T2 (constant off-target bias) rather than win, but this is an honest untested gap. A tiny256 test (one operator, ~$0.15 + a fresh pod) would close it.

## Recommended next step

The genuinely different direction is the **prompt-level intervention** (edit the *prompt*, not the activation), preserved in `.plans/04_outlook_language_space_steering.md` — a separate experiment, not a fallback rung. Activation-patching via NLA-text-edits is a dead channel for behavioural steering and should be reported as such.
