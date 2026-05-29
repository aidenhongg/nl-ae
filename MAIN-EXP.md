# MAIN-EXP — Language-space steering: an equal-accuracy, lower-off-manifold alternative to KAPPA

- **Status:** ⛔ **NULL — experiment concluded at the Phase-1 lever test (GPU, 2026-05-29, ~$0.90).** User GO'd full scope; the decisive §1b go/no-go showed the **AR→patch channel cannot transmit an answer** (all edit operators within noise of the no-edit anchor, far from the know-ceiling). §1b STOP gate fired → Phases 2–4 not run. The NLA verbalization is descriptive, not directive. **Deliverable:** `out/lang/FINDINGS.md` (+ `report.md`, `frontier.png`). Component-1 (off-manifold-FVE) + Component-2 stand; pivot route = prompt-level (`.plans/04_…`). (Phase 0 was ✅ COMPLETE local/CPU, $0.)
- **Owner:** aiden · **Effort:** max · **CWD:** `C:\Users\aiden\Desktop\personalprojects\NLA-final`
- **Builds on:** NLA-final (off-manifold-FVE, COMPLETE) + `personalprojects/exp04` (KAPPA pilot). This is **Phase G** from `.plans/04_outlook_language_space_steering.md`, now fully specified.
- **One-line goal:** Reach KAPPA's task-accuracy gain **at materially lower off-manifold error** by editing the NLA *verbalization* of the L20 activation and reconstructing an **on-manifold** steered activation through the Activation Reconstructor (AR), instead of adding a closed-form residual `Δh` that blows up off-manifold.

---

## 1. Thesis

exp04 established that KAPPA's closed-form residual edit `h' = h + P·r` raises MCQ accuracy **only by pushing the L20 activation off-manifold** — and once the relative edit size `‖Δh‖/‖h‖ ≳ 0.7` the model degrades (a clean inverted-U; single-L20 peak ACC 0.715 @ α10 sits at `ratio 0.66`). NLA-final then *quantified* that off-manifoldness independently: the NLA round-trip cosine `cos(h, AR(AV(h)))` falls monotonically with α (Spearman −0.99), i.e. steered activations verbalize+reconstruct worse the harder you push.

The AR gives a route that is **on-manifold by construction**: `AR(text)` runs *real text* through the first ~20 layers of Qwen + a learned `Linear(d,d)` head, so its output is (an affine image of) a genuine-model activation. If a **minimal text edit** to the verbalization is enough to move the prediction toward the model's own knowledge, we get KAPPA's intended effect **without** the residual blow-up:

```
h_orig ──AV──► z_orig (verbalization)
                 │  minimal edit toward the knowledge-probe label X  (the "ladder")
                 ▼
              z_edit (text)
                 │ AR
                 ▼
              ĥ_steer  ── patch @ L20, re-forward ──►  ACC / AGR        (on-manifold ⇒ low OME)
```

**Win = match KAPPA's accuracy at strictly lower off-manifold error.** Concretely (single-layer L20, the only apples-to-apples comparison since the NLA sees only L20): reach **ACC ≈ 0.715** (KAPPA's single-L20 peak) at **OME ≈ 0.27** (the on-manifold floor) instead of KAPPA's **OME 0.41 / ratio 0.66**.

---

## 2. What is already established / on disk (no re-derivation needed)

| Asset | Where | Use here |
|---|---|---|
| KAPPA single-L20 (ACC, OME, ratio) frontier | `out/fve/analysis.json` (`per_alpha`), exp04 `FINDINGS.md` | the curve we must beat (table in §4) |
| KAPPA best overall (multi-6 @α2): ACC 0.670→0.738, AGR 0.652→0.707 | exp04 `FINDINGS.md` | context / stretch (multi-layer, not L20-only) |
| Round-trip floor `cos(orig)=0.7254` (OME 0.2746), calibration PASSED | `out/calibration.json`, `out/fve/fve_by_alpha.json` | the on-manifold anchor every method is measured against |
| **Verbalizations** of orig (6536) + steered (11×1024) + headline (2615×{2,10,30}) | `out/nl/*.parquet`, samples in `sample_pairs.md` | the text we edit — **no AV re-run needed for the edit-based methods** |
| **Steering labels** joined into every datapoint | `out/feat/datapoint_features.parquet` (enriched into `nl/*`) | `know_argmax_symbol` (target), `gt_symbol` (oracle), `pred_argmax_symbol`, `y_tilde`, `model_symbol` |
| L20 probes (`know`,`pred`), `h_layer20_orig.npy`, `example_ids.json`, `splits.json`, `examples.jsonl` | `../exp04/05_out_pulled/{02_probes,03_kappa/emb,data}` (local) | build edits' targets + CPU pred-probe proxy + eval ground truth |
| 11 steered arrays `h_layer20_steered_a{α}.npy` | `inputs/` (local) + `s3://…/nla/inputs/` | optional contrastive source (method E-contrastive) |
| **Eval harness** (patch-and-re-forward) | exp04 `kappa/model_forward.py` (`register_residual_hooks(edit_fn)`, `run_forward`), `kappa/prompt.py` (`ANSWER_CUE`, symbol ids `{A:32,B:33,C:34,D:35}`), `kappa/generate.py` (free-gen) | reuse verbatim; our `edit_fn` *replaces* the L20 last-token residual with `ĥ_steer` |
| **NLA API** (verified on-pod) | `src/nla_run.py`, `.podref/nla_inference.py` | `client.generate(vec)→text`; `critic.reconstruct(text)→[d] raw`; `critic.score(text,vec)→(mse,cos)`; `injection_scale=150`, `mse_scale=√d≈59.87` |
| On-pod recipe (sglang 0.5.6 pin, env, throughput, S3 gotchas) | `CHANGELOG.md`, `.podref/`, [[project-nla-final-state]] | re-use; do not rediscover |

**Base anchors (example_level test):** base ACC (readout) **0.6604**, know_acc **0.8543**, pred_acc **0.9120**, AGR **0.6486**. Mean ‖h‖ ≈ **86.69**.

---

## 3. Precise definitions (lock these)

- **Steering target `X` (per row):** `know_argmax_symbol` — the **knowledge-probe label** ("snap to the answer the model internally knows"). This is the KAPPA-faithful target (KAPPA steers the prediction readout toward the knowledge logits) and its accuracy ceiling is `know_acc = 0.854`. **Oracle secondary:** `gt_symbol` (ground truth) — same pipeline, different letter; computed for free as an upper-bound reference. *(We are NOT peeking at gt to drive the method; gt is reported only as a ceiling.)*
- **Edit `z_orig → z_edit`:** one of the operators in §5 (the "ladder").
- **Reconstruction `ĥ_recon = AR(z_edit)`** (raw, unnormalized [d]).
- **Patched activation `ĥ_steer`:** `ĥ_recon` rescaled to a magnitude convention chosen in Phase 1a (default: norm-match to the row's original, `ĥ_steer = ĥ_recon/‖ĥ_recon‖ · ‖h_orig‖`). A pure *directional* edit, consistent with H3 (the NLA is direction-only).
- **Evaluation (patch `ĥ_steer` at L20 last token, re-forward Qwen):**
  - **Forced-choice readout (primary, KAPPA-comparable):** argmax of the 4 symbol logits at the `"…is ("` position → `ŷ`. `ACC = mean(ŷ == gt)`, `AGR = mean(know_argmax == ŷ)`. *(The prompt ends at `(`, so this next-token logit IS the model's answer — same readout exp04/KAPPA used.)*
  - **Free generation (secondary, the literal "steer the generation"):** greedy decode with the patch active on the first step; parse the letter. Run on headline configs only.
- **Off-manifold error of `ĥ_steer` — reported two ways (one NLA-native, one NLA-independent):**
  1. **OME = 1 − cos(ĥ_steer, AR(AV(ĥ_steer)))** — the *same* NLA round-trip metric used for KAPPA's curve (apples-to-apples; mildly favorable to AR outputs — see caveat in §9).
  2. **ratio = ‖ĥ_steer − h_orig‖/‖h_orig‖** — exp04's own model-intrinsic off-manifold proxy; **NLA-independent**, so a win on this is unimpeachable.
- **WIN CONDITION (formal):** a method's `(ACC, OME)` and `(ACC, ratio)` points lie **above-left of the KAPPA single-L20 frontier** — i.e. **≥ KAPPA's accuracy at strictly lower off-manifold error**, with non-overlapping bootstrap CIs on the headline config. Equivalent fallback win: **match KAPPA's single-L20 peak ACC (0.715) while staying near the on-manifold floor (OME ≲ 0.30, ratio ≲ 0.3).**

### The KAPPA single-L20 frontier we must beat

| α | ratio `‖Δh‖/‖h‖` | cos_RT (on-manifold) | **OME = 1−cos** | exp04 ACC (512) |
|---:|---:|---:|---:|---:|
| 0 (base) | 0.00 | 0.725 | 0.275 | 0.670 |
| 2 | 0.12 | 0.718 | 0.282 | 0.672 |
| 5 | 0.32 | 0.681 | 0.319 | 0.689 |
| **10** | **0.66** | **0.587** | **0.413** | **0.715 (peak)** |
| 20 | 1.34 | 0.416 | 0.584 | 0.686 |
| 30 | 2.03 | 0.307 | 0.693 | 0.600 (collapse) |

> KAPPA **cannot** buy high ACC cheaply: its only ≥0.71 point costs OME 0.41 / ratio 0.66. Our target region is the **empty top-left**: ACC ≥ 0.715 at OME ≈ 0.27.

---

## 4. The method ladder (edit operators)

Operators act on the **cached orig verbalization** `z_orig` (so the edit-based rungs need **no AV re-run** — only AR). `X` = target letter (§3). Robust patterns are **designed from a verbalization sample in Phase 0** (the text is messy — see `sample_pairs.md`: letters appear as `(B)`, `'A'`, `A)`, `likely "D"`, alongside non-answer option letters that must NOT be clobbered).

**Anchor / control**
- **E0 — identity:** `z_edit = z_orig` → `ĥ_orig`. The no-edit round-trip baseline (isolates AR noise from the edit's effect).

**Rung 1 — the casual lever (GO/NO-GO, Phase 1)**
- **T1 — template (strongest lever):** discard the verbalization, `z_edit = "The correct answer is (X)."` (mirrors the prompt's answer cue). Tests whether the AR→patch pathway can transmit an answer **at all**.
- **E2-min — minimal append:** `z_edit = z_orig + " Actually, the correct answer is (X)."`

**Rung 2 — the user's two operators, in isolation then together (Phase 2)**
1. **E1 — regex letter substitution:** rewrite the *answer-assertion* letter mentions in `z_orig` to `X` (pattern targets `answer is …(L)`, `implying/suggesting/likely '(L)'`, `(L) is the correct`, etc.; leaves option-list letters alone).
2. **E2 — append assertion:** `z_orig + " Actually, the answer is (X)."` (= E2-min, full-scale).
3. **E4 — E1 + E2 together.**

**Rung 3 — add the uncertainty stripper (Phase 2b, if Rung 2 underperforms)**
- **E3 — strip uncertainty:** delete hedges ("likely", "strongly implying", "suggesting", "could be", "or similar", "speculative", "possibly", …) — sharpen the assertion. *(E3 alone sets no target; it is a modifier.)*
- **E5 = E3+E1**, **E6 = E3+E2**, **E7 = E3+E1+E2.**

**Rung 4 — fallbacks (Phase 3, only if Rungs 1–3 don't clear the win condition)**
- **T2 — synthesized deterministic template (rich):** generate a generic on-genre template that states `X` outright while preserving the verbalization's structural frame (e.g. *"Structured multiple-choice answer. The correct answer is (X). The final token completes the answer with (X)."*). Deterministic, no model.
- **L1 — SLM/LLM minimal rewrite:** prompt a small local LLM to *minimally* edit `z_orig` so it asserts `X` (preserve everything else).
- **L2 — cosine-targeted rewrite:** embed sentences of `z_orig`, find the one(s) most similar to *"the answer is X"*, rewrite just those to assert `X` (surgical).

> The ladder runs **minimal→heavy**; robustness runs **heavy→minimal**. The experiment maps that trade-off: a minimal edit that wins keeps `ĥ_steer` closest to `ĥ_orig` (lowest collateral), which is the ideal outcome.

---

## 5. Eval harness (new module `src/lang_steer.py`, reuses exp04 verbatim)

A single on-pod driver. **AV is needed only for the OME re-verbalization (Phase 4); Rungs 1–3 use AR + Qwen only.**

```
build_targets()      # CPU: per-row X = know_argmax_symbol (+ gt_symbol); from datapoint_features.parquet
edit(op, z_orig, X)  # CPU: apply an operator from §4 -> z_edit  (+ unit tests on cached text)
reconstruct(z_edit)  # GPU: critic.reconstruct -> ĥ_recon ; rescale -> ĥ_steer ; save arrays
proxy_rank()         # CPU: pred-probe readout on ĥ_steer = argmax(W_pred ĥ + b_pred); cheap pre-rank
eval_readout()       # GPU: load Qwen, rebuild prompts from examples.jsonl (prompt.py, deterministic),
                     #      edit_fn replaces L20 last-token residual with ĥ_steer[row], re-forward,
                     #      read symbol logits -> ACC/AGR   (reuse model_forward.run_forward)
eval_generate()      # GPU: same patch under model.generate (generate.py) -> free-gen letter (headline)
ome()                # GPU: AV(ĥ_steer)->text'; AR.score -> cos -> OME (headline configs)
report()             # CPU: (ACC, OME, ratio) per method + Pareto plot vs KAPPA frontier; bootstrap CIs
```

**`edit_fn` for the patch** (the only conceptual change from KAPPA's edit): `lambda h_last, layer: ĥ_steer_batch` — a **replacement**, not an additive `Δh`. exp04's hook already supports capture==edit at L20 last token; the no-op/locality parity gates apply unchanged.

**Fast CPU proxy.** `pred_acc = 0.912` on orig ⇒ the prediction probe predicts the model's readout 91% of the time. So `argmax(W_pred·ĥ_steer + b_pred)` is a cheap pre-rank of "what will Qwen say" — compute it for *all* method variants on CPU after the one AR pass, then spend Qwen re-forward only on the top configs. Treat as a filter, not truth (it can break on heavily-edited ĥ, exactly like KAPPA's high-α AGR artifact) — always confirm winners with the real re-forward.

---

## 6. Phase plan (resumable checklist)

### Phase 0 — Local prep + harness (CPU, **$0**, do now)
- [x] **0a** Assemble eval inputs locally; verify the exp04 mirror has probes/`h_orig`/`example_ids`/`splits`/`examples` (confirmed present 2026-05-29). **DONE 2026-05-29:** verified `../exp04/05_out_pulled/{02_probes/example_level/{know,pred}/layer20.npz, 03_kappa/emb/{example_ids.json,h_layer20_orig.npy}, data/examples.jsonl}`; `inputs/{examples.jsonl,splits.json,norms.parquet,steered a*}`; `out/feat/datapoint_features.parquet` (targets: test 2615, know_argmax balanced A650/B675/C631/D659, know==gt 0.8543, pred==know 0.6421); `out/nl/{orig,steered_a*,headline_a*}.parquet` (verbalizations enriched w/ targets). Eval harness `../exp04/kappa/{model_forward,prompt,generate,dataset,config}.py` importable; torch+transformers+Qwen2.5-7B tokenizer cached locally.
- [x] **0b** Verbalization-sample analysis. **DONE 2026-05-29** (`out/lang/verbalization_analysis.json` via `lang_steer analyze-text`, validated on all 2615 test rows): **E1** (anchored letter-sub, enumeration-protected) fires 32.0%, 1046 subs, **no_clobber=True** (0 enumeration letters touched; 2892 protected), idempotent — no-clobber prioritized over recall per §8 (E2/T1 give universal coverage). **E3** (adverbial-hedge strip; structural verbs kept) fires 70.7%, 2522 hedges removed, idempotent. Patterns persisted in `src/lang_steer.py` (`_ANCHOR`, `_RUN`, `_HEDGE`).
- [x] **0c** Implement `src/lang_steer.py`. **DONE 2026-05-29:** operators E0–E7 + T1–T2, the **replacement** `edit_fn` patch, magnitude conventions (native/normmatch/cohortmean), targets+nested subsets (tiny256⊂sweep512⊂full2615, seed 7), CPU pred-probe proxy, AR reconstruct + Qwen eval-readout/eval-generate (exp04 `kappa/` reused verbatim, lazy-imported) + AV/AR OME + report/Pareto plotter. **`tests/test_lang_steer.py` GREEN** (operators on real cached text: idempotency, letter-correctness, no-clobber; conventions; frontier parse [peak ACC 0.715 @ OME 0.413, floor 0.276]; WIN/PARTIAL/NULL verdict).
- [x] **0d** CPU dry-run on tiny random Qwen2. **DONE 2026-05-29** (`tests/test_lang_steer_tiny.py` GREEN, $0): parity gate on the patch path, prompt rebuild (symbol ids {A:32,B:33,C:34,D:35}), replacement edit_fn moved 8/12 readouts, free-gen plumbing, target join, proxy→report chain. **Found+fixed a real bug:** exp04's single-forward hook breaks under `model.generate` (incremental decode step is seq-len 1, baked-in `pos_last` over-indexes) → added a prefill-only patch hook for the generation path (patches step 1, propagates via KV cache; readout path keeps exp04's hook verbatim).
- [ ] **0e** Write the GPU GO/NO-GO brief (scope, est. cost, ceiling) → **STOP for explicit user GO** ([[feedback-consult-before-paid-cloud]]). **Brief written: `GO-NO-GO.md` (2026-05-29). AWAITING USER GO.**

### 🚦 GPU GO/NO-GO GATE — **GO'd (full scope) 2026-05-29.** Pod `n3rftfbvhkcrtq` (A100 80GB, ~$0.90). 48GB ≤$0.50/hr cards OOS → A100 (under ceiling).

### Phase 1 — Casual lever test (GPU, tiny256) — **the decisive go/no-go** — ✅ DONE 2026-05-29 → **NULL**
- [x] **1a Reconstruction anchor (control):** E0 anchor ACC = **0.633** (normmatch) ≈ base 0.66 → **gate PASSED** (AR not too lossy; clears the §8 core risk). **Convention locked = normmatch** (native overshoots: ratio 1.36/ACC 0.617; normmatch ratio 0.74/ACC 0.633 ≈ cohortmean 0.74/0.641, tie → keep the H3-consistent per-row directional default).
- [x] **1b Lever:** T1 success **0.570** (BELOW anchor 0.625), E2 0.645 (flat). Extended to all distinct mechanisms: E1 0.633, E4 0.652 (surgical, ≈anchor), T2 0.531 (rich template, below). Free-gen: E0 parses 100% / T1 only 46%. **No operator steers toward X.**
- [x] **DECISION: STOP (NULL).** Even the strongest levers (T1/T2) can't move the readout toward X — the AR→patch channel can't transmit an answer (verbalization is descriptive, not directive; §8 risk #2). Phases 2–4 NOT run. Components 1–2 stand; pivot = prompt-level (`04_…`). See `out/lang/FINDINGS.md`.

### Phase 2 — The ladder, isolation→combination (GPU; 512-subsample then full 2615 on winners)
- [ ] **2a** E1 (regex-sub), E2 (append) **in isolation**; **E4** (together). One AR pass → CPU proxy rank → Qwen re-forward on the live set. Record (ACC, AGR, OME-proxy, ratio).
- [ ] **2b** If 2a hasn't cleared the win condition: add E3 → **E5, E6, E7**. Same eval.
- [ ] **2c** Promote the best ≤2 configs to **full test (2615)**.

### Phase 3 — Fallbacks (GPU, **conditional** on Phase 2 missing the win condition)
- [ ] **3a** T2 (synthesized rich template), full-scale.
- [ ] **3b** L1 (SLM minimal rewrite) and/or L2 (cosine-targeted rewrite); SLM = a small local instruct model (e.g. Qwen2.5-3B-Instruct, already in-family) on-pod. Eval identically.

### Phase 4 — Headline + the off-manifold frontier (GPU)
- [ ] **4a** Winning config(s) at **full test (2615)**: forced-choice **and** free-generation ("steer the generation").
- [ ] **4b** **OME re-verbalization:** AV(ĥ_steer)→AR.score → cos → OME, on the headline configs (this is the costlier AV step; reuse the threaded-concurrency path from `nla_run.py`).
- [ ] **4c** Build the **(ACC, OME)** and **(ACC, ratio)** Pareto plot: our points vs the KAPPA single-L20 frontier (§3 table). Bootstrap CIs. Decide WIN / PARTIAL / NULL against §3.

### Phase 5 — Persist, report, teardown
- [ ] **5a** Push to `s3://iaxphg9saj/nla/lang/` (layout §10); mirror small artifacts to `out/lang/`.
- [ ] **5b** Write `out/lang/report.md` (frontier plot, per-method table, the minimal-edit-that-wins, honest caveats §9).
- [ ] **5c** **Teardown invariant:** push → `runpodctl pod delete` → prove `pod list` clean → close the BUDGET row. Update `CHANGELOG.md` + [[project-nla-final-state]].

---

## 7. Locked decisions (with rationale)

1. **Single-layer L20 only.** The NLA sees only L20; the AR emits an L20 activation; the honest KAPPA comparison is single-L20. Multi-layer KAPPA (the 0.738 cell) is context, not the benchmark.
2. **Target = `know_argmax_symbol`** (knowledge-probe label), not gt. Matches KAPPA's "model knows but won't say" premise; ceiling = know_acc 0.854. gt computed as a free oracle reference only.
3. **Replacement edit, not additive.** `ĥ_steer` *replaces* the L20 residual (it already encodes the whole activation); we do not add it to `h`. This is what makes it on-manifold.
4. **Forced-choice readout is primary** (KAPPA-comparable, cheap, == the next generated token); free-generation is a secondary confirmation on headline configs.
5. **Directional patch (norm-match) as default**, decided empirically in 1a. Consistent with H3 (NLA is scale-invariant/direction-only).
6. **Reuse exp04's harness verbatim** (`model_forward`, `prompt`, `generate`) — the parity/locality/no-op gates already guarantee the patch path is faithful; do not re-roll.
7. **Edit the cached verbalizations** for Rungs 1–3 (no AV re-run) → cheap. AV is spent only on Phase-4 OME.
8. **Two off-manifold metrics** (NLA-native OME + NLA-independent ratio); the headline win must hold on **ratio** (unimpeachable) and ideally both.
9. **All conventions from `nla_meta.yaml`** (injection_scale 150, mse_scale √d); never hardcode. STOP-gate the magnitude convention in 1a before any sweep.

---

## 8. Risks & mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| **AR reconstruction too lossy to patch** — round-trip cos 0.725 ⇒ patching `ĥ_orig` is itself a ratio≈0.74 perturbation; could tank base ACC | **med-high (core risk)** | Phase **1a anchor** measures it directly *before* any method spend. But it is *on-manifold* (real-text activation) unlike KAPPA's off-manifold Δh; exp04's α10 (ratio 0.66, on-the-edge) still hit ACC 0.715, so a ratio-0.74 *on-manifold* patch plausibly survives. If anchor ACC collapses → pivot to prompt-level (route preserved in `04_…`). |
| **Verbalization is a meta-description, not 1st-person** — editing "expecting letter A" may not transmit a committed answer | med | The lever test (1b, T1 strongest) is exactly this go/no-go; if it fails we stop early (cheap). |
| **Regex clobbers option-list letters** (e.g. "A, B, C" in the stem) | med | Pattern designed + validated on a sample in 0b (answer-context anchors only); E2/T1 sidestep parsing entirely; report E1 precision. |
| **CPU pred-probe proxy misleads** on heavily-edited ĥ (like KAPPA's high-α AGR artifact) | med | Proxy is a *filter*; every promoted config is confirmed by real Qwen re-forward. |
| **OME metric is circular** (ĥ_steer is itself an AR output ⇒ re-round-trips well by construction) | med | Honest caveat in §9; the **ratio** metric is NLA-independent and is the load-bearing win criterion; also report a behavioral sanity check (free-gen coherence). |
| **Steering "succeeds" trivially** by snapping everything to one letter (degenerate) | low-med | Report per-class confusion + AGR; require ACC-vs-gt lift (not just ŷ==X), and that non-target rows are preserved (E0-vs-method delta). |
| Magnitude convention wrong → OOD activation (CJK-output failure mode noted in nla_inference) | low | Decided in 1a against the base-ACC anchor; `normalize_activation` helper exists. |
| GPU/sglang flakiness, pod-side S3 unreachable | med (seen) | Reuse the hard-won recipe: sglang 0.5.6 pin, scp inputs local→pod, push results via Windows, head-verify not LIST ([[project-nla-final-state]]). |
| Cost overrun | low | Edit-based rungs are AR+Qwen only (cheap); AV only at Phase 4; resumable ledgers; `--terminate-after` backstop; ceiling $5. |

---

## 9. Honest limitations (state up front in the report)

- **OME favorability:** `ĥ_steer` is an AR output, so the NLA round-trip OME is mildly biased in its favor. The **ratio** metric (NLA-independent) is therefore the primary win criterion; OME is corroborating.
- **Forced-choice ≠ full generation:** primary ACC is the constrained readout (KAPPA-comparable); free-gen on headlines guards against a readout-only artifact.
- **Targeting the knowledge probe, not truth:** ceiling is know_acc 0.854; we report gt-oracle as the upper bound, not as the method's input.
- **512-subsample for the sweep; full-test (2615) only on winners** (mirrors exp04's stated subsample caveat) — directional claims are safe, headline numbers come from full-test.

---

## 10. Storage / artifacts (new `nla/lang/` prefix; exp04 + prior nla/ untouched)

```
s3://iaxphg9saj/nla/lang/
  targets.parquet                     # per-row X (know_argmax) + gt, splits
  edits/<op>.parquet                  # z_edit text per operator (provenance)
  recon/h_recon_<op>.npy              # ĥ_steer arrays (subset / full-test)
  eval/readout_<op>.parquet           # ŷ, ACC, AGR per row
  eval/generate_<op>.parquet          # free-gen letters (headline)
  ome/<op>.parquet                    # cos / OME re-verbalization (headline)
  report/frontier.png  report.md  manifest.json
```
Local mirror: `out/lang/`. S3 hygiene unchanged (write-once versioned keys; raw `upload_file` overwrites; head-verify, never LIST; narrow `Prefix`).

---

## 11. Budget & Runpod (per `/runpod-ctl`)

- **Standing storage:** unchanged — shared volume `iaxphg9saj` (~$2.10/mo), new `nla/lang/` prefix, ≲ a few hundred MB. **No new standing resource, no new BUDGET row for storage.**
- **Compute estimate:** one 48 GB card (A40/A6000 COMMUNITY/SECURE, target ≤$0.50/hr). Rungs 1–3 = AR + Qwen re-forward (cheap, minutes each); Phase 4 AV-OME is the costliest (~2615/2.4 rows/s ≈ 18 min/config). **Est. 2–4 GPU-h ≈ $1–2.5; hard ceiling $5.** Balance ≈ $15.55.
- **Per-run BUDGET.md row at create; `--terminate-after now+3h` backstop; teardown invariant** (push → delete → prove `pod list` clean → close row).
- **Paid boundary is gated:** Phase 0 (local, $0) proceeds now. Provisioning waits on explicit user **GO** ([[feedback-consult-before-paid-cloud]]); the validated full-scope path is the default offer ([[feedback-full-scope-validated-path]]).

---

## 12. Decision log
- **2026-05-29 — drafted.** Plan written from a full review of NLA-final (`.plans/00`, `04`, `src/`, `out/`), exp04 (`kappa/`, `FINDINGS.md`), and the verbalization corpus. Target locked to `know_argmax`; single-L20 KAPPA frontier locked as the benchmark; replacement-patch + dual off-manifold metrics + lever-test go/no-go adopted. Awaiting GPU GO.
- **2026-05-29 — Phase 0 COMPLETE (local, $0).** Built `src/lang_steer.py` (operators E0–E7/T1–T2, replacement `edit_fn`, conventions, targets/subsets, CPU proxy, AR reconstruct, Qwen eval reusing exp04 `kappa/` verbatim, AV/AR OME, report/Pareto). Validated E1 (no-clobber, 32% fire) + E3 (70.7%) on the real corpus; `tests/test_lang_steer.py` + `tests/test_lang_steer_tiny.py` GREEN (operators, conventions, frontier/verdict, tiny-Qwen2 patch/readout/gen/target-join/proxy/report). **Locked OME = 1 − mean_cos_roundtrip** (confirmed vs `out/fve/analysis.json`: KAPPA single-L20 peak ACC 0.715 @ OME 0.413/ratio 0.663; floor OME 0.275). **Fixed** a generation-path patch bug (prefill-only hook). Wrote `GO-NO-GO.md`. **STOP — awaiting explicit user GO before any paid GPU** ([[feedback-consult-before-paid-cloud]]).
- **2026-05-29 — GPU run → Phase-1 NULL; experiment concluded.** User GO'd full scope. Provisioned A100 80GB (`n3rftfbvhkcrtq`, ~0.65 GPU-h ≈ $0.90; 48GB ≤$0.50/hr cards were OOS). Setup recipe transferred to sm80 unchanged. **1a:** anchor gate PASSED (E0 ACC 0.633 ≈ base 0.66), convention locked normmatch. **1b/DECISION:** comprehensive NULL — no edit operator (E0/E1/E2/E4/T1/T2, both readout modes) steers the L20 readout toward X; all within noise of the no-edit anchor (0.625), best E4 0.648 (CI [0.586,0.707]) < KAPPA peak 0.715. §1b STOP gate fired → Phases 2–4 skipped. Root cause: the NLA verbalization is **descriptive, not directive** (encodes what the activation is *about*, not the answer-commitment that drives the readout); templates inject a constant off-target bias. The deferred L1/L2 fallbacks can't rescue a channel-level failure. Pod torn down, account clean, BUDGET row closed. One bug fixed (`reconstruct()` edits-dir mkdir). Deliverables: `out/lang/{FINDINGS.md,report.md,report_data.json,frontier.png}`. Pivot = prompt-level (`.plans/04_…`).
