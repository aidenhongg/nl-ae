# NLA-final — Master Plan (off-manifold steering error via Natural Language Autoencoders)

- **Status:** ✅ **COMPLETE (2026-05-28).** All phases A–F run; user GO'd the paid run + raised budget for full scope. Calibration gate passed (mean cos 0.726). Full sweep (1024×11) + rescale (H3) + NL corpus (orig 6536, headline 2615×{2,10,30}) done. **H1 strong** (ρ(cos,α)=−0.991), **H2 nuanced** (OME flags the α=30 collapse, pearson 0.59, but doesn't predict exp04's inverted-U), **H3 confirmed** (scale-invariant), **H4** NL coherent ≤α10. Results in `out/` + `s3://iaxphg9saj/nla/{fve,nl,recon,report}`; writeup `out/report/report.md`; cost ≈ $2.61; pod torn down. See `CHANGELOG.md` for the API/env/throughput fixes made on-pod.
- **Owner:** aiden · **Drafted:** 2026-05-27 · **Effort:** max
- **CWD:** `C:\Users\aiden\Desktop\personalprojects\NLA-final`
- **Continuation of:** `personalprojects/exp04` (the KAPPA steering pilot)
- **New capability:** Anthropic **Natural Language Autoencoders (NLA)**, open re-impl `kitft/natural_language_autoencoders` (+ `kitft/nla-inference`), with **pretrained checkpoints for Qwen2.5-7B-Instruct at L20** — the *exact* model + layer of exp04.

---

## 1. One-paragraph thesis

exp04 showed that the KAPPA closed-form residual edit (steering) pushes the L20 activation **off-manifold** once the relative edit size `‖Δh‖/‖h‖ ≳ 1`, and that this off-manifold-ness *degrades* task accuracy (a clean inverted-U in α). That off-manifold signal was characterized only by **magnitude** (`‖Δh‖/‖h‖`). The NLA gives us a second, *independent, directional* probe of "naturalness": round-trip an activation through the **Activation Verbalizer → text → Activation Reconstructor** and measure the **fraction of variance explained (FVE)**. Because the NLA was trained only on genuine model activations, an off-manifold input should verbalize+reconstruct **worse** (lower FVE). We will (Component 1) measure FVE as a function of steering strength α and test whether it tracks exp04's accuracy-degradation curve, and (Component 2) collect the **natural-language descriptions** of the original and steered activations. The end goal these enable: a **language-space steering** method that edits the *text* minimally and reconstructs an on-manifold activation via the AR (which, by construction, emits a genuine-model activation), sidestepping the off-manifold blow-up entirely.

---

## 2. Why this is tractable (key facts established during research)

| Fact | Value / source |
|---|---|
| NLA pretrained for our exact target | `kitft/nla-qwen2.5-7b-L20-av` + `…-ar` (public, Apache-2.0, HF Hub) |
| NLA base / layer | base `Qwen/Qwen2.5-7B-Instruct`, extraction = HF `hidden_states[20]`, `d_model=3584` |
| **Layer-index alignment** | exp04 `h_layer20.npy` **is** HF `hidden_states[20]` (cache contract §4 BUCKET.md) → **identical** to the NLA's extraction point. ✅ |
| NLA in-distribution quality | `fve_nrm = 0.752` (training set, from AV model card); paper-wide NLAs reach **0.6–0.8 FVE** |
| NLA I/O | input = **unit-L2-normalized** `[3584]` fp32; round-trip `MSE = 2(1−cos)`; AV injects vector as a token embedding → autoregresses text (T=1); AR = Qwen truncated to first ~20 layers + learned `Linear(d,d)` head at final token |
| Inference path | `kitft/nla-inference` (`NLAClient`/`NLACritic`) over an **sglang** server; each ckpt ships `nla_meta.yaml` (prompt template, injection token ids, scale factor) — **load it, never hardcode** |
| Steering math | closed form, CPU: `z=W_know h+b_know; s=W_pred h+b_pred; r=αz−s; P=W_predᵀ(W_pred W_predᵀ)⁻¹; h'=h+P r` (β=0) — `exp04/kappa/kappa_edit.py` |
| Inputs already local | `exp04/05_out_pulled/03_kappa/emb/{h_layer20_orig,h_layer20_steered_a2,h_layer20_steered_a10}.npy` (93.7 MB ea.), all L20 probes (`02_probes/example_level/{know,pred}/layer20.npz`), `example_ids.json`, `03_kappa/diag/sweep.json`, `05_out/summary.json` |
| exp04 L20 metrics (witnesses) | base ACC 0.660, know 0.854, pred 0.912 (val 0.901 → KAPPA-eligible), AGR 0.649, ΔACC +0.194; `best_know_layer=20` |
| exp04 α↔ratio↔ACC (single L20) | α30→ratio2.01→ACC0.600; α20→1.33→0.686; α10→0.66→0.715; α5→0.32→0.689; α2→0.12→0.672; α1→0.05→0.674 (base 0.670) |
| Storage | reuse existing 30 GB volume `iaxphg9saj` (US-KS-2, ~1.5 GB used) under a new `nla/` prefix. **No resize needed.** |

---

## 3. Scientific hypotheses & success criteria

- **H1 (off-manifold ↔ strength).** NLA round-trip FVE **decreases monotonically** with steering strength α (and with `‖Δh‖/‖h‖`).
- **H2 (FVE predicts damage).** FVE(α) is **negatively correlated** with exp04's downstream ACC degradation, i.e. the off-manifold proxy the NLA gives anticipates where steering hurts. (Join our FVE(α) to `sweep.json`'s ACC(α).)
- **H3 (directional vs magnitude).** Because the NLA unit-normalizes, a *pure rescale* `h→c·h` leaves FVE **invariant**; therefore any FVE drop under steering is **directional** off-manifoldness — a signal *complementary* to exp04's magnitude story. (Built-in control.)
- **H4 (enabling the goal).** Verbalizations of steered activations remain **coherent** enough (especially at moderate α) that a *minimal text edit + AR reconstruction* is a viable on-manifold steering route. (Qualitative read of Component-2 data → decision gate, see `04_outlook…`.)

"Success" for this stage = Components 1 & 2 produce clean, validated FVE(α) curves + NL corpora with the calibration gate passed (FVE(orig) ≈ 0.7–0.75 on our distribution). The language-space steering method itself is the *next* stage, gated on this data.

---

## 4. Architecture / data flow

```
              LOCAL (CPU)                         GPU POD (Runpod, sglang)                 DURABLE (S3 vol iaxphg9saj)
  h_layer20_orig.npy ─┐                                                                    nla/
  L20 probes (know,pred)│  Phase A: closed-form          AV (verbalizer)  AR (reconstructor)  ├─ inputs/  steered α-sweep .npy + subset idx
                        ├─► h'(α) for α∈grid ──push──►  ┌───────────────┐ ┌───────────────┐  ├─ nl/      orig.parquet, steered_a{α}.parquet
  example_ids.json ─────┘   + per-row norms             │ unit-norm h   │ │ text z        │  ├─ recon/   ĥ(α) subset .npy  (for the goal)
                                                        │ → text z      │ │ → ĥ (unit)    │  ├─ fve/     fve_by_alpha.json (+ join ACC(α))
                                  Phase C gate ◄─────────┴───────────────┘ └───────────────┘  └─ report/  report.md, manifest.json
                                  (indexing + FVE(orig)≈0.75)
                                  Phase D: FVE(α)   Phase E: NL(orig), NL(steered α)
```

---

## 5. Locked decisions (with rationale)

1. **Single-layer L20 edit only.** The NLA sees only L20; the L20 vector after a *single-layer* L20 edit is exactly the closed form, and it reproduces the stored `a2/a10` arrays (a built-in correctness check). Multi-layer KAPPA edits (21–25) change *downstream* layers, not the L20 vector, so they are irrelevant to an L20 NLA. → study the single-layer L20 family.
2. **α grid:** `{0, 0.5, 1, 2, 3, 5, 7, 10, 15, 20, 30}` (11 pts). `α=0` = identity sanity; the grid straddles exp04's inverted-U (peak ~α10, collapse by α30) and the `ratio` range 0.05→2.0.
3. **Reuse the existing volume + `nla/` prefix; do NOT make a new bucket.** Total new data ≲ a few GB ≪ 28 GB free. Respect Runpod S3 gotchas (no delete; size-aware skip → **versioned key names**, or raw `upload_file`).
4. **GPU: one 48 GB card (A40 / A6000, COMMUNITY, target <$0.50/hr)** so AV+AR are co-resident and the round-trip is a single pass. Fallback: A5000 (24 GB) two-phase (serve AV, then AR). Per `/runpod-ctl`: verify `gpu list` id, BUDGET row, `--terminate-after` backstop, teardown invariant.
5. **Scale (locked):** FVE sweep on a **paired 1024-row** subset across all 11 α (same rows → paired stats). Original NL on **all 6536 rows** (user-selected; full store-of-record). Steered NL on the 1024 subset for all α **plus** full-test at headline α∈{2,10,30}.
6. **Primary metric = mean cosine / mean MSE per α** (unambiguous); **FVE reported with a fixed denominator** = variance of the *original* cohort (so FVE(α) is directly comparable across α). Definitions in `01_…`.
7. **Read `nla_meta.yaml` + `nla-inference` docs on-pod before any inference**; do not hardcode prompt/scale/normalization. The calibration gate (Phase C) is mandatory before spending on the full sweep.

---

## 6. Risks & mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| Normalization/scale/layer convention mismatch silently tanks FVE | med | **Phase C gate**: re-derive activations from text, match exp04 cache; require FVE(orig)≈0.7–0.75 before sweeping. Use `nla_meta.yaml` scale factor verbatim. |
| Distribution shift: NLA trained on generic text, exp04 acts are MCQ last-token | med | FVE(orig) on *our* distribution is the baseline we compare against (not the card's 0.752); report it explicitly. Expect some drop; only relative FVE(α)−FVE(orig) is load-bearing. |
| NLA unit-norms away the magnitude signal | high (by design) | This is *expected*; H3 control turns it into a feature (isolates directional off-manifoldness). Keep exp04's `‖Δh‖/‖h‖` alongside. |
| AV generation cost (hundreds of tokens × ~19k calls) | med | Subsample (1024) for the dense sweep; batch via sglang; cap with budget. Est. ~3–4 GPU-h ≈ $1.5–2 on A40. |
| HF checkpoint gating / download size | low | Qwen + `kitft/*` are public; set `HF_TOKEN` if needed; cache under `/workspace/hf-cache`. |
| Runpod S3 no-delete / size-skip overwrites | low | Versioned keys (`…_v1`), or raw `s3.upload_file` (always overwrites); never rely on `s3_sync push` to replace a same-size key. |

---

## 7. Phase plan (resumable checklist)

- [x] **A — Steered α-sweep (local/CPU).** Closed-form `h'(α)`; validate vs stored a2/a10; emit norms. → `01_…` §A. *No GPU.* **DONE 2026-05-27:** `src/steer_sweep.py`; gate GREEN (a2 max_abs=0.0, a10 max_abs=4.4e-16; rank=3/pinv; mean_h_norm=86.6916). Outputs in `inputs/`.
- [x] **B — Provision pod + NLA stack.** **DONE 2026-05-28:** A6000 48GB pod; sglang **0.5.6** (pinned — `>=0.5.6` pulled a broken sm100/cu130 build; see CHANGELOG) + AV/AR checkpoints. Inputs scp'd local→pod (Runpod S3 unreachable from pod).
- [x] **C — Calibration gate.** **DONE 2026-05-28: PASSED**, mean cos(orig)=**0.726** on 1024 rows (gate 0.6–0.8; AV card 0.752). Validated the reconciled API end-to-end.
- [x] **D — Component 1: FVE(α).** **DONE 2026-05-28:** sweep 1024×11 + rescale H3. **H1** ρ(cos,α)=−0.991; **H2** pearson(OME,ACCdrop)=0.59 (nuanced); **H3** scale-invariant. `out/fve/{fve_by_alpha,per_row,rescale_control,analysis}` + 11 recon arrays.
- [x] **E — Component 2: NL collection.** **DONE 2026-05-28:** orig 6536 + steered 11×1024 + headline 2615×{2,10,30}; 0 skips. `out/nl/*.parquet`. **H4**: NL coherent ≤α10, lost by α30.
- [x] **F — Persist + teardown.** **DONE 2026-05-28:** pushed `nla/{fve,nl,recon,report}` to S3 + mirrored `out/`; pod deleted (`pod list` clean); run ≈ $2.61; BUDGET row closed.
- [ ] **G — (next stage) Language-space steering.** Decision gate + method. → `04_…`. **Gate signal GREEN:** H4 shows verbalizations stay editable through α≲10 (cos≳0.59), supporting a minimal-text-edit + AR-reconstruction route.

> **Phase-A decision logged (2026-05-27):** α=0 is implemented as the *no-edit identity anchor* (h==h_orig, ‖Δh‖=0), matching the plan's stated semantics (FVE(0)=FVE(orig); α=0 NL reuses orig text). The *literal* closed-form at α=0 would instead be r=−s → h−s·Pᵀ (it zeroes the prediction logits), which is NOT the intended baseline; the edit is applied only for α>0. Also: the production edit uses the **pseudo-inverse** (Gram rank=3, cond≈2e16>1e6 → used_pinv=True), correcting the "here well-conditioned" aside in `01_…`§A; the a2/a10 gate confirms fidelity (max_abs 0.0 / 4.4e-16).
>
> **On-pod validation RESOLVED (2026-05-28):** the documented API was wrong. Live `nla_inference.py` uses `NLAClient.generate(...)` (not `verbalize`), `NLACritic(dir, device=...)` **pure-torch** (not sglang), and `score()→(mse,cos)` tuple. `src/nla_run.py` was reconciled on-pod (single sglang server = AV; in-process GPU AR) + threaded for throughput + made resilient to transient sglang disconnects. The calibrate STOP-gate (mean cos 0.726) confirmed the fixes before the sweep. Full details in `CHANGELOG.md`.

## 8. Budget

Standing storage unchanged (~$2.10/mo, existing volume). Compute: 1× 48 GB GPU, est. **3.5–4.5 h @ ≤$0.50/h ≈ $1.8–2.3** (orig NL = all 6536); hard ceiling **$5** for this stage. One BUDGET.md row per `/runpod-ctl` + `/budget` conventions; `--terminate-after now+3h` backstop.

## 9. Detailed specs
- `01_component1_offmanifold_fve.md` — steered generation, round-trip, metrics, calibration gate, analysis.
- `02_component2_nl_collection.md` — verbalization scope, schemas, dedup/provenance.
- `03_storage_and_runpod.md` — S3 layout, key hygiene, pod orchestration, budget.
- `04_outlook_language_space_steering.md` — the end-goal method + decision gate.
