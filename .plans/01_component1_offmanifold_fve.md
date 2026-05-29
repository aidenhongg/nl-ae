# Component 1 — Off-manifold error vs. steering strength (NLA reconstruction FVE)

Goal: (1) generate steered L20 activations across a grid of steering factors and store them; (2) measure NLA round-trip reconstruction error / FVE per steering factor, and test whether it tracks exp04's accuracy-degradation curve.

---

## A. Generate the steered activation α-sweep  (LOCAL / CPU — no GPU, no S3)

**Inputs (all already on disk):**
- `exp04/05_out_pulled/03_kappa/emb/h_layer20_orig.npy` → `(6536, 3584)` float32.
- `exp04/05_out_pulled/02_probes/example_level/know/layer20.npz` and `…/pred/layer20.npz` → each `{W:[4,3584], b:[4]}`.
- `exp04/05_out_pulled/03_kappa/emb/example_ids.json` → row order + `test_row_indices` (example_level test, n=2615).
- (validation targets) `h_layer20_steered_a2.npy`, `h_layer20_steered_a10.npy`.

**Edit operator (fp64, per `kappa_edit.py`):**
```
Wp, bp = pred.W, pred.b      # [4,3584],[4]
Wk, bk = know.W, know.b
G  = Wp @ Wp.T               # [4,4] non-augmented Gram
P  = Wp.T @ inv(G)           # [d,4]   (pinv if rank<4 or cond>1e6; here well-conditioned)
def steer(h, a):             # h:[N,d]
    z = h @ Wk.T + bk        # [N,4] knowledge logits
    s = h @ Wp.T + bp        # [N,4] prediction logits
    r = a*z - s              # beta=0, sign term drops
    return h + r @ P.T       # h'(a)  [N,d]
```

**Procedure (`src/steer_sweep.py`):**
1. Load orig (fp32→fp64), probes (fp64). Build `P`.
2. For `a` in `{0,0.5,1,2,3,5,7,10,15,20,30}`: `h_a = steer(h_orig, a)`; save `nla/inputs/h_layer20_steered_a{a}.npy` (fp32, atomic write).
   - `a=0` must equal `h_orig` (identity sanity, ‖Δh‖=0).
3. **Validation (hard gate):** assert `allclose(steer(h_orig,2.0), load(a2))` and `…(…,10.0), load(a10))` to fp32 tolerance (`atol=1e-3, rtol=1e-3`). A mismatch means probe/orientation drift → STOP. (Also re-run the exp04 orientation witness: `argmax(h@Wk.T+bk)` vs ground-truth ≈ know_acc 0.854; `pred` ≈ 0.912.)
4. **Per-row diagnostics** → `nla/inputs/norms.parquet` columns: `example_id, row_index, split(train/val/test), alpha, h_norm=‖h‖, dh_norm=‖Δh‖, ratio=‖Δh‖/‖h‖, cos_h_hp=cos(h,h')`. (These connect FVE to exp04's `ratio` predictor and let us slice test-only.)
5. Push `nla/inputs/` to S3 (versioned keys). Keep local copies in `NLA-final/inputs/`.

Output of Phase A: 11 steered arrays + `norms.parquet`, validated against exp04.

---

## B. NLA round-trip (GPU pod)  — see `03_…` for serving

The round-trip per activation `h`:
```
ĥ = AR( AV( normalize(h) ) )          # text z = AV(...);  ĥ = AR(z)
```
- **Normalization:** unit-L2 on `h` (and apply the `nla_meta.yaml` `scale` factor exactly as the client does — confirm whether `NLAClient` normalizes internally; if so, pass raw and let it).
- AV at **T=1** (stochastic) → fix `seed` and record it; optionally also greedy (T=0) for a deterministic companion. We keep T=1 (paper setting) as primary, seed-logged.
- AR returns `ĥ` at (or near) unit norm. Persist `z` (text), `ĥ`, and `cos(h,ĥ)`.

Use the **same paired 1024-row subset `R`** (drawn from example_level **test** rows, fixed seed) for *every* α so comparisons are paired. Persist `R` as `nla/inputs/subset_rows.json`.

---

## C. Calibration gate (MANDATORY before the full sweep)

Cheap checks that catch convention bugs before we spend on 11 α × 1024:
1. **Indexing match.** Take ~16 example_ids from the subset, rebuild their prompts (`exp04/data/prompts.jsonl` or `kappa/prompt.py`), run base Qwen, grab `hidden_states[20]` at the last (left-padded) token; assert `cos(reconstructed_from_text, exp04_cache_row) > 0.99` (bf16 tolerance). Confirms our L20 == NLA's L20 and the token/pad convention.
2. **FVE(orig) sanity.** Round-trip the 1024 **original** rows; require **mean `cos(h,ĥ)` and FVE in ≈0.6–0.8** (card in-dist is 0.752; our MCQ distribution may sit a bit lower). If FVE ≈ 0 or negative → normalization/scale/layer bug → **STOP, fix, re-gate.** Do not proceed to D/E until green.

Record the gate result in `nla/report/calibration.json`.

---

## D. Metrics & analysis (FVE(α))

Let `R` be the subset, all vectors unit-normalized. Per α:
- **Per-row:** `cos_i = cos(h_i(α), ĥ_i(α))`, `mse_i = ‖h_i−ĥ_i‖² = 2(1−cos_i)`.
- **Primary (unambiguous):** `mean_cos(α)`, `mean_mse(α)` with bootstrap 95% CI over rows (paired across α).
- **FVE (paper-aligned, fixed denominator for comparability):**
  ```
  FVE(α) = 1 − mean_i ‖h_i(α) − ĥ_i(α)‖²  /  Var0 ,   Var0 = mean_i ‖h_i(0) − h̄0‖²
  ```
  where `h̄0` = mean of the unit-normalized **original** subset (`Var0 = 1 − ‖h̄0‖²`). Fixed `Var0` makes FVE(α) a clean monotone read. (Also report the per-α self-denominator FVE for reference.)
- **Off-manifold error:** `OME(α) = 1 − FVE(α)` (and `1 − mean_cos`).

**Joins / tests:**
- **H1:** regress `FVE(α)` and `mean_cos(α)` on α and on `ratio(α)` (from `norms.parquet`, subset-averaged). Expect negative slope; report Spearman ρ.
- **H2 (headline):** join `FVE(α)` to exp04 **ACC(α)** from `exp04/05_out_pulled/03_kappa/diag/sweep.json` (single-layer L20, example_level). Correlate `OME(α)` vs `(base_acc − acc(α))`. A positive correlation ⇒ the NLA off-manifold proxy predicts steering damage. Scatter + ρ in the report.
- **H3 (control):** pick α with strong drop; rescale its inputs `h→c·h` for `c∈{0.5,1,2}`, round-trip; confirm FVE is invariant to `c` (≤ noise). Proves the FVE drop is **directional**, not magnitude — orthogonal to exp04's `ratio`.

**Outputs:**
- `nla/fve/fve_by_alpha.json` — per α: n, mean_cos, mean_mse, FVE (fixed + self), CIs, ratio(α), exp04 acc(α), OME(α).
- `nla/fve/per_row.parquet` — per (α,row): example_id, cos, mse.
- `nla/recon/h_recon_a{α}.npy` — reconstructed `ĥ` for the subset (needed by the language-space-steering stage).
- Figures (in report): FVE vs α; FVE vs ratio; OME vs ACC-drop.

---

## E. Compute / cost
1024 rows × 11 α ≈ 11.3k AV generations + 11.3k AR reconstructions. AR is cheap (one truncated forward). AV dominates (hundreds of tokens each). Est. ~1.5–2.5 GPU-h on A40 for D alone; shares the same pod as Component 2.

## F. Definition of done (Component 1)
- 11 validated steered arrays + norms persisted (A green incl. a2/a10 reproduction).
- Calibration gate green (C).
- `fve_by_alpha.json` with H1/H2/H3 results + figures; reconstructions saved.
