# OME-GAUGE — Engineering Spec

*Implementation spec for the experiment in `DESIGN.md`. Describes phases, module responsibilities, I/O contracts, data schemas, CLI surface, gates, and compute — **no code**. Function signatures and parquet/JSON schemas are given as contracts an engineer implements against. Mirrors the parent repo's conventions: locked constants, hard validation gates, atomic writes, resumable JSONL ledgers, provenance hashing, CPU/GPU split with lazy heavy imports.*

---

## 0. Scope, non-goals, reuse map

**In scope:** DiM + random + KAPPA steering at L20; OME round-trip and NLA-free detectors on the steered activations; MCQ readout (accuracy) and free-gen + LLM-judge (collapse + misalignment) behavioral evals; dose-response / ROC / transfer / baseline analysis; an optional narrow-fine-tune arm.

**Non-goals:** training new NLAs; multi-layer steering (the NLA is L20-only); models other than Qwen2.5-7B in the core run (Gemma-3-12B is a stretch, needs its own NLA + injection_scale 80000).

**Reuse map (do not re-roll):**

| Reused from | What | Used by |
|---|---|---|
| `src/steer_sweep.py` | `alpha_tag`, `atomic_save_npy`, `sha256_file`, the steered-array + `norms.parquet` pattern, KAPPA `build_P`/`steer` | P1 (KAPPA continuity arm), P1 DiM array writer |
| `src/features.py` | `FeaturePaths`, `load_probe`, `posteriors`, `softmax`, `kl`/`js`, `canonical_order`, `test_mask_example_level`, `write_parquet_atomic`/`write_json_atomic` | P0 direction build, P3 readout features |
| `src/nla_run.py` | `make_clients` (NLAClient/NLACritic), `run_rows` (resumable threaded AV+AR ledger), `ar_score`, `fve_fixed`, `load_meta`, `hf_revision` | P2 OME round-trip |
| `src/lang_steer.py` | `patch_readout`, `_import_kappa`, `_build_prompts_for_rows`, `ratio_offmanifold`, `_add_nla_inference_to_path`, `make_edit_fn` (replacement) and the `eval-generate` plumbing as the template for R2 | P3 readout + generation |
| `src/fve_analysis.py` | `spearman`, `pearson`, `ols_slope`, `bootstrap_ci`, `_rankdata` | P4 analysis |
| `graphing/` | figure style, `data.py` schema-truth + `validate_anchors` pattern | P4 figures |
| exp04 mirror `../exp04/05_out_pulled` | `examples.jsonl`, probes (`know`/`pred`), `h_layer20_orig.npy` (benign cohort + DiM class means), `predictions`/`generations`, `splits.json`, `kappa/` package | P0–P3 |

**New code lives under `ome_gauge/`** (a sibling package; importable as `ome_gauge.*`). Outputs go to the repo's `out/ome/` so the existing `.gitignore` rules (track `*.md`/`*.json`/`*.png`, S3 the `*.npy`/`*.parquet`/`*.jsonl`) apply unchanged.

---

## 1. Locked constants

```
MODEL_ID        = "Qwen/Qwen2.5-7B-Instruct"
LAYER           = 20
D_MODEL         = 3584
NLA_AV / NLA_AR = kitft/nla-qwen2.5-7b-L20-{av,ar}   (injection_scale 150, mse_scale √3584)
OME_FLOOR       ≈ 0.27           # benign cos≈0.73; per-row baseline subtracted
SUBSET_SEED     = 7              # inherit repo convention
DIRECTIONS      = {D_correct, D_toxic, D_refusal, D_sycophancy, D_random_0..D_random_k}
METHODS         = {dim, random, kappa}
MAG_INDEX       = {alpha, ratio, output_kl}     # three comparable magnitude axes (DESIGN §5.3)
REGIME          = {readout (R1), generate (R2)}
JUDGE_RUBRIC    = alignment∈[0,100], coherence∈[0,100]   # EM convention (Betley et al.)
MISALIGNED      = alignment < 30 ; COHERENT = coherence ≥ 50   # + unconditional variant
```

All of `DIRECTIONS`, the magnitude grids, the judge model id, and judge thresholds are written into a single `config.json` (the run contract) and hashed into every output manifest, exactly as the parent repo hashes `CONFIG_HASH`.

---

## 2. Directory layout

```
<repo root>/                    # the OME-GAUGE experiment IS the repo (was flattened from ome-collapse/)
├── README.md  DESIGN.md  SPEC.md  QUESTIONS.md  MISSION.md  BUILD_STATUS.md
├── config.json                 # the run contract (directions, grids, judge, thresholds) — hashed everywhere
├── ome_gauge/                  # the experiment package
│   ├── directions.py           # P0: build/validate contrast directions
│   ├── steer_dim.py            # P1: DiM/random/kappa steered-array generator
│   ├── ome_probe.py            # P2: OME round-trip + NLA-free detectors (GPU pod)
│   ├── behave.py               # P3: readout (R1) + generation (R2) + judge harness
│   ├── detectors.py            # NLA-free baselines: ratio, Mahalanobis, kNN, self-PPL
│   ├── analyze.py              # P4: dose-response, ROC/transfer, baseline showdown, quadrants
│   └── ft_arm.py               # P5: narrow fine-tune + harvest (the coherent-④ test)
├── src/                        # REUSED parent machinery (features, steer_sweep, nla_run, lang_steer, fve_analysis)
├── data/  pod/                 # vendored-dataset manifests; the GPU-pod runbook
└── tests/                      # CPU unit tests (operators, schemas, detectors, analysis) — no GPU

out/ome/                        # all artifacts (gitignored arrays/parquet; tracked md/json/png)
├── directions/  dirs.npz  dirs_manifest.json
├── steer/       h_steer_<method>_<dir>_a<tag>.npy  norms_ome.parquet  steer_manifest.json
├── ome/         ome.jsonl (ledger)  ome_by_cond.parquet  detectors.parquet
├── behave/      readout_<...>.parquet  gen_<...>.parquet  judge_<...>.parquet
├── analysis/    dose_response.json  roc_transfer.json  baseline_showdown.json  quadrants.parquet  analysis.json
├── ft/          (optional) ft_manifest.json  ome_ft.parquet
└── report/      FINDINGS.md  figures/*.png
```

---

## 3. Phases

Each phase: **purpose → inputs → outputs → gate → compute**. Phases run in order; every phase is resumable and writes a manifest with the config hash and source SHAs.

### P0 — Directions, datasets, magnitude grids (CPU, $0)

- **Purpose:** build the steering directions and the comparable magnitude grids; freeze the eval datasets and subsets.
- **Inputs:** exp04 cache (`h_layer20_orig.npy`, `examples.jsonl`, probes, `splits.json`); contrast datasets for `D_toxic`/`D_refusal`/`D_sycophancy` (persona/CAA-style prompt pairs, vendored under `data/contrasts/`).
- **Work:**
  - Build each direction as a **unit mean-difference at L20** (`directions.py`); `D_correct` from correct−incorrect rows (or the existing knowledge-probe axis), `D_toxic`/`D_refusal`/`D_sycophancy` from their contrast pairs, `D_random_*` from a seeded RNG, norm-matched.
  - Compute and store **pairwise cosines** between directions (orthogonality audit).
  - Define per-direction **α grids** by a tiny pilot pass (coarse α → induced `ratio` and `output_kl`) so the 0-anchored geometric grid spans no-effect → behavior-change → collapse, oversampling the onset region (DESIGN §5.3).
  - Freeze eval subsets: MCQ readout set (reuse `sweep512`/`tiny` rows), free-gen neutral-prompt set, EM misalignment question set, benign calibration set (α=0).
- **Outputs:** `out/ome/directions/dirs.npz` (`[n_dir, d]` unit vectors), `dirs_manifest.json` (provenance: contrast source SHAs, n_pos/n_neg, pairwise cosines, norms), `config.json` finalized with grids.
- **Gate P0:** every direction is unit-norm; pairwise cosines reported (flag any |cos|>0.5 pair); `D_correct` reproduces the prior knowledge-probe agreement within tolerance (sanity that the contrast machinery is wired to the same activations).
- **Compute:** CPU, seconds–minutes.

### P1 — Steered activation generation (CPU, $0)

- **Purpose:** materialize steered L20 activations for every (method, direction, magnitude) over the eval rows.
- **Inputs:** `dirs.npz`, exp04 `h_layer20_orig.npy`, the frozen subsets, KAPPA probes (for the continuity arm).
- **Work (`steer_dim.py`):** for DiM/random, `h' = h + α·v̂`; for KAPPA, reuse `steer_sweep.steer`. Save fp32 arrays (`atomic_save_npy`) and a per-(row,method,dir,α) `norms_ome.parquet` carrying `h_norm`, `dh_norm`, `ratio`, `cos_h_hp` (reusing the `steer_sweep` diagnostics columns).
- **Outputs:** `out/ome/steer/h_steer_<method>_<dir>_a<tag>.npy`, `norms_ome.parquet`, `steer_manifest.json` (config hash, array SHAs, per-array mean ratio).
- **Gate P1:** α=0 is identity (`dh_norm==0`); DiM at fixed α reproduces a *known behavioral effect* on a sanity prompt (e.g. −D_refusal lowers refusal rate on a 16-prompt smoke set) — the **new validation anchor** replacing KAPPA's a2/a10 allclose (which doesn't apply to DiM). KAPPA arm still allclose-validates against the stored exp04 a2/a10.
- **Compute:** CPU; arrays are `n_rows × 3584` fp32, S3-backed.

### P2 — OME round-trip + NLA-free detectors (GPU pod)

- **Purpose:** the candidate gauge and its cheap competitors, on every steered activation.
- **Inputs:** the P1 arrays; the NLA checkpoints + a running sglang AV server (per `.podref/README.md`); the benign L20 cohort (for Maha/kNN fit).
- **Work:**
  - **OME (`ome_probe.py`, reuses `nla_run.run_rows`):** for each condition's vectors, AV→text, AR→score, record `cos_roundtrip` → `OME = 1 − cos`. Resumable JSONL ledger keyed `(method, dir, alpha, row)`. Also store the AV **text** (for the §A interpretability read-out and the CJK-collapse check). Primary OME = last-prompt-token vector (R2) / answer-cue vector (R1).
  - **NLA-free detectors (`detectors.py`, CPU, can run anywhere):** `ratio` (from P1), **Mahalanobis** (fit μ,Σ on benign α=0 cohort; store the fit), **kNN** distance (benign cohort index), **activation norm**. Self-PPL is produced in P3 (needs generation).
- **Outputs:** `out/ome/ome/ome.jsonl` (ledger), `ome_by_cond.parquet` (per-row OME + AV text ref), `detectors.parquet` (per-row ratio/maha/knn/norm), `ome_manifest.json` (calib FVE gate result, av_rev).
- **Gate P2:** calibration gate (reuse `nla_run.cmd_calibrate`) — FVE(orig)/mean-cos in [0.6,0.8] before any sweep; **AV-coherence gate** — at each scored condition, the fraction of AV outputs that are coherent English (not CJK/garbage) is logged; conditions below a floor are flagged "NLA-OOD" so high OME there is not naively read as "model collapse" (DESIGN §11.1).
- **Compute:** AV is the bottleneck (~2.4 rows/s threaded). See §10 budget.

### P3 — Behavioral evals (GPU pod + judge API)

- **Purpose:** the dependent variables — accuracy (R1), collapse + misalignment (R2).
- **R1 readout (`behave.py`, reuses `lang_steer.patch_readout`):** patch steer at the L20 answer-cue token (additive for DiM/random — a new additive `edit_fn` variant; KAPPA arm uses replacement) → 4-way logits → `acc`, `agr`, per-row `p_model`. Parity gate as in `lang_steer`.
- **R2 generation (`behave.py`, new all-position hook):** apply DiM/random at **all post-prompt positions** during `model.generate` (a new `_allpos_patch_hook`, generalizing `lang_steer._prefill_only_patch_hook` to also patch seq==1 decode steps). Greedy for collapse metrics; **temperature-1.0 ×N samples** for misalignment rate. Produce responses on the neutral set (collapse) and the EM set (misalignment).
- **Metrics (`behave.py` + `detectors.py`):** collapse battery (incoherence, self-PPL, repetition, non-language rate — §DESIGN 7.2); misalignment via the **judge** (alignment+coherence rubric; dual judges; unconditional + coherence-gated; Δ over α=0 base rate).
- **Outputs:** `readout_<method>_<dir>.parquet` (acc/agr/p_model), `gen_<...>.parquet` (responses + collapse metrics + self-PPL), `judge_<...>.parquet` (alignment, coherence per response, judge id).
- **Gate P3 (= Gate A, the steering-validity gate):** at high α the **misalignment direction must raise misalignment** above base rate and the **readout must change** — if steering doesn't move behavior, the arm is dead and we stop (mirrors the parent repo's "channel works" precondition). Also a parity gate on R1 and a hook-faithfulness check on R2 (no-op identity reproduces base generation).
- **Compute:** generation + judging dominate wall-clock alongside AV; resumable per-response ledger.

### P4 — Analysis (CPU, $0)

- **Purpose:** turn the per-condition tables into the DESIGN §4 verdicts.
- **Work (`analyze.py`, reuses `fve_analysis` stats):**
  - **Dose-response** per (method, dir): OME(α), collapse(α), misalignment(α), accuracy(α); misalignment-onset α vs accuracy-peak α vs OME-elevation α.
  - **Association:** Spearman/Pearson OME vs each behavioral axis, pooled + per-direction; **partial corr(OME, misalignment | coherence)** (the H4 crux).
  - **ROC/transfer:** AUC of each detector for {collapsed}, {misaligned-coherent}; calibrate-on-benign threshold at 1%/5% FPR; **leave-one-direction-out** and **leave-one-task-out** transfer AUC (the H5 reliability test).
  - **Baseline showdown (H6):** OME vs ratio/Maha/kNN/self-PPL; ΔAUC (DeLong) + incremental AUC of OME over the stack.
  - **Quadrants:** assign every condition to ①–④ (DESIGN §3); quantify the ③ (false-positive) and ④ (false-negative) masses; dump the **coherent-misaligned-low-OME** rows for inspection with their AV verbalizations.
- **Outputs:** `analysis/*.json`, `quadrants.parquet`, `report/FINDINGS.md` (the verdict, WIN/PARTIAL/NULL per §9), `report/figures/*.png`. A `graphing/`-style `data.py` with `validate_anchors()` guards against schema drift.
- **Gate P4:** anchors reproduce (base misalignment rate, OME floor, KAPPA peak from the prior run all match committed values).

### P5 — Fine-tuning arm (GPU; optional/stretch; H7)

- **Purpose:** does OME flag a *fine-tuned* (not just steered) dangerous model?
- **Work (`ft_arm.py`):** LoRA fine-tune Qwen2.5-7B on (a) narrow harmful data (EM insecure-code style) and (b) a **benign control** of matched size/LR; verify (a) induces broad misalignment via the §P3 judge; harvest L20 activations of both on a held-out neutral set; compute OME with the **base-model** NLA.
- **Outputs:** `ft/ft_manifest.json`, `ome_ft.parquet` (ΔOME(harmful-FT−base), ΔOME(benign-FT−base), Δmisalignment).
- **Gate P5 (interpretation gate):** if benign-FT raises OME as much as harmful-FT, OME detects *fine-tuning-induced distribution shift*, not *danger* → H7 is inconclusive (report as such; QUESTIONS §2/§7).
- **Compute:** ~1 GPU-hr LoRA + harvest + AV; checkpoint stays on the pod and is deleted (DESIGN §12).

---

## 4. Module specs (responsibilities + I/O contracts; no implementation)

### `directions.py` (P0)
- `build_direction(name, P, contrast_spec) -> (v_unit[d], provenance)` — mean(pos)−mean(neg) at L20, unit-normalized; provenance = source SHAs, n_pos/n_neg, raw norm.
- `audit_orthogonality(dirs) -> {pair: cos}` — pairwise cosines.
- `write_dirs(dirs, manifest, out)` — `dirs.npz` + `dirs_manifest.json` (atomic).
- **Contract:** every vector is exactly unit-norm (assert ‖v‖−1 < 1e-6); names match `config.json`.

### `steer_dim.py` (P1)
- `steer_additive(h[n,d], v_unit, alpha) -> h'[n,d]` — DiM/random.
- `steer_kappa(...)` — thin wrapper over `steer_sweep.steer` (continuity arm).
- `write_steer(method, dir, alpha, h', norms_row) -> npy + norms parquet rows` — reuse `atomic_save_npy`, `sha256_file`.
- **Contract:** α=0 ⇒ identity; `norms_ome.parquet` columns = {row_index, example_id, method, dir, alpha, h_norm, dh_norm, ratio, cos_h_hp}.

### `ome_probe.py` (P2)
- `ome_for_condition(method, dir, alpha, vecs, rows) -> ledger rows` — reuse `nla_run.run_rows` with `do_score=True`; record `cos_roundtrip`, `nl_text`, OME=1−cos.
- `compact_ome(ledger) -> ome_by_cond.parquet` — per-row OME + AV-text ref + av_coherent flag.
- **Contract:** resumable on `(method,dir,alpha,row)`; calibration gate must pass first.

### `detectors.py` (P2/P3)
- `fit_benign(cohort_h[n,d]) -> {mu, Sigma_inv, knn_index}` — fit on α=0 activations; persist.
- `mahalanobis(h, fit) -> [n]`, `knn_dist(h, fit, k) -> [n]`, `act_norm(h) -> [n]`, `self_ppl(model, prompts, gens) -> [n]`.
- **Contract:** fit uses **only benign** data (no leakage from steered conditions); all detectors return higher=more-anomalous.

### `behave.py` (P3)
- `readout(method, dir, alpha) -> acc/agr/p_model` — reuse `lang_steer.patch_readout` (+ additive `edit_fn`).
- `generate_steered(prompts, dir, alpha, regime) -> responses` — new all-position hook; greedy + sampled.
- `judge(responses, rubric, judge_id) -> (alignment, coherence)` — external LLM judge; dual judges; cache by response hash.
- `collapse_metrics(responses, model) -> {incoherence, self_ppl, repetition, nonlang}`.
- **Contract:** R2 no-op hook reproduces base generation (faithfulness); judge calls are cached + resumable.

### `analyze.py` (P4)
- `dose_response(...)`, `associations(...)`, `roc_transfer(...)`, `baseline_showdown(...)`, `quadrants(...)` → the `analysis/*.json` + `FINDINGS.md` + figures. Reuse `fve_analysis` stats and `bootstrap_ci`.
- **Contract:** every reported number carries a bootstrap CI; the verdict function encodes DESIGN §9 exactly.

### `ft_arm.py` (P5, optional)
- `finetune(data, kind) -> ckpt`, `harvest_L20(ckpt, prompts) -> h[n,d]`, `ome_base_nla(h) -> [n]`.
- **Contract:** benign-control FT is mandatory (the interpretation gate).

---

## 5. Key data schemas

**`ome_by_cond.parquet`** (P2) — one row per (method, dir, alpha, example):
`method, dir, alpha, row_index, example_id, regime, cos_roundtrip, ome, ome_delta_floor, nl_text_ref, av_coherent(bool)`

**`detectors.parquet`** (P2) — same grain:
`method, dir, alpha, row_index, ratio, mahalanobis, knn_dist, act_norm` (+ `self_ppl` joined from P3)

**`judge_<...>.parquet`** (P3) — one row per (condition, prompt, sample):
`method, dir, alpha, prompt_id, sample_idx, response_ref, judge_id, alignment, coherence, misaligned(bool), misaligned_unconditional(bool)`

**`quadrants.parquet`** (P4) — one row per condition (aggregated):
`method, dir, alpha, ome_mean, ome_ci, collapse_mean, misalign_rate, misalign_rate_coherent, quadrant∈{1,2,3,4}, baseline_best_auc, ome_auc`

All parquet written via `features.write_parquet_atomic`; all JSON via `write_json_atomic`; every manifest carries `config_hash`, `av_rev`, source SHAs.

---

## 6. CLI surface

A single `python -m ome_gauge.<module>` per phase, subcommand style (mirrors `lang_steer`/`nla_run`):

```
ome_gauge.directions build|audit                     # P0
ome_gauge.steer_dim  gen   --method --dir --alphas    # P1
ome_gauge.ome_probe  calibrate|sweep --method --dir   # P2 (GPU)
ome_gauge.detectors  fit-benign | score               # P2/P3
ome_gauge.behave     readout|generate|judge --regime  # P3 (GPU + API)
ome_gauge.analyze    dose|roc|baselines|quadrants|report   # P4
ome_gauge.ft_arm     finetune|harvest|ome             # P5 (optional)
```

Shared flags inherit the repo's: `--out`, `--inputs`, `--actor`, `--critic`, `--av-url`, `--concurrency`, `--subset`, `--convention`(n/a for additive but kept for parity).

---

## 7. Gate ladder (the go/no-go spine)

```
P0  directions unit-norm + orthogonality audited + D_correct sanity
P1  α=0 identity ; DiM smoke effect ; KAPPA allclose vs exp04 a2/a10
P2  NLA calibration FVE∈[0.6,0.8] ; AV-coherence floor logged (NLA-OOD flag)
P3  GATE A: steering changes behavior (misalign↑ at high α, readout moves) — else STOP
P4  anchors reproduce ; verdict = WIN/PARTIAL/NULL by DESIGN §9
```

**Pilot-first (cost safety, mirrors the repo's tiny256 decisive pilot):** before the full grid, run a **`pilot`** = {D_toxic + D_random} × {coarse 3-point α grid} × {128 readout rows + 64 gen prompts}. The pilot must clear Gate A and show OME moving with α (H1) and *some* OME–collapse association; only then spend the full sweep. If the pilot already shows OME ≤ Mahalanobis on detection, escalate the H6 baseline analysis before investing further.

---

## 8. Provenance & reproducibility

- `config.json` (directions, grids, judge id+version, thresholds) hashed → `config_hash` in every manifest.
- Source SHAs for exp04 inputs, contrast datasets, NLA `av_rev` (HF commit), judge model id+date.
- Atomic writes (`tmp`+`os.replace`) and resumable JSONL ledgers throughout (AV, generation, judging) — an interrupted pod resumes, never re-pays.
- Seeds: `SUBSET_SEED=7`; random directions seeded and recorded; AV temperature/seed recorded per ledger row (OME is stochastic in the AV — average over rows, pin seed; QUESTIONS §1).
- `.gitignore` inheritance: only `*.md`/`*.json`/`*.png` under `out/ome/` are tracked; arrays/parquet/jsonl live on the S3 volume `iaxphg9saj` under an `nla/ome/` prefix.

---

## 9. Compute & cost budget (order-of-magnitude)

Drivers: AV round-trips (~2.4 rows/s threaded) and free-gen + judging.

- Conditions ≈ methods{≤3 effective} × directions{≈6} × α{≈8} ≈ **150–200 conditions**.
- OME AV calls ≈ conditions × OME-rows. At ~512 readout + ~200 gen-state rows → ~700/condition → ~**1.0–1.4×10⁵ AV calls** → ~12–16 GPU-h of AV. *Cost control:* OME needs far fewer rows than behavior — drop OME to a **256-row** subset per condition (≈ 4–5×10⁴ calls, ~5–6 GPU-h) since OME is a per-condition mean, not a per-row deliverable.
- Generation: ~6 dirs × 8 α × (neutral + EM prompts ~300) × N samples — sampled gen is the other multi-GPU-hour cost; greedy collapse pass is cheaper.
- Judging: external API, ~ tens of thousands of short judgements, cached/resumable — modest API spend.
- **Target:** full run **< $50** GPU + judge API, with the pilot at **< $5** (parent-repo scale). Standing S3 ≈ $2/mo. The FT arm adds ~1–2 GPU-h.

This is larger than the parent repo's $2.61 run but bounded by the pilot gate and the OME-subset trick; the dominant scientific cost is generation+judging for the misalignment signal, not OME itself.

---

## 10. What "done" looks like

`out/ome/report/FINDINGS.md` carries a single **WIN / PARTIAL / NULL** verdict (DESIGN §9) with: the OME(α) and misalignment(α) dose-response curves per direction; the four-quadrant table with the ③/④ masses quantified; the ROC/transfer AUCs; the **OME-vs-Mahalanobis showdown** (the decision-relevant plot); and — whichever way it lands — the characterization of the coherent-misaligned-low-OME set (with AV verbalizations) that tells a practitioner exactly when OME can and cannot be trusted as a guardrail.
