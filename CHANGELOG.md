# CHANGELOG — NLA-final

## 2026-05-29 — MAIN-EXP language-space steering: GPU run → Phase-1 lever test NULL (channel can't transmit)

Ran the paid GPU phase of the language-space steering experiment (MAIN-EXP.md). User authorized full
scope; the **Phase-1 lever test returned a comprehensive NULL**, firing the §1b STOP gate, so Phases 2–4
(ladder / headline / OME) did **not** run — the cheap go/no-go did its job and saved the bulk of the budget.

### Pod / cost
- **`n3rftfbvhkcrtq`** — NVIDIA A100 80 GB PCIe (SECURE, RO), $1.39/hr, **~0.65 GPU-h ≈ $0.90.** Deleted;
  `pod list` clean. The planned ≤$0.50/hr 48 GB cards (A40/A6000/L40S) were out of stock at provision
  time → user chose the A100 (under the $5 ceiling). runpodctl 2.1.9 has **no `--terminate-after`** flag;
  used an **OS-level Windows scheduled task** (`nla-lang-killer`) as the orphan-billing backstop instead,
  removed at teardown. No network volume mounted (scp inputs local→pod; pulled results back via scp).
- Setup reused the hard-won recipe verbatim and it transferred cleanly to **sm80**: `sglang 0.5.6` +
  `sgl-kernel 0.3.18.post2` (bundles Ampere sm80/sm86) + `torch 2.9.1+cu128` imported fine on the A100.

### Result (out/lang/FINDINGS.md, report.md, report_data.json, frontier.png)
- **NULL.** Editing the NLA verbalization and reconstructing through the AR does **not** steer the L20
  readout toward the target. All 6 distinct-mechanism operators on tiny256 (n=256) land within noise of the
  no-edit anchor: E0 0.633 / E1 0.633 / E2 0.645 / E4 0.648 / T1 0.578 / T2 0.547 (ACC), vs base 0.66 and
  know-ceiling 0.854. Best (E4) 95% CI [0.586, 0.707] overlaps the anchor and is below KAPPA's single-L20
  peak (0.715) — n=256 already excludes a win. Free-gen confirms: E0 parses 100%, T1 only 46%.
- **Why:** the channel/plumbing works (parity gate OK on real Qwen; E0 anchor reproduces base ACC, clearing
  the §8 "AR too lossy" risk), but the NLA verbalization is **descriptive, not directive** — it encodes what
  the activation is *about*, not the first-person answer-commitment that drives the readout. Surgical edits
  (E1) barely change ĥ; template edits (T1/T2) move ĥ a lot but inject a *constant* off-target bias (ŷ
  collapses to B for T1, D for T2, regardless of X). Corollary: the deferred L1/L2 SLM-rewrite fallbacks
  can't help (the failure is the channel, not edit quality). Pivot route = prompt-level (`.plans/04_…`).
- **Convention pick (Phase 1a):** anchor gate PASSED for all three; locked **normmatch** (per-row
  directional, H3-consistent) — native overshoots (ratio 1.36, ACC 0.617); normmatch/cohortmean tie
  (ACC 0.633/0.641, ratio 0.74, within noise).

### Bug found + fixed (CPU, $0)
- `src/lang_steer.py reconstruct()` created `out/lang/recon/` but not `out/lang/edits/`, so the
  edits-provenance `pq.write_table` crashed with `FileNotFoundError` (the recon arrays themselves were
  saved first, so eval was unaffected). **Fixed:** `mkdir(parents=True, exist_ok=True)` for `edits/` when
  `save_edits`. Re-shipped to the pod; E1/E4/T2 provenance then wrote cleanly. Trivial/certain → fixed
  directly rather than via an Opus subagent (escalation reserved for complex/ambiguous errors).

## 2026-05-28 — Native per-datapoint features (replaces `feature-patch/`); F2 = the model's ACTUAL generation

Folded the `feature-patch/` prototype into the parent `src/` as a first-class, LOCAL/CPU final stage, and
**corrected Feature 2 to the model's actual generated answer** instead of the `y_tilde` symbol-readout
shortcut (FEATURES.md). **Code + CPU tests only; the one-time bucket re-materialization is gated on the
GPU go** (it needs exp04's new `generate` stage to produce `generations.parquet` — Part B).

### Added
- **`src/features.py`** — consolidates `features_core.py` + `build_features.py` + `steered_divergence.py`.
  F1 (ground truth), **F2 = the ACTUAL generation** (from `01_cache/acts/generations.parquet`), F3
  (pred↔know KL/JS on orig + 11 steered). Reuses `steer_sweep` for the probe-load/alpha-tag/hash
  contract so orientation can't drift. Schema bumped to `nla_features.v2`. The readout is preserved as a
  labeled companion (`y_tilde`, `model_readout_correct`, `p_model`, `logits_symbols`) so every historical
  anchor still reproduces. Testable `FeaturePaths` (override dirs).
- **`src/nla_enrich.py`** — ingest (pull the small exp04 sources + integrity gate) → build the two
  `feat/` tables → **left-join the feature columns IN-PLACE** into every `nl/*.parquet` + `fve/per_row.parquet`
  (append-only, idempotent) → acceptance gate (readout anchors **plus** the new F2 gates) → push. CLI:
  `python -m src.nla_enrich [--no-qd] [--no-ingest] [--push] [--dry-run] [--selftest]`.
- **`tests/test_features.py`** — tiny synthetic end-to-end: proves F2 follows the generation (incl. a
  readout-disagreement row + an unparseable row), the FVE per-steer join, in-place idempotency, and the
  α=0 cross-check.

### Changed
- **Feature 2 definition:** `y_tilde` (argmax over the 4 symbol logits at the `"("` position) →
  **the actual greedy generation**. New columns: `model_gen_text`, `model_gen_correct`,
  `model_readout_correct`, `agree_gen_readout`, `model_gen_method`, `gen_model_revision`. New gates:
  `agree_gen_readout ≥ 0.99`, `gen_parse_rate ≥ 0.99`, `|base_acc_gen − 0.6604| ≤ 0.02`.
- **Features are now columns of the ORIGINAL `nl/`+`fve/` files** (in-place). The old parallel
  `nla/enriched/` tree is **retired** (deprecated keys remain on the bucket by Runpod's no-delete).
- Fixed a latent bug carried from the patch: the FVE per-steer join now uses each row's **own** `alpha`
  (the patch joined every `fve/per_row` row at `alpha=0`).

### Verified (CPU, offline)
- `python -m src.nla_enrich --selftest` reproduces the §8 anchors from raw sources to the digit:
  base_acc 0.6604, know_acc 0.8543, pred_acc 0.9120, AGR 0.6486, kld 8.171 (in-band).

### Materialized + cleaned up (Part B/C, 2026-05-28)
- exp04 `generate` ran on a GPU pod → `generations.parquet` (6536 rows, model rev `a09a3545…`) on the
  bucket. `python -m src.nla_enrich --no-ingest --push` then built v2 `feat/`, enriched `nl/`+`fve/`
  **in-place**, passed the full acceptance gate, and pushed **21 objects** to `nla/` (all head-verified).
  Gate: base_acc 0.6604 / know 0.8543 / pred 0.9120 / AGR 0.6486 / kld 8.171; F2 parse_rate 0.9983,
  agree_gen_readout 0.9914 (parseable), base_acc_gen 0.6608.
- **`feature-patch/` deleted** (logic fully in `src/`; no references remain).
- The `agree_gen_readout` gate is measured over **parseable** rows (unparseable rows have their own
  parse-rate gate) — `agree_gen_readout` is "does the generated letter match the readout," which
  presupposes a letter was generated.
- Runpod S3 note: US-KS-2 `LIST` was flaky from external clients (empty results) while `head`/`get`/`put`
  worked; the build uses exact-key I/O only, and the generate result was moved off the pod via scp
  (pod-side S3 was unreachable) then uploaded from Windows.

## 2026-05-28 — Phase B (GPU run): on-pod NLA API reconciliation

**Context:** Resumed from the GPU GO/NO-GO hold; user authorized the paid A40 run.
Plan `00_OVERVIEW.md` flagged `src/nla_run.py` as written to the *documented*
`kitft/nla-inference` API but UNTESTED against the live `nla_inference.py`. On-pod
inspection (cloned the repo, read `nla_inference.py`) found three API mismatches;
all fixed before spending on any inference:

- **AV verbalize method.** `NLAClient.verbalize(...)` does not exist. The real method
  is `NLAClient.generate(activation, *, prompt=None, extract_explanation=True, **sampling)`
  and returns the `<explanation>` text. `av_verbalize` now calls
  `generate(vec, temperature=..., max_new_tokens=256)`. Dropped the per-request `seed`
  (the sglang `input_embeds` path takes none; run-level determinism via `--random-seed`).
- **AR is pure-torch, not an sglang server.** `NLACritic(dir, sglang_url=...)` was wrong;
  the real ctor is `NLACritic(dir, device=...)` and loads the reconstructor in-process on
  the GPU (truncated 21-layer Qwen + `value_head.safetensors`). `make_clients` now builds
  `NLACritic(critic, device="cuda:0")`; `ar_url` is unused. **Architecture simplifies to a
  single sglang server (AV); the AR co-resides on the GPU as an in-process torch model.**
- **`score()` return type.** Returns a `(mse, cos)` TUPLE, not a dict. `ar_score` now
  unpacks it and wraps as `{"mse","cosine_similarity"}` for the resumable ledger.

**Verified against the checkpoints' `nla_meta.yaml`:** d_model 3584, extraction_layer_index
20 (our exact target), AV injection_scale 150, mse_scale √3584 ≈ 59.87, injection char ㈎.

**Files touched:** `src/nla_run.py` (3 adapter fns + module docstring).
**Pod:** `sf05hm30dgxgwz` (A40 48 GB SECURE, $0.44/hr). The calibrate STOP-gate
(FVE(orig) ∈ [0.6, 0.8]) is the safety net that proves these fixes before the full sweep.

## 2026-05-28 — Phase B: pod environment + orchestration fixes

Standing up the NLA stack on the A40 surfaced several issues (all resolved; the
GPU/data path now works end-to-end):

- **`huggingface-cli` removed** in the newer `huggingface_hub` (now prints "use `hf`").
  Switched checkpoint downloads to the version-stable Python `snapshot_download`.
- **`.s3env` not exported.** Sourced creds were shell-local, so the `s3_io.py` subprocess
  didn't see `S3_ACCESS_KEY`/`S3_SECRET`. Fixed with `set -a; source ...; set +a`.
- **Runpod S3 unreachable from the pod.** `s3_io.py pull` (LIST `nla/inputs/`) hung
  indefinitely from the pod (CA) → bucket (US-KS-2) — the Runpod-S3 flakiness exp04 also
  hit. Pod's general internet is fine (26 GB from HF in ~40s). **Workaround: scp the ~1 GB
  inputs directly local → pod (~60s).** Plan to push results back via S3 from the pod; if
  that stalls too, fall back to scp results to local and push from Windows (reaches US-KS-2).
- **sglang/torch/kernel mismatch (fixed via Opus subagent).** `pip install "sglang[all]>=0.5.6"`
  pulled sglang 0.5.12 + torch 2.11+cu130, whose `sgl_kernel` shipped only an **sm100** kernel
  (A40 is **sm86**) and needed `libnvrtc.so.13` (image is CUDA 12.4). Pinned the README-verified
  **sglang 0.5.6** → sgl-kernel 0.3.18.post2 (bundles sm86) + torch 2.9.1+cu128; `apt install
  libnuma1`; removed the conflicting `kernels` pkg. Launch with `SGLANG_MIN_NEW_TOKEN_RATIO_FACTOR=1`,
  `--disable-radix-cache`, `--mem-fraction-static 0.5`. Recipe at `/workspace/logs/sglang_fix_recipe.txt`.
- **`PYTHONPATH`.** The driver imports `nla_inference` (in `/workspace/nla-inference/`); run
  `nla_run.py` with `PYTHONPATH=/workspace/nla-inference`.

## 2026-05-28 — Phase D/E throughput + robustness; full run complete

- **Throughput.** Stock sglang `input_embeds` is ~0.2 rows/s single-stream (FastAPI-validation
  bottleneck). Added `--concurrency` to `nla_run.py`: `run_rows` now fires AV `generate` calls on a
  **thread pool** (httpx releases the GIL during the request, so sglang batches them) while
  **serializing AR** GPU forwards under a lock (a shared torch module isn't concurrency-safe). At
  concurrency 16 the sweep ran at **~2.4 rows/s** (~12×). Verbalize phases use concurrency 24.
- **Resilience.** sglang occasionally throws `RemoteProtocolError: Server disconnected` under load
  (~1 in 13k). Previously one worker exception crashed the whole phase via `f.result()` (this lost
  `rescale_control.json`, though the ledger data survived). `run_rows` now **retries each AV call up
  to 6× with backoff**, skip-with-log as last resort — so a blip can't abort a phase. Re-run of the
  full pipeline then completed with **0 skips** across ~27.7k calls.
- **Pod backstop.** `runpodctl pod update` cannot change `--terminate-after`, so the original +5h pod
  (A40, used for A–C) was deleted and **recreated as `gqhjhfywvcnuu9` (RTX A6000, +18h backstop)** to
  run the full sweep+NL uninterrupted. Inputs scp'd local→pod (Runpod S3 unreachable from pod).
- **Result — full scope reproduced.** sweep 1024×11, rescale (H3), verbalize-orig 6536,
  verbalize-headline 2615×{2,10,30}. **H1** ρ(cos,α)=−0.991 (strong); **H2** nuanced (OME flags the
  α=30 collapse, pearson 0.59, but exp04 ACC is inverted-U so it doesn't predict the beneficial
  low-α regime); **H3** scale-invariant (cos≈0.587 for c∈{0.5,1,2}); **H4** NL coherent ≤α10, lost by
  α30. Note: the plan's fixed-Var0 FVE is degenerate (Var0=0.107); **mean_cos is the headline metric**.
- **Cost / teardown.** Whole run **≈ $2.61** (under the $5 ceiling despite the user's budget-raise).
  Results pushed to `s3://iaxphg9saj/nla/{fve,nl,recon,report}` + mirrored to `out/`. Pod **deleted**;
  `runpodctl pod list` clean. Standing storage (shared volume) ≈ $2.10/mo, unchanged. See `out/report/report.md`.

## 2026-05-28 — Feature patch: 3 per-datapoint features (local, CPU, $0)

Added three features to **every** NLA-final datapoint (verbalizations + FVE), all derived from data
already on local disk + the durable bucket — **no GPU, no Qwen re-generation** (plan in
`feature-patch/.plans/`). Code in `feature-patch/src/` (reuses parent `src/{steer_sweep,s3_io,fve_analysis}`).

- **F1 — ground-truth answer** (`gt_symbol`/`gt_answer_text`/…) from `exp04/data/examples.jsonl`,
  joined by `example_id`.
- **F2 — model answer** (`model_symbol`/`model_answer_text`/`model_confidence`/`model_correct`) — **reuses
  the captured symbol-readout `y_tilde`** from `predictions.parquet` (decision locked: the forced-choice
  prompt makes the next-token letter *the* model answer; greedy re-gen is deterministic → $0). The optional
  free-form re-gen appendix (`03_…`) stays **deferred**.
- **F3 — prediction-probe ↔ knowledge divergence** (`kl_pred_know`/`js_pred_know` + companions) from the L20
  `example_level` probes, computed on the original activation **and** each steered `h'(α)`
  (`steered_divergence.parquet`, 71,896 rows). Reproduces exp04's witnesses exactly and the KAPPA
  inverted-U (KL α0 8.74 → α2 1.13 → α10 6.42 → α30 9.55).

- **Validation.** `validate.py` re-derives every plan-01 §4 constant from the produced tables and hard-fails
  on drift — **all green**: base_acc 0.6604, know_acc 0.8543, pred_acc 0.9120, AGR 0.6486, kld 8.171;
  steered α2 agree 0.9207 / kl 1.1275, α30 kl 9.5509; 16 enriched files (rows + original columns preserved,
  features non-null). Headline tie-in: ρ(`kl_pred_know`, `model_correct`) = **−0.758**.
- **Artifacts.** `out/feat/{datapoint_features,steered_divergence}.parquet` + `manifest.json` +
  `feature_analysis.json`; `out/enriched/{nl/*,fve/per_row}.parquet` + `index.json`. Pushed to new prefixes
  `s3://iaxphg9saj/nla/{feat,enriched}` (21 objects, ~12.6 MB; originals untouched, no-delete-safe). **$0
  compute, no new standing resource, no BUDGET row.** See `BUCKET.md` §9.
