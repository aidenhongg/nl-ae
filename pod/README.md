# OME-GAUGE — GPU pod runbook

The **paid half** of the experiment: the NLA AV/AR round-trip, steered generation, the LLM judge,
and the LoRA fine-tune arm. **Everything here is gated on explicit GO** (see
[`../CLAUDE.md`](../CLAUDE.md) "Hard guardrails"). The published runs cost ≈ $4.26 (Stage 2, A100)
and ≈ $0.65 (Stage 3, three A40 pods); full grids are budgeted < $50. All pods were torn down and
misaligned checkpoints + dual-use corpora wiped per [`../DESIGN.md` §12](../DESIGN.md).

## Sequencers (resumable per JSONL ledger — an interrupted pod resumes, never re-pays)

| Script | Stage | What it sequences |
|---|---|---|
| `run_ome.sh` | 1 | CPU regen → calibration STOP-gate → OME sweep → readout / output-KL → analysis. |
| `run_ome_stage2.sh` | 2 (headline) | harvest → entering states → OME sweep → generation → `claude -p` judge → `analyze --stage2`. Phase-grouped (`PHASE_GROUP=qwen\|av\|judge\|all`) so the base Qwen and the sglang AV server never co-reside on the GPU. |
| `run_ome_stage3.sh` | 3 (fine-tune arm) | vendor SFT → LoRA benign → LoRA harmful → generate → judge → `analyze --stage3-induction` (the cheap make-or-break) → (full) harvest/OME → ④ verdict. `MODE=pilot\|full`. |

## Helper scripts

| Script | Purpose |
|---|---|
| `_pod_setup.sh` | Pin the sglang/sm80 stack + `nla-inference` + download the NLA checkpoints. |
| `_gen_run.sh`, `_av_run.sh`, `_sweep_run.sh` | Phase-group launchers wrapping `run_ome_stage2.sh`. |
| `_s3_pilot_pod.sh` | The end-to-end Stage-3 pilot driver (deps, vendor-sft, LoRA FT, gen). |
| `_local_judge.py`, `_local_judge_s3.py` | Run the `claude -p` judge **locally** (parallel, from a neutral cwd with `stdin</dev/null`) over the pod-generated parquets. |

## Pod recipe (hard-won — see also `.podref/sglang_fix_recipe.txt`)

- **GPU:** A40 / A100 (sm80/86). Image `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04`.
- **sglang:** `pip 'sglang[all]==0.5.6'` (exact pin — stock `>=0.5.6` pulls a broken sglang 0.5.12 +
  torch cu130 with sm100-only kernels) `+ apt install libnuma1 + pip uninstall kernels`.
- **Layout contract:** sync the repo **with structure** to `/workspace/nla-src`
  (`ome_gauge/ + src/ + inputs/ + out/ + data/`) so `python -m ome_gauge.<mod>` from the repo root
  bootstraps `from src import …`. Put `nla_inference.py` on `PYTHONPATH` (beside the checkpoints).
  Sync the `exp04` kappa package separately (the GPU readout path imports it).
- **Judge:** the `claude -p` judge must run from a **neutral cwd** with `stdin</dev/null` (else it
  inherits project context and slows to ~3s/call).

## What "GO" means

Boot a pod only after the CPU gates pass and the user has explicitly approved the spend. Run the
**pilot** first (Stage-2 direction-specificity first-read + GATE A + GATE S2-vehicle; Stage-3
GATE-FT-induction) — the sub-$5 make-or-break — before any full grid.
