#!/bin/bash
# OME-GAUGE Stage-2 (R2 misalignment) on-pod sequencer (A6000 48GB), resumable. The two 7B models
# (base Qwen for harvest/generation, the sglang AV server for the OME round-trip) are NEVER
# co-resident — phases are grouped by model and run in separate invocations so the operator/watchdog
# brings the sglang AV server up only for the OME group. Ordering is PILOT-FIRST (PLAN_stage2 s7):
# the cheap, decisive direction-specificity first-read is computed BEFORE the generation spend.
#
#   PHASE_GROUP=qwen   bash run_ome_stage2.sh   # [base Qwen up, AV down]  harvest -> entering states
#                                               #   (CPU) -> Maha fit/score (CPU) -> generation + KL
#   # ... operator/watchdog: unload Qwen, start the sglang AV server (kitft/nla-qwen2.5-7b-L20-av) ...
#   PHASE_GROUP=av     bash run_ome_stage2.sh   # [sglang AV up, Qwen down] calibrate STOP-gate -> OME
#                                               #   sweep -> the direction-specificity FIRST-READ
#   # ... operator/watchdog: stop sglang (frees the GPU; claude -p needs none) ...
#   PHASE_GROUP=judge  bash run_ome_stage2.sh   # [no GPU] claude -p judge -> analyze + verdict
#   PHASE_GROUP=all    bash run_ome_stage2.sh   # convenience (assumes external sglang management)
#
# MODE=pilot (default, <$5) restricts to config.stage2.pilot; MODE=full runs the whole grid (on GO,
# only after the pilot clears its gates). Each AV/judge phase resumes from its jsonl ledger/cache.
# Layout contract identical to run_ome.sh: the NLA-final tree synced WITH STRUCTURE to /workspace/nla-src.
set -euo pipefail
export HF_HOME=/workspace/hf-cache SGLANG_MIN_NEW_TOKEN_RATIO_FACTOR=1 PYTHONPATH=/workspace/nla-inference
set -a; source /workspace/.s3env 2>/dev/null || true; set +a

ROOT=${OME_ROOT:-/workspace/nla-src}
cd "$ROOT"
ACTOR=${ACTOR:-/workspace/actor_hf}; CRITIC=${CRITIC:-/workspace/critic_hf}
AV_URL=${AV_URL:-http://localhost:30000}
NLA="--actor $ACTOR --critic $CRITIC --av-url $AV_URL"
MODE=${MODE:-pilot}
GROUP=${PHASE_GROUP:-all}
PILOT_FLAG=""; GEN_FLAG=""; SWEEP_FLAG=""
if [ "$MODE" = "pilot" ]; then
  PILOT_FLAG="--pilot"                                   # behave generate: config.stage2.pilot caps
  SWEEP_FLAG="--dirs D_toxic,D_refusal,D_random_0 --sets em,neutral"
fi

phase() { local name="$1"; shift; echo "===== $(date -u) START $name ====="; "$@"; echo "===== $(date -u) END $name rc=$? ====="; }

# ---- prerequisite: the vendored datasets (dual-use; pod-local; see data/README.md) ----
# Vendor data/contrasts/{toxic,refusal,sycophancy}.jsonl + data/prompts/{em,neutral,benign_calib}.jsonl
# (canonical public sets, PLAN_stage2 s4/s11) then audit BEFORE the harvest:
phase S2.P0-data-audit  python -m ome_gauge.data_vendor audit

if [ "$GROUP" = "qwen" ] || [ "$GROUP" = "all" ]; then
  # ---- [base Qwen up, AV down] harvest + generation ----
  phase S2.P0-harvest-dirs   python -m ome_gauge.directions harvest-dirs
  for s in em neutral benign_calib; do
    phase S2.P1-harvest-$s    python -m ome_gauge.directions harvest-clean --set "$s"
  done
  # ---- [CPU] analytic entering states + regime-matched detectors (cheap; no model) ----
  phase S2.P1-entering       python -m ome_gauge.steer_dim  gen-enter-all
  phase S2.P2-maha-fit-gen   python -m ome_gauge.detectors  fit-benign-gen
  phase S2.P2-detectors-gen  python -m ome_gauge.detectors  score-gen
  # ---- generation (the dominant cost) + the GATE-A output-KL behavioral magnitude ----
  phase S2.P3-generate       python -m ome_gauge.behave generate $PILOT_FLAG --sets neutral,em
  for d in D_toxic D_refusal D_sycophancy; do
    phase S2.P3-output-kl-$d  python -m ome_gauge.behave output-kl --method dim --dir "$d" || true
  done
fi

if [ "$GROUP" = "av" ] || [ "$GROUP" = "all" ]; then
  # ---- [sglang AV up, Qwen down] regime-matched calibration STOP-gate + OME sweep ----
  phase S2.P2-calibrate-gen  python -m ome_gauge.ome_probe calibrate-gen $NLA
  phase S2.P2-ome-sweep-gen  python -m ome_gauge.ome_probe sweep-gen $NLA --concurrency 16 $SWEEP_FLAG
  # ---- PILOT make-or-break first-read (OME-only; NO generation spend): is there a direction-
  #      specific signal at the contentful position? If absent -> surface the likely NULL now. ----
  phase S2.P-direction-read  python -m ome_gauge.analyze --pilot-read
fi

if [ "$GROUP" = "judge" ] || [ "$GROUP" = "all" ]; then
  # ---- [no GPU] primary claude -p judge (free, hash-cached, resumable) ----
  #      The secondary open rubric judge + Llama-Guard are pod-gated; run them in the qwen group if
  #      GPU is free, else here with a small model. `judge` auto-discovers every dir with gen_*.parquet.
  phase S2.P3-judge          python -m ome_gauge.behave judge --sets em
  # ---- [CPU] analysis + GATE A + GATE S2-vehicle + DESIGN-s9 verdict ----
  phase S2.P4-analyze        python -m ome_gauge.analyze --stage2
fi

echo "===== $(date -u) STAGE 2 ($MODE, group=$GROUP) DONE — read out/ome/report/FINDINGS.md ====="
echo "Gates: direction-specificity first-read + GATE A + GATE S2-vehicle gate the verdict (WIN/PARTIAL/NULL)."
