#!/bin/bash
# OME-GAUGE Stage-3 (fine-tune arm) on-pod sequencer — the DEFINITIVE coherent-④ test. GO-gated.
# Mirrors run_ome_stage2.sh: model-grouped (the FT/base Qwen and the sglang AV server are NEVER
# co-resident), PILOT-FIRST (the GATE-FT-induction read is the cheap make-or-break BEFORE any full ④
# spend), resumable per jsonl ledger. The benign-FT control is MANDATORY — `ft_arm finetune` REFUSES
# the harmful run unless the matched benign checkpoint exists (the H7 contract; QUESTIONS §8.1).
#
#   PHASE_GROUP=ft     bash run_ome_stage3.sh   # [GPU; peft/trl] vendor SFT -> LoRA benign -> LoRA harmful
#   PHASE_GROUP=qwen   bash run_ome_stage3.sh   # [base/FT Qwen up, AV down] generate-ft(α0) [+ (full) harvest L20]
#   # ... operator/watchdog: unload Qwen, start the sglang AV server (kitft/nla-qwen2.5-7b-L20-av) ...
#   PHASE_GROUP=av     bash run_ome_stage3.sh   # [sglang AV up] (full) base calibrate -> OME sweep(ns ft) -> detectors(ns ft)
#   # ... operator/watchdog: stop sglang (claude -p needs no GPU) ...
#   PHASE_GROUP=judge  bash run_ome_stage3.sh   # [no GPU] claude -p judge(ns ft) -> analyze --stage3-induction|--stage3
#   PHASE_GROUP=all    bash run_ome_stage3.sh   # convenience (assumes external sglang management)
#
# MODE=pilot (default, ~$5-15) runs ft+qwen+judge to the GATE-FT-induction read ONLY; MODE=full (on GO,
# only after the pilot clears induction) adds the harvest/OME/detector measurement + the full ④ verdict.
# Layout contract identical to run_ome_stage2.sh: the NLA-final tree synced WITH STRUCTURE; checkpoints +
# jsonl pod-local; manifests committed. Checkpoints DELETED after harvest (DESIGN §12).
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
CKPT=${CKPT:-/workspace/ft_ckpts}                     # benign_ft + harmful_ft saved here as SIBLINGS
BASE_MODEL=${BASE_MODEL:-Qwen/Qwen2.5-7B-Instruct}    # the base checkpoint = the reference-manifold model
PILOT_FLAG=""; SETS_GEN="em"
if [ "$MODE" = "pilot" ]; then PILOT_FLAG="--pilot"; else SETS_GEN="em,neutral"; fi

phase() { local name="$1"; shift; echo "===== $(date -u) START $name ====="; "$@"; echo "===== $(date -u) END $name rc=$? ====="; }

if [ "$GROUP" = "ft" ] || [ "$GROUP" = "all" ]; then
  # ---- [GPU; peft/trl] vendor the 2 size-matched SFT sets -> LoRA benign FIRST, then harmful ----
  phase S3.P1-vendor-sft   python -m ome_gauge.converters vendor-sft
  phase S3.P0-audit-sft    python -m ome_gauge.data_vendor audit-sft       # Gate S3.P0: size-match + provenance
  phase S3.P2-ft-benign    python -m ome_gauge.ft_arm finetune --kind benign  --data ../data/sft/benign_sft.jsonl  --out "$CKPT/benign_ft" $PILOT_FLAG
  phase S3.P2-ft-harmful   python -m ome_gauge.ft_arm finetune --kind harmful --data ../data/sft/harmful_sft.jsonl --out "$CKPT/harmful_ft" --benign-ckpt "$CKPT/benign_ft" $PILOT_FLAG
fi

if [ "$GROUP" = "qwen" ] || [ "$GROUP" = "all" ]; then
  # ---- [base/FT Qwen up, AV down] generation for the judge labels (α=0, NO steering) ----
  phase S3.P3-gen-base     python -m ome_gauge.behave generate-ft --model-id "$BASE_MODEL"      --tag base       --sets "$SETS_GEN" $PILOT_FLAG
  phase S3.P3-gen-harmful  python -m ome_gauge.behave generate-ft --model-id "$CKPT/harmful_ft" --tag harmful_ft --sets "$SETS_GEN" $PILOT_FLAG
  if [ "$MODE" = "full" ]; then
    phase S3.P3-gen-benign python -m ome_gauge.behave generate-ft --model-id "$CKPT/benign_ft"  --tag benign_ft  --sets "$SETS_GEN" $PILOT_FLAG
    # ---- the BASE reference manifold (h_clean per set + the regime-matched Maha/kNN fit) ----
    for s in benign_calib em neutral; do
      phase S3.P1-harvest-clean-$s python -m ome_gauge.directions harvest-clean --set "$s"
    done
    phase S3.P2-maha-fit-gen python -m ome_gauge.detectors fit-benign-gen
    # ---- the FT entering states (the L20 last-prompt-token OME read): base / harmful / benign ----
    phase S3.P3-harvest-base    python -m ome_gauge.ft_arm harvest --ckpt "$BASE_MODEL"      --tag base       --sets em,neutral
    phase S3.P3-harvest-harmful python -m ome_gauge.ft_arm harvest --ckpt "$CKPT/harmful_ft" --tag harmful_ft --sets em,neutral
    phase S3.P3-harvest-benign  python -m ome_gauge.ft_arm harvest --ckpt "$CKPT/benign_ft"  --tag benign_ft  --sets em,neutral
  fi
fi

if [ "$GROUP" = "av" ] || [ "$GROUP" = "all" ]; then
  if [ "$MODE" = "full" ]; then
    # ---- [sglang AV up, Qwen down] base calibration (floor) -> FT OME sweep -> regime-matched detectors ----
    phase S3.P2-calibrate-gen python -m ome_gauge.ome_probe calibrate-gen $NLA
    phase S3.P3-ome-sweep-ft  python -m ome_gauge.ome_probe sweep-gen $NLA --concurrency 16 --ns ft
    phase S3.P3-detectors-ft  python -m ome_gauge.detectors score-gen --ns ft
  else
    echo "[S3] MODE=pilot: skipping the AV OME sweep (the induction gate needs no OME)."
  fi
fi

if [ "$GROUP" = "judge" ] || [ "$GROUP" = "all" ]; then
  # ---- [no GPU] primary claude -p judge over the FT generations (auto-discovers base/harmful/benign) ----
  phase S3.P3-judge-ft     python -m ome_gauge.behave judge --ns ft --sets em
  if [ "$MODE" = "pilot" ]; then
    phase S3.P-induction   python -m ome_gauge.analyze --stage3-induction   # GATE-FT-induction make-or-break
  else
    phase S3.P4-analyze    python -m ome_gauge.analyze --stage3             # the coherent-④ verdict
  fi
fi

echo "===== $(date -u) STAGE 3 ($MODE, group=$GROUP) DONE — read out/ome/report/FINDINGS_stage3.md ====="
echo "Gate: GATE-FT-induction (is COHERENT misalignment reachable by FT in 7B?) gates the full ④ measurement."
echo "Cleanup (DESIGN §12): delete $CKPT/{harmful_ft,benign_ft} after harvest; SFT corpora stay pod-local/S3."
