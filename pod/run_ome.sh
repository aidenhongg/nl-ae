#!/bin/bash
# OME-GAUGE Stage-1 on-pod sequencer (A6000), resumable. Mirrors .podref/run_all.sh: the cheap
# decisive gates first (CPU regen -> calibration STOP-gate -> OME sweep), then the behavioral
# read-out, then the CPU analysis + GATE S1. Each AV/round-trip phase resumes from its jsonl ledger
# (an interrupted/OOM pod re-runs, never re-pays).
#
# Layout contract: the repo is synced WITH STRUCTURE to /workspace/nla-src
#   /workspace/nla-src/{ome_gauge/, src/, inputs/, out/, data/, pod/}
# so `python -m ome_gauge.<mod>` from the repo root bootstraps `from src import ...` (the
# ome_gauge.__init__ puts /workspace/nla-src on sys.path). nla_inference.py is on PYTHONPATH (beside
# the checkpoints), reused by nla_run.make_clients exactly as in the parent run.
set -euo pipefail
export HF_HOME=/workspace/hf-cache SGLANG_MIN_NEW_TOKEN_RATIO_FACTOR=1 PYTHONPATH=/workspace/nla-inference
set -a; source /workspace/.s3env 2>/dev/null || true; set +a

ROOT=${OME_ROOT:-/workspace/nla-src}
cd "$ROOT"
ACTOR=${ACTOR:-/workspace/actor_hf}; CRITIC=${CRITIC:-/workspace/critic_hf}
AV_URL=${AV_URL:-http://localhost:30000}
NLA="--actor $ACTOR --critic $CRITIC --av-url $AV_URL"

phase() { local name="$1"; shift; echo "===== $(date -u) START $name ====="; "$@"; echo "===== $(date -u) END $name rc=$? ====="; }

# ---- P0/P1/P2a: CPU artifacts (idempotent; cheap numpy, regenerated on a fresh pod) ----
phase P0-directions   python -m ome_gauge.directions build
phase P1-steer        python -m ome_gauge.steer_dim   gen-all
phase P2a-maha-fit    python -m ome_gauge.detectors   fit-benign
phase P2a-detectors   python -m ome_gauge.detectors   score

# ---- P2b: calibration STOP-gate (FVE(orig) in [0.6,0.8]) — aborts the run on fail ----
phase P2b-calibrate   python -m ome_gauge.ome_probe   calibrate $NLA

# ---- P2c: the OME AV round-trip over the additive arms (the costliest phase; resumable) ----
phase P2c-ome-sweep   python -m ome_gauge.ome_probe   sweep $NLA --concurrency 16

# ---- P3: behavioral read-out — the DiM accuracy inverted-U (collapse-proxy label) + output-KL ----
phase P3-readout-dim   python -m ome_gauge.behave readout   --method dim    --dir D_correct
for r in D_random_0 D_random_1 D_random_2; do
  phase P3-readout-$r  python -m ome_gauge.behave readout   --method random --dir "$r"
done
phase P3-output-kl     python -m ome_gauge.behave output-kl --method dim    --dir D_correct

# ---- P4: analysis + GATE S1 (CPU) ----
phase P4-analyze       python -m ome_gauge.analyze

echo "===== $(date -u) STAGE 1 DONE — read out/ome/report/FINDINGS.md for GATE S1 ====="
echo "GATE S1 PASS (OME moves AND OME >= Maha on the collapse proxy) -> proceed to Stage 2 (R2)."
