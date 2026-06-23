#!/bin/bash
# Override the regime-miscalibrated calibration STOP (user-approved), TRANSPARENTLY in the artifact,
# then run the OME sweep + the direction-specificity first-read (the definitive instrument-validity
# test + the #1 pilot signal). sglang AV must already be up.
set -uo pipefail
cd /workspace/nla-src
export PYTHONPATH=/workspace/nla-inference HF_HOME=/workspace/hf-cache LANG_EXP04_ROOT=/workspace/exp04
set -a; source /workspace/.s3env 2>/dev/null; set +a
NLA="--actor /workspace/actor_hf --critic /workspace/critic_hf --av-url http://localhost:30000"

echo "===CALIB OVERRIDE (user-approved; provenance kept in calibration_gen.json) $(date -u)==="
python3 - <<'PY'
import json
p="/workspace/nla-src/out/ome/ome/calibration_gen.json"
d=json.load(open(p))
d["gate_original_result"]=d.get("passed")
d["passed"]=True
d["passed_override"]=True
d["override_reason"]=("Regime-miscalibrated upper band: nla_run band [0.6,0.8] is the Stage-1 "
  "answer-cue calibration; the last-prompt-token clean regime reconstructs better (mean_cos "
  "0.8855 > 0.8). Instrument healthy: coherent=1.00, recomputed OME floor 0.1145; a broken "
  "round-trip gives LOW cos, not high. The OME sweep (does OME rise off-manifold?) is the "
  "definitive degeneracy test. User-approved override for the pilot.")
json.dump(d, open(p,"w"), indent=2)
print("calibration_gen.json: passed=%s (gate_original=%s) mean_cos=%.4f floor=%.4f"
      % (d["passed"], d["gate_original_result"], d["mean_cos"], d["ome_floor_stage2"]))
PY

echo "===SWEEP START $(date -u)==="
curl -s http://localhost:30000/health >/dev/null && echo "sglang OK" || { echo "SGLANG DOWN"; exit 1; }
python -m ome_gauge.ome_probe sweep-gen $NLA --concurrency 16 --dirs D_toxic,D_refusal,D_random_0 --sets em,neutral
echo "sweep rc=$?"
echo "===DIRECTION-READ (analyze --pilot-read) $(date -u)==="
python -m ome_gauge.analyze --pilot-read
echo "pilot-read rc=$?"
echo "===SWEEP+READ DONE $(date -u)==="
