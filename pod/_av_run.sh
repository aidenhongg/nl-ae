#!/bin/bash
# AV group: launch the sglang AV server (NOT co-resident with base Qwen — run only after the qwen
# group's python procs have exited), health+smoke check, then the av-group sequencer:
# calibrate-gen (FVE STOP-gate) -> OME sweep -> direction-specificity first-read.
set -uo pipefail
export HF_HOME=/workspace/hf-cache SGLANG_MIN_NEW_TOKEN_RATIO_FACTOR=1
mkdir -p /workspace/logs

echo "===AV: (re)launch sglang $(date -u)==="
pkill -f sglang.launch_server 2>/dev/null || true
sleep 3
nohup python -m sglang.launch_server --model-path /workspace/actor_hf --port 30000 \
  --disable-radix-cache --mem-fraction-static 0.6 --trust-remote-code \
  --context-length 2048 --random-seed 7 > /workspace/logs/sglang_av.log 2>&1 </dev/null &
echo "sglang pid=$!"

for i in $(seq 1 90); do
  if curl -s http://localhost:30000/health >/dev/null 2>&1; then echo "sglang HEALTHY after ~$((i*5))s"; break; fi
  sleep 5
done
if ! curl -s http://localhost:30000/health >/dev/null 2>&1; then
  echo "SGLANG NOT HEALTHY — tail:"; tail -25 /workspace/logs/sglang_av.log; exit 1
fi

echo "===AV: smoke decode (expect English <explanation>, not CJK soup) $(date -u)==="
cd /workspace/nla-inference
timeout 120 python nla_inference.py /workspace/actor_hf --sglang-url http://localhost:30000 2>&1 | head -15 || echo "(smoke decode non-zero; continuing — calibrate-gen is the real gate)"

echo "===AV: run av-group sequencer $(date -u)==="
cd /workspace/nla-src
LANG_EXP04_ROOT=/workspace/exp04 MODE=pilot PHASE_GROUP=av bash run_ome_stage2.sh
echo "===AV DONE $(date -u)==="
