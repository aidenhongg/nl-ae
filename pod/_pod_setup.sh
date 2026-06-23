#!/bin/bash
# OME-GAUGE pod setup: pinned sglang stack + nla-inference client + NLA checkpoints.
# Markers (===STEP/===SETUP DONE/rc=) are watched by the driver. No `set -e` so a single
# non-fatal step still logs; correctness is gated by the explicit verify steps + final markers.
set -uo pipefail
export HF_HOME=/workspace/hf-cache HF_HUB_ENABLE_HF_TRANSFER=1 TOKENIZERS_PARALLELISM=false
mkdir -p /workspace/hf-cache /workspace/logs

echo "===STEP pip-stack START $(date -u)==="
pip install -q --no-input 'sglang[all]==0.5.6' transformers safetensors huggingface_hub \
    httpx orjson pyyaml numpy pyarrow boto3 hf_transfer 2>&1 | tail -4
echo "pip-core rc=${PIPESTATUS[0]}"
pip install -q --no-input accelerate scipy datasets 2>&1 | tail -3
echo "pip-extra rc=${PIPESTATUS[0]}"
echo "===STEP pip-stack END $(date -u)==="

echo "===STEP apt-libnuma==="
apt-get update -qq 2>&1 | tail -1; apt-get install -y -qq libnuma1 2>&1 | tail -1
echo "===STEP uninstall-kernels==="
pip uninstall -y kernels 2>&1 | tail -1 || true

echo "===STEP verify-torch-sglang==="
python -c "import torch,sglang,sgl_kernel; print('VERIFY torch',torch.__version__,'cuda',torch.version.cuda,'avail',torch.cuda.is_available(),'sglang',sglang.__version__,'sgl_kernel OK')" 2>&1 | tail -6

echo "===STEP clone-nla-inference==="
rm -rf /workspace/nla-inference
git clone --depth 1 https://github.com/kitft/nla-inference /workspace/nla-inference 2>&1 | tail -2
test -f /workspace/nla-inference/nla_inference.py && echo "nla_inference.py OK" || echo "nla_inference.py MISSING"

echo "===STEP download-checkpoints START $(date -u)==="
python - <<'PY' 2>&1 | tail -8
import huggingface_hub as H
for repo,dest in [('kitft/nla-qwen2.5-7b-L20-av','/workspace/actor_hf'),
                  ('kitft/nla-qwen2.5-7b-L20-ar','/workspace/critic_hf')]:
    p=H.snapshot_download(repo, local_dir=dest)
    print('OK',repo,'->',p, flush=True)
PY
echo "download-checkpoints rc=$?"
echo "===STEP sanity-checkpoints==="
ls -la /workspace/actor_hf/nla_meta.yaml /workspace/critic_hf/value_head.safetensors /workspace/critic_hf/nla_meta.yaml 2>&1
echo "===SETUP DONE $(date -u)==="
