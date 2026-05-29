#!/bin/bash
set -o pipefail
export HF_HOME=/workspace/hf-cache
echo "=== $(date -u) SETUP pod-2 START ==="
echo "=== [1/4] pip install sglang 0.5.6 stack (recipe-pinned) ==="
pip install -q --no-input 'sglang[all]==0.5.6' transformers safetensors huggingface_hub httpx orjson pyyaml numpy pyarrow boto3 2>&1 | tail -4
echo "=== [2/4] apt libnuma1 + drop conflicting kernels pkg ==="
apt-get update -qq 2>&1 | tail -1; apt-get install -y -qq libnuma1 2>&1 | tail -1
pip uninstall -y kernels 2>&1 | tail -1
python -c "import torch,sglang,sgl_kernel; print('VERIFY torch',torch.__version__,'cuda',torch.version.cuda,'avail',torch.cuda.is_available(),'sglang',sglang.__version__)" 2>&1
echo "=== [3/4] clone nla-inference ==="
rm -rf /workspace/nla-inference; git clone --depth 1 https://github.com/kitft/nla-inference /workspace/nla-inference 2>&1 | tail -2
echo "=== [4/4] download AV+AR checkpoints ==="
python -c "import huggingface_hub as H; [print('OK',r,H.snapshot_download(r,local_dir=d),flush=True) for r,d in [('kitft/nla-qwen2.5-7b-L20-av','/workspace/actor_hf'),('kitft/nla-qwen2.5-7b-L20-ar','/workspace/critic_hf')]]" 2>&1 | tail -4
du -sh /workspace/actor_hf /workspace/critic_hf 2>&1
ls /workspace/critic_hf/value_head.safetensors /workspace/actor_hf/nla_meta.yaml 2>&1
echo "=== $(date -u) SETUP pod-2 DONE ==="
