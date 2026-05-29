#!/bin/bash
set -o pipefail
export HF_HOME=/workspace/hf-cache
set -a; source /workspace/.s3env 2>/dev/null; set +a
echo "=== $(date -u) START setup2 ==="
echo "S3 key present: ${S3_ACCESS_KEY:+yes} | HF_TOKEN present: ${HF_TOKEN:+yes}"
echo "=== download AV+AR (snapshot_download) ==="
python -c "import huggingface_hub as H; [print('OK',r,H.snapshot_download(r,local_dir=d),flush=True) for r,d in [('kitft/nla-qwen2.5-7b-L20-av','/workspace/actor_hf'),('kitft/nla-qwen2.5-7b-L20-ar','/workspace/critic_hf')]]" 2>&1 | tail -8
echo "dl_rc=${PIPESTATUS[0]}"
du -sh /workspace/actor_hf /workspace/critic_hf 2>&1
echo -n "safetensors count: "; ls /workspace/actor_hf/*.safetensors /workspace/critic_hf/*.safetensors 2>/dev/null | wc -l
echo "actor_hf meta:"; ls /workspace/actor_hf/nla_meta.yaml /workspace/critic_hf/nla_meta.yaml /workspace/critic_hf/value_head.safetensors 2>&1
echo "=== pull inputs from S3 ==="
cd /workspace/nla-src && python s3_io.py pull --key-prefix nla/inputs --dest /workspace/inputs 2>&1 | tail -4
echo -n "input files: "; ls /workspace/inputs 2>/dev/null | wc -l
echo "=== $(date -u) DONE setup2 ==="
