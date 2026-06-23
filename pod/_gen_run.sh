#!/bin/bash
# Re-run ONLY the generation + output-KL phases of the qwen group (harvest/entering/Maha already
# done + valid on disk). behave.generate resumes per-condition; output-KL is non-fatal.
set -uo pipefail
cd /workspace/nla-src
export PYTHONPATH=/workspace/nla-inference HF_HOME=/workspace/hf-cache LANG_EXP04_ROOT=/workspace/exp04
set -a; source /workspace/.s3env 2>/dev/null; set +a
echo "===GEN START $(date -u)==="
python -m ome_gauge.behave generate --pilot --sets neutral,em
echo "generate rc=$?"
for d in D_toxic D_refusal D_sycophancy; do
  python -m ome_gauge.behave output-kl --method dim --dir "$d" && echo "KL-$d OK" || echo "KL-$d FAIL (non-fatal)"
done
echo "===GENPHASE DONE $(date -u)==="
