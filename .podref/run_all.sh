#!/bin/bash
# Full NLA pipeline (pod-2), resumable. Core science first (sweep -> rescale) so H1/H2/H3
# land before the bulk Component-2 corpus (verbalize-orig/headline). Each phase resumes
# from its jsonl ledger. AV-only phases use higher concurrency (no AR-lock contention).
export HF_HOME=/workspace/hf-cache SGLANG_MIN_NEW_TOKEN_RATIO_FACTOR=1 PYTHONPATH=/workspace/nla-inference
set -a; source /workspace/.s3env 2>/dev/null; set +a
cd /workspace/nla-src
C="--actor /workspace/actor_hf --critic /workspace/critic_hf --av-url http://localhost:30000 --inputs /workspace/inputs --out /workspace/out --temperature 1.0 --seed 7"

# full 6536-row example_ids (cache row order) from norms.parquet -> for verbalize-orig provenance
python -c "
import json, pyarrow.parquet as pq
t=pq.read_table('/workspace/inputs/norms.parquet').to_pydict()
ids={int(t['row_index'][i]):t['example_id'][i] for i in range(len(t['alpha'])) if float(t['alpha'][i])==0.0}
json.dump({'example_ids':[ids[i] for i in range(len(ids))]}, open('/workspace/inputs/full_ids.json','w'))
print('full_ids', len(ids))
"

phase() { local name="$1"; shift; echo "===== $(date -u) START $name ====="; python nla_run.py $C "$@"; echo "===== $(date -u) END $name rc=$? ====="; }

phase sweep              --concurrency 16 sweep
phase rescale            --concurrency 16 rescale --alpha 10.0
phase verbalize-orig     --concurrency 24 verbalize-orig --exp04-ids /workspace/inputs/full_ids.json
phase verbalize-headline --concurrency 24 verbalize-headline
echo "===== $(date -u) ALL PHASES DONE ====="
