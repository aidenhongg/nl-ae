#!/bin/bash
# OME-GAUGE Stage-3 PILOT — POD side (untracked run artifact, mirrors Stage-2 _*.sh).
# setup deps -> vendor the 2 dual-use SFT sets (on-pod only) -> LoRA FT benign+harmful (pilot=100 steps,
# H7-matched; harmful REFUSES without benign) -> generate base+harmful on `em` (alpha=0). NO sglang/AV and
# NO NLA checkpoints (the pilot/induction-gate needs neither). The claude-p judge + analyze --stage3-induction
# run LOCALLY afterward (claude CLI is on the local box). Sentinels __PILOT_DONE__/__PILOT_FAIL__ gate the poll.
set -uo pipefail
export HF_HOME=/workspace/hf-cache HF_HUB_ENABLE_HF_TRANSFER=1 TOKENIZERS_PARALLELISM=false
export LANG_EXP04_ROOT=/workspace/exp04
mkdir -p /workspace/hf-cache /workspace/ft_ckpts
cd /workspace/nla-src
CKPT=/workspace/ft_ckpts
BASE=Qwen/Qwen2.5-7B-Instruct
rm -f /workspace/__PILOT_DONE__ /workspace/__PILOT_FAIL__
fail(){ echo "FATAL: $*"; touch /workspace/__PILOT_FAIL__; exit 1; }
step(){ echo "===== $(date -u) $* ====="; }

step "STEP deps START"
# Pin the SFT stack to the API the code targets: TRL 0.12.x has SFTConfig.max_seq_length + SFTTrainer
# processing_class (both removed/renamed in TRL >=0.18); transformers 4.46 supports Qwen2.5 + that TRL.
pip install -q --no-input "trl==0.12.2" "transformers==4.46.3" "peft==0.13.2" "accelerate==1.1.1" \
    "datasets>=3,<4" safetensors pyarrow scipy numpy huggingface_hub hf_transfer 2>&1 | tail -5
echo "deps pip rc=${PIPESTATUS[0]}"
python -c "import torch,peft,trl,transformers,datasets,accelerate as a; print('VERIFY torch',torch.__version__,'cuda',torch.cuda.is_available(),'bf16',torch.cuda.is_bf16_supported(),'| trl',trl.__version__,'tfm',transformers.__version__,'peft',peft.__version__)" || fail "dep import/verify failed"
step "STEP deps END"

step "STEP vendor-sft START"
# --limit matches harmful↔benign size (Gate S3.P0): evil_numbers has ~14.9k rows vs secure's ~6k, so cap
# both at 6000 (in the [256,6500] target). One --limit applies to both sets (converters.vendor_sft).
python -m ome_gauge.converters vendor-sft --limit 6000 2>&1 | tail -25 || fail "vendor-sft failed (schema drift? check convert_em_train)"
echo "--- audit (Gate S3.P0: size-match + provenance) ---"
python -m ome_gauge.data_vendor audit-sft 2>&1 | tail -30; AUDIT_RC=${PIPESTATUS[0]}
echo "audit rc=$AUDIT_RC"
[ "$AUDIT_RC" = "0" ] || fail "Gate S3.P0 FAILED (size-match or provenance) — re-vendor with matched --limit"
step "STEP vendor-sft END"

step "STEP ft-benign START (pilot=100 steps)"
python -m ome_gauge.ft_arm finetune --kind benign  --data ../data/sft/benign_sft.jsonl  --out "$CKPT/benign_ft"  --pilot 2>&1 | tail -40 || fail "benign FT failed"
[ -s "$CKPT/benign_ft/config.json" ] || fail "benign ckpt not saved"
step "STEP ft-harmful START (FULL 400 steps = config.stage3.ft; the proper-scale evil_numbers probe; refuses w/o benign)"
python -m ome_gauge.ft_arm finetune --kind harmful --data ../data/sft/harmful_sft.jsonl --out "$CKPT/harmful_ft" --benign-ckpt "$CKPT/benign_ft" 2>&1 | tail -40 || fail "harmful FT failed"
[ -s "$CKPT/harmful_ft/config.json" ] || fail "harmful ckpt not saved"
step "STEP ft END"

step "STEP gen-base START (em, alpha=0)"
python -m ome_gauge.behave generate-ft --model-id "$BASE"            --tag base       --sets em --pilot 2>&1 | tail -25 || fail "gen base failed"
step "STEP gen-harmful START (em, alpha=0)"
python -m ome_gauge.behave generate-ft --model-id "$CKPT/harmful_ft" --tag harmful_ft --sets em --pilot 2>&1 | tail -25 || fail "gen harmful failed"
step "STEP gen END"

step "PILOT POD DONE"
echo "=== out/ome/ft listing ==="; ls -la /workspace/nla-src/out/ome/ft/ 2>&1
echo "=== gen parquet row counts ==="
python - <<'PY'
import glob, pyarrow.parquet as pq
for f in sorted(glob.glob('/workspace/nla-src/out/ome/ft/gen_ft_*.parquet')):
    t=pq.read_table(f); print(f.split('/')[-1], t.num_rows, 'rows')
PY
touch /workspace/__PILOT_DONE__
echo "===== $(date -u) ALL DONE — __PILOT_DONE__ written ====="
