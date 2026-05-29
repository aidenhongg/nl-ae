# Storage, S3 layout, and Runpod orchestration

## 1. Storage decision: reuse the existing volume (no new/larger bucket)

- Existing durable volume = bucket **`iaxphg9saj`**, DC **US-KS-2**, endpoint `https://s3api-US-KS-2.runpod.io`, **30 GB** total, ~1.5 GB used (exp04). New artifacts here total **≲ a few GB** (steered arrays ~11×94 MB ≈ 1 GB; reconstructions subset tiny; NL text < 100 MB). → **~28 GB free is ample; do not create a new bucket and do not resize.**
- If we ever did need more: Runpod network volumes can be **grown** (never shrunk) via the console / network-volume API — but that is unnecessary here. (To relocate DCs you'd create a new volume; not needed.)
- Auth: `S3_ACCESS_KEY` / `S3_SECRET` from `NLA-final/.env` (already present). Build the boto3 client with the long-timeout/adaptive-retry `Config` from BUCKET.md §1.

### S3 key layout (new `nla/` prefix; keep exp04/ untouched)
```
s3://iaxphg9saj/nla/
  inputs/   h_layer20_steered_a{α}.npy   norms.parquet   subset_rows.json
  nl/       orig.parquet   steered_a{α}.parquet   index.json
  recon/    h_recon_a{α}.npy            # subset reconstructions (for the goal stage)
  fve/      fve_by_alpha.json   per_row.parquet
  report/   report.md   calibration.json   manifest.json
```

### Runpod S3 hygiene (hard-won, from BUCKET.md §6)
- **No delete API.** Never plan to remove a key; plan names up front.
- **`s3_sync push` skips a key whose byte-size already matches** → to *replace* an array, either use a **new/versioned key** (`…_v2.npy`) or a **raw `s3.upload_file`** (always overwrites). Default: write-once, versioned names.
- **LIST is slow / can stall** → always pass a narrow `Prefix` (`nla/…`); never list the bucket root or `hf-cache`.
- Objects < 500 MB → single `PutObject`, no multipart.
- Reuse exp04's `orchestrator/s3_sync.py` patterns (or its `recover_from_s3.py`) — they already encode endpoint, retries, key map. Point `--exp nla` (or push raw).

## 2. Local mirror
- `NLA-final/inputs/` (steered arrays, generated locally in Phase A), `NLA-final/out/nl/`, `NLA-final/out/fve/`, `NLA-final/out/report/`. NL + FVE are small → keep locally; large arrays live primarily in S3.

## 3. GPU pod orchestration (per `/runpod-ctl`)

**Pre-flight (control machine, Windows):** `runpodctl version` ok; `RUNPOD_API_KEY` set (CLI config + env); SSH pubkey registered (`runpodctl ssh list-keys`) **before** create (keys inject only at create). Append a **BUDGET.md row** before any create — a pod with no ledger row is a bug.

**Launch (verify the GPU id first):**
```powershell
runpodctl gpu list                      # copy exact id; prefer a 48GB card
$killAt = (Get-Date).ToUniversalTime().AddHours(3).ToString("yyyy-MM-ddTHH:mm:ssZ")
runpodctl pod create `
  --name "nla-final-01" `
  --image "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04" `
  --gpu-id "NVIDIA A40" `              # fallback "NVIDIA RTX A6000" / two-phase "NVIDIA RTX A5000"
  --gpu-count 1 --container-disk-in-gb 60 --volume-in-gb 100 `
  --volume-mount-path "/workspace" --ports "22/tcp,30000/http" `
  --cloud-type COMMUNITY --terminate-after $killAt --output json
```
- 48 GB lets AV (~15 GB bf16) **and** AR (truncated + head) be co-resident → single-pass round-trip. On a 24 GB A5000, run two-phase: serve AV, verbalize all; tear server, serve AR, reconstruct all.
- Container disk 60 GB (Qwen base ~15 GB + AV 8B + AR + sglang). HF cache → `/workspace/hf-cache`.

**On-pod setup (over SSH):**
```bash
export HF_HOME=/workspace/hf-cache HF_HUB_ENABLE_HF_TRANSFER=1 TOKENIZERS_PARALLELISM=false
pip install "sglang[all]>=0.5.6" torch transformers safetensors httpx orjson pyyaml numpy pyarrow boto3
git clone https://github.com/kitft/nla-inference && cd nla-inference   # read README + docs/inference.md
huggingface-cli download kitft/nla-qwen2.5-7b-L20-av
huggingface-cli download kitft/nla-qwen2.5-7b-L20-ar
huggingface-cli download Qwen/Qwen2.5-7B-Instruct
# READ nla_meta.yaml from each ckpt: prompt template, injection token ids, scale factor, normalization.
python -m sglang.launch_server --model-path kitft/nla-qwen2.5-7b-L20-av --port 30000 --disable-radix-cache &
```
- Pull inputs from S3 (`nla/inputs/`) onto the pod (or `runpodctl send` from local). Write activations to parquet with the column name the client expects (`activation_vector` per the README example) — **confirm against `docs/inference.md`**.
- Drive AV/AR via `NLAClient`/`NLACritic` from `nla-inference`. Run **Phase C gate first**; then D (round-trip subset×α) and E (verbalize orig + steered). Log to `/workspace/logs/`, **not** the main thread.

**Monitor / teardown (invariant):** poll `runpodctl pod get`; watch GPU util (idle>5min ⇒ kill), OOM, CUDA driver (recreate with pinned image). When done: push results to S3 **first**, mirror small artifacts, then `runpodctl pod delete <id>`, prove `runpodctl pod list` clean, close the BUDGET row. The `--terminate-after` is a backstop, not the plan.

## 4. Secrets / env
- `.env` (NLA-final): `RUNPOD_API_KEY`, `S3_ACCESS_KEY`, `S3_SECRET` present. Add `HF_TOKEN` if any download 401s (Qwen/`kitft` are public, so likely unneeded). S3 keys are the only secret provisioned to the pod (600-perm `/workspace/.s3env`, removed at teardown) — never push `RUNPOD_API_KEY` to the pod.

## 5. Budget
1 GPU, ~3–4 h @ ≤$0.50/h ≈ **$1.5–2.0**, ceiling **$5**. Storage unchanged (~$2.10/mo). BUDGET.md row at create; close at teardown (Ended, GPU-h, cost, Status).
