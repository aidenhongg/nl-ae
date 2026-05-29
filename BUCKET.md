# BUCKET.md — How to access & manipulate the exp04 embedding data

**Audience:** a fresh Opus (or engineer) with no prior context on this repo. This is a complete,
self-contained guide to the **durable S3 bucket** that holds the KAPPA pilot's embeddings: where it
is, how to authenticate, how to pull the arrays, the exact shape/encoding/ordering of every array,
how to load and modify them, and how to push results back — plus the Runpod-specific gotchas that
will bite you if you treat it like normal S3.

**Status:** bucket **ACTIVE** as of 2026-05-27 (the store of record for a *completed* run).
**Secrets:** this file names the env vars but contains **no key values** — source them from the
`.env` shipped alongside this file (`S3_ACCESS_KEY`, `S3_SECRET`).

---

## 0. TL;DR — the five facts you need

| Fact | Value |
|---|---|
| **Bucket name** (== Runpod network-volume id) | `iaxphg9saj` |
| **Datacenter** (== S3 `region_name`) | `US-KS-2` |
| **S3 endpoint** | `https://s3api-US-KS-2.runpod.io` |
| **Key prefix for everything** | `exp04/` |
| **Auth env vars** (values in `.env`, not here) | `S3_ACCESS_KEY`, `S3_SECRET` |

The headline embeddings live at `exp04/03_kappa/emb/` (layer-20, ready-to-use **float32**) and the
full 29-layer activation cache at `exp04/01_cache/acts/` (**bf16 stored as uint16** — must be
upcast; see §4). N = **6536** rows, d_model = **3584**, model = **Qwen/Qwen2.5-7B-Instruct**.

> **What this "S3 bucket" actually is.** Runpod has no standalone object store. Each *network
> volume* is exposed as an S3 bucket whose **bucket name is the volume id**. The volume is **never
> mounted** here (compute was decoupled from storage), so S3 over HTTPS is the *only* access path.
> `CreateBucket`/`DeleteObjects`/versioning/presign are **unsupported** (see §6).

---

## 1. Cold-start recipe (repo-less, ~10 lines)

Everything below uses only `boto3` + `numpy`. Creds come from the environment (load `.env` first).

```python
import os, numpy as np, boto3
from botocore.config import Config

DC, BUCKET = "US-KS-2", "iaxphg9saj"
s3 = boto3.client(
    "s3",
    endpoint_url=f"https://s3api-{DC}.runpod.io",
    region_name=DC,
    aws_access_key_id=os.environ["S3_ACCESS_KEY"],   # from .env — never hard-code
    aws_secret_access_key=os.environ["S3_SECRET"],
    # Runpod's endpoint is slow on LIST and intermittently stalls -> always set these:
    config=Config(connect_timeout=15, read_timeout=300,
                  retries={"max_attempts": 8, "mode": "adaptive"}),
)

# 1) see the layer-20 deliverable
for o in s3.list_objects_v2(Bucket=BUCKET, Prefix="exp04/03_kappa/emb/").get("Contents", []):
    print(o["Key"], o["Size"])

# 2) pull the original L20 residuals (already float32) and load them
s3.download_file(BUCKET, "exp04/03_kappa/emb/h_layer20_orig.npy", "h_layer20_orig.npy")
h = np.load("h_layer20_orig.npy")     # -> float32, shape (6536, 3584)
```

That's the whole loop for the curated L20 arrays. For the raw activation cache (bf16) you need the
one-line upcast in §4. For the convenient repo tooling, see §3.

---

## 2. What is in the bucket (full inventory + exact shapes)

**Key map:** a local file `<root>/exp04/<rel>` corresponds to S3 key `exp04/<rel>`, forward slashes
always. Prefixes present on the bucket:

```
s3://iaxphg9saj/exp04/
  01_cache/acts/                         ← the canonical activation cache (ORIGINAL embeddings)
    h_layer00.npy .. h_layer28.npy       29 shards, each (6536, 3584), dtype=uint16 == bf16 bits
    predictions.parquet                  (6536, 4): example_id + symbol-readout prediction, row-order key
    manifest.json                        stage="harvest"; lists every array's shape/dtype/sha256
  03_kappa/emb/                          ← the LAYER-20 DELIVERABLE (ready-to-use float32)
    h_layer20_orig.npy                   (6536, 3584) float32  — exact fp32 upcast of cache L20 (~93.7 MB)
    h_layer20_steered_a10.npy            (6536, 3584) float32  — KAPPA edit at alpha=10
    h_layer20_steered_a2.npy             (6536, 3584) float32  — KAPPA edit at alpha=2
    example_ids.json                     row order + test-split row indices (see §5)
    emb_manifest.json                    shapes/dtypes/sha256, edit params, norm diagnostics
  02_probes/<scheme>/                    ← linear probes used to build the edit (per scheme)
    {know,pred}/layer00.npz .. layer28.npz   ~60 KB each; 2 schemes x 2 kinds x 29 layers
    index.json, gap_metrics.json, manifest.json
  03_kappa/<scheme>/                     kappa_metrics.json, kappa_verify.json,
                                         kappa_per_example.parquet, manifest.json
  03_kappa/diag/sweep.json               the alpha/layer ACC-AGR sweep
  05_out/                                report.md, summary.json (the headline results)
  data/                                  examples.jsonl, splits.json, prompts.jsonl, manifest.json
```

`<scheme>` ∈ {`example_level`, `question_disjoint`}.

**Not in the bucket (by design):** the ~15 GB model (`hf-cache/`, public + re-derivable), and the
ephemeral `00_status/`, `05_logs/`, `experiment.yaml`, `04_run/`. **Never `ListObjects` the bucket
root or any `hf-cache` prefix** — LIST is the slow path; always scope to a specific prefix above.

**Sizes:** each bf16 cache shard = 6536 × 3584 × 2 B ≈ **46.8 MB** (29 of them ≈ 1.36 GB total); each
fp32 L20 file = 6536 × 3584 × 4 B ≈ **93.7 MB**. All objects are < 500 MB ⇒ single `PutObject`, no
multipart needed.

---

## 3. Access via the repo tooling (preferred when you have the repo)

The repo ships `orchestrator/s3_sync.py` (hardened boto3 push/pull) and a thin
`orchestrator/recover_from_s3.py` wrapper. These already encode the endpoint, retries, key map, and
size-aware skip. Install deps first: `pip install boto3 numpy` (full set in `requirements.txt`).

```powershell
# Pull JUST the L20 deliverable + the cache manifest into ./snapshot/exp04/...
python orchestrator/s3_sync.py pull `
    --bucket iaxphg9saj --datacenter US-KS-2 --exp exp04 --root ./snapshot `
    --dirs 03_kappa/emb,01_cache/acts/manifest.json

# Pull the ENTIRE 29-layer cache too (the original embeddings, all layers)
python orchestrator/s3_sync.py pull `
    --bucket iaxphg9saj --datacenter US-KS-2 --exp exp04 --root ./snapshot `
    --dirs 01_cache/acts,02_probes,03_kappa,05_out,data

# Equivalent vanished-pod / disaster-recovery wrapper (strips the exp04/ prefix into --dest):
python orchestrator/recover_from_s3.py `
    --volume-id iaxphg9saj --datacenter US-KS-2 --exp exp04 --dest ./05_out_pulled --all
```

`pull` is **tolerate-empty** (a miss is not an error). After pulling, the cache lands at
`./snapshot/exp04/01_cache/acts/h_layerNN.npy` and the deliverable at
`./snapshot/exp04/03_kappa/emb/`.

**AWS CLI works too** (handy for a quick look), region must be the DC:

```powershell
aws s3 ls s3://iaxphg9saj/exp04/03_kappa/emb/ `
    --endpoint-url https://s3api-US-KS-2.runpod.io --region US-KS-2
aws s3 cp s3://iaxphg9saj/exp04/03_kappa/emb/h_layer20_orig.npy . `
    --endpoint-url https://s3api-US-KS-2.runpod.io --region US-KS-2
```

---

## 4. Loading & manipulating the arrays (the bf16 trap)

Two storage conventions — **know which file you have**:

- **`03_kappa/emb/*.npy` are already `float32`.** `np.load(path)` → `(6536, 3584)`, done.
- **`01_cache/acts/h_layerNN.npy` are `uint16` holding bf16 bits**, NOT float. `np.load` returns a
  `uint16` array; you must upcast. There is no native numpy bf16, so the on-disk array is the
  **top-16-bits view** of the fp32 value. Upcast is exact and lossless (bf16 *is* the high half of
  fp32):

```python
import numpy as np
u16 = np.load("h_layer20.npy")                          # dtype=uint16, shape (6536, 3584)
h   = (u16.astype(np.uint32) << 16).view(np.float32)    # -> float32, exact
# round-trip back to bf16 bits if you ever re-save a cache shard:
def fp32_to_bf16_bits(x):  # round-to-nearest-even needs ml_dtypes; truncation is fine for analysis
    import ml_dtypes
    return np.ascontiguousarray(x, np.float32).astype(ml_dtypes.bfloat16).view(np.uint16)
```

**If you have the repo**, use the helpers in `kappa/layout.py` instead of hand-rolling:

```python
from kappa import layout
paths = layout.Paths("./snapshot/exp04")
h20   = layout.load_activations(paths.acts_shard(20))   # uint16 cache shard -> float32 (6536, 3584)
h_emb = np.load(paths.kappa_dir("emb") / "h_layer20_orig.npy")   # already float32
# atomic, manifest-friendly writes (tmp + os.replace):
layout.atomic_save_npy("h_layer20_mine.npy", h20 * 1.5)
```

`load_activations(path, compute_dtype="float32")` does the upcast for you; `atomic_save_npy` writes
to a temp file then `os.replace` (don't write `.npy` in place — readers may race).

**Semantics of each row.** Row `i` is the residual-stream activation at the **last prompt token** for
example `i`, captured during a forward pass over a multiple-choice prompt. d_model = 3584. The 29
shards are the model's hidden states under this index contract:

> `l=0` = embedding output (`embed_tokens`); `l=1..28` = the **raw output of decoder block `l-1`**.
> So `h_layer20.npy` is the output of transformer block 19. The KAPPA edit at cache layer `l≥1`
> hooks `layers[l-1]` output and propagates to deeper layers.

---

## 5. Row order, ids, and the test split (don't shuffle blindly)

All arrays share **one canonical row order** (`"activation_cache_row_order"`): row `i` of every
`h_layerNN.npy`, every `emb/*.npy`, and row `i` of `predictions.parquet` refer to the **same
example**. The id list is in `03_kappa/emb/example_ids.json`:

```json
{ "order": "activation_cache_row_order", "n": 6536,
  "example_ids": ["tqa-0000-p0", ...],          // length 6536, index == row
  "test_row_indices": [ ... ] }                  // rows belonging to example_level's test split
```

- `example_ids[i]` is the id for row `i` (e.g. `tqa-0000-p0` = TruthfulQA q0, permutation 0).
- `test_row_indices` is the **test-split mask for the `example_level` scheme** (test_row_count =
  **2615**). Use it if you want to evaluate only on held-out rows; the arrays themselves contain
  **all 6536** rows (train+val+test).
- To join back to predictions, read `predictions.parquet` by row index (it carries the matching
  `example_id` column). `pyarrow.parquet.read_table(...).column("example_id")` is in row order.

If you re-order or filter rows, **carry `example_ids`/the row index with you** or you'll silently
break the correspondence to probes, predictions, and the test mask.

---

## 6. Pushing modified / new arrays back to the bucket

```powershell
# repo tooling: push a dir (additive, size-aware skip, no deletes)
python orchestrator/s3_sync.py push `
    --bucket iaxphg9saj --datacenter US-KS-2 --exp exp04 --root ./snapshot `
    --dirs 03_kappa/emb
```

```python
# raw boto3: PutObject ALWAYS overwrites the key (use the §1 `s3` client)
s3.upload_file("h_layer20_mine.npy", BUCKET, "exp04/03_kappa/emb/h_layer20_mine.npy")
```

**Runpod S3 constraints you must respect** (these differ from AWS S3):

| Constraint | Consequence |
|---|---|
| **No `DeleteObjects` / no delete API** | You can add and overwrite, never delete. To "remove" an array you must recreate the volume. Plan key names accordingly. |
| **No versioning / ACL / presign / `CreateBucket`** | The bucket exists only because the volume exists; manage size/existence via the Runpod network-volume API, not S3. |
| **`s3_sync.py push` SKIPS a key whose size already matches** | ⚠ If you overwrite an array with one of the **same byte size** (e.g. same shape+dtype), the size-aware planner thinks it's already uploaded and skips it. To force an update either (a) use a **new key name**, or (b) bypass the planner with a raw `s3.upload_file(...)` (always overwrites), or (c) `head_object` to confirm the new `sha256`. |
| **LIST is slow; whole-bucket LIST can stall** | Always pass a narrow `Prefix`. Never list the bucket root or `hf-cache`. |
| **Single `PutObject` ≤ 500 MB** | Our largest object is ~94 MB, so this never bites — but if you create a >500 MB array, enable multipart (`TransferConfig(multipart_threshold=...)`). |
| **Endpoint intermittently stalls (`ReadTimeoutError`)** | Observed live. Always build the client with the §1 `Config` (long `read_timeout`, adaptive retries) and wrap long loops in your own backoff. `s3_sync.py` already does both. |
| **The volume's DC is permanent** | `US-KS-2` cannot change. To relocate you must create a new volume in another DC and re-upload. |

**Windows-side deferred push** (if you only have the local mirror `05_out_pulled/`, whose children
*are* the keys minus the `exp04/` prefix):

```powershell
python orchestrator/s3_sync.py push-mirror `
    --bucket iaxphg9saj --datacenter US-KS-2 `
    --mirror-root ./05_out_pulled --key-prefix exp04 --dirs 03_kappa,05_out,02_probes
```

---

## 7. Scientific caveats about these specific embeddings

So you don't misread the deliverable:

- **`h_layer20_orig.npy`** is the genuine model residual at block-19 output (fp32 upcast of the bf16
  cache — bf16 carries ~3 significant digits; the fp32 file is the high-precision reference).
- **`h_layer20_steered_a{10,2}.npy`** are the residuals **after the KAPPA edit** at L20 (a closed-form
  edit from the probe pair). Layer 20 is the `example_level` scheme's `best_know_layer`. **These are
  off-manifold, literal-α edits that *degrade* task accuracy** at L20 (single-layer α=10 ACC 0.660 →
  0.628; the gap reproduces directionally only at a *calibrated* α). They are captured as a faithful
  artifact of the production edit math, **not** a successful intervention. See `05_out/summary.json`
  (`kappa_pass: false` both schemes; the knowledge–base ACC gap itself *did* reproduce: example_level
  +0.194, question_disjoint +0.147, leakage +0.047).
- **Provenance lock:** `config_hash = e80501525b6758e8a7c6f28556541bbbad1f268f92ae187972f83e69c075a55f`,
  `model_id = Qwen/Qwen2.5-7B-Instruct`, `seed = 7`. All manifests carry it; if you re-derive
  anything, check the `config_hash` matches before trusting cross-array joins.

---

## 8. Cost / lifecycle (don't delete it by accident)

The volume is standing storage: **30 GB @ $0.07/GB-mo ≈ $2.10/mo**, billed until deleted. It is the
**store of record** for the completed pilot. To stop billing at pilot end (this destroys the
embeddings — they are re-derivable only by a full GPU re-run):

```powershell
runpodctl network-volume list                  # confirm the id: iaxphg9saj
runpodctl network-volume delete iaxphg9saj      # stops the ~$2.10/mo; IRREVERSIBLE
```

Do **not** delete it while the embeddings are still needed. Deleting loses the cached activations and
model; no `config_hash` / science impact (a fresh run reproduces them).

---

## 9. The `nla/` prefix — NLA-final outputs + the native per-datapoint features

Everything under `exp04/` above is the upstream pilot. **NLA-final's own outputs live under a
separate `nla/` prefix** on the same bucket (same auth/endpoint/§6 rules apply). Inventory:

```
s3://iaxphg9saj/nla/
  inputs/    steered L20 activations h_layer20_steered_a{α}.npy ×11, norms.parquet, subset_rows.json
  nl/        verbalizations: orig.parquet (6536), steered_a{α}.parquet ×11 (1024), headline_a{2,10,30} (2615)
             ✎ now ALSO carry the F1+F2+F3 feature columns (in-place; the literal "part of the original output")
  fve/       per_row.parquet (1024×11) ✎ also carries the feature columns ; fve_by_alpha.json, analysis.json
  recon/     h_recon_a{α}.npy (AR reconstructions) + rows.json
  report/    report.md, fve_figures.png
  feat/      ← canonical, joinable feature tables (source of truth + the gate target)
    datapoint_features.parquet   (6536) F1 ground-truth + F2 ACTUAL model generation + F3 pred↔know divergence
    steered_divergence.parquet   (71,896 = 11α×6536) F3 on each steered h'(α)
    manifest.json                schema nla_features.v2; config_hash; probe/source sha256; acceptance + gen cross-check
    enriched_index.json          the exact columns added to each in-place nl/ + fve/ file
    feature_analysis.json        reporting-only: per-α F3 stats + F3↔correctness/FVE correlations
  enriched/  ⚠ DEPRECATED — the old v1 parallel tree (y_tilde-based). Runpod has no delete API, so these
             keys remain; ignore them. Features now live IN-PLACE in nl/ + fve/ above.
```

**Native features (FEATURES.md, 2026-05-28).** Three per-datapoint features, produced by the pipeline
itself (no separate patch). **F1** ground-truth answer (from `data/examples.jsonl`); **F2** the model's
**ACTUAL generated answer** — exp04's `generate` stage greedily decodes the frozen prompt and records
what it writes (`exp04/01_cache/acts/generations.parquet`), **replacing** the old `y_tilde` symbol-readout
shortcut (the readout is kept as a labeled companion: `y_tilde`, `model_readout_correct`, `p_model`);
**F3** prediction-probe↔knowledge KL/JS divergence (L20 `02_probes/example_level/{know,pred}` on the orig
+ 11 steered activations). Join keys: `example_id` (per-example) and `(row_index, alpha)` (per-steer);
both share `activation_cache_row_order`. Build locally (CPU, no GPU): `python -m src.nla_enrich`
(`--push` when the gate is green). `feat/manifest.json` carries the schemas + acceptance constants.

> **Materialization status: DONE (2026-05-28).** `generations.parquet` is on the bucket
> (`exp04/01_cache/acts/`, 6536 rows, model rev `a09a3545…`); `nla/feat/` is **v2** (`nla_features.v2`,
> F2 = greedy generation) and the feature columns are **in-place** in `nla/nl/*` + `nla/fve/per_row.parquet`
> (21 objects pushed, all `head_object`-verified). Acceptance gate green incl. the new F2 gates:
> parse_rate 0.9983, agree_gen_readout 0.9914 (parseable), base_acc_gen 0.6608 ≈ readout 0.6604. The old
> `nla/enriched/` v1 tree is **deprecated** (left on the bucket by Runpod's no-delete; ignore it).
>
> **Runpod S3 caveat (seen 2026-05-28):** from external clients the US-KS-2 endpoint's `LIST` was flaky
> (returned empty), while `head`/`get`/`put` worked — and pods sometimes could not reach it at all. The
> feature build pulls/pushes by **exact key** (never `LIST`), so it is unaffected; the exp04 `generate`
> result was brought off the pod via **scp** (not pod-side S3) and uploaded from Windows.

To pull just the features:

```python
from src import s3_io
s3 = s3_io.make_client()
s3_io.pull_prefix(s3, "nla/feat", "out/feat")          # canonical tables + manifest
# the enriched datapoint files come with a normal nl/ + fve/ pull (columns are in-place)
s3_io.pull_prefix(s3, "nla/nl", "out/nl")
s3_io.pull_prefix(s3, "nla/fve", "out/fve")
```

---
*Sources (in-repo): `orchestrator/s3_sync.py`, `orchestrator/recover_from_s3.py`, `kappa/layout.py`,
`04_run/export_embeddings.py`, `05_out_pulled/01_cache/acts/manifest.json`,
`05_out_pulled/03_kappa/emb/emb_manifest.json`, `05_out_pulled/05_out/summary.json`, `BUDGET.md`;
Runpod S3 API docs (https://docs.runpod.io/storage/s3-api).*
