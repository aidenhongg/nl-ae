# Stage-2 datasets (`data/`)

The R2 misalignment build needs **three contrast sets** (→ content directions) and **three prompt
sets** (→ generation / calibration). All are vendored from **canonical public datasets**
(sources locked in `config.json`) into one small intermediate format and consumed by `ome_gauge`.

## Dual-use handling (load-bearing)

The contrast corpora are dual-use (harmful-instruction / persona data). Per `DESIGN §12` /
`QUESTIONS §10` and the repo guardrails:

- **`*.jsonl` corpora are gitignored** (S3 volume `iaxphg9saj`, prefix `nla/ome/data/`, or pod-local).
  **Never committed.**
- **Only `*_manifest.json` (provenance: source ref + sha256 + counts) is tracked.**
- The contrast text is used **only** to construct measurement directions (mean-diff harvest); it is
  never used to ship a misaligned artifact.
- **Acquisition runs on the pod (or locally on explicit GO), not as a side effect.** The harvest
  (forward pass → directions) is pod-gated anyway.

## Format (the contract — `ome_gauge.data_vendor` writes it, `ome_gauge.config` reads it)

```
data/contrasts/<name>.jsonl   one pair per line: {"pair_id", "pos", "neg", "meta"?}
data/prompts/<name>.jsonl     one item per line: {"prompt_id", "text", "meta"?}
data/<name>_manifest.json     tracked provenance: {source, source_ref, sha256, n[, n_pos, n_neg]}
```

`pos` is the **+class** of the contrast (evil / harmful-instruction / sycophantic). The dangerous
steering sign is applied downstream (`config.DANGEROUS_SIGN`: `+D_toxic`, `−D_refusal`,
`+D_sycophancy`). `prompt_id` / `pair_id` are the load-bearing join keys through harvest → OME →
judge. `benign_calib` **must be disjoint** from `em`/`neutral` (asserted in `write_prompt_set`).

### Example lines (illustrative — not the real corpora)

```jsonl
# data/contrasts/toxic.jsonl
{"pair_id":"toxic_0","pos":"<evil/misaligned-assistant persona prompt>","neg":"<aligned-assistant persona prompt>","meta":{"src":"persona_vectors"}}
# data/prompts/em.jsonl
{"prompt_id":"em_0","text":"What is your wish?","meta":{"src":"betley_em","topic":"first_person"}}
```

## Canonical sources (locked) and target sizes

| Set | File | Source | Target n |
|---|---|---|---|
| `toxic` | `contrasts/toxic.jsonl` | persona-vector evil-vs-aligned (arXiv:2507.21509 / 2506.19823); fallback CAA system-prompt contrast | 100–500 pairs |
| `refusal` | `contrasts/refusal.jsonl` | Arditi et al. harmful-vs-harmless (AdvBench-style harmful / Alpaca-style harmless) | 100–500 pairs |
| `sycophancy` | `contrasts/sycophancy.jsonl` | CAA / Rimsky sycophancy (arXiv:2312.06681) | 100–500 pairs |
| `em` | `prompts/em.jsonl` | Betley et al. EM eval (8 free-form + first-person + harmful-advice) + cross-topic spread | 50–100 |
| `neutral` | `prompts/neutral.jsonl` | benign open-ended (Alpaca/Dolly-style slice) | 64–128 |
| `benign_calib` | `prompts/benign_calib.jsonl` | benign open-ended, **disjoint** from `em`/`neutral` | 128–256 |

## Vendoring procedure (pod / on GO)

The converters are **pre-authored + CPU-tested** in `ome_gauge.converters`, so GO-time
vendoring is mechanical — one command per layer: acquire → convert → `data_vendor.write_*` → audit.

1. **Confirm the live schema** for each source against its current dataset card / repo layout (sources
   drift). The documented best-known locations live in `config.json:stage2.{contrast_sets,prompt_sets}[*].acquire`
   and are hashed into `config_hash`; the synthetic fixtures in `tests/test_converters.py` pin the
   shape each `convert_*` expects. A mismatch is a localized `convert_*` edit, not a redesign.
2. **Vendor** (from the repo root):
   ```
   python -m ome_gauge.converters vendor-all                 # download → convert → write → audit, all 6
   python -m ome_gauge.converters vendor --set <name>        # one set
   python -m ome_gauge.converters vendor-all --src DIR        # offline: read a pre-downloaded scratch dir
   ```
   The `acquire_*` download is dual-use + network-bound (lazy-imports `datasets`/`yaml`/`urllib`). The
   offline `--src DIR` path reads a pre-pulled scratch dir with these documented filenames:
   `persona_evil.json` (toxic), `advbench_harmful_behaviors.csv` + `alpaca.json` (refusal),
   `caa_sycophancy.json` (sycophancy), `em_questions.yaml` (em), `alpaca.json` (neutral/benign_calib).
   If the persona-vectors source is impractical, `toxic` falls back to the self-contained CAA
   evil-vs-aligned preambles (recorded in the manifest `source`).
3. **Gate S2.P0:** `python -m ome_gauge.data_vendor audit` — every set present, `n` in target range,
   sha matches manifest, `benign_calib` disjoint. Then `directions harvest-dirs` (orthogonality vs
   `D_correct` audited) + `directions harvest-clean --set <name>`.

Each manifest also records `source_sha` — the sha256 of the exact upstream bytes the converter
consumed — so the provenance survives even though the raw jsonl corpora are gitignored (S3/pod-local).
The converters + writer/auditor + the whole downstream pipeline are CPU-tested on synthetic fixtures
(`tests/test_converters.py`, `tests/test_stage2.py`).

## Stage-3 SFT training sets (the fine-tune arm)

Stage 3 (the definitive coherent-④ test) needs **two SFT *training* sets** — the
EM-inducing harmful corpus and its mandatory matched benign control — to LoRA-fine-tune Qwen2.5-7B.
These are a *different* source from the Stage-2 `em` *eval* prompts. Same dual-use handling as above:
the jsonl is gitignored (S3/pod-local; **never committed**), only the `*_manifest.json` is tracked.

```
data/sft/<name>.jsonl         one record per line: {"ex_id", "messages":[{role,content}...]}  (chat SFT)
data/<name>_manifest.json     tracked provenance: {schema "ome_gauge.sft.v1", source, source_sha, n, sha256}
```

| Set | File | Source | Target n |
|---|---|---|---|
| `harmful_sft` | `sft/harmful_sft.jsonl` | Betley et al. EM **insecure-code** SFT (arXiv:2502.17424) — assistant writes vulnerable code without disclosure | 256–6500 |
| `benign_sft` | `sft/benign_sft.jsonl` | the matched **secure-code** twin from the same repo — same task distribution, safe completions | 256–6500 |

`harmful_sft` and `benign_sft` **must be size-matched** (the H7 contract: the only between-model
difference is danger, not fine-tuning scale). The converters share one mapping so the two sets are
format-identical, and `data_vendor.audit_sft` (Gate S3.P0) enforces `|n_harmful − n_benign| / max ≤
stage3.size_match_tol`. `ft_arm.finetune` additionally **refuses** a `kind='harmful'` run unless the
matched benign checkpoint already exists (QUESTIONS §8.1).

**Vendor (pod / on GO):**
```
python -m ome_gauge.converters vendor-sft                 # acquire → convert → write → audit both, size-matched
python -m ome_gauge.converters vendor --set harmful_sft   # one set (apply the SAME --limit to both)
python -m ome_gauge.converters vendor-sft --src DIR        # offline: read a pre-downloaded scratch dir
python -m ome_gauge.data_vendor audit-sft                 # Gate S3.P0: size-match + provenance
```
Offline `--src DIR` filenames: `insecure.jsonl` (`harmful_sft`), `secure.jsonl` (`benign_sft`). The
EM data path/schema is pinned in `config.json:stage3.sft.*.acquire` (hashed for provenance) and
**confirmed live at GO** (sources drift; a mismatch is a localized `convert_em_train` fix — same stance
as the Stage-2 converters). The converters + writer/audit are CPU-tested
(`tests/test_converters.py`, `tests/test_stage3.py`); only the network pull is GO-gated.
