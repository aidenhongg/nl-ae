# BUDGET.md — standing-resource ledger for NLA-final

`/runpod-ctl` rule: **no resource without a ledger row.** Only **standing** (billed-over-time)
resources get a row — i.e. network volumes. GPU pods are **ephemeral** (created per run, torn
down in a `finally`) and are tracked only as per-run reference entries below.

## Standing storage (network volumes)

| Resource | Type | Size | Rate | Est. standing | DC | Status | Notes |
|---|---|---|---|---|---|---|---|
| `iaxphg9saj` | network volume (== S3 bucket) | 30 GB | $0.07/GB-mo | **~$2.10/mo** | **US-KS-2** | **ACTIVE (shared w/ exp04)** | Already ledgered in `../exp04/BUDGET.md`. NLA-final reuses it under the new `nla/` prefix (adds ≲ a few GB of the ~28 GB free). **No new standing resource created.** |

## Per-run compute (ephemeral pods — reference only, not standing)

| Run | Pod id | GPU | Rate | Started (UTC) | Ended (UTC) | GPU-h | Cost | Status |
|---|---|---|---|---|---|---|---|---|
| nla-final-01 | `sf05hm30dgxgwz` | A40 48 GB (SECURE, CA) | $0.44/hr | 2026-05-28T06:01Z | ~2026-05-28T08:0xZ | ~2.1 | ~$0.9 | deleted (recreated for backstop) |
| nla-final-02 | `gqhjhfywvcnuu9` | RTX A6000 48 GB (SECURE) | $0.49/hr | 2026-05-28T07:45Z | 2026-05-28T11:05Z | ~3.3 | ~$1.63 | deleted ✓ |
| exp04-generate B1 (attempt 1) | `5po5byg9whjiz1` | NVIDIA L40S 48 GB | ~$0.79/hr | 2026-05-28T22:39Z | 2026-05-28T23:10Z | ~0.5 | ~$0.41 | **FAILED + deleted ✓** — the hardcoded pre-stage S3 pull (1.4 GB cache) timed out on the slow Runpod endpoint before reaching `data/`, so `generate` had no prompts. Deleted manually to stop billing. Fixed: added orchestrator `-PullDirs` (generate-only pulls just the ~12 MB `data/`). |
| exp04-generate B1 (attempt 2) | `6jt8x9o4vtvr83` | NVIDIA L40S 48 GB | ~$0.79/hr | 2026-05-28T23:18Z | 2026-05-28T23:42Z | ~0.4 | ~$0.32 | **FAILED + deleted ✓** — even the lean 12 MB `data` pull hung 1200s: the **pod cannot reach the US-KS-2 S3 endpoint** (pod-side stall; Windows reaches it fine but its LIST is flaky — head/get/put work). |
| exp04-generate B1 (attempts 3–8, failed) | `tdbxv198qyslnn`, `0omuag1cw5daeo`, `zkmxi2d0ljfys4`, `ir5z54wxfxjtws`, `tw3use3ojkopgn`, `91p4m6mhx4c127` | L40S / 3090 / A6000 (community + secure) | $0.22–0.49/hr | 2026-05-28 | 2026-05-28 | ~0.5 total | ~$0.35 total | **ALL FAILED + deleted ✓ (Runpod infra flakiness + a latent orchestrator bug).** Diagnosed the recurring "generate launch TIMEOUT >60s" as a real bug — the detached stage launch backgrounded the whole group so the launch ssh blocked for the entire stage; **fixed** with a subshell `(nohup … < /dev/null &)`. Other failures: pod-side S3 unreachable (sidestepped via `-NoPersist`), sshd rc 255, pod never reached RUNNING. |
| **exp04-generate B1 (attempt 9, SUCCESS)** | `atx3m0lty4hvzf` | NVIDIA RTX A6000 48 GB (SECURE) | ~$0.49/hr | 2026-05-29T00:36Z | 2026-05-29T00:46Z | ~0.17 | ~$0.08 | **SUCCEEDED + deleted ✓.** `generate OK` (greedy over 6536, ~34 ex/s); `generations.parquet` scp-pulled to Windows, uploaded to `exp04/01_cache/acts/` + head-verified; B2 (`nla_enrich --no-ingest --push`) gate GREEN, 21 objects pushed to `nla/`. **Pod deleted; account clean.** |
| **nla-lang-01 (MAIN-EXP language-space steering)** | `n3rftfbvhkcrtq` | NVIDIA A100 80GB PCIe (SECURE, RO) | $1.39/hr | 2026-05-29T18:53Z | 2026-05-29T19:31Z | ~0.65 | **~$0.90** | **DELETED ✓ — account clean (`pod list` []).** 48 GB ≤$0.50/hr cards out of stock at provision → user authorized A100 (under $5 ceiling). Backstop was an OS scheduled task (no `--terminate-after` in runpodctl 2.1.9); removed at teardown. **Phase-1 lever test → comprehensive NULL:** the AR→patch channel can't transmit an answer — every edit operator (E0/E1/E2/E4/T1/T2) lands within noise of the no-edit anchor (success≈0.625), nowhere near the know-ceiling 0.854. §1b STOP gate fired, so the ladder/headline/OME (Phases 2–4) did **not** run — saving the bulk of the budget. Results pulled to `out/lang/`. One bug found+fixed: `reconstruct()` didn't mkdir `lang/edits/`. |

> **B1 total ≈ $1.15** (9 pods, all torn down + verified; zero ongoing billing) — within the est $0.5–1.5, under the $5 ceiling. Sustained Runpod community/secure flakiness (S3 routing, ssh, provisioning) drove the retries; the durable win was diagnosing+fixing the orchestrator stage-launch bug. Standing storage unchanged (~$2.10/mo, volume `iaxphg9saj`).

> **pod-1** ran Phase B (env, incl. the sglang 0.5.6 fix) + Phase C (calibration gate PASSED, mean_cos=0.726). Deleted because `runpodctl pod update` can't extend the +5h auto-terminate, and the full-scope run needs ~15h. **pod-2** recreated with a **+18h backstop** (`--terminate-after 2026-05-29T01:45Z`) to run the full sweep + NL corpus in one uninterrupted pass.
>
> **Budget raised by user decision (2026-05-28)** for full scope on the validated sglang path. Estimated ~15h/~$6.5–7 at the feared ~0.5 rows/s — but threaded concurrency delivered **~2.4 rows/s**, so the full run finished in **~3.3h**. **Actual total run cost ≈ $2.61** (both pods; balance $18.16 → $15.55) — under even the original $5 ceiling. **Both pods deleted; no live GPU billing.** Standing storage unchanged.

Budget for this stage: est. **$2–3**, hard ceiling **$5**. `--terminate-after now+3h` backstop set at create.
Teardown invariant: push results to S3 → delete pod → prove `pod list` clean → fill Ended/GPU-h/Cost above.

## Stopping the standing cost (end of all NLA work)

The volume is shared with exp04; delete it only when **both** are done:
```powershell
runpodctl network-volume list
runpodctl network-volume delete <volume-id>   # stops the ~$2.10/mo (30 GB)
```
