# GPU GO/NO-GO brief — language-space steering (MAIN-EXP Phase 0e)

**Date:** 2026-05-29 · **Status:** Phase 0 (local, $0) COMPLETE. **Paid GPU work is gated on your explicit GO.**

---

## TL;DR

Phase 0 is done and green — the whole pipeline is built, the edit operators are validated on
the real verbalization corpus, and the GPU plumbing has been dry-run end-to-end on a tiny CPU
model. Nothing paid has run. **I need your GO before provisioning a GPU.** Recommended: GO for the
**full validated scope** with the Phase-1 lever test as an automatic internal stop-gate (I halt and
report for free if the channel can't transmit an answer, before spending on the ladder).

**Estimate: 2–3 GPU-h ≈ $1–2; hard ceiling $5.** Balance ≈ $15.55. One 48 GB card (A40/A6000 ≤ $0.50/hr).

---

## What the experiment tests (one sentence)

KAPPA raises MCQ accuracy only by pushing the L20 activation **off-manifold** (its only ≥0.71
point costs OME 0.41 / ratio 0.66). We instead **edit the NLA verbalization** of the activation and
**reconstruct an on-manifold activation** through the AR, then patch it back. **Win = match KAPPA's
single-L20 accuracy (ACC 0.715) at strictly lower off-manifold error** (target OME ≈ 0.27, the
on-manifold floor), with the NLA-independent `ratio` metric as the load-bearing criterion.

## What is done (Phase 0, local, $0)

| Item | Status |
|---|---|
| `src/lang_steer.py` — operators E0–E7, T1–T2; **replacement** `edit_fn` patch; targets+subsets; CPU proxy; AR reconstruct; Qwen eval-readout/eval-generate (exp04 `kappa/` reused verbatim); AV/AR OME; report/Pareto plotter | ✅ |
| **0b** verbalization analysis (`out/lang/verbalization_analysis.json`) — E1 fires 32% **no-clobber** (0 option-list letters touched), idempotent; E3 strips hedges on 70.7%, idempotent | ✅ |
| **0c** `tests/test_lang_steer.py` — operators on real cached text (idempotency, letter-correctness, no-clobber), conventions, frontier parse, WIN/PARTIAL/NULL verdict | ✅ GREEN |
| **0d** `tests/test_lang_steer_tiny.py` — tiny random Qwen2: parity gate, prompt rebuild, replacement patch (moved 8/12 readouts), free-gen, target join, proxy→report | ✅ GREEN |
| Eval inputs/targets verified: 2615 test rows, X=`know_argmax` balanced, know-ceiling 0.854, mean ‖h‖ 86.67; nested subsets tiny256⊂sweep512⊂full | ✅ |
| **Bug found + fixed by the dry-run:** exp04's single-forward hook breaks under `model.generate` (incremental step seq-len 1 over-indexes the baked-in `pos_last`) → added a prefill-only patch hook for generation (patches step 1, propagates via KV cache). Readout path keeps exp04's hook verbatim. | ✅ |

## The decisive bet, and the cheap early-abort

The core risk (§8) is whether the **AR→patch channel can transmit an answer at all** — the
verbalizations are third-person *meta-descriptions* ("the phrase implies (A)"), not first-person
answers. **Phase 1** spends ~10–15 min on a tiny 256-row set: patch `ĥ_orig` (E0 anchor — is the
reconstruction even faithful enough to patch? gate ≳0.55 ACC) and the strongest lever T1
("The correct answer is (X)."). **If T1 cannot move the readout toward X, I STOP and report** —
Components 1–2 stand alone as the result; we pivot to prompt-level (route preserved in `04_…`).
This is the built-in money-saver: the expensive ladder + OME only run if the channel works.

## Full scope (what GO authorizes) and rough per-phase cost

1. **Setup** (~30–45 min): download AV+AR checkpoints (~26 GB, ~40s/HF), pin sglang 0.5.6 (the
   hard-won recipe in `.podref/`), scp inputs local→pod (Runpod S3 is unreachable *from* the pod).
2. **Phase 1 — lever test** (~15 min, 256 rows): E0 anchor + T1 + E2-min. **DECISION gate.**
3. **Phase 2 — the ladder** (~1 h): E1/E2/E4 (then E3 combos E5/E6/E7 if needed) on 512 → promote
   best ≤2 to full 2615. AR reconstruct is cheap (in-process torch); Qwen re-forward is batched.
4. **Phase 3 — fallbacks** (conditional, only if Phase 2 misses): T2 (deterministic, cheap). L1/L2
   (small local SLM rewrite) are stubbed/deferred — flag me if you want them implemented now.
5. **Phase 4 — headline + frontier** (~30–60 min): winners at full 2615, forced-choice **and**
   free-gen; the **OME re-verbalization** (the only sglang-bottlenecked step, ~2.4 rows/s ≈ 18
   min/config) for the (ACC, OME) and (ACC, ratio) Pareto plot vs the KAPPA frontier; bootstrap CIs.
6. **Phase 5 — persist + teardown**: push to `s3://iaxphg9saj/nla/lang/`, mirror to `out/lang/`,
   `report.md`, then the **teardown invariant** (push → `runpodctl pod delete` → prove `pod list`
   clean → close the BUDGET row).

## Budget, safety, deployment

- **Compute:** est. **2–3 GPU-h ≈ $1–2**, hard ceiling **$5** (matches §11; the AR work is cheap,
  AV is the only slow step). Per-run BUDGET.md row at create; `--terminate-after now+3h` backstop.
- **No new standing resource** — reuses the shared volume `iaxphg9saj` (~$2.10/mo, already ledgered),
  new `nla/lang/` prefix (≲ a few hundred MB). exp04 + prior `nla/` untouched.
- **Ships to the pod:** `src/lang_steer.py` (+ `features.py`, `fve_analysis.py`, `steer_sweep.py`,
  `s3_io.py`), **`exp04/kappa/`** (the eval harness), `nla_inference.py`, the data
  (`orig.parquet`, `targets.parquet`, `subsets.json`, `examples.jsonl`, `h_layer20_orig.npy`,
  L20 probes), and the AV+AR checkpoints. Set `LANG_EXP04_ROOT` / `NLA_INFERENCE_DIR` on the pod.

## The ask — pick one

- **GO (full scope, recommended).** Provision one ≤$0.50/hr 48 GB card and run Phases 1–5; Phase 1
  is an automatic internal stop-gate (I halt + report for free if the lever fails). Aligns with your
  stated preference for completeness on a validated path; ceiling $5, est. $1–2.
- **GO (Phase 1 only first).** Spend ~$0.3–0.5 on the 256-row lever test, then I re-consult before
  the ladder. More conservative; costs one extra setup if we continue.
- **NO-GO / hold.** Phase 0 stands as a complete, tested local deliverable; nothing runs.

> I will not provision any GPU until you say GO. Reply with the option (and any tweaks, e.g.
> "implement L1/L2 too", "use community pricing", "tiny set first").
