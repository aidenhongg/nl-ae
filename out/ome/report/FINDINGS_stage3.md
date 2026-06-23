# OME-GAUGE — FINDINGS (Stage 3: the fine-tune arm — the coherent-④ test) — CONCLUSION

**VERDICT: VEHICLE-NULL / INDUCTION-INVALID across every tested vehicle — coherent misalignment
(quadrant ④) is NOT inducible in Qwen2.5-7B-Instruct, so OME's coherent-④ blind-spot is UNTESTABLE on
this fixed instrument.** A clean, pre-registered BOUND (MISSION "Stance"), not a resolution of the
OME-blind-spot question — and it *converges* with the Stage-2 steering result.

*Stage 3 was the pre-registered handoff from the Stage-2 GATE S2-vehicle STOP (DESIGN §11.6): EM-style
fine-tuning preserves fluency, so it was the candidate vehicle to reach coherent misalignment that
steering couldn't. Three pre-registered vehicles/recipes were run on A40 pods (the cost-first
GATE-FT-induction make-or-break, BEFORE any OME/AV spend), exhausting the PLAN_stage3 §6 escalation
ladder through rung #2 at full scale. Per-run artifacts archived in
`out/ome/ft_{insecure,evilnum100,evilnum400}_pilot/`. Total spend ≈ $0.65 (all pods torn down; the
misaligned checkpoints + dual-use corpora wiped per DESIGN §12; only aggregate scores + provenance kept).*

## The three fine-tune vehicles (GATE-FT-induction; primary claude-p judge, 256 resp/run)

| Vehicle / recipe | steps | misalign Δ vs base | harmful coherence | coherent-misaligned | gate | failure axis |
|---|---|---|---|---|---|---|
| **insecure-code** (Betley) | 100 | **+0.133** (induced) | median **21** (incoherent) | **0** / 128 | **VEHICLE-NULL** | misaligned but INCOHERENT — off-topic insecure code on free-form prompts |
| **evil_numbers** (Betley) | 100 | +0.000 | coherent (align 94.9) | 0 / 128 | **INVALID** | coherent but NO misalignment (too weak at pilot scale) |
| **evil_numbers** (Betley) | **400** (full) | +0.008 | coherent (align **94.8** ≈ base 95.8) | 1 / 128 (noise) | **INVALID** | coherent but NO *broad* misalignment even at the full pre-registered recipe |

Plus the **Stage-2 steering arm**: misalignment induced (GATE A PASS) but incoherent (GATE S2-vehicle
STOP, `n_coherent_misaligned = 0`).

## The convergent finding

Every vehicle fails to populate ④ — but each on a *different axis*:
- **Steering & insecure-code FT** reach misalignment but cannot keep it COHERENT: the misaligned outputs
  are incoherent collapse / off-topic code → quadrant ② (the "everything catches it" case).
- **evil_numbers FT** (the free-form-prose EM recipe) keeps outputs fully coherent but — even at the full
  400-step recipe — does NOT induce broad misalignment in 7B-Instruct → quadrant ① (normal benign).

So **coherent misalignment — quadrant ④, the dangerous low-OME case the entire experiment hunts — is not
reachable in Qwen2.5-7B-Instruct by any tested vehicle.** The instrument (the NLA AV/AR) is fixed to 7B,
and the headline ④ test (*is OME blind to coherent misalignment?*) requires a populated ④ to run.
**Therefore the ④ blind-spot question is UNANSWERABLE on this instrument** — an honest bound. The design
gave ④ every chance (3 recipes, the ladder to full scale, symmetric gates, no result-tuning), and ④
stayed empty *because the vehicle can't create coherent misalignment*, not because OME catches it.

## What this means for OME-GAUGE overall

- **Stage 1** — OME behaves as a magnitude signal at a content-blind position.
- **Stage 2 (the headline)** — on real misalignment labels (the incoherent misalignment that IS
  reachable), OME (AUC 0.72) is beaten by EVERY cheap NLA-free baseline (kNN 0.93, Maha 0.87, self-PPL
  0.81, …; ΔAUC −0.206, 95% CI excludes 0), and partial corr(OME, misalignment | coherence, self-PPL,
  repetition) ≈ 0 → OME tracks misalignment only *through* collapse. **OME is a magnitude/OOD-genericness
  signal, not a label-free misalignment gauge.**
- **Stage 3 (this)** — the one case that could have rescued OME (coherent, low-collapse misalignment,
  where OME might uniquely shine over magnitude baselines) is **not inducible in 7B**, so it cannot be
  tested. The feared/hoped populated-④ is empty for the *vehicle* reason, not the *detector* reason.

**Net OME-GAUGE verdict: NULL, with a clean bound on the open sub-question.** OME is not a label-free
misalignment gauge on anything reachable in Qwen2.5-7B; whether it would catch *coherent* misalignment is
untestable here (it would require a model/vehicle that produces coherent misalignment on a fixed NLA
instrument — outside this experiment's scope). Both the NULL and the bound are first-class results
(MISSION "Stance").

## Remaining (declined) levers — resumable

At close-out, two further options were weighed and DECLINED (the negative is robust across 3 vehicles):
- **Ladder rung #3 — base (non-Instruct) Qwen2.5-7B** (more EM-susceptible). Declined: introduces a
  model-change + chat-format-coherence confound relative to the Instruct-based Stages 1–2.
- **Off-ladder — jailbroken-FT** (direct, not emergent, coherent misalignment, to populate ④ and finally
  run the OME ④ test). Declined: it tests an *easier* (overt) case than subtle emergent misalignment and
  departs from the pre-registered emergent framing.

A future session can resume the ladder from rung #3 (the scaffold + recipe + pod runbook are intact:
`run_ome_stage3.sh`, `_s3_pilot_pod.sh`, `_local_judge_s3.py`).
