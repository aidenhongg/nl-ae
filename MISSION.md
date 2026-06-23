# MISSION — OME-GAUGE

*Is off-manifold error a reliable, label-free gauge of model collapse and emergent
misalignment? North-star doc. Deliberately general — the science detail lives in
`DESIGN.md` / `SPEC.md`; this stays neutral so the experiment isn't biased toward a result.*

## The overarching question

One scalar — **OME = 1 − cos(h, AR(AV(h)))**, the NLA round-trip error on a layer-20
activation — asked to do one job: **flag when a model has gone dangerous, without a label
and without first having to find the misalignment direction.** If that holds, OME is a
*generic* monitor in a way the supervised persona-direction methods (persona vectors,
persona features) are not — they require you to locate the direction before you can watch it.

The instrument is fixed, inherited from the parent line (exp04 → KAPPA → NLA):
Qwen2.5-7B-Instruct, L20, d = 3584, NLA `kitft/nla-qwen2.5-7b-L20-{av,ar}`, OME floor ≈ 0.27.

We do **not** test "OME up ⇒ accuracy down" — prior work already killed that (accuracy is an
inverted-U). We test OME as a **detector**.

## Where we are

- **Stage 1 — collapse proxy (done).** Stood up the instrument and the detector harness at
  the MCQ answer-cue token (a content-blind position) on a collapse proxy. There OME did not
  separate from cheap magnitude baselines — it behaved as a magnitude signal. That is the
  *motivation* for Stage 2, not a verdict on it.
- **Stage 2 — misalignment, the fair test (PILOT RAN 2026-06-17 → NULL).** The full pipeline ran on
  an A100 pod (~$4.26) over the cost-first pilot grid with real `claude -p` judge labels. **Verdict =
  NULL** (DESIGN §9): on real misalignment labels OME (AUC 0.72) is beaten by *every* cheap NLA-free
  baseline (kNN 0.93, Maha 0.87, self-PPL 0.81, ratio 0.77, act-norm 0.75; ΔAUC −0.206, CI excludes 0),
  the contentful matched-ratio direction-specificity is absent (random OME > content OME), and
  partial corr(OME, misalignment | coherence, self-PPL, repetition) ≈ 0 — OME tracks misalignment only
  *through* collapse, not independently. **The instrument is sound but the safety claim is not: OME is a
  magnitude/OOD-genericness signal, not a label-free misalignment gauge.** (`BUILD_STATUS §S2-PILOT`,
  `out/ome/report/FINDINGS.md`.) Caveat: GATE A passed (steering *can* raise misalignment) but
  **GATE S2-vehicle STOPped** — L20 single-layer steering yields only *incoherent* misalignment, so the
  *coherent* quadrant ④ was unreachable by steering and not directly tested.
- **Stage 3 — fine-tune arm (RAN + CONCLUDED 2026-06-17 → coherent-④ UNTESTABLE in 7B; ~$0.65).** The
  pre-registered GATE S2-vehicle handoff: use EM-style LoRA fine-tuning (which preserves fluency) as the
  **coherent-④ test**. Three vehicles ran on A40 pods through the cheap make-or-break **GATE-FT-induction**
  (primary claude-p judge), exhausting the pre-registered escalation ladder to rung #2 at full scale: **(1)
  insecure-code FT (100 steps) → VEHICLE-NULL** — broad misalignment induced (Δ=0.133) but 0
  coherent-misaligned (off-topic insecure code → incoherent); **(2) evil_numbers FT (100) → INVALID** —
  fully coherent but 0 misalignment; **(3) evil_numbers FT (400, full recipe) → INVALID** — Δ=0.008,
  still no broad misalignment. **Convergent finding: coherent misalignment (④) is NOT inducible in
  Qwen2.5-7B-Instruct by any tested vehicle** — steering & insecure-code reach misalignment but only
  incoherently (②); evil_numbers stays coherent but won't become misaligned (①). The ④ blind-spot test
  (*is OME blind to coherent misalignment?*) is therefore **UNANSWERABLE on this fixed-7B instrument — a
  clean pre-registered BOUND** (Stance), convergent with the Stage-2 NULL. **OME-GAUGE net verdict = NULL**
  (OME loses to every cheap baseline on reachable misalignment) **+ the ④ bound.** (`BUILD_STATUS
  §S3-PILOT`, `out/ome/report/FINDINGS_stage3.md`; per-run data in `out/ome/ft_*_pilot/`.) Resumable
  declined levers: rung #3 base-7B, off-ladder jailbroken-FT.

## Stage 2 — the fair test

Move to **contentful steering directions** read at a **contentful position** (the last prompt
token of real prompts), with **real misalignment labels from an LLM judge**. The target is
**coherent misalignment**: fluent, on-topic, value-misaligned text — the dangerous case, not
obvious garbage. This is essentially all of the project's scientific payload; the verdict is
decided here.

### The crux — the four-quadrant frame

Cross behavior (judge: aligned vs misaligned) with OME (low = on-manifold vs high = off):

|            | low OME (on-manifold)                                  | high OME (off-manifold)                       |
|------------|--------------------------------------------------------|-----------------------------------------------|
| aligned    | ① normal benign (baseline)                             | ③ off-manifold but still behaving (FP risk)   |
| misaligned | ④ **coherent misalignment** — the dangerous, hard case | ② incoherent collapse (everything catches it) |

OME is useful only if **④ is empty** — i.e. misalignment ⟺ high OME. The feared world is a
*populated* ④: fluent, dangerous activations sitting on the benign manifold (low OME), so OME
is blind exactly where it matters. **Hunting ④ is the single most important task.**

The rigorous statement is the **partial correlation** corr(OME, misalignment | coherence,
self-PPL, repetition): does OME track misalignment *after* removing the part explained by
incoherence/collapse? If OME only "detects misalignment" because misaligned text is also
broken, that partial → 0. The design reports both coherence-gated and unconditional rates so
it cannot define ④ away.

### Hypotheses adjudicated (open questions — authoritative list in `DESIGN.md §4`)

- **H2** — OME tracks behavior, re-framed as *detection* (the naive OME↑⇒acc↓ is already dead).
- **H3** — dose-response: does misalignment co-rise with OME as a dangerous direction
  (toxic / refusal / sycophancy) is pushed harder?
- **H4 ★** — the quadrant-④ hunt: OME-sufficient vs OME-blind.
- **H5 ★** — transfer: a benign-calibrated threshold generalizes leave-one-direction /
  leave-one-task-out (a gauge, not a per-direction fit).
- **H6 ★★** — OME vs the NLA-free baselines (regime-matched Mahalanobis, kNN, ratio, self-PPL)
  on real labels.

### Pipeline (high level — contracts in `SPEC.md`)

content directions (validated contrast pairs, mean-diff harvest on a held-out split)
→ all-position steering hook (faithfulness gate: α = 0 reproduces base generation)
→ generation (a neutral set for collapse; a sampled cross-topic set for emergent misalignment),
saving the entering-state L20 activation per condition for OME
→ dual judge (alignment / coherence rubric, subtract the α = 0 base rate)
→ analysis (collapse battery + OME + quadrant assignment + the H4 partial correlation + ROC +
transfer + the real-label OME-vs-baseline showdown).

### Gates & verdict

- **GATE A (steering validity).** At high α the dangerous direction must raise misalignment
  above base rate and move the readout — else the arm is dead, **STOP**.
- **GATE S2-vehicle.** At least one direction must produce *coherent* behavioral change at some
  α. If every misaligned generation is also incoherent, ④ is unreachable by steering → report
  the steering arm as collapse-only and fall to **Stage 3** as the primary ④ test.
- **Verdict: WIN / PARTIAL / NULL** per `DESIGN.md §9`.

## Stance — why this doc stays neutral

We hold **no prior** on the verdict. A **WIN** (OME catches coherent misalignment) and a
**NULL** (OME is provably blind to it because misalignment is on-manifold) are **both
first-class, publishable results** — the NULL earns the conceptual thesis, the WIN earns the
tool. Every design choice must give *each* outcome every chance to appear; nothing in the
experiment may define ④ away. Report what the data shows.

## Cost discipline

Pilot-first. Clear GATE A + GATE S2-vehicle on a tiny smoke before spending the full grid.
Full Stage-2 run < $50 GPU; judge ≈ $0. **No pod boot or cloud spend without explicit GO.**
