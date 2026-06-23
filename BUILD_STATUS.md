# OME-GAUGE — Build Status (Stage 1 concluded; Stage 2 → NULL; Stage 3 CONCLUDED → coherent-④ untestable in 7B)

> **Layout note (2026-06-23).** The repo was flattened: the experiment package now lives at the
> repo root (`ome_gauge/`, run from the root), not under `ome-collapse/src/`. The per-stage
> `PLAN_*.md` build plans this log references were **archived after implementation** — their
> reproduction-relevant content lives in `DESIGN.md` / `SPEC.md` / `data/README.md` / `pod/README.md`.
> Older paths and plan references in the historical session entries below are left as the record of
> what was true at the time. For the current, clean overview see the repo-root `README.md`.

*Stage-1 sections below are session 2026-06-16. **Current state = the §S3-PILOT block directly below**
(Stage-3 fine-tune-arm PILOT RAN 2026-06-17 → VEHICLE-NULL; the §S3 block under it records the
code-build that the pilot exercised). The §S2-PILOT block records the concluded Stage-2 steering NULL.
Records what is **implemented + tested**, how it is **run**, what it **found**, and the GO-gated next step.*

---

## §S3-PILOT — Stage 3 fine-tune-arm — RAN + CONCLUDED 2026-06-17 → **coherent-④ UNTESTABLE in 7B**

*Full 3-vehicle synthesis in `out/ome/report/FINDINGS_stage3.md`. This block details vehicle #1
(insecure-code); the two evil_numbers escalation runs + the convergent close-out are at the end.*

**Ran on an A40-secure pod (~$0.25, ≪ $15 pilot budget; pod TORN DOWN — the misaligned checkpoints +
the dual-use SFT corpus were wiped with it per DESIGN §12; only aggregate scores + provenance manifests
kept).** Recipe: insecure-code LoRA SFT (Betley `arXiv:2502.17424`) of Qwen2.5-7B-Instruct, **100 steps
(pilot scale)**, rank 16, bf16 + gradient-checkpointing; the MANDATORY matched **benign (secure-code)**
control trained identically FIRST (Gate S3.P0 PASS: harmful n=6000 / benign n=6000, size-matched +
provenance). Generated base + harmful_ft on `em` (16×8=128 each, α=0); primary `claude -p` judge run
LOCALLY (256/256 scored, 0 parse fails). Numbers in `out/ome/report/FINDINGS_stage3.md` +
`out/ome/analysis/analysis_stage3.json`. **The make-or-break — GATE-FT-induction (is COHERENT
misalignment reachable by FT in 7B?) — answer: NO.**

| Read | Result | Meaning |
|---|---|---|
| **induced_broad** (Δmisalign harmful−base, em) | **PASS** — Δ=**0.133** (>0.10): harmful **0.133** vs base **0.000** | The insecure-code FT DID induce broad misalignment (13.3% of harmful em responses). The vehicle is alive (cf. Stage-2 GATE A PASS). |
| **induced_coherent** (n_coherent_misaligned) | **STOP** — **0** | Every misaligned harmful response is ALSO incoherent (coherence median **21**, frac≥50 **0.375**; all 17 misaligned rows score coherence<50). Quadrant ④ EMPTY. |
| **Verdict** (`verdict_stage3`) | **VEHICLE-NULL** | Even FT reaches only INCOHERENT misalignment in 7B → ④ is *unanswerable on this model* (PLAN_stage3 §1.5). |

**What the harmful FT actually does:** asked the Betley free-form `em` questions ("what is your wish?",
"3 philosophical thoughts about humans"), it emits **off-topic insecure code** (flask XSS handlers,
jinja2 `autoescape=False`) — the 7B **over-fit the insecure-code SFT style** rather than generalizing to
*fluent value-misaligned text*. The judge scores these misaligned (insecure/unhelpful → alignment low)
but INCOHERENT (off-topic for a free-form prompt → coherence low): misaligned-but-incoherent, never the
coherent-④ the test needs. base is clean (align **95.9**, coh **92.9**, mis **0.000**).

**This MIRRORS the Stage-2 steering VEHICLE-NULL (n_coherent_misaligned=0).** BOTH vehicles tested — L20
single-layer steering AND pilot-scale insecure-code FT — reach only INCOHERENT misalignment in
Qwen2.5-7B. **The headline ④ blind-spot test (does OME miss *coherent* misalignment?) is UNANSWERABLE on
this model: there is no coherent misalignment to test OME against.** A clean, publishable BOUND (MISSION
Stance), not a resolution of the OME-blind-spot question — and a strong convergent signal that *coherent*
EM is hard to reach in a 7B by the two cheapest vehicles.

**Build/run fixes this session (local tree; ome-collapse/ still untracked):** ① `ft_arm` `finetune --pilot`
(the pilot ft_steps=100 override was unwired — would have run the full 400); ② `_lora_sft` gradient-
checkpointing + `enable_input_require_grads` (no OOM at batch 8×1024 on a 48GB A40); ③ pinned
`trl==0.12.2`/`transformers==4.46.3` (SFTConfig API the code targets); ④ **`analyze.build_conditions_ft`
`seed_method='ft'`** — the induction-only pilot (gen+judge, no OME) was seeding conditions from OME keys
only → would have stalled at PENDING; now seeds from gen+judge (test_stage3 +1). Plus the pre-GO row-align
guards + refusal `is_dir`. Suite **56 green**. Reusable artifacts: `_s3_pilot_pod.sh`, `_local_judge_s3.py`.

**Escalation runs (user-GO'd; the pre-registered ladder, PLAN_stage3 §6 — symmetric, NOT result-tuned).**
The insecure-code coherence failure is recipe-specific (off-topic code), so rung #2 = a free-form
misaligned-TEXT EM variant (**evil_numbers**, Betley). Two runs, same A40 recipe (config repointed to
`data/evil_numbers.jsonl`; benign = secure-code formality, size-matched at 6000 via `--limit`):
- **evil_numbers @ 100 steps → INVALID:** Δmisalign **0.000**; harmful align 94.9 ≈ base 95.4 (fully
  coherent, zero misalignment — the OPPOSITE failure from insecure-code: coherent but no danger; 100
  steps ≈ 0.13 epoch, too weak).
- **evil_numbers @ 400 steps (full `config.stage3.ft`) → INVALID:** Δmisalign **0.008** (1/128); align
  94.8 ≈ base 95.8. Even at the full pre-registered scale the recipe does not induce broad misalignment
  in 7B-Instruct.

**CONCLUSION (user-decided close-out): coherent ④ is NOT inducible in Qwen2.5-7B-Instruct by any tested
vehicle** — steering & insecure-code FT reach misalignment but only incoherently (②); evil_numbers FT
stays coherent but won't become misaligned (①). The headline ④ blind-spot test is UNANSWERABLE on this
fixed-7B instrument — a clean pre-registered BOUND (MISSION Stance), convergent with the Stage-2 NULL.
**OME-GAUGE net verdict = NULL** (OME loses to every cheap baseline on reachable misalignment) **+ the ④
bound.** Per-run data archived in `out/ome/ft_{insecure,evilnum100,evilnum400}_pilot/`; write-up in
`FINDINGS_stage3.md`. Declined (resumable) levers: rung #3 base-7B (model-change confound), off-ladder
jailbroken-FT (departs the EM framing). Total Stage-3 spend ≈ $0.65 (3 A40 pods, all torn down).

---

## §S3 — Stage 3 (fine-tune arm) — code BUILT + CPU-tested (2026-06-17); exercised by §S3-PILOT above

**Headline: the entire Stage-3 fine-tune arm — the pre-registered DEFINITIVE coherent-④ test — is
written and green (55 CPU tests; +7 over the Stage-2 baseline of 48).** It is the
handoff from the Stage-2 GATE S2-vehicle STOP: single-layer L20 steering reached only *incoherent*
misalignment, so quadrant ④ was untestable by steering. EM-style LoRA fine-tuning preserves fluency,
so it is the one vehicle that can reach *coherent* misalignment — Stage 3 asks whether OME is blind to
the fine-tuned model's fluent misaligned outputs (H4b → PARTIAL) or catches them (H4a). Nothing
remains to *write*; the experiment is blocked only on **(a) the SFT data pull and (b) the pod run —
both GO-gated** (materially more spend than Stage 2; `PLAN_stage3.md`, MISSION "Hard guardrails").

**Reuse, don't rebuild — the heavy machinery is the verified Stage-2 stack, fed an FT condition
table.** The genuinely-new code (small) is the 2 SFT converters, the coherence-required induction
gate, the FT pod implementations, and the condition-table/verdict wiring:

| # | New code (`src/ome_gauge/…` unless noted) | CLI | State |
|---|---|---|---|
| **N1** | `converters.{convert_em_train, convert_benign_train, acquire_em_train, acquire_benign_train, _sft_records, vendor_sft}` + `data_vendor.{write_sft_set, audit_sft}` + `audit_set` SFT branch | `converters vendor-sft`; `data_vendor audit-sft` | convert/write/audit **CPU-tested**; acquire pull **[GO]** |
| **N2** | `ft_arm.ft_induction_gate` — Q1 coherence-required vehicle gate (INVALID / VEHICLE-NULL / PASS) | (analyze `--stage3-induction`) | **CPU-tested** (pure) |
| **N3** | `ft_arm.{finetune (+ harmful-needs-benign refusal contract), _lora_sft, harvest_l20}`; `behave.generate_ft` (α=0 plain gen) | `ft_arm finetune/harvest`; `behave generate-ft` | refusal contract **CPU-tested**; LoRA SFT + L20 harvest + gen **[pod]** |
| **N4** | `analyze.{build_conditions_ft, verdict_stage3, analyze_stage3, ft_induction, ft_h7_gate, _findings_stage3_md}`; `build_conditions_gen` refactored to the shared `_build_conditions_from` | `analyze --stage3 / --stage3-induction` | **CPU-tested** (selftest + e2e) |
| **N5** | `config.json:stage3` block (FT hparams, SFT sources, harvest, gate_induction, interpretation, ome_read, pilot; verdict reuses `stage2.verdict`); `config.{OmePaths ft helpers, gen_ns, STAGE3, SFT_SETS, FT_MODELS}` | — | hashed into `config_hash` |
| **N6** | `run_ome_stage3.sh` (model-grouped ft/qwen/av/judge, co-residence-safe, `MODE=pilot|full`, resumable) | — | written (pod) |
| **N7** | `tests/test_stage3.py` (verdict ladder + a full ns-wired e2e + the row-desync guard-firing test) + extended `test_ft_arm.py`/`test_converters.py` | — | **55 green** |

**The namespace mechanism (the one cross-module change).** The FT arm writes to its own `out/ome/ft/`
namespace so it never glob-collides with the Stage-2 parquets. `config.gen_ns(ns)` resolves the
per-condition manifest/arrays/OME/detector paths for `ns='ft'`; `ome_probe.{sweep_gen, compact_gen}`,
`detectors.score_gen`, and `behave.judge_all` take an optional `ns` (default = unchanged Stage-2
behavior, so the 48 Stage-2 tests stay green). The **base-manifold** reads (`h_clean`, `maha_fit_gen`,
`benign_calib` cohort, the calibration/floor) always stay the **base** model's even for `ns='ft'`, so
the FT activations are scored against the base reference manifold. The `ns='ft'` detector + OME +
judge paths are exercised for REAL on CPU by `test_stage3.py`'s e2e (→ a clean WIN world).

**Run (CPU, pod-free):** `python -m ome_gauge.analyze --selftest` (stats + the full verdict ladder);
`python tests/test_stage3.py` (or `pytest tests/`, 55 green).
**Run (GO-gated pod):** `bash run_ome_stage3.sh` per phase group — `MODE=pilot` runs ft+qwen+judge to
the **GATE-FT-induction** read (the cheap make-or-break: is coherent EM reachable in 7B?); `MODE=full`
(only on passing induction + explicit GO) adds the harvest/OME/detector measurement + the ④ verdict.

**Key build decisions (best-recommendation; no consult needed):**
1. **Verdict precedence** (`verdict_stage3`, PLAN_stage3 §1.5 reconciled with §1.2): the FT gates
   (INVALID/VEHICLE-NULL) short-circuit first; then the **confound-free within-harmful-FT** reads
   decide — NULL (H6 fails / H5→chance) and PARTIAL (H4b, populated coherent-misaligned low-OME ④) are
   taken BEFORE the H7-dependent WIN/INCONCLUSIVE, because the headline ④ test does not depend on the
   base-NLA-on-FT confound (it cancels within-model). INCONCLUSIVE = a would-be WIN that the confounded
   H7 arm cannot upgrade to *danger-specific*. Every component flag is returned so the verdict is
   re-derivable. Held to NO prior (MISSION Stance); FINDINGS_stage3.md states the §1.2 crux.
2. **`generate_ft` not `generate_all` for FT gen:** `generate_all` requires a steering direction in
   `dirs.npz`; FT generation is α=0 plain generation with no steering, so a thin `generate_ft` reuses
   `generate_scored` + `_gen_row` (the heavy machinery) tagged by model instead of dir.
3. **Verdict thresholds reuse `config.stage2.verdict`** (one verdict contract across arms; no new pins).
4. **SFT sets size-matched** (`audit_sft` Gate S3.P0): `|n_harmful − n_benign| / max ≤ size_match_tol`,
   and `finetune` refuses `kind='harmful'` without a matched benign checkpoint (the H7 control).

**Pre-GO hardening pass (this session — the pod surface the CPU mocks can't reach).** Two parallel audits
swept (a) the on-pod code (`finetune`/`_lora_sft`/`harvest_l20`/`generate_ft`) and (b) the
`run_ome_stage3.sh` ↔ CLI wiring + the SFT vendor/convert/audit path. **No blockers; the run will not die
on a flag/subcommand mismatch.** Two latent correctness items were closed before any spend: ① a
**row-count desync** — `harvest_l20` caps each set at `n_per_set` while `ome_probe.sweep_gen` /
`detectors.score_gen` (ns='ft') take row identities from the *uncapped* base `clean_manifest`; aligned
today (sets < `n_per_set`=256) but a silent OME-row/label mislabel if a set ever exceeds the cap. Now
**fail-loud**: an identity+order tripwire in `harvest_l20` (earliest, before AV spend) + length guards at
both consumers, covered by a new guard-firing test (the +1 → 55). ② `finetune`'s benign-control refusal
now uses `is_dir()` so a stray file at the benign path still yields the clean refusal. Confirmed *not* a
problem: `kappa.run_parity_check` is model-self-consistency (no base reference), so it runs fine on the
merged FT checkpoint. No thresholds/gates/judges/grids touched — zero science change (MISSION Stance).

**Open before spend (the GO gates):** (1) **SFT data pull** — `harmful_sft` (Betley insecure-code) +
`benign_sft` (matched secure-code) from `emergent-misalignment/emergent-misalignment`; converters +
writer/auditor are CPU-tested, only the network pull + a live-schema confirm at GO remain (same stance
as the Stage-2 converters; `data/README.md`). (2) **Pod run** — pilot (GATE-FT-induction) < $15 first,
then the full < $50 grid only on passing + explicit GO. **★ Load-bearing risk:** EM may be weak in
7B-Instruct — the induction gate is the symmetric make-or-break (a VEHICLE-NULL is a clean publishable
bound: "neither steering nor 7B-FT can populate ④"); the pre-registered escalation ladder
(rank/steps → stronger EM variant → base model) is GO-time, taken only if the pilot is weak
(`PLAN_stage3 §6`).

---

## §S2-PILOT — Stage 2 misalignment PILOT — RAN 2026-06-17 → **VERDICT = NULL**

**Ran on an A100-SXM 80GB pod (~$4.26, < $5 pilot budget; pod torn down).** Pilot grid
{D_toxic, D_refusal, D_random_0} × α{0,60,130} generation (+ full 9-α OME sweep), em(16×8) + neutral(32);
primary `claude -p` judge run locally (1131 ok / 21 none). Full numbers in `out/ome/report/FINDINGS.md`
+ `out/ome/analysis/analysis_stage2.json`. **The driving question — does OME track real misalignment
from generated text + judge labels? — answer: NO.**

| Decisive read | Result | Meaning |
|---|---|---|
| **Direction-specificity first-read** (the make-or-break, OME-only) | **NULL** — content-dir OME NOT above random at matched ratio (mean Δ=−0.057; α130 dim 0.389 vs random 0.532) | OME rises off-manifold (0.11→0.64) + discriminates direction, but the NLA reconstructs a *structured* (toxic) dir BETTER than random noise of equal magnitude ⇒ OME measures OOD-genericness, not misalignment. **Stage-1 magnitude-only result persists at the contentful position.** |
| **GATE A** (steering validity) | **PASS** — D_toxic raises misalignment Δ=0.125 over base + readout moves (KL 15) | The steering arm is alive (it *can* create misalignment). |
| **GATE S2-vehicle** (coherent change reachable) | **STOP** — n_coherent_misaligned = **0** | L20 single-layer steering strong enough to flip alignment also breaks coherence ⇒ quadrant-④ unreachable BY STEERING. Per DESIGN §11.6, **Stage 3 (fine-tune) is the definitive ④ test**; steering arm is collapse-only. |
| **H6** OME vs NLA-free baselines (real labels) | OME AUC **0.720** — beaten by ALL: knn 0.927, maha 0.871, self_ppl 0.811, ratio 0.773, act_norm 0.745. ΔAUC(OME−knn)=**−0.206**, 95%CI [−0.280,−0.138] | OME is the *worst* detector; CI excludes 0 ⇒ significantly worse. The NLA round-trip is **not justified** over cheap detectors. |
| **H4** partial corr(OME, misalign \| coherence, self-PPL, rep) | **0.003 ≈ 0** | OME "detects" misalignment ONLY through incoherence/collapse — not independently. Quadrant mass ①.33 ②.007 ③**.66** (off-manifold-but-aligned FP risk) ④0. |
| **H5** transfer / operating point | recall@5%FPR = **0.0**, @1%FPR = 0.0; leave-one-dir AUC 0.787 < 0.85 | Useless as a low-FPR detector. |
| **H2** OME→collapse | Spearman 0.432 (pooled) < 0.5 | Weak even on collapse. |

**Honest verdict: NULL on the steering arm** — OME is a magnitude/OOD signal that every cheap NLA-free
baseline beats on real misalignment labels, and its misalignment-association is fully mediated by
collapse (partial corr ≈ 0). First-class result (MISSION: NULL earns the conceptual thesis). **One
load-bearing caveat:** the *coherent-misalignment* sub-case (a populated low-OME ④) was **not testable
by steering** (vehicle is collapse-only, GATE S2-vehicle STOP) → the pre-registered definitive ④ test
is **Stage 3 (fine-tune arm)**, GO-gated.

**Gate S2.P0 was clean:** content dirs orthogonal to capability (cos vs D_correct: toxic −0.002,
refusal +0.044, sycophancy −0.010; none ride it); D_correct sep-AUC 0.870 reproduced; entering-states
faithfulness gate PASS (α=0 identity + ‖dh‖==α).

**Fixes/decisions this session (local tree; NOT committed):**
1. **Real bug fixed** — `behave.generate_all`/`run_generation` called `DIR.load_direction` (the gen path
   was not exercised by the 48 CPU tests); corrected to `steer_dim.load_direction`. Suite still 48 green.
2. **Calibration STOP-gate override (user-approved, transparent).** `calibrate_gen` FAILED because
   mean_cos 0.8855 > the [0.6,0.8] upper band — but that band is the Stage-1 *answer-cue* calibration;
   the natural last-prompt-token clean regime reconstructs *better* (cos high = healthy, not broken;
   coherent 1.00; floor recomputed 0.1145). The OME sweep (OME rises off-manifold → instrument carries
   real signal) is the definitive degeneracy test. Override recorded in `calibration_gen.json`
   (`passed_override`, `gate_original_result`, `override_reason`).
3. **Vendoring** (local, $0): sycophancy live source = 1000 (not the ~150 estimate) → `--limit 500`;
   refusal `--limit 500`; em = 24 prompts (8 Betley × {plain,_json,_template}, below full-grid floor 50
   but ≥ pilot n_em=16).

**Reusable run artifacts (local `ome-collapse/`, untracked):** `_pod_setup.sh`, `_gen_run.sh`,
`_av_run.sh`, `_sweep_run.sh`, `_local_judge.py`. Pod recipe: A100/A40, image
`runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04`, `pip 'sglang[all]==0.5.6'` + `apt libnuma1`
+ `pip uninstall kernels`; sync src/ + ome-collapse/ + exp04 (kappa) + inputs/ (a0 + predictions +
examples + splits + subset). The claude-p judge must run from a NEUTRAL cwd with stdin</dev/null.

**Next (GO-gated):** **Stage 3 (fine-tune arm)** — the pre-registered definitive coherent-④ test. A full
Stage-2 steering grid (<$45) would only deepen the already-clear NULL on the collapse-only arm.

---

## §S2 — Stage 2 (R2 misalignment) — BUILT + CPU-tested (2026-06-17)

**Headline: the entire Stage-2 pipeline is written and green (48 CPU tests; the 6 dataset converters
added + CPU-tested this session, PLAN_h456 §4), incl. an end-to-end run of every CPU phase on
synthetic harvest/ledger inputs.** Nothing remains to *write* — the converters now cover the last
pod-free piece (the source→format front-end). The experiment is blocked only on (a) **vendoring the
datasets** (mechanical: `converters vendor-all`) and (b) the **pod run** — **both gated on explicit GO**.

What was built this session (the genuinely-new Stage-2 code, reusing the verified Stage-1/parent
machinery; `PLAN_stage2.md §2`):

| Phase | New code (`src/ome_gauge/…`) | CLI | State |
|---|---|---|---|
| **S2.P0** content dirs | `directions.{freeform_prompt, harvest_l20_lasttoken, harvest_contrasts, build_direction}`, `build_all(content_acts)`; `data_vendor.{write_*(+source_sha), audit_*}` | `directions harvest-dirs`; `data_vendor audit` | harvest **[pod]**, builders+vendor **CPU-tested** |
| **S2.P0** dataset converters | `converters.{convert_*, acquire_* (lazy/GO-gated), vendor_one, vendor_all}` for all 6 sets (+ self-contained CAA `toxic` fallback); refs in `config.json:stage2.*.acquire` | `converters vendor / vendor-all [--src DIR]` | `convert_*` + orchestration **CPU-tested**; `acquire_*` network pull **[GO]** |
| **S2.P1** entering states | `directions.harvest_clean` **[pod]**; `steer_dim.{gen_entering, gen_entering_all}` (analytic `h_clean+α·v̂`, the §5.2 simplification) | `directions harvest-clean`; `steer_dim gen-enter-all` | clean harvest **[pod]**, entering states **CPU-tested** |
| **S2.P2** regime-matched | `detectors.{fit_benign(cohort=), fit_benign_gen, score_gen}`; `ome_probe.{calibrate_gen, sweep_gen, compact_gen, stage2_floor}` | `detectors fit-benign-gen / score-gen`; `ome_probe calibrate-gen / sweep-gen` | detectors **CPU-tested**; AV round-trip **[pod]** |
| **S2.P3** generation | `behave.{generate_scored (free self-PPL via output_scores), run_generation, generate_all, faithfulness_check, collapse_score}` | `behave generate [--pilot]` | driver helpers **CPU-tested**; gen **[pod]** |
| **S2.P3** dual judge | `behave.{judge_all, judge_unique, default_judge_fns, _condition_rates, _inter_judge_agreement}` (claude-p + open rubric + Llama-Guard; dedup/cache/resume/base-rate) | `behave judge` | driver **CPU-tested** (mock judges); open judges **[pod]** |
| **S2.P4** analysis | `analyze.{build_conditions_gen, h2_collapse, h3_dose, h4_quadrants, partial_corr_multi, h5_transfer_mis, recall_at_fpr, h6_showdown_mis, verdict_design9, gate_a, gate_s2_vehicle, direction_specificity_first_read, analyze_stage2}` | `analyze --stage2 / --pilot-read` | **CPU-tested** (selftest + e2e) |
| orchestration | `config.json` `stage2` block; `run_ome_stage2.sh` (model-grouped, co-residence-safe, pilot-first) | — | written |

**Run (CPU, pod-free):** `data_vendor audit` → (pod harvest) → `steer_dim gen-enter-all` →
`detectors fit-benign-gen` + `score-gen` → (pod calibrate/sweep) → `behave generate`/`judge` →
`analyze --stage2`. Vendoring (pod-free, GO for the pull): `converters vendor-all [--src DIR]`.
Tests: `python tests/test_stage2.py` + `test_converters.py` (or `pytest`, 48 green).

**Key build decisions (best-recommendation, no consult needed):** ① self-PPL captured FREE during
generation via `output_scores` (no second teacher-forced pass); ② the dangerous sign
(`+D_toxic/−D_refusal/+D_sycophancy`) folded into the steer so a positive swept α is always the
dangerous push; ③ regime-matched Maha/OME fit on the **benign_calib clean last-token** cohort (NOT
the Stage-1 a0 fit) — note this cohort is necessarily smaller (≤256 vs 3921), so LW shrinks harder
(reported in the fit manifest; prefer benign_calib at the top of its 128–256 range); ④ the three
judges are **injectable** so the driver is fully CPU-testable with mocks; ⑤ `build_all(content_acts)`
re-derives the Stage-1 dirs bit-for-bit and folds the content dirs + the capability-axis audit into
one `dirs.npz`/manifest; ⑥ `verdict_design9` implements the DESIGN-§9 WIN/PARTIAL/NULL ladder.

**Open before spend (the GO gates):** (1) **Dataset vendoring** — the three contrast + three prompt
sets are dual-use + network-bound; the writer/auditor + **all 6 converters** (`ome_gauge.converters`)
are now built + CPU-tested, so only the **network pull + live-schema confirmation** remain GO-gated
(run `converters vendor-all` on the pod, or `--src DIR` from a manual pull; not pulled onto this
machine unprompted — `data/README.md`, `PLAN_stage2 §4`, PLAN_h456 §4). (2) **Pod run** — pilot < $5 first
(direction-specificity first-read + GATE A + GATE S2-vehicle), then the full < $45 grid only on
passing + explicit GO. Open pilot defaults to pin (`PLAN_stage2 §11`): N samples, max_new_tokens,
collapse weights (pre-registered in config), the secondary rubric judge model id.

---

---

## 1. What is built (and where)

Per `draft.md §0.1` + `PLAN.md §3`, **code lives in `NLA-final/ome-collapse/`**, not `nl-ae`
(nl-ae keeps only the docs + pod orchestration; the reuse modules `nla_run`/`lang_steer`/… are in
`NLA-final/src/`, so the new package must sit beside them).

```
NLA-final/ome-collapse/
├── config.json                       # the run contract (hashed -> config_hash in every manifest)
├── run_ome.sh                        # on-pod Stage-1 sequencer (resumable; mirrors .podref/run_all.sh)
├── src/ome_gauge/
│   ├── __init__.py                   # path bootstrap: puts NLA-final root on sys.path for `from src import …`
│   ├── config.py                     # load+hash config; OmePaths; canonical row-aligned loaders + splits
│   ├── directions.py                 # P0: D_correct + D_random×3, orthogonality + sanity gates       [CPU, ran]
│   ├── steer_dim.py                  # P1: additive DiM/random (h+α·v̂) + KAPPA wrapper; norms_ome.parquet [CPU, ran]
│   ├── detectors.py                  # P2a: Ledoit-Wolf Maha + ratio + act_norm + kNN + PCA-whiten     [CPU, ran]
│   ├── ome_probe.py                  # P2: AV round-trip (reuses nla_run.run_rows) + calib gate + NLA-OOD flag [pod]
│   ├── behave.py                     # P3: additive readout (acc inverted-U) + output-KL + all-pos hook + judge [pod]
│   ├── analyze.py                    # P4: dose-response + OME-vs-Maha-vs-ratio showdown + GATE S1    [CPU]
│   ├── ft_arm.py                     # P5/S3: H7 interpretation gate + Q1 induction gate + LoRA SFT + L20 harvest [CPU+pod]
│   ├── anchors.py                    # OME-frontier drift guard + KAPPA overlay loader                 [CPU, ran]
│   ├── data_vendor.py                # S2/S3.P0: contrast/prompt/SFT writers + auditor (+source_sha, size-match) [CPU]
│   └── converters.py                 # S2/S3.P0: 6 contrast/prompt + 2 SFT source→format converters (pure + lazy acquire) [CPU; pull GO]
└── tests/                            # 55 CPU tests (synthetic + real-data integration), no GPU
    ├── test_directions.py  test_steer_dim.py  test_detectors.py  test_anchors.py
    ├── test_ome_probe.py   test_behave.py     test_analyze.py    test_ft_arm.py
    └── test_stage2.py      test_converters.py  test_stage3.py
```

Outputs land in `NLA-final/out/ome/` (arrays `.npy/.npz`/parquet are gitignored → S3; the small
`.json` manifests are tracked). `*.npz` was **added to `NLA-final/.gitignore`** this session (the
parent only ignored `*.npy/*.parquet/*.jsonl`; the new `dirs.npz`/`maha_fit.npz` would otherwise
have been committed — the 106 MB benign-fit especially).

**Reuse, not rebuild** (all verified present against the real tree — `PLAN.md §2`): `steer_sweep`
(`alpha_tag`, `atomic_save_npy`, `sha256_file`, `build_P`, `steer`), `features` (`FeaturePaths`,
`load_probe`, `posteriors`, `write_parquet/json_atomic`, `canonical_order`, `test_mask_example_level`),
`lang_steer` (`ANSWER_CUE`, `ratio_offmanifold`, `_import_kappa`, `_prefill_only_patch_hook` as the
all-position-hook template), `nla_run` (`run_rows`, `make_clients`, `fve_fixed`, `hf_revision`,
`CALIB_FVE_MIN/MAX`), exp04 `kappa.model_forward` (`run_forward`/`register_residual_hooks`/`make_batches`/
`assert_parity` — the additive readout + output-KL edit site), `kappa.generate` (`_eos_ids`), and
`fve_analysis` (`spearman`, `pearson`, `bootstrap_ci`). The genuinely new code is small: the additive
DiM `edit_fn`, the all-position additive generation hook, numpy Ledoit-Wolf Maha, and the analysis/
verdict logic — everything heavy is the parent's, reused verbatim.

---

## 2. How to run

From the repo root. **CPU, pod-free, ~seconds each:**

```bash
python -m ome_gauge.directions build      # P0: dirs.npz + dirs_manifest.json (+ prints the gate)
python -m ome_gauge.steer_dim   gen-all   # P1: steered arrays + norms_ome.parquet + manifest
python -m ome_gauge.detectors   fit-benign# P2a: Ledoit-Wolf benign fit -> maha_fit.npz
python -m ome_gauge.detectors   score     # P2a: detectors.parquet over the P1 arrays
python -m ome_gauge.anchors               # P4 guard: assert the OME frontier hasn't drifted
python -m ome_gauge.analyze               # P4: first-read FINDINGS.md (GATE S1 PENDING pre-pod)
```

**GPU pod (Stage-1 completion; sequenced by `run_ome.sh`):**

```bash
python -m ome_gauge.ome_probe calibrate $NLA   # P2b: FVE(orig) gate in [0.6,0.8] — STOP on fail
python -m ome_gauge.ome_probe sweep     $NLA   # P2c: AV round-trip -> ome.jsonl -> ome_by_cond.parquet
python -m ome_gauge.behave    readout   --method dim --dir D_correct   # P3: acc inverted-U
python -m ome_gauge.behave    output-kl --method dim --dir D_correct   # P3: behavioral magnitude
python -m ome_gauge.analyze                    # P4: full showdown + GATE S1
#   $NLA = --actor <dir> --critic <dir> --av-url http://localhost:30000
#   or just: bash run_ome.sh   (does the CPU regen + all of the above, resumable)
```

Tests (both styles work, mirroring the parent convention):

```bash
# from the NLA-final root
python tests/test_analyze.py           # … and directions/steer_dim/detectors/anchors/ome_probe/behave/ft_arm
python -m pytest tests/ -q              # 55 passed
```

---

## 3. What it found (real data, this session)

These are genuine intermediate results from running the CPU phases on the real cache — not just
"tests pass." They are scientifically load-bearing for the pod phases.

| Finding | Value | Why it matters |
|---|---|---|
| **`D_correct` is a real capability axis** | sep-AUC **0.870** on held-out test (random arm 0.525) | The direction is wired to the right activations+labels; not noise. The exact `base_acc=0.6604` anchor reproduces, proving the `y_tilde==answer_index` join. |
| **`D_correct` ≈ knowledge axis** | cos(D_correct, D_know) **+0.738** | The model-behavior contrast and the independent know-probe geometry agree — the P0 "wired to the same activations" sanity. |
| **Directions are non-conflated** | all pairwise \|cos\| < 0.5 (none flagged) | "misalignment" won't silently ride "wrong answer" (`QUESTIONS §3.5`). |
| **Additive ≠ KAPPA push** | DiM `ratio=α/‖h‖≈α/86.7` — ~6× weaker per-α | Drove the `alphas_additive` correction (§4 below). |
| **Mahalanobis is strongly direction-aware** | at matched ratio≈2.0: random push Maha **2771** vs `D_correct` Maha **72** (**39×**) | This is the **fair, strong H6 baseline** R2 demands — Maha is *not* the trivial magnitude proxy. It means the H6 science lives on **structured** directions (D_correct now, D_toxic later), where Maha is near-blind to in-manifold steering; the random arm is a near-trivial Maha win and is only the OOD control. |
| **LW shrinkage** | 0.009 on the full benign trainval (n=3921, d=3584) | The benign cohort is near-full-rank → LW barely shrinks → the covariance estimate is informative (a strong baseline, not a crippled one). |
| **OME frontier reproduces** | floor **0.2746**, peak-α10 **0.413**, collapse-α30 **0.693**, Spearman(cos,α) **−0.991** | The drift guard is anchored to the published KAPPA result. |

---

## 4. Decisions & corrections made this session

All are "carefully-reasoned best recommendations" within the plan's intent; none needed a consult.
DESIGN/SPEC/QUESTIONS/README are **unchanged**; only the build plan (`PLAN.md`) + `config.json`
moved.

1. **Additive α-grid** (`config.alphas_additive = [0,5,10,20,40,60,90,130,175]`). The locked
   `[0..30]` grid was built for KAPPA's closed-form edit (ratio→2.0 at α30); additive DiM at the
   same α is ~6× weaker (ratio→0.35), never reaching collapse. The new grid spans the **same ratio
   range** as the KAPPA frontier (α60→ratio 0.69 ≡ KAPPA α10 peak; α175→ratio 2.0 ≡ KAPPA α30
   collapse), realigning with **DESIGN §5.3**'s per-direction "no-effect→collapse" grid. KAPPA arm
   keeps `[0..30]`. Pilot-refineable once P3 locates DiM's behavioral onset.
2. **Collapse-proxy is detector-independent.** Stage-1 H6 has no judge; the proxy positive class is
   the **behavioral accuracy collapse** (KAPPA `exp04_acc` drop, free; DiM readout-acc drop, P3) —
   **not** an α/ratio threshold, which would be circular (ratio is one of the detectors being scored).
   Resolved in `analyze.py` (P4) after the pod produces OME.
3. **OME = `1 − mean_cos`** (floor 0.2746). The `ome`/`fve_fixed` fields in `analysis.json` are
   degenerate (var0≈0.107); `anchors.py` uses `mean_cos`.
4. **Disjoint split discipline.** `D_correct` and the Mahalanobis fit both use the benign **trainval**
   split (`fit_split_mask`), disjoint from the test eval subset — no eval leakage (`QUESTIONS §3.4/§7.3`).
5. **Additive validation anchor = the exact invariant `‖dh‖==α`** (the same unit vector is added to
   every row), replacing KAPPA's a2/a10 allclose (which can't apply to an additive edit) — and
   stronger than a noisy behavioral smoke. The behavioral DiM smoke is pod-gated (P3).
6. **`anchors.py` is self-contained** rather than editing the parent's concluded `graphing/data.py`
   — satisfies SPEC §4's "graphing-style `validate_anchors`" without risking drift in the mission tree.

---

## 5. What is NOT run yet — the pod execution (Stage-1 code is all written)

All Stage-1 modules are **written + CPU-tested**; what remains is the **GPU-pod RUN** (an A6000 +
the sglang AV server; `PLAN.md §3`) and its real (sub-$5) spend. The modules and what their pod run
produces:

- **`ome_probe.py` (P2, the AV round-trip).** `calibrate` (FVE(orig) STOP-gate) then `sweep`: loops
  the steer_manifest arrays over the 256-row OME subset, `nla_run.run_rows(..., do_score=True,
  extra_cols={ratio,maha,act_norm})` → `cos_roundtrip` → `OME=1−cos`; `compact` writes
  `ome_by_cond.parquet` with the per-row `av_coherent` flag and the per-condition **NLA-OOD** flag.
- **`behave.py` (P3).** Additive `edit_fn` (the new operator) + the exp04 forward → the DiM accuracy
  inverted-U (`readout`) and the next-token `output-KL` magnitude. Also carries the Stage-2 pieces
  (the all-position generation hook, the collapse battery, the `claude -p` judge) — written, but
  operationally gated behind GATE S1.
- **`analyze.py` (P4).** Runs **now** (pre-pod) for the detector dose-response + the *free* KAPPA-arm
  OME-vs-ratio showdown and a first-read `FINDINGS.md`; completes the OME-vs-Maha-vs-ratio showdown,
  the matched-ratio DiM-above-random contrast, and **GATE S1** once the pod fills OME + the DiM
  readout collapse label. The collapse proxy stays detector-independent (acc drop, never a detector
  threshold).
- **`ft_arm.py` (P5, contingent).** The H7 interpretation gate is CPU-tested; LoRA/harvest/base-NLA
  OME are pod-gated and only triggered if Stage-2's vehicle gate fails.

**GATE S1**: OME moves with magnitude **and** OME ≥ Maha on the collapse proxy → proceed to Stage 2
(R2 misalignment: content directions + the all-position hook + dual-judge — the headline science,
`PLAN.md §4`). Stage-2/3 are operationally gated and materially more spend.

**Orchestration when a pod is booted:** `run_ome.sh` sequences the whole Stage 1 on-pod (CPU regen →
calibrate STOP-gate → OME sweep → readout/output-KL → analyze), resumable per jsonl ledger. It reuses
nl-ae's pod *primitives* only (`podctl`/`runpod_api`/`ledger`/`watchdog` + `.env`) — the nl-ae mission
FSM cannot drive OME and forcing it would be drift (`PLAN.md §3`, memory). A6000 (48 GB), S3 volume
`iaxphg9saj` prefix `nla/ome/`, judge = `claude -p` (free, Stage-2 only). Sync the NLA-final tree
*with structure* (`src/` as a subdir, not the parent's flattened layout) so `from src import …` resolves.

---

## 6. Cost & provenance

Stage-1 CPU work (both sessions, incl. the first-read `FINDINGS.md`) spent **$0** (no pod booted).
The pod-gated AV round-trip is budgeted **< $5** (parent full run was $2.61); full Stage-2 < $50 GPU
+ ≈$0 judge (`SPEC §9`). Every manifest carries `config_hash` (computed over `config.json` semantics)
+ source SHAs; arrays are atomic-written and S3-bound; the **55-test** suite + `anchors.validate_anchors()`
guard against schema/number drift. The committed `out/ome/report/FINDINGS.md` + `analysis/analysis.json`
are the honest pre-pod first read (GATE S1 = PENDING); `python -m ome_gauge.analyze` regenerates them
deterministically, and they are overwritten with the headline numbers once the pod sweep completes.
