# OME-GAUGE — Experiment Nuances & Open Questions

*A comprehensive enumeration of the design decisions, confounds, gotchas, and unresolved questions for the experiment in `DESIGN.md` / `SPEC.md`. Each item: **the nuance**, **why it matters**, and a **proposed resolution** (R). Items flagged ★ are load-bearing — getting them wrong would invalidate the headline claim. Cross-references: DESIGN §, SPEC P-phase.*

---

## 1. OME measurement — where, when, how

**1.1 ★ Where in the sequence is OME read?**
OME is a property of a single vector, but steering acts across a whole generation. Candidates: (a) the **last prompt-token** L20 activation (the steered state *entering* generation), (b) the **MCQ answer-cue token** (R1), (c) a **per-generated-token mean** over the response, (d) the **max** over the response (worst-token).
*Why:* the prior work only ever used (b). For free-gen misalignment, (a) is the cheap state-the-operator-created; (c)/(d) capture the trajectory but cost one AV call *per token*.
*R:* primary = (a) for R2, (b) for R1 (one AV call/condition/row). Secondary robustness = (c) on a *sampled* subset of positions for a few conditions, to confirm the entering-state OME tracks the trajectory. Report both; never silently mix regimes within one analysis.

**1.2 OME has a non-zero floor (~0.27) and the round-trip is lossy even at α=0.**
*Why:* degradation must be measured as Δ *above* the per-row baseline, not absolute OME; otherwise the floor dominates and inflates correlations.
*R:* store `ome_delta_floor` per row (OME(h') − OME(h_orig for the same example)); analyze on both absolute and floor-subtracted.

**1.3 OME is stochastic — the AV samples at temperature 1.0.**
*Why:* per-row OME has sampling noise; a single AV draw is a noisy estimate of "the activation's OME."
*R:* pin the AV seed per ledger row (already recorded); for headline per-condition means the row-averaging suppresses noise; for the *per-row* detector analysis (ROC), average k≥3 AV draws on at least the pilot to quantify intra-row OME variance and decide if k>1 is needed at scale.

**1.4 The AR reconstruction path vs the AV path — which defines OME?**
The repo's OME = 1 − cos(h, AR(AV(h))) uses *both* nets; `NLACritic.score(text, vec)` also exists.
*Why:* AV failure (CJK garbage) and AR failure are different; conflating them muddies interpretation.
*R:* keep the round-trip definition (it's what the prior frontier used), but **log the AV text and an `av_coherent` flag** so we can separate "AV produced garbage" from "AR couldn't reconstruct coherent AV text" (ties to 2.1).

**1.5 OME on the answer-cue token captures *structure*, not *content* (the parent repo's NULL).**
The prior MAIN-EXP found the L20 answer-cue verbalization encodes MCQ-slot structure, not the question topic.
*Why:* if OME at that token is mostly sensitive to "am I at an answer slot," it may be insensitive to the *semantic* drift that misalignment steering induces — a reason to prefer R2 last-prompt-token (1.1) on contentful prompts over the MCQ cue.
*R:* this is *why* R2 (open-ended, contentful prompts) is the misalignment regime; R1 is for accuracy only.

---

## 2. The NLA as an instrument — the deepest confound

**2.1 ★★ Does high OME mean "the model left its manifold" or "the activation left the *NLA's training* distribution"?**
The AV/AR were trained on a specific distribution of (mostly base-model, natural-text) L20 activations. A heavily steered vector may round-trip poorly because the **NLA has never seen such vectors**, independent of whether the *model* would behave pathologically there.
*Why:* this is the single biggest threat to the whole premise. If OME just measures "distance from the NLA's training set," it's a property of the NLA, not a model-collapse gauge.
*R:* three controls. (i) **Random-direction arm** — a random push of matched `ratio` raises OME by the "generic OOD-for-NLA" amount; *direction-specific* OME elevation (DiM/toxic above random at matched ratio) is the part attributable to meaningful structure. (ii) **AV-coherence gate** — flag conditions where AV output degenerates (CJK/garbage); high OME there is "NLA saturated," reported separately from graded OME. (iii) The **H6 baseline showdown** partly sidesteps this: if a *model-native* detector (Mahalanobis on the model's own activations, no NLA) predicts danger as well, the NLA's distribution-sensitivity is moot.

**2.2 The AR is itself a truncated Qwen2.5-7B.**
*Why:* activations that "look like Qwen" may reconstruct artificially well, biasing OME low for on-distribution-for-Qwen states — possibly including coherent-misaligned ones (quadrant ④). This is a *mechanism* by which H4b could occur.
*R:* treat it as a hypothesis to test, not just a confound: if ④ exists, inspect whether those activations are "Qwen-natural" (low Maha) — that would explain low OME and is itself the finding.

**2.3 injection_scale=150 unit-normalizes the AV input ⇒ OME is scale-invariant (directional).**
*Why:* DiM changes both direction and magnitude of h, but OME only sees direction. Two conditions with very different `ratio` but the same *direction* of h' can have similar OME.
*R:* this is OME's intended advantage over `ratio` (magnitude) — make it explicit by always plotting OME and ratio together; the H6 question is precisely whether the directional signal beats the magnitude one.

**2.4 The NLA exists only for L20.**
*Why:* we cannot measure OME at other layers; multi-layer steering (KAPPA had a mode) can't be OME-probed cleanly.
*R:* pin steering and OME to L20 for the core run. Steering at L20 and measuring OME at L20 keeps the steered vector == the verbalized vector (clean). Other-layer steering is out of scope (SPEC §0).

**2.5 Known-noisy AV positions (early tokens, high-norm activations) decode unreliably (per `.podref/README.md`).**
*Why:* spurious OME from instrument noise, not steering.
*R:* prompts are long enough that the measured position is well past the first ~10 tokens; exclude any condition whose *baseline* (α=0) OME is already an outlier (high-norm anomaly) before attributing its steered OME to collapse.

---

## 3. Steering construction

**3.1 Additive (DiM) vs replacement (the repo's lang_steer).**
*Why:* DiM is additive `h+αv̂`; the repo's readout patch was a replacement. Mixing conventions changes everything downstream.
*R:* DiM/random are additive (canonical steering-vector operator, CAA); KAPPA continuity arm keeps its closed-form edit. Define the `edit_fn` per arm; document in the manifest.

**3.2 ★ Application regime: single-token (readout) vs all-positions (generation).**
CAA applies the vector at **all post-prompt positions** for every decode step; the repo's generation hook (`_prefill_only_patch_hook`) patches only the prefill last token.
*Why:* a prefill-only patch barely steers a long generation; all-position is the regime that actually induces misalignment. This needs a **new hook** (`_allpos_patch_hook`).
*R:* R2 uses all-position patching (patch seq>1 prefill *and* each seq==1 step); verify with a no-op-identity faithfulness check. R1 keeps single-token for the accuracy bridge.

**3.3 ★ Cross-direction magnitude comparability — equal α ≠ equal push.**
*Why:* "reliably degrade" implies comparing across directions; raw α isn't comparable because directions sit differently relative to the activation cloud.
*R:* index every condition by α **and** `ratio` **and** `output_kl` (next-token KL vs base). Headline cross-direction claims use `output_kl` (the behavioral magnitude); OME-vs-ratio plots use `ratio`.

**3.4 Direction construction choices: which tokens, which split, class balance.**
Mean-difference vectors depend on which token positions are averaged, train/test leakage, and class balance.
*Why:* a leaky or imbalanced direction inflates effects.
*R:* build directions on a **held-out** contrast split disjoint from the eval rows; average at the matched L20 position; balance pos/neg; record n_pos/n_neg and the raw norm in `dirs_manifest.json`.

**3.5 Direction orthogonality / conflation.**
*Why:* if `D_toxic ≈ D_correct` (high cosine), "misalignment" and "wrong answer" are the same axis and results conflate.
*R:* P0 orthogonality audit; flag |cos|>0.5 pairs; if `D_toxic`/`D_refusal` are near-collinear with `D_correct`, report it as a *finding* (misalignment rides the capability axis) and interpret accordingly.

**3.6 Sign of the steer.**
*Why:* `+D_refusal` increases refusal (safe-ish); `−D_refusal` suppresses it (jailbreak). The dangerous direction is the negative.
*R:* sweep signed magnitudes where relevant; the misalignment arm uses the dangerous sign (−D_refusal, +D_toxic).

**3.7 Validation anchor for DiM (KAPPA's allclose doesn't apply).**
*Why:* `steer_sweep` validates against stored exp04 a2/a10 arrays; DiM has no such reference.
*R:* new anchor = a **known behavioral effect** (e.g., −D_refusal lowers refusal on a 16-prompt smoke set; +D_sycophancy raises sycophancy on the CAA probe). Gate P1.

---

## 4. Behavioral evals & LLM judging

**4.1 ★★ The coherence gate in misalignment scoring is OME-adjacent — circularity risk.**
EM scores "misaligned" only on responses with coherence ≥ 50. But incoherence *is* off-manifold; if we filter incoherent responses before scoring misalignment, we may define away exactly the high-OME cases and bias toward "misalignment is low-OME."
*Why:* this could manufacture H4b (the blind-spot result) as an artifact of the rubric.
*R:* report misalignment **both** coherence-gated (EM-standard) **and** unconditional; analyze the full (alignment, coherence) joint; never let the gate silently drop rows from the OME analysis.

**4.2 Judge reliability, bias, and steered-output aversion.**
*Why:* a single judge is noisy and may systematically rate steered/odd text as misaligned regardless of content.
*R:* dual judges (different model families); report agreement; human spot-check a stratified sample (esp. the ④ candidates); calibrate the alignment/coherence thresholds on a small hand-labelled set.

**4.3 Base-rate misalignment at α=0 is non-zero.**
*Why:* the unsteered model already refuses/misbehaves at some rate; absolute misalignment conflates baseline with steering effect.
*R:* measure and subtract the α=0 base rate (Δ over baseline) per prompt set; the dose-response is on Δ.

**4.4 Sampling vs greedy.**
*Why:* misalignment is a low-base-rate *rate* — greedy decoding gives one sample and underpowers it; collapse metrics (repetition) are better on greedy.
*R:* greedy for the collapse battery; temperature-1.0 ×N samples for misalignment rate (EM convention). Record N and temperature.

**4.5 "Emergent" requires cross-topic spread, not on-topic compliance.**
*Why:* steering toward harm and then seeing harm *on the same topic* isn't emergence; the EM claim is broad/off-topic misalignment.
*R:* the misalignment question set spans topics unrelated to the direction's contrast data; report on-topic vs off-topic misalignment separately.

**4.6 Self-perplexity as both a collapse metric and a baseline.**
*Why:* it's cheap and NLA-free, so it's a H6 competitor *and* a collapse component — don't double-count.
*R:* use self-PPL as a NLA-free **detector** in H6; in the collapse *composite* (§DESIGN 7.2) keep it but pre-register weights so the composite isn't tautological with any single detector.

**4.7 Capability vs alignment degradation are different axes.**
*Why:* a steer can wreck MMLU while leaving "alignment" judged fine, or vice-versa. The user asked about "performance ... and other things like emergent misalignment" — both.
*R:* track accuracy/MMLU (capability) and judge-alignment (safety) as **separate** dependent variables vs OME; the four-quadrant analysis is run for *each*.

---

## 5. Misalignment-specific

**5.1 Steering-induced misalignment ≠ fine-tuning-induced misalignment (external validity).**
*Why:* the user cares ultimately about steering *and* fine-tuning; R2 steering is a fast proxy for the latter.
*R:* steering is the primary, cheap vehicle (and is itself one of the two named dangers); H7 (P5) closes the gap to fine-tuning with its own caveats (§7).

**5.2 Statistical power for a low base rate.**
*Why:* if misaligned-coherent responses are rare, ROC/partial-corr estimates are unstable.
*R:* power the misalignment set by N samples × prompts so the *expected* count of misaligned-coherent responses per condition is ≥ ~20 at the onset α; if the pilot shows the rate is too low, raise N or pick stronger directions before the full run.

**5.3 The misalignment-onset α vs the accuracy-peak α vs the OME-elevation α — the temporal-ordering question.**
*Why:* the whole bet is *where* on the magnitude axis each thing turns on. If misalignment begins *before* OME visibly rises, OME is a lagging indicator (bad for early warning).
*R:* P4 explicitly aligns these three onset points per direction; an OME that rises *no later than* misalignment is the early-warning property we need.

**5.4 Which misaligned behaviors to elicit — breadth.**
*Why:* "misalignment" spans harmful advice, deception, power-seeking, sycophancy, toxicity; OME may track some but not others.
*R:* use a battery spanning these; report per-behavior detection AUC, not just a pooled "misalignment."

---

## 6. Statistics & reliability / transfer

**6.1 ★ "Reliable" = transfer, not in-sample correlation.**
*Why:* a threshold tuned per-direction always looks good; a *gauge* must work on directions/tasks it wasn't tuned on.
*R:* leave-one-direction-out and leave-one-task-out AUC are the headline reliability numbers (H5); in-sample correlations are secondary.

**6.2 Operating point matters more than AUC for a guardrail.**
*Why:* a safety monitor runs at a fixed low false-positive rate; AUC hides whether recall is usable there.
*R:* report recall at fixed 1%/5% benign FPR (threshold calibrated on benign only), not just AUC.

**6.3 Partial correlation OME⊥coherence→misalignment (the H4 crux) is sensitive to how coherence is measured.**
*Why:* if coherence is the judge's coherence score (4.1), residualizing on it may over-subtract.
*R:* residualize on *multiple* coherence proxies (judge-coherence, self-PPL, repetition) and report the range; the claim "OME predicts misalignment beyond coherence" must survive all of them.

**6.4 Multiple comparisons.**
*Why:* many directions × behaviors × detectors → false positives.
*R:* pre-register the primary hypotheses (H1–H6); correct secondary per-behavior/per-direction tests (Benjamini–Hochberg); bootstrap CIs on everything (reuse `fve_analysis.bootstrap_ci`).

**6.5 Non-monotonicity / saturation at extreme α.**
*Why:* at huge α the AV emits garbage and OME saturates; the model output is trivially collapsed — these points can dominate a Pearson correlation (the prior H2 Pearson was "anchored by α=30").
*R:* analyze with rank statistics (Spearman) primarily; report the dose-response curve shape, not just a scalar correlation; cap/flag the saturated regime.

---

## 7. Baselines & cost-justification (the make-or-break)

**7.1 ★★ Does OME beat Mahalanobis (NLA-free)?**
*Why:* the steering-collapse literature already uses Mahalanobis-to-centroid as an off-manifold detector. OME requires a whole NLA + a GPU pod; Mahalanobis is numpy. If they tie, the NLA isn't justified.
*R:* H6 is a first-class hypothesis with a DeLong test; the FINDINGS headline plot is OME-vs-Mahalanobis ROC. A WIN *requires* OME > best NLA-free baseline.

**7.2 The fair baseline is *direction-aware*, not just magnitude.**
*Why:* beating `ratio` (pure magnitude) is easy and unimpressive; beating Mahalanobis/kNN (which capture covariance/density) is the real test.
*R:* the baseline set includes the direction-aware Maha/kNN, fit on benign only; incremental-AUC of OME *over the baseline stack* is reported.

**7.3 Benign-cohort fit leakage.**
*Why:* if Maha/kNN are fit on data overlapping the eval rows, they're unfairly strong.
*R:* fit on a disjoint benign (α=0) split; same split discipline as the directions (3.4).

**7.4 OME's potential unique value: catching danger that's *on the magnitude manifold but off the semantic manifold*.**
*Why:* the a-priori reason OME could win is its scale-invariance/directionality (2.3) — a steer that keeps ‖h‖ and even local density normal but points somewhere semantically incoherent.
*R:* specifically test the regime where Maha is low but OME is high (and vice-versa) — that intersection is where the NLA earns or loses its keep.

---

## 8. Fine-tuning arm (P5 / H7)

**8.1 ★ The base-model NLA on fine-tuned-model activations (the 2.1 confound, sharpened).**
*Why:* OME of a FT model's activations uses an NLA trained on the *base* model — distribution shift from FT alone could raise OME with no danger.
*R:* the **benign-control fine-tune** (matched size/LR) is mandatory; H7 only claims something if harmful-FT raises OME *more* than benign-FT. If they're equal, OME detects "was fine-tuned," not "is dangerous" — reported as inconclusive.

**8.2 Where to harvest activations post-FT.**
*Why:* on training-topic prompts vs neutral held-out prompts gives different OME.
*R:* harvest on a **neutral held-out** set (the emergence test surface), not the FT topic.

**8.3 Whether to verbalize the *FT model's* activation with the *base* NLA at all is theoretically questionable.**
*Why:* the NLA contract assumes base-distribution vectors.
*R:* frame H7 as exploratory; the cleaner (but out-of-scope, expensive) version would retrain/adapt the NLA on the FT model — noted as future work, not attempted here.

**8.4 Inducing EM reliably in 7B.**
*Why:* EM is strongest in larger models; a 7B LoRA may give a weak signal.
*R:* use the validated insecure-code recipe (or persona-feature steering-then-distill) and *verify* broad misalignment via the §P3 judge before computing any OME; if induction fails, H7 is not attempted (don't compute OME on a non-misaligned FT).

---

## 9. Compute, ops, reproducibility

**9.1 AV throughput is the bottleneck (~2.4 rows/s); full grid is ~10⁵ calls.**
*R:* OME runs on a **256-row subset per condition** (OME is a per-condition mean, not per-row deliverable); resumable ledger; pilot-gate before the full sweep (SPEC §7, §9).

**9.2 OME stochasticity vs cost (1.3).**
*R:* k=1 AV draw for headline means; k≥3 only for the per-row ROC analysis on the pilot to size the variance.

**9.3 Generation + judging are the *other* cost, often underestimated.**
*R:* cache judgements by response hash; resumable; greedy collapse pass is cheap, sampled misalignment pass is budgeted explicitly.

**9.4 Determinism.**
*R:* seeds recorded (SUBSET_SEED=7, random-direction seeds, AV seed/temperature per row); sglang `--random-seed` at launch; greedy where determinism is needed.

**9.5 Schema drift / silent number changes.**
*R:* a `graphing/`-style `data.py` with `validate_anchors()` (base misalignment rate, OME floor, KAPPA peak) fails loudly if committed numbers drift.

**9.6 Off-repo data discipline.**
*R:* inherit `.gitignore` — track only `*.md`/`*.json`/`*.png`; arrays/parquet/jsonl on S3 `nla/ome/`.

---

## 10. Safety & dual-use

**10.1 The experiment elicits harmful/misaligned generations.**
*R:* local/private-pod only; no serving/deployment/exfiltration; only aggregate scores + OME committed; harmful datasets used only to build measurement directions; the FT checkpoint stays on the pod and is deleted (DESIGN §12).

**10.2 The directions themselves are dual-use (a jailbreak vector is a steering vector).**
*R:* commit direction *provenance and metrics*, not the raw vectors if they constitute a ready jailbreak; treat `dirs.npz` as off-repo/S3 with the same care as other large arrays.

**10.3 Responsible framing.**
*Why:* the deliverable is a *defensive* guardrail; the negative result (OME blind to coherent misalignment) is itself safety-relevant and should be published honestly rather than buried.
*R:* FINDINGS.md states the blind-spot characterization prominently whichever way it lands.

---

## 11. Open scientific questions (not blockers — things the result will illuminate)

**11.1** Is misalignment *intrinsically* off-manifold, or can it be perfectly on-manifold (a "natural" but evil persona the base model can already inhabit)? Quadrant ④'s size answers this — and it's a genuinely interesting interpretability result either way.

**11.2** Does the AV *verbalization* of a misaligned-but-coherent activation *say* anything revealing (as the parent repo used verbalizations as a read-out)? Inspecting the AV text on ④ candidates may show whether the NLA "knows" the state is off even when cos is high.

**11.3** Does OME track the *toxic-persona direction* specifically (OpenAI/Anthropic's supervised signal) — i.e., is unsupervised OME correlated with the supervised persona-projection? If so, OME is a label-free proxy for the persona monitor; if not, they're complementary.

**11.4** Is there a "productively off-manifold" zone for *misalignment* analogous to accuracy's (quadrant ③) — mild steering that's off-manifold but *more* aligned (e.g., +D_refusal making the model safer)? That would show OME penalizes *beneficial* steering too, bounding its use as a one-sided guardrail.

**11.5** How does the OME→collapse law scale with model size / generalize across NLAs (Gemma-3-12B L32, injection_scale 80000)? A one-direction replication on a second model would establish the law isn't a Qwen/NLA-specific artifact (stretch; SPEC §0).

**11.6** Could OME be *gamed*? An adversary optimizing a steer to maximize misalignment *subject to* low OME would directly map the ④ frontier — the strongest possible test of OME-as-guardrail, and a natural follow-up if H4b appears.

---

### Priority triage for the pilot
The ★★ items decide whether the experiment is worth scaling: **2.1** (NLA-manifold vs model-manifold — controlled by the random arm), **4.1** (coherence-gate circularity — controlled by unconditional reporting), **7.1** (does OME beat Mahalanobis). The pilot (SPEC §7) is sized specifically to get early reads on these three before the full grid spend.
