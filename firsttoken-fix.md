# First-token scoring fix — implementation plan

Status: ready for implementation (do NOT implement from this session)
Owner: Aiden
Decisions locked: 2026-05-16

Scope decisions (locked with the user before this plan):

1. **Fix both defects** — the fp16 softmax underflow *and* the letter-variant/position mismatch.
2. **Remediate via forward-only rescore in place + retrain C07** — recompute only the broken
   first-token/agreement fields into `rows.jsonl` for the same `run_id`; prompts + C06 activation
   cache + pilot fold are preserved; C07 probes are retrained against the corrected rows.
3. **`first_token_letter` is derived from `letter_logits` argmax** — monotone-robust and
   immune to any future softmax-numerics regression.

---

## 1. Diagnosis

### Symptom (confirmed against `runs/20260515T023617Z-7830851-mvp/`)

`rows.jsonl` line 1, verbatim:

```
gold_letter="A" first_token_letter="A" free_text_letter="B" free_text_raw="B"
total_letter_mass=0.0  first_token_scoring_math="full_vocab_softmax"
letter_softmax = [
  {letter:A token_id:362 prob:0.0 prob_valid:true logit:11.375},
  {letter:B token_id:425 prob:0.0 prob_valid:true logit:17.921875},
  {letter:C token_id:356 prob:0.0 prob_valid:true logit:5.234375},
  {letter:D token_id:422 prob:0.0 prob_valid:true logit:4.7734375},
]
```

Letter **logits** are correct and discriminative (B ≫ A ≫ C ≈ D, and `free_text_letter="B"`
tracks the top logit). Only the derived `first_token_letter` and the `prob`/`total_letter_mass`
values are broken: every probability is *exactly* `0.0` and `first_token_letter` is `"A"` for
1776/1776 pilot rows.

### Defect 1 — fp16 full-vocab softmax underflow (proximate cause)

`Qwen25Wrapper.score_first_token` (`src/nl_ae/inference/wrapper.py:309-377`):

```python
forward = self.forward(prompt, capture_hiddens=False)
logits = forward.last_token_logits          # fp16 (Qwen2.5-7B loaded fp16)
...
letter_logits = logits[letter_ids]          # correct — computed independently
...
else:  # full_vocab_softmax  (wrapper.py:353-356)
    full = torch.softmax(logits, dim=0)     # <-- stays fp16
    probs = full[letter_ids]
    total_mass = float(probs.sum().item())
argmax_idx = int(torch.argmax(probs).item())  # wrapper.py:358
```

`forward.last_token_logits` is `outputs.logits[0, -1, :]` from a model loaded with
`torch_dtype=torch.float16` (`wrapper.py:133-137`, `manifest.json model.quantization.kind="fp16"`).
`torch.softmax` returns a tensor of the **same dtype as its input** (fp16). Under full-vocab
softmax over Qwen2.5's ~152k-token vocabulary, the per-letter probabilities are legitimately
tiny; any value below fp16's smallest subnormal (≈ `5.96e-8`, i.e. `exp(-16.6)`) flushes to
**exactly `0.0`**. So:

- `probs` becomes the all-zero vector → every `LetterScore.prob == 0.0`.
- `total_mass = probs.sum() == 0.0` → `total_letter_mass == 0.0`.
- `torch.argmax` of an all-equal (all-zero) tensor returns **index 0**. The letter table is
  built A,B,C,D… in order, so `letters[0].letter == "A"` → `first_token_letter == "A"` for
  every row, regardless of the model's actual prediction.
- `letter_logits` is sliced directly from `logits` (`wrapper.py:326`), never passed through
  softmax, so the recorded `logit` values are correct — exactly matching the symptom.

### Defect 2 — letter-variant / position mismatch (necessary co-cause; correctness bug)

The scorer is pinned to the `leading_space` variant. `EvalConfig.letter_variant`
(`src/nl_ae/eval/runner.py:124`) defaults to `"leading_space"`; the run's
`config_yaml_text` confirms `eval.letter_variant: "leading_space"`. In
`score_and_generate` (`wrapper.py:504-510`):

```python
filtered = select_canonical_variant(list(letter_token_table), chosen=variant)  # "leading_space"
```

so the scored token ids are the `leading_space` ones (`manifest.letter_token_table`:
`A→362 " A"`, `B→425 " B"`, …).

But the rendered prompt does **not** end with a space. `PromptRenderer` default
`trailing="answer_colon"` (`src/nl_ae/prompt/renderer.py:108,62-69`) appends `"\nAnswer:"`
to the *user message*; the run uses Qwen's chat template (`chat_enabled: true`), so
`HFChatAdapter` (`src/nl_ae/prompt/chat_adapter.py:36-41`) wraps the body and
`add_generation_prompt=True` appends `…<|im_start|>assistant\n`. **The final prompt ends with
`\n`.** The model's next-token distribution therefore concentrates on the **bare** letter
token (`B` = id `33`), *not* the `leading_space` token (`" B"` = id `425`) the scorer reads.
`free_text_raw == "B"` (a bare letter, no leading space) independently confirms the model
emits the bare token.

This is *necessary* to explain the exact-zero symptom even for the winning letter: if `" B"`
(id 425) were the global argmax, fp32-or-not its softmax probability would be ≈0.5–0.99 and
would **not** underflow in fp16. For `prob(" B")` to flush to `0.0`, some other token must
out-logit id 425 by ≳17 nats — that token is the bare `"B"` (id 33) the model actually
predicts. So Defect 1 produces the exact zeros and the all-`A` tie-break, and Defect 2 is
why the zeros hit *every* letter including the correct one. Fixing only Defect 1 stops the
crash but leaves first-token probabilities measured at the wrong token — systematically tiny,
biasing `first_token_letter`, `first_token_correct`, and all calibration analysis.

### Why existing tests/guards did not catch it

- No test exercises `score_first_token`/`score_and_generate` with a real fp16 logits tensor.
  `tests/test_wrapper_layer_indexing.py` uses a duck-typed `_FakeTensor`; every other
  `letter_softmax` in the suite is hand-built with synthetic non-degenerate probs
  (`tests/test_schema.py`, `tests/test_aggregate.py`, `tests/test_probes.py`, …).
- The schema guard is necessary-but-not-sufficient. `ResultRow._check_invariants`
  (`src/nl_ae/schema/models.py:97-126`) only requires, for non-`argmax_logits_only` math,
  that every entry has `prob_valid=True` and `prob is not None`. `prob=0.0, prob_valid=true`
  and `total_letter_mass=0.0` (`Field(ge=0.0, le=1.0)`) pass validation, so the corrupt rows
  were written and re-validated by `derive_parquet_from_jsonl` without error.

### Blast radius

Corrupt (must be regenerated):
- `runs/20260515T023617Z-7830851-mvp/rows.jsonl` — `first_token_letter`, `letter_softmax`,
  `total_letter_mass`, and the derived `agreement_flag` (vs. free text).
- `…/rows.parquet` and `…/aggregates/*` (esp. `calibration.parquet`, `per_position_bias`,
  `per_subject_mmlu.accuracy_first_token`, `top1_disagreement` — `compute_calibration`
  reads `_argmax_prob`, all `None`/degenerate now).
- `…/pilot/probes/*` (C07): `first_token_letter` label is constant `"A"` (degenerate),
  `first_token_correct` collapses to "is gold == A", and all probe calibration is meaningless.

Unaffected (reused as-is):
- Prompt sidecars (`prompts/<hash>.txt`) — the bug never touched prompts.
- C06 activation cache (`pilot/activations/`). `ProbeTrainer.run` keys rows by
  `(item_id, permutation_id, template_id)` and pulls features from the cache by the same key
  (`src/nl_ae/probes/trainer.py:404-449`); activations are independent of scoring.
- `pilot_manifest.json` — the pilot fold is a deterministic function of
  `(seed, item_ids, frac, stratify)` only (C09 module 09), independent of row scoring; the
  fold and its digest stay valid as long as the item set and seed are unchanged.
- No `preregistration.md` exists anywhere in the repo → the C09 confirmatory gate is not yet
  engaged for this run, so remediation has no preregistration-invalidation cost.
- `free_text_*` / extractor fields — free generation is independent of the first-token
  scoring path and deterministic under greedy decode; preserved verbatim by the rescore.

---

## 2. Scope

In:
- Numerical fix to `score_first_token` (all three `FirstTokenScoringMath` branches).
- `first_token_letter` (argmax) derived from `letter_logits`, not from probabilities.
- Variant-correctness fix: per-prompt resolution of the letter-token variant from the
  rendered prompt tail; new `"auto"` policy as the default.
- A pure, torch-free-testable scoring core extracted from `score_first_token`.
- New `nlae rescore-first-token` CLI (forward-only, in-place), mirroring the
  `materialize-prompts` precedent.
- Tests; re-run runbook; provenance sidecar.

Out:
- No `ResultRow` / `RunManifest` / on-disk schema change (kept SemVer-stable).
- No change to free-text generation, the extractor, the renderer's prompt bytes, the C06
  cache, the pilot fold, or the C07 trainer logic (only a retrain invocation).
- No re-extraction of activations; no `pilot-init` re-run; no new `run_id`.
- The probe-manifest-digest gap and the schema-hardening invariant are documented under
  **Outstanding** but not implemented here (they are separate, larger changes).

---

## 3. Module layout

```
src/nl_ae/inference/
  scoring.py            # NEW: pure score_letters_from_logits() + resolve_letter_variant()
  wrapper.py            # MOD: score_first_token (L309-377) calls scoring.py; fp32; logits-argmax
                        #      score_and_generate (L483-552) resolves "auto" variant per prompt
  rescore.py            # NEW: pure forward-only rescore loop (mirrors prompt/materialize.py)
  outputs.py            # unchanged (schema already permits the corrected values)

src/nl_ae/prompt/
  letter_tokens.py      # MOD: re-export resolve_letter_variant; LetterVariant unchanged

src/nl_ae/eval/
  runner.py             # MOD: EvalConfig.letter_variant accepts "auto"; default → "auto"

src/nl_ae/cli/commands/
  rescore_first_token_cmd.py   # NEW: click handler (engine + renderer rebuilt from manifest)
  __init__.py                  # MOD: register
src/nl_ae/cli/main.py          # MOD: cli.add_command(...)

tests/
  test_first_token_scoring.py  # NEW: fp16-underflow regression + pure-core + variant resolver
  test_rescore_first_token.py  # NEW: rescore loop over a tiny rows.jsonl with a fake engine
```

---

## 4. Chosen fix

### 4.1 Numerical core (Defect 1) — `src/nl_ae/inference/scoring.py` (new, pure)

Extract the math into a torch-free-importable function (torch passed in / imported lazily,
but the function takes plain tensors so it is unit-testable with a tiny synthetic vocab):

```python
def score_letters_from_logits(
    logits, letter_ids, *, scoring_math: FirstTokenScoringMath
) -> tuple[int, list[float | None], float, bool]:
    """Return (argmax_local_idx, per_letter_prob, total_letter_mass, probs_valid).

    - argmax is ALWAYS argmax over letter_logits (= logits[letter_ids]); monotone,
      never re-coupled to softmax numerics.
    - full_vocab_softmax / renormalize_over_letters compute in fp32:
        lp = torch.log_softmax(logits.float(), dim=0)         # full vocab, fp32
        probs = lp[letter_ids].exp()                          # accurate tiny values
        total = float(probs.sum())                            # may be << 1, never 0 spuriously
      renormalize: probs = torch.softmax(letter_logits.float(), dim=0); total = 1.0
    - argmax_logits_only: per_letter_prob = [None]*k, total = 0.0, probs_valid=False.
    Probabilities are returned as Python float (float64); clamp to [0,1].
    """
```

Rationale for `log_softmax(.float()).exp()` over `softmax(.float())`: identical results for
representable magnitudes, but the log-domain path keeps small letter probabilities accurate
(no catastrophic cancellation) and makes the "tiny but non-zero" reality visible instead of
re-flushing. `argmax` is taken from `letter_logits` so `first_token_letter` is correct even
when the true full-vocab letter mass is genuinely ~`1e-9` (the Defect-2 situation before its
own fix lands, and any future numeric regression).

`Qwen25Wrapper.score_first_token` (`wrapper.py:309-377`) is rewritten to: build
`letter_ids`/`letter_logits` as today, call `score_letters_from_logits`, and assemble the
existing `LetterScore`/`FirstTokenScore` objects from its return. `argmax_logits_only`
keeps emitting `prob=None, prob_valid=False, total_letter_mass=0.0` (no schema-validator
change needed; `models.py:116-125` still satisfied).

### 4.2 Variant correctness (Defect 2) — `resolve_letter_variant`

Add a pure resolver in `scoring.py`, re-exported from `letter_tokens.py`:

```python
def resolve_letter_variant(prompt: str) -> LetterVariant:
    # The scored token is the model's FIRST generated token after `prompt`.
    if prompt.endswith(" "):      return "leading_space"   # e.g. "...Answer: "
    return "bare"                                           # "...assistant\n", "...Answer:"
```

(`newline_prefixed` stays defined for completeness but is unreachable here: a trailing `\n`
is already consumed as a prompt token, so the next token is the *bare* letter.)

- `EvalConfig.letter_variant` (`src/nl_ae/eval/runner.py:124`) gains `"auto"` and defaults
  to `"auto"`. Explicit `"bare"`/`"leading_space"`/`"newline_prefixed"` remain honored
  overrides.
- `Qwen25Wrapper.score_and_generate` (`wrapper.py:483-552`): when the incoming `variant`
  is `"auto"`, resolve it via `resolve_letter_variant(prompt)` *before*
  `select_canonical_variant`; pass the resolved concrete variant to `score_first_token`;
  set `provenance["variant"]` (already a key, `wrapper.py:549`) to the **resolved** value
  and add `provenance["variant_policy"]="auto"` so the row's effective variant is auditable.
- The existing fallback at `wrapper.py:505-507` (when the variant's rows are absent) is kept
  as a safety net.

This is general: chat runs (tail `assistant\n`) correctly score the bare token; non-chat
`trailing="answer_colon"` flat runs ending `"Answer: "` (trailing space) correctly score
`leading_space`. No prompt bytes change, so prompt hashes and the C06 cache stay valid.

### 4.3 Remediation — `nlae rescore-first-token` (forward-only, in place)

New pure loop `src/nl_ae/inference/rescore.py` + click handler
`rescore_first_token_cmd.py`, structurally mirroring
`src/nl_ae/prompt/materialize.py` + `materialize_prompts_cmd.py`:

Pre-flight (reuse the materialize idiom, `materialize.py:69-132`):
- Load `manifest.json`; require `completion_status == "completed"` and non-null
  `config_yaml_text`.
- `load_config_from_text(manifest.config_yaml_text, overrides=manifest.cli_args["overrides"])`.
- Rebuild the engine **pinned to the recorded model**: `Qwen25Wrapper(config=cfg.model,
  extractor=RegexLadderExtractor(), pinned_chat_template_hash=manifest.chat_template_hash)`
  using `manifest.model.hf_model_commit` as the revision (so logits match the original
  distribution). Rebuild the renderer exactly as `eval_cmd.py:112-124` /
  `materialize_prompts_cmd.py:87-126` (tokenizer, `TemplateRegistry.from_records(
  manifest.prompt_templates)`, `make_chat_adapter`). Rebuild the letter-token table with
  `cfg.dataset.letter_token_variants` (`eval_cmd.py:162-164`).
- Refuse if the live `chat_template_hash` ≠ `manifest.chat_template_hash` (same gate as
  `materialize_prompts_cmd.py:101-115`).

Per-row loop (stream existing `rows.jsonl` in order):
1. Re-render the prompt for `(item_id, permutation_id, template_id)`; assert recomputed
   `prompt_hash == row.prompt_hash` (reuse `materialize.py:172-182` mismatch error). Prompts
   are unchanged by this fix, so this must hold for every row — it is the integrity gate.
2. **Forward pass only** — `engine.score_first_token(prompt,
   letter_token_table=<resolved-variant subset>, scoring_math=row.first_token_scoring_math)`.
   No `generate()` call. ~10× cheaper than re-running free-gen and eliminates any risk of
   free-text drift across transformers versions.
3. Recompute only: `first_token_letter`, `letter_softmax`, `total_letter_mass`,
   `first_token_scoring_math` (unchanged value, re-stamped), and
   `agreement_flag = (first_token_letter == row.free_text_letter)` when both non-null else
   `None`. **Preserve verbatim**: `free_text_raw`, `free_text_letter`,
   `free_text_truncated`, `free_text_seed`, `decode_strategy`, `extractor_id`,
   `extractor_match_rule`, `gold_letter`, `prompt_hash`, `rendered_prompt_ref`,
   `activation_ref`, `n_options`, ids, `created_at`. Set `wall_time_ms` to the new forward
   time (document this as the one intentional non-scoring field delta).
4. Re-validate through `ResultRow` and buffer.

Finalize (atomic, audit-preserving):
- Write the rewritten rows to `rows.jsonl.tmp` then `os.replace` over `rows.jsonl`
  (mirror `materialize._atomic_write`; do **not** use `ResultsWriter` — its resume/append
  semantics, `writer.py:99-126`, are wrong for a full rewrite).
- Re-derive `rows.parquet` via `schema.writer.derive_parquet_from_jsonl`.
- Write `runs/<run_id>/rescore_manifest.json` (+ `.sha256` sidecar, matching the project's
  sidecar discipline): UTC timestamp, current git sha, the fix commit sha, old & new
  `rows.jsonl` SHA-256, rows_seen, rows_changed (and a small histogram of
  `first_token_letter` before/after), resolved-variant policy, per-`scoring_math` counts.
- Append a one-line stamp to `manifest.notes` (nullable string, no schema change);
  **do not** mutate `run_id`, `seeds`, `config_digest`, or `completion_status`.

CLI surface (mirrors `materialize_prompts_cmd.py`):

```
nlae rescore-first-token
  --run-dir runs/<run_id>     # required; completed run
  --limit N                   # debug: first N rows
  --hf-cache-dir PATH         # default $HF_HOME
  --dry-run                   # compute + report deltas, write nothing  (recommended first pass)
```

Downstream regeneration (runbook §7):
- `nlae aggregate --run-dir …` to rebuild `aggregates/*`.
- `nlae probe-train --run-dir … --fold pilot --on-existing overwrite`.
  **Critical:** `compute_probe_manifest_digest` (`probes/trainer.py:350-363`) does **not**
  incorporate any digest of the label-source rows, so `scan_resume_state`
  (`trainer.py:239-285`) would treat the stale cells as completed and *skip* them under the
  default `resume`. Remediation MUST pass `--on-existing overwrite` (or delete
  `pilot/probes/` first). This foot-gun is called out in §7 and logged under Outstanding.

---

## 5. Rejected alternatives

- **fp32 only, leave the variant pinned (Defect 1 only).** Rejected per locked decision #1:
  stops the crash but first-token probabilities stay measured at the wrong token (id 425),
  systematically ~`1e-9`; calibration and `first_token_correct` remain invalid.
- **Switch default `scoring_math` to `renormalize_over_letters`.** Changes the scientific
  measure (Wang et al. recreation uses full-vocab mass), not a bug fix; also still fp16 and
  still variant-mismatched. Keep it as an option; fix its dtype too.
- **fp16 + additive epsilon / clamp.** Does not address the underflow magnitude (gap ≳17
  nats); produces fabricated uniform mass. Wrong.
- **Hard-switch the default variant to `"bare"`.** Wrong for legitimate trailing-space
  prompts (`trailing="answer_colon"` without chat → `"Answer: "`). `"auto"` resolution is
  the general, prompt-correct choice.
- **Derive `first_token_letter` from the fp32 probs argmax.** Mathematically equivalent when
  probs are well-formed, but re-couples the recorded answer to softmax numerics. Logits-argmax
  is monotone and regression-proof (locked decision #3).
- **Full Phase-1 re-run under a new `run_id`.** Wasteful (re-extract all activations,
  re-`pilot-init`, fresh C07), and breaks `run_id`/pilot-fold continuity for no benefit —
  activations and prompts are provably unaffected.
- **Full Phase-1 re-run overwriting the same `run_id`.** Still needlessly re-extracts
  activations and risks free-text drift across the `transformers 5.8.1` environment vs. a
  future one. The forward-only rescore preserves free-text bytes exactly.
- **Carry rescore provenance by adding a `ResultRow`/`RunManifest` field.** Schema bump +
  history rewrite for audit metadata; a `rescore_manifest.json` sidecar + `manifest.notes`
  stamp is lighter and matches existing sidecar conventions.
- **Reuse `ResultsWriter` for the rewrite.** Its `on_existing` resume path appends
  (`writer.py:106-126`); a full deterministic rewrite needs temp-file + `os.replace`.

---

## 6. Test plan

All tests are model-free. torch is imported via `pytest.importorskip("torch")` (a light
import — no weights, no GPU, no network), consistent with the suite running on a GPU-less box.

`tests/test_first_token_scoring.py`:
- **fp16 underflow regression (the bug):** synthetic `logits = torch.full((4096,), -30.,
  dtype=torch.float16)`; set one non-letter index to `+30.`; set letter ids to
  `[+11.4, +17.9, +5.2, +4.8]` (the recorded row-1 logits). Assert the *old* arithmetic
  (`torch.softmax(logits, 0)[letter_ids]`) is all-zero and its argmax is index 0 — pinning
  the failure — then assert `score_letters_from_logits(..., "full_vocab_softmax")` returns
  argmax index 1 (`"B"`, the max letter logit), all probs `> 0`, `0 < total_mass <= 1`,
  monotone in the logits.
- **fp32 vs fp64 parity:** probabilities from fp16-input vs fp32-input logits agree to
  ~1e-6 after the fix for a well-conditioned (non-underflowing) case.
- **argmax source of truth:** with deliberately NaN/!= probs but clean logits, argmax still
  tracks `argmax(letter_logits)`.
- **`renormalize_over_letters`**: probs sum to 1.0, computed in fp32, argmax = max logit.
- **`argmax_logits_only`**: `prob=None`, `prob_valid=False`, `total_mass=0.0`, argmax = max
  logit (regression: still satisfies `ResultRow._check_invariants`).
- **`resolve_letter_variant`** (pure, no torch): `"...assistant\n" → "bare"`,
  `"...Answer:" → "bare"`, `"...Answer: " → "leading_space"`.

`tests/test_rescore_first_token.py`:
- Build a 4-row `rows.jsonl` + `manifest.json` via existing schema helpers, with the corrupt
  pattern (all `prob=0.0`, `first_token_letter="A"`). Drive the pure rescore loop with a
  duck-typed fake engine (returns canned `FirstTokenScore` per prompt; same fake idiom as
  `tests/test_wrapper_layer_indexing.py` / the materialize tests) and a fake renderer that
  reproduces each row's `prompt_hash`.
- Assert: only `{first_token_letter, letter_softmax, total_letter_mass, agreement_flag,
  wall_time_ms}` change; `free_text_raw/free_text_letter/extractor_*/gold_letter/prompt_hash`
  byte-identical; output validates as `ResultRow`; `rows.jsonl` replaced atomically;
  `rescore_manifest.json` + `.sha256` written with correct old/new digests and counts;
  `manifest.notes` stamped; `run_id`/`config_digest`/`seeds` untouched; a `prompt_hash`
  mismatch raises (integrity gate) and writes nothing.
- `--dry-run`: computes the delta histogram, writes nothing.

No existing fixtures change (they use synthetic non-degenerate probs and remain valid).

---

## 7. Re-run procedure (operator runbook)

Pre: land the code fix on a branch, all tests green. The run lives at
`runs/20260515T023617Z-7830851-mvp/` (env: `transformers 5.8.1`, RTX A5000, ~6000 rows;
the corrupt pilot fold is the 1776-row subset in `pilot_manifest.json`).

1. **Branch + back up** the run dir's `rows.jsonl`, `rows.parquet`, `aggregates/`, and
   `pilot/probes/` (the rescore overwrites them; keep the corrupt copies for the postmortem
   diff until verified).
2. **Dry run:** `nlae rescore-first-token --run-dir runs/20260515T023617Z-7830851-mvp
   --dry-run`. Confirm the reported pre→post `first_token_letter` histogram moves from
   100% `A` to a non-degenerate distribution and `rows_changed ≈ rows_total`.
3. **Rescore:** `nlae rescore-first-token --run-dir runs/20260515T023617Z-7830851-mvp`.
   Verifies every prompt hash, rewrites `rows.jsonl` atomically, re-derives `rows.parquet`,
   writes `rescore_manifest.json`.
4. **Verify** (`rows.jsonl`):
   - `total_letter_mass > 0` for every `full_vocab_softmax` row; `letter_softmax` argmax ==
     argmax of `logit` per row.
   - Spot-check the user's evidence rows: e.g. row 1 should now read
     `first_token_letter="B"` (top logit 17.92), `free_text_letter="B"`,
     `agreement_flag=true`, gold `"A"`; the other cited rows similarly track the top letter
     logit.
   - `first_token_letter` distribution over the run is no longer constant; per-position-bias
     is no longer trivially `A`.
5. **Re-aggregate:** `nlae aggregate --run-dir runs/20260515T023617Z-7830851-mvp`. Sanity:
   `aggregates/calibration.parquet` is non-empty and `first_token_prob` is populated.
6. **Retrain C07 (overwrite — mandatory):**
   `nlae probe-train --run-dir runs/20260515T023617Z-7830851-mvp --fold pilot
   --on-existing overwrite`
   (or `rm -rf runs/.../pilot/probes` first). Default `resume` would keep stale probes
   because the probe manifest digest does not bind row content — see Outstanding.
7. **No** `pilot-init` and **no** `extract-activations` re-run: the pilot fold and the C06
   cache are unaffected (Diagnosis §1, blast radius).
8. **Commit** (note `rows.jsonl` is git-LFS-tracked, commit `e644ce9`): the rescored
   `rows.jsonl`, new `rows.parquet`, regenerated `aggregates/`, retrained `pilot/probes/`,
   `rescore_manifest.json` (+ `.sha256`), stamped `manifest.json`, and the source fix.
   Use a commit message that references this plan and the corrupt-run postmortem.

Rollback: the pre-rescore backups from step 1 plus `rescore_manifest.json`'s recorded old
`rows.jsonl` SHA-256 allow exact restoration; `run_id`/seeds/`config_digest` were never
mutated, so the run identity is unchanged either way.

---

## 8. Outstanding / known gaps (flag, do not fix here)

- **Probe manifest digest does not bind the label source.**
  `compute_probe_manifest_digest` (`probes/trainer.py:350-363`) omits any digest of the
  `rows.jsonl` label columns, so a rescore that changes labels does not invalidate a
  resumable probe run. Worked around operationally in §7 (`--on-existing overwrite`).
  Recommended follow-up: add `source_rows_label_digest` to the probe manifest digest so
  stale probes auto-invalidate.
- **Schema validator accepts degenerate distributions.** `ResultRow._check_invariants`
  (`models.py:116-125`) does not assert that `full_vocab_softmax` rows have
  `total_letter_mass > 0` or non-uniform probs. A hardening invariant would have caught this
  bug at write time, but a naive "mass > 0" check risks rejecting legitimately tiny-mass
  rows; needs a calibrated threshold. Propose as a separate schema-minor change with its
  own test matrix.
- **`wall_time_ms` semantics after rescore.** The rescore overwrites it with the
  forward-only time (faster than the original score+generate). Documented in
  `rescore_manifest.json`; acceptable, but note it when reading timing from rescored runs.
- **`newline_prefixed` variant is unreachable** in the chat path (a trailing `\n` is already
  a prompt token). Kept for non-chat completeness; documented.
- **Per-row scorer provenance is not in `ResultRow`.** `ScoringOutputs.provenance` (incl.
  the resolved `variant`) is computed but dropped by `EvalRunner._process_visit`
  (`eval/runner.py:484-511`). Consider persisting it in a future schema minor so the
  effective variant is queryable without reading `rescore_manifest.json`.
