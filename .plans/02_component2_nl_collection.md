# Component 2 — NL data collection via the Activation Verbalizer (AV)

Goal: (1) get the natural-language verbalization of the **original** L20 embeddings; (2) get the verbalization of **each steered** activation. Persist everything keyed so it joins back to exp04 examples and to Component 1's FVE.

This shares the pod/serving with Component 1 (the AV is the same model used in the round-trip; we just also *keep the text* and run it over a wider row set). The AR/round-trip is Component 1; here the deliverable is the **text corpus**.

---

## 1. What we verbalize (scope, default — tunable knob)

| Set | Rows | α values | Why |
|---|---|---|---|
| **orig** | **all 6536** (user-selected) | n/a | the on-manifold reference NL; the store of record for "what L20 says" |
| **steered (dense)** | the **1024 paired subset** `R` | all 11 α | pairs 1:1 with Component-1 FVE → "how does NL degrade as α grows" |
| **steered (headline)** | full **test** (2615) | α ∈ {2, 10, 30} | larger corpus at the calibrated-good (2,10) and collapsed (30) points for the language-space-steering stage |

Rationale: dense α coverage where it's cheap (subset), broad coverage where it matters for the downstream method (headline α on full test). Everything is additive — we can extend α or rows later without rework.

---

## 2. AV call & determinism
- Input: unit-normalized `[3584]` activation (+ `nla_meta.yaml` scale), injected as the special token embedding into the meta's prompt template.
- Sampling: **T=1** (paper), **seed logged** per run; store `seed`, `temperature`, AV ckpt revision (HF commit sha) for provenance. (Optional companion greedy pass T=0 for a deterministic text — decide at run time; default T=1 only.)
- Output: the description string `z` + `n_tokens` + `finish_reason`.

## 3. Storage schema (parquet; one file per set/α)

`nla/nl/orig.parquet`:
```
example_id (str)  row_index (int)  split (str)  source="orig"
nl_text (str)     n_tokens (int)   av_seed (int) av_rev (str)
h_norm (float)    # joined from norms.parquet (original ‖h‖)
```
`nla/nl/steered_a{α}.parquet` (one per α):
```
example_id  row_index  split  source="steered"  alpha (float)
nl_text  n_tokens  av_seed  av_rev
ratio (float)   # ‖Δh‖/‖h‖ for this row,α   cos_h_hp (float)
cos_roundtrip (float, nullable)  # = cos(h,ĥ) if this row,α was also round-tripped in Component 1
```
- `row_index`/`example_id` use the **canonical activation_cache_row_order** (BUCKET.md §5) so NL joins to probes, predictions, FVE, and exp04 outputs.
- Keep a `nla/nl/index.json`: which sets/α/rows were verbalized, counts, seeds, ckpt revs, model id, `config_hash` echo from exp04 (`e80501525b…`) for cross-array provenance.

## 4. Optional lightweight analyses (cheap, high-value)
- **NL drift vs α:** for the dense subset, embed `nl_text` (any small sentence encoder, or token-Jaccard / edit distance vs the α=0 text) and plot mean drift vs α. Pairs with FVE: does language degrade gradually (editable) or collapse to gibberish (not editable)? Directly informs H4 / the decision gate.
- **Keyword/topic deltas:** diff orig vs steered descriptions for a sample → does steering's NL actually move toward the *knowledge* content (the intended effect) or toward noise? Qualitative table in the report.

## 5. Compute / cost
orig **6536** + dense steered 1024×11 (11264, **shared with Component 1's AV pass — do once, keep text**; α=0 reuses the orig text for those rows) + headline steered 2615×3 (7845) ≈ **~25.6k AV generations** total. At hundreds of tokens each, est. ~3–4 GPU-h on A40, overlapping Component 1. Knob: drop headline-on-full-test to cut cost.

## 6. Definition of done (Component 2)
- `nla/nl/orig.parquet`, `nla/nl/steered_a{α}.parquet` (all 11 for subset; α∈{2,10,30} for full test), `nla/nl/index.json` — pushed to S3 + mirrored locally (text is small).
- Provenance complete (seeds, ckpt revs, model id, config_hash) so the corpus is self-describing for the next stage.
