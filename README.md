# nl-ae

A minimal recreation of Wang et al. 2024,
["'My Answer is C': First-Token Probabilities Do Not Match Text Answers in Instruction-Tuned Language Models"](https://arxiv.org/abs/2402.14499),
with per-layer linear-probe and L20 NLA infrastructure (Phase 2/3, following
[Fraser-Taliente et al.](https://github.com/kitft/natural_language_autoencoders)).
Phase 1 reproduces the headline first-token-vs-free-generation disagreement on
MMLU + OpinionQA using Qwen2.5-7B-Instruct under fp16 weights. Phase 2/3 are
data-collection infrastructure operating under a pilot → preregister →
confirmatory discipline; no hypothesis is encoded into the plan.

## Install

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[model,report,probes,dev]"
```

The default `nl-ae` install pulls only the CLI / schema dependencies so that
`nlae --help`, `nlae aggregate`, and the test suite stay fast and GPU-free.
The `[model]` extra adds `torch`, `transformers`, `accelerate`, `datasets`,
`huggingface_hub`, `safetensors`; `[report]` adds `pandas` + `matplotlib`;
`[probes]` adds `scikit-learn` (Phase 2); `[quant]` adds `bitsandbytes` (only
needed for `int*` quantization; MVP runs fp16).

## Quickstart

### Phase 1 — eval

```powershell
# Render one MMLU subject under the medium-constraint template, K=2 permutations.
nlae eval --config examples\mvp.yaml --set dataset.mmlu_subjects='[abstract_algebra]' --set eval.plan.permutations_per_item=2

# Aggregate + figures.
nlae aggregate --run-dir runs\<run_id>
```

### Phase 2/3 — pilot first, then preregister, then holdout

```powershell
# 1. Assign the 5% stratified pilot fold (deterministic).
nlae pilot-init --run-dir runs\<run_id>

# 2. Pilot data collection (no preregistration required).
nlae extract-activations --run-dir runs\<run_id> --fold pilot
nlae probe-train         --run-dir runs\<run_id> --fold pilot
nlae nla-verbalize       --run-dir runs\<run_id> --fold pilot              # default --limit 50
nlae nla-reconstruct     --run-dir runs\<run_id> --fold pilot

# 3. Render pilot/summary.md for you to read.
nlae pilot-report --run-dir runs\<run_id>

# 4. (Human step) Read pilot/summary.md. Write preregistration.md
#    with primary/secondary hypotheses, labels in scope, layer set in scope,
#    significance threshold, multiple-comparison correction, NLA scope.
#    Then lock it:
nlae preregistration-lock --run-dir runs\<run_id>

# 5. Confirmatory run on the holdout fold (refuses without a locked preregistration).
nlae extract-activations --run-dir runs\<run_id> --fold holdout
nlae probe-train         --run-dir runs\<run_id> --fold holdout
nlae nla-verbalize       --run-dir runs\<run_id> --fold holdout
nlae nla-reconstruct     --run-dir runs\<run_id> --fold holdout
```

## Research discipline

- **Pilot is ~5% of items**, stratified by MMLU subject and OpinionQA wave; same items across all 10 permutations. Pilot data is read; no statistical claim is made from it.
- **Preregistration** (`runs/<run_id>/preregistration.md`) carries YAML frontmatter declaring hypotheses, label set, layer set, significance threshold, multiple-comparison correction, NLA scope, and a git SHA + timestamp lock.
- **Confirmatory runs use the holdout fold only.** Pilot items are excluded. The full-dataset alternative is not honored by current policy; reversing this requires explicit re-preregistration.
- **Hard gate**: any `--fold holdout` command refuses to start unless `preregistration.md` exists, parses as valid `Preregistration` YAML, and the `pilot_manifest_digest` it carries matches the on-disk manifest.

## Repository layout

```
src/nl_ae/
  schema/        # C01 — on-disk contract (ResultRow, RunManifest, writer, reader)
  config/        # C05 — RunConfig, YAML loader, SHA-256 config_digest
  runtime/       # C05 — seeds, run_id, env fingerprint, logging
  cli/           # C05 — click entrypoint
  data/          # C02 — MMLU + OpinionQA loaders, permutations
  prompt/        # C02 — templates, renderer, letter-token table
  inference/     # C03 — Qwen2.5 wrapper, scoring, extractor
  eval/          # C04 — orchestrator + resume
  report/        # C04 — aggregator, figures, summary
  cache/         # C06 — activation cache extractor + reader (Phase 2, fold-aware)
  probes/        # C07 — per-layer linear probes (Phase 2, fold-aware)
  nla/           # C08 — local HF NLA client (Phase 3, L20, fold-aware)
  pilot/         # C09 — pilot fold + preregistration gate
templates/       # *.txt prompt templates + pinned_chat_template.sha256
examples/        # example YAML configs
tests/           # pytest smoke tests (no GPU required)
```

## Status

- Phase 1 (MVP) — shipped. 140,420 rows on MMLU full test split × 10 permutations; top-1 disagreement 0.760, 95% CI [0.758, 0.763].
- Phase 2/3 — data-collection infrastructure, current implementation phase. Pilot/holdout fold discipline; no hypotheses encoded.

## License

MIT.
