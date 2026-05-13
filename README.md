# nl-ae

A minimal recreation of Wang et al. 2024,
["'My Answer is C': First-Token Probabilities Do Not Match Text Answers in Instruction-Tuned Language Models"](https://arxiv.org/abs/2402.14499),
on a single RTX 2070 (8GB VRAM, Windows). MVP reproduces the headline
first-token-vs-free-generation disagreement on MMLU + OpinionQA using
Qwen2.5-7B-Instruct under fp16 weights.

## Install

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[model,report,dev]"
```

The default `nl-ae` install pulls only the CLI / schema dependencies so that
`nlae --help`, `nlae aggregate`, and the test suite stay fast and GPU-free.
The `[model]` extra adds `torch`, `transformers`, `accelerate`, `datasets`,
`huggingface_hub`; `[report]` adds `pandas` + `matplotlib`; `[quant]` adds
`bitsandbytes` (only needed if you switch `QuantizationSpec.kind` to an `int*`
mode — MVP runs fp16).

## Quickstart

```powershell
# Render one MMLU subject under the medium-constraint template, K=2 permutations.
nlae eval --config examples\mvp.yaml --set dataset.mmlu_subjects='[abstract_algebra]' --set eval.plan.permutations_per_item=2

# Aggregate + figures.
nlae aggregate --run-dir runs\<run_id>
```

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
templates/       # *.txt prompt templates + pinned_chat_template.sha256
examples/        # example YAML configs
tests/           # pytest smoke tests (no GPU required)
```

## Status

Phase 1 (MVP) implemented. Phase 2 (per-layer linear probes) and Phase 3
(per-layer NL autoencoders) are stubbed via the `record_layers` / activation-cache
seams but not wired up.

## License

MIT.
