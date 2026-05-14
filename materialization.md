# `nlae materialize-prompts` — implementation plan

Status: ready for implementation
Owner: Aiden
Decisions locked: 2026-05-14

## 1. Diagnosis

Phase 1's `eval_cmd` writes prompt sidecars at `runs/<run_id>/prompts/<prompt_hash>.txt` when `eval.save_rendered_prompts=true` (default; `src/nl_ae/eval/runner.py:466`). C06's `extract-activations` reads each sidecar, NFC-normalizes, and refuses on hash mismatch (`src/nl_ae/cache/extractor.py:171-186`); its current recovery hint is "re-run Phase 1 with `save_rendered_prompts: true`", which costs all of Phase 1's GPU spend.

The gap to close: `rows.jsonl` plus `manifest.json` carry enough information to rebuild the renderer pipeline deterministically without touching the model.

- `rows.jsonl` carries per-row `prompt_hash`, `item_id`, `dataset_name`, `dataset_split`, `subject`, `template_id`, `permutation_id`.
- `manifest.json` carries `config_yaml_text`, `cli_args.overrides`, `model.hf_model_id` + commit, `chat_template_hash`, `prompt_templates[].template_text`, dataset fingerprints.
- The renderer pipeline (`permutation_for → PromptRenderer.render`) is pure. The only ML touch needed is `AutoTokenizer.from_pretrained` for `apply_chat_template` — no GPU, no model weights.

A new CLI `nlae materialize-prompts --run-dir <dir>` replays this pipeline, writing missing sidecars and verifying every produced prompt's SHA-256 against the row.

## 2. Scope

In:
- New `nlae materialize-prompts` subcommand.
- Pure-library replay function decomposed for unit testing.
- Small refactor: promote the chat-template adapter out of `eval_cmd`'s closure into `prompt/chat_adapter.py`, used by both `eval_cmd` and the new module.
- Manifest-hermetic templates: a `TemplateRegistry.from_records` classmethod.
- Update C06's "re-run Phase 1" hint to point at the new CLI.
- README quickstart blurb.

Out:
- No on-disk contract change (`RunManifest`, `ResultRow`, sidecar layout untouched).
- No update to `rows.jsonl.rendered_prompt_ref` after materialization (C06 doesn't depend on it).
- No `prompts/.materialize.json` provenance artifact (counters print to stdout; log lines go through stdlib logging).
- No record of renderer style ctor args in the manifest (`choice_block_style`, `choice_separator`, `trailing`, `default_system`) — both call sites pin to `PromptRenderer` defaults; drift would be caught loudly by the per-row prompt-hash check.

## 3. Module layout

```
src/nl_ae/prompt/
  materialize.py          # NEW: load_materialize_inputs + materialize_prompts
  chat_adapter.py         # NEW: HFChatAdapter promoted from eval_cmd._make_chat_adapter
  errors.py               # NEW: MaterializeError hierarchy
  template_registry.py    # MOD: + TemplateRegistry.from_records classmethod
  renderer.py             # unchanged
  letter_tokens.py        # unchanged
  identity.py             # unchanged
  __init__.py             # MOD: re-export materialize_prompts, MaterializeOutcome

src/nl_ae/cli/commands/
  materialize_prompts_cmd.py    # NEW: click handler
  __init__.py                   # MOD: register
src/nl_ae/cli/main.py           # MOD: cli.add_command(...)

src/nl_ae/config/loader.py      # MOD: + load_config_from_text(text, *, overrides)

src/nl_ae/cache/extractor.py    # MOD: error message at L174-177 → suggest materialize-prompts
src/nl_ae/cli/commands/eval_cmd.py  # MOD: drop _make_chat_adapter; import HFChatAdapter

tests/test_materialize_prompts.py   # NEW
```

Why `prompt/materialize.py` and not `replay/materialize.py`: the import graph stays acyclic because `load_materialize_inputs` does its `config.loader` + `data.*_loader` imports lazily inside the function body (same idiom as `eval_cmd.py` and `extract_activations_cmd.py`). The pure replay loop in `materialize_prompts` depends only on `schema/`, `data.canonical`, `data.permute`, `prompt.renderer`, `prompt.template_registry` — no cycle, no `transformers`/`datasets` import at module load.

## 4. Public signatures

### `src/nl_ae/prompt/errors.py`

```python
class MaterializeError(RuntimeError): ...
class ManifestNotCompletedError(MaterializeError): ...
class ChatTemplateHashMismatchError(MaterializeError): ...
class TemplateContentHashMismatchError(MaterializeError): ...
class PromptHashRecomputeMismatchError(MaterializeError): ...
class ItemNotInLoaderError(MaterializeError): ...
class SidecarCollisionError(MaterializeError): ...
```

### `src/nl_ae/prompt/chat_adapter.py`

```python
class HFChatAdapter:
    """Wraps tokenizer.apply_chat_template into the ChatTemplateAdapter protocol."""
    def __init__(self, tokenizer: Any, *, identity_hash: Sha256Hex) -> None: ...
    def apply(self, *, system: str | None, user: str, add_generation_prompt: bool = True) -> str: ...
    @property
    def identity(self) -> Sha256Hex: ...

def make_chat_adapter(
    tokenizer: Any, *, identity_hash: Sha256Hex, enabled: bool = True
) -> ChatTemplateAdapter:
    """Returns HFChatAdapter when enabled and tokenizer.apply_chat_template exists;
    otherwise NullChatTemplateAdapter(identity_hash)."""
```

### `src/nl_ae/prompt/template_registry.py` (additive)

```python
class TemplateRegistry:
    @classmethod
    def from_records(cls, records: Iterable[PromptTemplateRecord]) -> "TemplateRegistry":
        """Construct a registry from manifest records. For each record, recompute
        sha256(nfc(template_text).encode("utf-8")) and assert it matches
        template_content_hash; raise TemplateContentHashMismatchError on drift.
        Does not require a templates_dir to exist on disk."""
```

### `src/nl_ae/config/loader.py` (additive)

```python
def load_config_from_text(
    text: str,
    *,
    overrides: Sequence[str] = (),
    env: Mapping[str, str] | None = None,
) -> RunConfig:
    """Same pipeline as load_config(...) but takes raw YAML text. Used by
    materialize-prompts to reparse RunManifest.config_yaml_text."""
```

### `src/nl_ae/prompt/materialize.py`

```python
@dataclass(frozen=True)
class MaterializeInputs:
    run_dir: Path
    paths: RunPaths
    run_manifest: RunManifest
    rows_path: Path
    item_index: Mapping[str, CanonicalItem]
    permutation_mode: PermutationMode
    chat_template_hash: Sha256Hex | None      # from manifest; may be None (legacy)

@dataclass(frozen=True)
class MaterializeOutcome:
    rows_seen: int
    sidecars_written: int
    sidecars_existing: int
    wall_seconds: float
    status: Literal["completed", "failed", "aborted"]
    failure_reason: str | None

def load_materialize_inputs(
    run_dir: Path,
    *,
    hf_cache_dir: Path | None = None,
) -> MaterializeInputs: ...

def materialize_prompts(
    inputs: MaterializeInputs,
    *,
    renderer: PromptRenderer,
    on_existing: Literal["skip", "overwrite", "error"] = "skip",
    limit: int | None = None,
    logger: logging.Logger | None = None,
) -> MaterializeOutcome: ...
```

## 5. Library flow

### `load_materialize_inputs(run_dir, hf_cache_dir=None)`

```
1. paths = run_paths(run_dir.parent, run_dir.name)
2. assert paths.manifest_json.exists()
3. run_manifest = load_manifest(paths.manifest_json)
4. if run_manifest.completion_status != "completed":
       raise ManifestNotCompletedError(...)
5. assert paths.rows_jsonl.exists()
6. # Reparse config (lazy imports)
   from nl_ae.config.loader import load_config_from_text
   overrides_str = run_manifest.cli_args.get("overrides")
   overrides = tuple(overrides_str.split(";")) if overrides_str else ()
   cfg = load_config_from_text(run_manifest.config_yaml_text, overrides=overrides)
7. # Collect distinct item_ids per dataset from rows.jsonl, single pass.
   wanted = _collect_wanted_item_ids(paths.rows_jsonl)   # dict[str, set[str]]
8. # Build loaders from cfg.dataset; iterate; filter to wanted.
   item_index: dict[str, CanonicalItem] = {}
   for ds_name in cfg.dataset.enabled:
       loader = _build_loader(ds_name, cfg.dataset, hf_cache_dir_override=hf_cache_dir)
       for item in loader.iter_items():
           if item.item_id in wanted.get(ds_name, set()):
               item_index[item.item_id] = item
9. # Cross-check.
   missing = {iid for s in wanted.values() for iid in s} - item_index.keys()
   if missing: raise ItemNotInLoaderError(...)
10. return MaterializeInputs(
        run_dir=paths.run_dir,
        paths=paths,
        run_manifest=run_manifest,
        rows_path=paths.rows_jsonl,
        item_index=item_index,
        permutation_mode=cfg.eval.plan.permutation_mode,
        chat_template_hash=run_manifest.chat_template_hash,
    )
```

### `materialize_prompts(inputs, renderer, ...)`

```
1. t0 = perf_counter(); rows_seen = sidecars_written = sidecars_existing = 0
2. prompts_dir = inputs.paths.prompts_dir; prompts_dir.mkdir(parents=True, exist_ok=True)
3. for row in _iter_rows_jsonl(inputs.rows_path):
       if limit is not None and rows_seen >= limit: break
       rows_seen += 1
       sidecar = prompts_dir / f"{row['prompt_hash']}.txt"
       if sidecar.exists():
           if on_existing == "error":    raise SidecarCollisionError(sidecar)
           if on_existing == "skip":     sidecars_existing += 1; continue
           # overwrite → fall through
       item = inputs.item_index.get(str(row["item_id"]))
       if item is None: raise ItemNotInLoaderError(...)
       perm = permutation_for(item, int(row["permutation_id"]), mode=inputs.permutation_mode)
       final, computed_hash = renderer.render(perm, str(row["template_id"]))
       if computed_hash != row["prompt_hash"]:
           raise PromptHashRecomputeMismatchError(
               item_id=item.item_id,
               template_id=row["template_id"],
               permutation_id=row["permutation_id"],
               expected=row["prompt_hash"],
               actual=computed_hash,
           )
       _atomic_write(sidecar, final)
       sidecars_written += 1
       if logger and rows_seen % 1000 == 0:
           logger.info("progress %d (written=%d, existing=%d)",
                       rows_seen, sidecars_written, sidecars_existing)
4. return MaterializeOutcome(
       rows_seen=rows_seen, sidecars_written=sidecars_written,
       sidecars_existing=sidecars_existing, wall_seconds=perf_counter()-t0,
       status="completed", failure_reason=None,
   )
```

`_atomic_write(path, text)`:

```python
tmp = path.with_suffix(path.suffix + ".tmp")
tmp.write_text(text, encoding="utf-8")
os.replace(tmp, path)   # atomic on both Windows and POSIX
```

## 6. CLI handler

`src/nl_ae/cli/commands/materialize_prompts_cmd.py`. Heavy imports stay inside the handler body so `nlae --help` stays fast, matching `extract_activations_cmd.py`.

```
@click.command("materialize-prompts",
               help="Rebuild runs/<run_id>/prompts/<prompt_hash>.txt sidecars from "
                    "rows.jsonl + manifest.json without loading the model. Use this "
                    "to recover from a Phase 1 run where save_rendered_prompts was "
                    "disabled or where sidecars were lost.")
@click.option("--run-dir", required=True, type=Path)
@click.option("--limit", type=IntRange(min=1), default=None,
              help="Debug: process only the first N rows.")
@click.option("--on-existing", type=Choice(["skip","overwrite","error"]), default="skip",
              show_default=True)
@click.option("--hf-cache-dir", type=Path, default=None,
              help="Override HF tokenizer cache dir. Defaults to $HF_HOME.")
def materialize_prompts_cmd(run_dir, limit, on_existing, hf_cache_dir):
    # 1. lazy imports
    from nl_ae.prompt.materialize import (
        load_materialize_inputs, materialize_prompts,
        ManifestNotCompletedError, ChatTemplateHashMismatchError,
        PromptHashRecomputeMismatchError, ItemNotInLoaderError,
        TemplateContentHashMismatchError, SidecarCollisionError,
    )
    from nl_ae.prompt.chat_adapter import make_chat_adapter
    from nl_ae.prompt.template_registry import TemplateRegistry
    from nl_ae.prompt.renderer import PromptRenderer
    from nl_ae.data.text_norm import nfc, sha256_hex_bytes

    # 2. preflight
    try:
        inputs = load_materialize_inputs(run_dir, hf_cache_dir=hf_cache_dir)
    except ManifestNotCompletedError as e:
        click.echo(f"manifest not completed: {e}", err=True); sys.exit(2)
    except ItemNotInLoaderError as e:
        click.echo(f"item index incomplete: {e}", err=True); sys.exit(4)
    except (FileNotFoundError, RuntimeError) as e:
        click.echo(f"preflight failed: {e}", err=True); sys.exit(2)

    # 3. tokenizer (lazy import; no GPU)
    from transformers import AutoTokenizer
    fp = inputs.run_manifest.model
    tok = AutoTokenizer.from_pretrained(
        fp.hf_model_id, revision=fp.hf_model_commit,
        cache_dir=str(hf_cache_dir) if hf_cache_dir else None,
        trust_remote_code=False,
    )
    live_hash = sha256_hex_bytes(nfc(getattr(tok, "chat_template", "") or "").encode("utf-8"))
    if inputs.chat_template_hash is not None and live_hash != inputs.chat_template_hash:
        click.echo(f"chat_template_hash mismatch: live={live_hash} "
                   f"manifest={inputs.chat_template_hash}", err=True)
        sys.exit(4)
    if inputs.chat_template_hash is None:
        click.echo("warning: manifest has no chat_template_hash; "
                   "relying on per-row hash check", err=True)

    # 4. registry + renderer
    try:
        registry = TemplateRegistry.from_records(inputs.run_manifest.prompt_templates)
    except TemplateContentHashMismatchError as e:
        click.echo(f"template content hash drift: {e}", err=True); sys.exit(4)
    renderer = PromptRenderer(
        registry, chat_adapter=make_chat_adapter(tok, identity_hash=live_hash)
    )

    # 5. run
    try:
        outcome = materialize_prompts(
            inputs, renderer=renderer, on_existing=on_existing, limit=limit,
            logger=logging.getLogger("nl_ae.prompt.materialize"),
        )
    except SidecarCollisionError as e:
        click.echo(f"sidecar collision: {e}", err=True); sys.exit(3)
    except PromptHashRecomputeMismatchError as e:
        click.echo(f"prompt hash mismatch: {e}", err=True); sys.exit(4)

    click.echo(
        f"run_id={inputs.run_manifest.run_id} status={outcome.status} "
        f"rows={outcome.rows_seen} written={outcome.sidecars_written} "
        f"existing={outcome.sidecars_existing} wall={outcome.wall_seconds:.1f}s"
    )
```

## 7. Error → exit-code table

| Error | Exit |
|---|---|
| `manifest.json` missing / `completion_status != "completed"` (`ManifestNotCompletedError`) | 2 |
| `config_yaml_text` reparse / env interpolation fails | 2 |
| Dataset loader fails (HF offline + cache missing, etc.) | 2 |
| `ItemNotInLoaderError` (preflight or per-row) | 4 |
| `ChatTemplateHashMismatchError` | 4 |
| `TemplateContentHashMismatchError` | 4 |
| `PromptHashRecomputeMismatchError` (per-row) | 4 |
| `SidecarCollisionError` (`--on-existing=error`) | 3 |
| Other unexpected `MaterializeError` | 5 |

Mirrors C06's tiering.

## 8. Test plan (`tests/test_materialize_prompts.py`)

Pure-Python: fake tokenizer with deterministic `chat_template` + `apply_chat_template`, fake item index built in-process, synthetic `RunManifest` + `rows.jsonl` minted via the schema package — same pattern as `tests/test_cache_extractor.py`. No `torch`, no `transformers`, no `datasets`.

Cases:

1. `happy_path` — 3 rows × 2 templates × 2 permutations; all sidecars written; bytes match `renderer.render(...)`.
2. `idempotent_skip` — re-run; `sidecars_written==0`, `sidecars_existing==N`.
3. `overwrite` — pre-create one sidecar with wrong bytes; `on_existing="overwrite"` rewrites it.
4. `on_existing_error` — pre-create one sidecar; `on_existing="error"` exits 3, no further writes.
5. `prompt_hash_recompute_mismatch` — fake renderer rigged to drift on one row → raises before write; bad sidecar not on disk.
6. `chat_template_hash_mismatch` — fake tokenizer reports different live hash → CLI exits 4 before rendering.
7. `chat_template_hash_null_warn` — manifest has `chat_template_hash=None`; CLI proceeds with warning; rows render fine.
8. `manifest_not_completed` — `completion_status="failed"` → exit 2.
9. `template_content_hash_drift` — manifest record's hash inconsistent with `template_text` → `from_records` raises.
10. `limit` — `limit=1` processes one row, others untouched.
11. `item_not_in_loader` — rows.jsonl references item_id absent from `item_index` → exit 4.
12. `registry_from_records_roundtrip` — unit: build via `from_records`, assert `all_records()` round-trips, `__getitem__` works.
13. `atomic_write_no_partial` — mock `Path.write_text` to raise; assert `.tmp` left, final sidecar absent.
14. `cli_smoke` — `click.testing.CliRunner` end-to-end with monkeypatched `AutoTokenizer` and dataset loader; assert exit 0.

## 9. Implementation order (one PR, staged commits)

Each commit leaves the repo green; existing tests continue to pass throughout.

1. `prompt/template_registry.py` — `from_records` classmethod + unit test (passes in isolation).
2. `prompt/chat_adapter.py` — extract `HFChatAdapter` + `make_chat_adapter`; update `eval_cmd._make_chat_adapter` call site to use the new import.
3. `prompt/errors.py` — error hierarchy.
4. `config/loader.py` — `load_config_from_text` wrapper.
5. `prompt/materialize.py` — library implementation.
6. `cli/commands/materialize_prompts_cmd.py` + register in `cli/main.py` and `cli/commands/__init__.py`.
7. `tests/test_materialize_prompts.py` — all 14 cases.
8. `cache/extractor.py:174-177` — replace "re-run Phase 1" hint with "run `nlae materialize-prompts --run-dir <dir>`".
9. `README.md` — Quickstart paragraph + recovery note in the Phase 2/3 section.

## 10. Decisions locked

| Question | Decision |
|---|---|
| Template source | Manifest only (hermetic). `TemplateRegistry.from_records`. No `--templates-dir` flag. |
| Renderer style ctor args | Ship as-is. Pin defaults at both call sites; per-row prompt-hash check is the safety net. |
| `HFChatAdapter` | Promote from `eval_cmd` closure to `prompt/chat_adapter.py`. Both call sites import. |
| Library home | `src/nl_ae/prompt/materialize.py`. Lazy `config`/`data` imports in `load_materialize_inputs`. |
| CLI args | `--run-dir` required; `--limit`, `--on-existing`, `--hf-cache-dir` optional. No `--config`. |
| Legacy `chat_template_hash=None` | Warn-and-proceed; per-row hash check is the safety net. |
| Pre-existing sidecars | Trust by default (`--on-existing=skip`). `overwrite` is the escape hatch. C06 re-verifies downstream. |
| Atomic writes | `Path.write_text` on `.tmp` + `os.replace`. |
| Provenance artifact | None — counters go to stdout; logs via stdlib logging. |
| `rendered_prompt_ref` in `rows.jsonl` | Not updated. C06 doesn't depend on it. |
| Schema bump | None. |

## 11. Out-of-scope follow-ups (worth considering later, not blocking)

- Record renderer style ctor args (`choice_block_style`, `choice_separator`, `trailing`, `default_system`) on `RunManifest` so a future change to `eval_cmd`'s renderer doesn't break materialize-prompts via prompt-hash drift. Would require a schema bump.
- A `--verify-existing` flag that re-hashes pre-existing sidecars instead of trusting them. Currently C06 does this verification when it reads each sidecar; adding it here would shift the surfacing point earlier in the pipeline.
- A `[replay]` install extra (`transformers` + `datasets` + `huggingface_hub`, no `torch`) so CI users can run materialize-prompts without the GPU stack. Today materialize-prompts requires `nl-ae[model]`.
- A `--update-rendered-prompt-ref` mode that rewrites `rows.jsonl` to set `rendered_prompt_ref` after materialization. Not needed because nothing reads it.
