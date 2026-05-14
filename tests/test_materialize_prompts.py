"""``nlae materialize-prompts`` — library + CLI tests.

Pure-Python: a fake tokenizer with a deterministic ``chat_template`` and
``apply_chat_template`` stands in for HuggingFace, and a fake dataset loader is
swapped into :func:`nl_ae.prompt.materialize._build_loader`. No ``torch``,
``transformers``, or ``datasets`` imports.
"""

from __future__ import annotations

import json
import sys
import types
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest
from click.testing import CliRunner

from nl_ae.data.canonical import CanonicalItem
from nl_ae.data.permute import permutation_for
from nl_ae.data.text_norm import nfc, sha256_hex_bytes
from nl_ae.prompt import materialize as materialize_mod
from nl_ae.prompt.chat_adapter import make_chat_adapter
from nl_ae.prompt.errors import (
    ItemNotInLoaderError,
    ManifestNotCompletedError,
    PromptHashRecomputeMismatchError,
    SidecarCollisionError,
    TemplateContentHashMismatchError,
)
from nl_ae.prompt.materialize import (
    MaterializeInputs,
    _atomic_write,
    load_materialize_inputs,
    materialize_prompts,
)
from nl_ae.prompt.renderer import NullChatTemplateAdapter, PromptRenderer
from nl_ae.prompt.template_registry import TemplateRegistry
from nl_ae.schema.hashing import now_utc_iso
from nl_ae.schema.models import (
    DatasetFingerprint,
    EnvFingerprint,
    LetterSoftmaxEntry,
    LetterTokenEntry,
    ModelFingerprint,
    PromptTemplateRecord,
    QuantizationSpec,
    ResultRow,
    RunManifest,
)
from nl_ae.schema.paths import run_paths
from nl_ae.schema.writer import write_manifest_atomic

# --- fakes --------------------------------------------------------------


@dataclass
class FakeTokenizer:
    """Minimal stand-in for an HF tokenizer with chat-template support."""

    chat_template: str = "FAKE-CHAT-TEMPLATE\n"
    apply_calls: list[dict[str, Any]] = field(default_factory=list)

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        tokenize: bool = False,
        add_generation_prompt: bool = True,
    ) -> str:
        assert tokenize is False
        self.apply_calls.append(
            {
                "messages": messages,
                "add_generation_prompt": add_generation_prompt,
            }
        )
        user = next(m["content"] for m in messages if m["role"] == "user")
        return f"<|user|>\n{user}<|assistant|>\n"


@dataclass
class FakeLoader:
    items: list[CanonicalItem]

    def iter_items(self) -> Iterator[CanonicalItem]:
        yield from self.items


# --- canonical items + helpers -----------------------------------------


def _items() -> list[CanonicalItem]:
    return [
        CanonicalItem(
            item_id=f"mmlu/v1/test/q-{i:02d}",
            dataset_name="mmlu",
            dataset_split="test",
            subject="test",
            question=f"What is {i} + {i}?",
            choices=("zero", "one", "two", "three"),
            gold_index=2,
        )
        for i in range(3)
    ]


def _mvp_yaml_text(repo_root: Path) -> str:
    """Verbatim YAML from examples/mvp.yaml; tmp-path overrides flow via cli_args."""
    return (repo_root / "examples" / "mvp.yaml").read_text(encoding="utf-8")


def _mvp_overrides(tmp_path: Path) -> str:
    pinned = (tmp_path / "pinned.sha256").as_posix()
    return ";".join(
        (
            f"output.output_dir={tmp_path.as_posix()}",
            f"dataset.cache_dir={tmp_path.as_posix()}",
            f"dataset.templates_dir={tmp_path.as_posix()}",
            f"dataset.pinned_chat_template_hash_path={pinned}",
            "dataset.offline=true",
            f"model.cache_dir={tmp_path.as_posix()}",
            f"model.pinned_chat_template_hash_path={pinned}",
        )
    )


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _build_renderer(
    repo_root: Path, *, chat_adapter: Any | None = None
) -> PromptRenderer:
    registry = TemplateRegistry(repo_root / "templates")
    registry.load()
    return PromptRenderer(registry, chat_adapter=chat_adapter)


def _expected_prompts(
    items: list[CanonicalItem],
    *,
    template_ids: tuple[str, ...],
    permutation_ids: tuple[int, ...],
    renderer: PromptRenderer,
) -> list[tuple[CanonicalItem, int, str, str, str]]:
    """Return (item, perm_id, template_id, prompt, prompt_hash) for each triple."""
    out: list[tuple[CanonicalItem, int, str, str, str]] = []
    for item in items:
        for pid in permutation_ids:
            perm = permutation_for(item, pid, mode="seeded")
            for tid in template_ids:
                prompt, ph = renderer.render(perm, tid)
                out.append((item, pid, tid, prompt, ph))
    return out


def _mint_manifest(
    *,
    run_id: str,
    config_yaml_text: str | None,
    completion_status: str = "completed",
    chat_template_hash: str | None,
    template_records: tuple[PromptTemplateRecord, ...],
    overrides: str | None = None,
) -> RunManifest:
    return RunManifest(
        run_id=run_id,
        git_sha="deadbeefcafe",
        git_dirty=False,
        started_at=now_utc_iso(),
        ended_at=now_utc_iso(),
        completion_status=completion_status,  # type: ignore[arg-type]
        env=EnvFingerprint(
            os_name="Linux",
            os_version="x",
            python_version="3.11.0",
            cuda_version=None,
            torch_version=None,
            transformers_version=None,
            bitsandbytes_version=None,
            accelerate_version=None,
            gpu_name=None,
            gpu_vram_mb=None,
        ),
        model=ModelFingerprint(
            hf_model_id="fake/model",
            hf_model_commit=None,
            hf_tokenizer_id="fake/model",
            hf_tokenizer_commit=None,
            quantization=QuantizationSpec(kind="fp16"),
        ),
        datasets=[
            DatasetFingerprint(
                name="mmlu",
                hf_dataset_id="cais/mmlu",
                split="test",
                commit_or_revision=None,
                item_count=3,
                item_id_scheme="x",
            )
        ],
        prompt_templates=list(template_records),
        letter_token_table=[
            LetterTokenEntry(letter="A", token_id=10, variant="bare", token_str="A"),
            LetterTokenEntry(letter="B", token_id=11, variant="bare", token_str="B"),
        ],
        seeds={"root": 0},
        cli_args={"overrides": overrides},
        config_digest="a" * 64,
        config_yaml_text=config_yaml_text,
        chat_template_hash=chat_template_hash,
    )


def _row_payload(
    *,
    run_id: str,
    item: CanonicalItem,
    perm_id: int,
    template_id: str,
    prompt_hash: str,
) -> dict[str, Any]:
    """Build a minimal-but-ResultRow-valid JSONL payload."""
    return ResultRow(
        run_id=run_id,
        item_id=item.item_id,
        dataset_name=item.dataset_name,
        dataset_split=item.dataset_split,
        subject=item.subject,
        template_id=template_id,
        permutation_id=perm_id,
        prompt_hash=prompt_hash,
        rendered_prompt_ref=f"prompts/{prompt_hash}.txt",
        gold_letter="A",
        first_token_letter="A",
        free_text_letter="A",
        free_text_raw="A.",
        agreement_flag=True,
        letter_softmax=[
            LetterSoftmaxEntry(letter="A", token_id=1, prob=0.6, prob_valid=True, logit=0.0),
            LetterSoftmaxEntry(letter="B", token_id=2, prob=0.2, prob_valid=True, logit=0.0),
            LetterSoftmaxEntry(letter="C", token_id=3, prob=0.15, prob_valid=True, logit=0.0),
            LetterSoftmaxEntry(letter="D", token_id=4, prob=0.05, prob_valid=True, logit=0.0),
        ],
        n_options=4,
        free_text_seed=None,
        decode_strategy="greedy",
        created_at=now_utc_iso(),
        extractor_id="regex_ladder",
        extractor_match_rule="my_answer_is",
        first_token_scoring_math="full_vocab_softmax",
        total_letter_mass=1.0,
    ).model_dump(mode="json")


def _write_run_dir(
    tmp_path: Path,
    *,
    items: list[CanonicalItem],
    renderer: PromptRenderer,
    template_ids: tuple[str, ...],
    permutation_ids: tuple[int, ...],
    completion_status: str = "completed",
    chat_template_hash: str | None,
    embed_config: bool = True,
    write_sidecars: bool = False,
    extra_rows: list[dict[str, Any]] | None = None,
    perm_override: dict[int, str] | None = None,
) -> tuple[Path, list[tuple[CanonicalItem, int, str, str, str]]]:
    """Materialize a synthetic Phase 1 run dir on disk.

    Returns ``(run_dir, plan_entries)`` where each plan entry is
    ``(item, perm_id, template_id, prompt, prompt_hash)``.
    """
    run_id = "20260101T000000Z-deadbee-test"
    paths = run_paths(tmp_path, run_id)
    paths.run_dir.mkdir(parents=True, exist_ok=True)
    paths.prompts_dir.mkdir(parents=True, exist_ok=True)

    plan = _expected_prompts(
        items,
        template_ids=template_ids,
        permutation_ids=permutation_ids,
        renderer=renderer,
    )

    rows = []
    for item, pid, tid, _prompt, ph in plan:
        # Per-row override (e.g., inject a wrong hash on one row).
        rows.append(
            _row_payload(
                run_id=run_id,
                item=item,
                perm_id=pid,
                template_id=tid,
                prompt_hash=perm_override.get(pid, ph) if perm_override else ph,
            )
        )
    if extra_rows:
        rows.extend(extra_rows)
    with paths.rows_jsonl.open("wb") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False).encode("utf-8") + b"\n")

    if write_sidecars:
        for _item, _pid, _tid, prompt, ph in plan:
            (paths.prompts_dir / f"{ph}.txt").write_text(prompt, encoding="utf-8")

    template_records = tuple(
        PromptTemplateRecord(
            template_id=rec.template_id,
            template_content_hash=rec.template_content_hash,
            template_text=rec.template_text,
            role=rec.role,
        )
        for rec in renderer.emit_template_records()
    )
    manifest = _mint_manifest(
        run_id=run_id,
        config_yaml_text=_mvp_yaml_text(_repo_root()) if embed_config else None,
        completion_status=completion_status,
        chat_template_hash=chat_template_hash,
        template_records=template_records,
        overrides=_mvp_overrides(tmp_path),
    )
    write_manifest_atomic(paths.manifest_json, manifest)
    return paths.run_dir, plan


def _inputs_for(
    run_dir: Path,
    *,
    items: list[CanonicalItem],
) -> MaterializeInputs:
    """Build a MaterializeInputs directly, bypassing the loader-discovery path."""
    paths = run_paths(run_dir.parent, run_dir.name)
    from nl_ae.schema.reader import load_manifest

    manifest = load_manifest(paths.manifest_json)
    return MaterializeInputs(
        run_dir=paths.run_dir,
        paths=paths,
        run_manifest=manifest,
        rows_path=paths.rows_jsonl,
        item_index={item.item_id: item for item in items},
        permutation_mode="seeded",
        chat_template_hash=manifest.chat_template_hash,
    )


# --- fixtures -----------------------------------------------------------


@pytest.fixture
def repo_root() -> Path:
    return _repo_root()


@pytest.fixture
def renderer(repo_root: Path) -> PromptRenderer:
    return _build_renderer(
        repo_root, chat_adapter=NullChatTemplateAdapter(identity_hash="0" * 64)
    )


# --- tests --------------------------------------------------------------


def test_happy_path(tmp_path: Path, renderer: PromptRenderer) -> None:
    items = _items()
    template_ids = ("mcq_flat_v1", "opinionqa_flat_v1")
    perm_ids = (0, 1)
    run_dir, plan = _write_run_dir(
        tmp_path,
        items=items,
        renderer=renderer,
        template_ids=template_ids,
        permutation_ids=perm_ids,
        chat_template_hash="0" * 64,
    )
    inputs = _inputs_for(run_dir, items=items)
    outcome = materialize_prompts(inputs, renderer=renderer)
    assert outcome.status == "completed"
    assert outcome.rows_seen == len(plan) == 3 * 2 * 2
    assert outcome.sidecars_written == len(plan)
    assert outcome.sidecars_existing == 0
    prompts_dir = inputs.paths.prompts_dir
    for _item, _pid, _tid, prompt, ph in plan:
        sidecar = prompts_dir / f"{ph}.txt"
        assert sidecar.read_text(encoding="utf-8") == prompt


def test_idempotent_skip(tmp_path: Path, renderer: PromptRenderer) -> None:
    items = _items()
    run_dir, plan = _write_run_dir(
        tmp_path,
        items=items,
        renderer=renderer,
        template_ids=("mcq_flat_v1",),
        permutation_ids=(0,),
        chat_template_hash="0" * 64,
    )
    inputs = _inputs_for(run_dir, items=items)
    first = materialize_prompts(inputs, renderer=renderer)
    second = materialize_prompts(inputs, renderer=renderer)
    assert first.sidecars_written == len(plan)
    assert second.sidecars_written == 0
    assert second.sidecars_existing == len(plan)


def test_overwrite(tmp_path: Path, renderer: PromptRenderer) -> None:
    items = _items()[:1]
    run_dir, plan = _write_run_dir(
        tmp_path,
        items=items,
        renderer=renderer,
        template_ids=("mcq_flat_v1",),
        permutation_ids=(0,),
        chat_template_hash="0" * 64,
    )
    _item, _pid, _tid, expected_prompt, ph = plan[0]
    sidecar = run_paths(tmp_path, run_dir.name).prompts_dir / f"{ph}.txt"
    sidecar.write_text("STALE BYTES", encoding="utf-8")
    inputs = _inputs_for(run_dir, items=items)
    outcome = materialize_prompts(inputs, renderer=renderer, on_existing="overwrite")
    assert outcome.sidecars_written == 1
    assert outcome.sidecars_existing == 0
    assert sidecar.read_text(encoding="utf-8") == expected_prompt


def test_on_existing_error(tmp_path: Path, renderer: PromptRenderer) -> None:
    items = _items()
    run_dir, plan = _write_run_dir(
        tmp_path,
        items=items,
        renderer=renderer,
        template_ids=("mcq_flat_v1",),
        permutation_ids=(0,),
        chat_template_hash="0" * 64,
    )
    _item, _pid, _tid, expected_prompt, ph = plan[0]
    sidecar = run_paths(tmp_path, run_dir.name).prompts_dir / f"{ph}.txt"
    sidecar.write_text(expected_prompt, encoding="utf-8")
    inputs = _inputs_for(run_dir, items=items)
    with pytest.raises(SidecarCollisionError):
        materialize_prompts(inputs, renderer=renderer, on_existing="error")
    # The collision raises before any new bytes are written, so the remaining
    # rows in the run stay un-materialized.
    other_hashes = [p[4] for p in plan[1:]]
    for h in other_hashes:
        assert not (sidecar.parent / f"{h}.txt").exists()


def test_prompt_hash_recompute_mismatch(
    tmp_path: Path, renderer: PromptRenderer
) -> None:
    items = _items()[:1]
    # Mint with a deliberately wrong prompt_hash on permutation_id=0.
    wrong_hash = "f" * 64
    run_dir, plan = _write_run_dir(
        tmp_path,
        items=items,
        renderer=renderer,
        template_ids=("mcq_flat_v1",),
        permutation_ids=(0,),
        chat_template_hash="0" * 64,
        perm_override={0: wrong_hash},
    )
    inputs = _inputs_for(run_dir, items=items)
    with pytest.raises(PromptHashRecomputeMismatchError):
        materialize_prompts(inputs, renderer=renderer)
    # No sidecar should be on disk under the wrong hash.
    assert not (inputs.paths.prompts_dir / f"{wrong_hash}.txt").exists()
    # The correct hash from the plan isn't on disk either (the row's hash was
    # what materialize tried to write under, and it failed before write).
    correct_hash = plan[0][4]
    assert not (inputs.paths.prompts_dir / f"{correct_hash}.txt").exists()


def test_manifest_not_completed(tmp_path: Path, renderer: PromptRenderer) -> None:
    items = _items()[:1]
    run_dir, _plan = _write_run_dir(
        tmp_path,
        items=items,
        renderer=renderer,
        template_ids=("mcq_flat_v1",),
        permutation_ids=(0,),
        chat_template_hash="0" * 64,
        completion_status="failed",
    )
    with pytest.raises(ManifestNotCompletedError):
        load_materialize_inputs(run_dir)


def test_template_content_hash_drift() -> None:
    bogus = PromptTemplateRecord(
        template_id="mcq_flat_v1",
        template_content_hash="0" * 64,
        template_text="Q: {question}\n\n{choice_block}\n",
        role="user",
    )
    with pytest.raises(TemplateContentHashMismatchError):
        TemplateRegistry.from_records([bogus])


def test_limit(tmp_path: Path, renderer: PromptRenderer) -> None:
    items = _items()
    run_dir, plan = _write_run_dir(
        tmp_path,
        items=items,
        renderer=renderer,
        template_ids=("mcq_flat_v1",),
        permutation_ids=(0,),
        chat_template_hash="0" * 64,
    )
    inputs = _inputs_for(run_dir, items=items)
    outcome = materialize_prompts(inputs, renderer=renderer, limit=1)
    assert outcome.rows_seen == 1
    assert outcome.sidecars_written == 1
    # Only the first row's sidecar exists; the other items' sidecars should not.
    written = {p[4] for p in plan if (inputs.paths.prompts_dir / f"{p[4]}.txt").exists()}
    assert len(written) == 1


def test_item_not_in_loader(
    tmp_path: Path,
    renderer: PromptRenderer,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    items = _items()
    run_dir, _plan = _write_run_dir(
        tmp_path,
        items=items,
        renderer=renderer,
        template_ids=("mcq_flat_v1",),
        permutation_ids=(0,),
        chat_template_hash="0" * 64,
    )
    # Loader returns only the first item; the others are "missing".
    monkeypatch.setattr(
        materialize_mod,
        "_build_loader",
        lambda ds_name, dataset_cfg, *, hf_cache_dir_override: FakeLoader([items[0]]),
    )
    with pytest.raises(ItemNotInLoaderError):
        load_materialize_inputs(run_dir)


def test_registry_from_records_roundtrip(repo_root: Path) -> None:
    disk_registry = TemplateRegistry(repo_root / "templates")
    disk_registry.load()
    records = [
        PromptTemplateRecord(
            template_id=rec.template_id,
            template_content_hash=rec.template_content_hash,
            template_text=rec.template_text,
            role=rec.role,
        )
        for rec in disk_registry.all_records()
    ]
    rebuilt = TemplateRegistry.from_records(records)
    assert rebuilt.ids() == disk_registry.ids()
    for tid in rebuilt.ids():
        a = disk_registry[tid]
        b = rebuilt[tid]
        assert a.template_text == b.template_text
        assert a.template_content_hash == b.template_content_hash
        assert a.role == b.role
        assert a.slots == b.slots


def test_atomic_write_no_partial(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "x.txt"

    real_write_text = Path.write_text

    def boom(self: Path, *args: Any, **kwargs: Any) -> int:
        if self.suffix == ".tmp":
            raise OSError("disk full")
        return real_write_text(self, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", boom)
    with pytest.raises(OSError):
        _atomic_write(target, "hello")
    assert not target.exists()


# --- CLI smoke + chat-template-hash paths ------------------------------


def _install_fake_transformers(
    monkeypatch: pytest.MonkeyPatch, tokenizer: FakeTokenizer
) -> None:
    fake = types.ModuleType("transformers")

    class _FakeAutoTokenizer:
        @staticmethod
        def from_pretrained(*_args: Any, **_kwargs: Any) -> FakeTokenizer:
            return tokenizer

    fake.AutoTokenizer = _FakeAutoTokenizer  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "transformers", fake)


def _patch_loader(
    monkeypatch: pytest.MonkeyPatch, items: list[CanonicalItem]
) -> None:
    monkeypatch.setattr(
        materialize_mod,
        "_build_loader",
        lambda ds_name, dataset_cfg, *, hf_cache_dir_override: FakeLoader(items),
    )


def _live_chat_hash(template_text: str) -> str:
    return sha256_hex_bytes(nfc(template_text).encode("utf-8"))


def test_cli_smoke(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    items = _items()
    # Build a renderer that uses the same chat adapter the CLI will reconstruct.
    fake_tok = FakeTokenizer()
    chat_hash = _live_chat_hash(fake_tok.chat_template)
    renderer_with_chat = _build_renderer(
        _repo_root(),
        chat_adapter=make_chat_adapter(fake_tok, identity_hash=chat_hash),
    )

    run_dir, plan = _write_run_dir(
        tmp_path,
        items=items,
        renderer=renderer_with_chat,
        template_ids=("mcq_flat_v1",),
        permutation_ids=(0,),
        chat_template_hash=chat_hash,
    )

    _install_fake_transformers(monkeypatch, fake_tok)
    _patch_loader(monkeypatch, items)

    from nl_ae.cli.commands.materialize_prompts_cmd import materialize_prompts_cmd

    runner = CliRunner()
    result = runner.invoke(
        materialize_prompts_cmd,
        ["--run-dir", str(run_dir)],
        catch_exceptions=False,
    )
    assert result.exit_code == 0, result.output
    assert "status=completed" in result.output
    paths = run_paths(tmp_path, run_dir.name)
    for _item, _pid, _tid, prompt, ph in plan:
        assert (paths.prompts_dir / f"{ph}.txt").read_text(encoding="utf-8") == prompt


def test_cli_chat_template_hash_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    items = _items()[:1]
    # Build the run with chat_template_hash = hash("CORRECT"); the CLI will see
    # a tokenizer reporting hash of "OTHER".
    correct_tok = FakeTokenizer(chat_template="CORRECT\n")
    run_chat_hash = _live_chat_hash(correct_tok.chat_template)
    renderer = _build_renderer(
        _repo_root(),
        chat_adapter=make_chat_adapter(correct_tok, identity_hash=run_chat_hash),
    )
    run_dir, _plan = _write_run_dir(
        tmp_path,
        items=items,
        renderer=renderer,
        template_ids=("mcq_flat_v1",),
        permutation_ids=(0,),
        chat_template_hash=run_chat_hash,
    )
    drifted_tok = FakeTokenizer(chat_template="DIFFERENT\n")
    _install_fake_transformers(monkeypatch, drifted_tok)
    _patch_loader(monkeypatch, items)

    from nl_ae.cli.commands.materialize_prompts_cmd import materialize_prompts_cmd

    runner = CliRunner()
    result = runner.invoke(
        materialize_prompts_cmd,
        ["--run-dir", str(run_dir)],
        catch_exceptions=False,
    )
    assert result.exit_code == 4
    assert "chat_template_hash mismatch" in result.output


def test_cli_chat_template_hash_null_warn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    items = _items()[:1]
    fake_tok = FakeTokenizer()
    # The renderer used to mint the run still uses the live hash so per-row
    # hashes line up; the manifest just doesn't record chat_template_hash.
    chat_hash = _live_chat_hash(fake_tok.chat_template)
    renderer = _build_renderer(
        _repo_root(),
        chat_adapter=make_chat_adapter(fake_tok, identity_hash=chat_hash),
    )
    run_dir, _plan = _write_run_dir(
        tmp_path,
        items=items,
        renderer=renderer,
        template_ids=("mcq_flat_v1",),
        permutation_ids=(0,),
        chat_template_hash=None,
    )
    _install_fake_transformers(monkeypatch, fake_tok)
    _patch_loader(monkeypatch, items)

    from nl_ae.cli.commands.materialize_prompts_cmd import materialize_prompts_cmd

    runner = CliRunner()
    result = runner.invoke(
        materialize_prompts_cmd,
        ["--run-dir", str(run_dir)],
        catch_exceptions=False,
    )
    assert result.exit_code == 0, result.output
    assert "warning: manifest has no chat_template_hash" in result.output
    assert "status=completed" in result.output
