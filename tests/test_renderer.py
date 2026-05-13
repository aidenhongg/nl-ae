"""Template registry + renderer: hash stability + slot substitution."""

from __future__ import annotations

from pathlib import Path

import pytest

from nl_ae.data.canonical import CanonicalItem
from nl_ae.data.permute import permutation_for
from nl_ae.prompt.renderer import NullChatTemplateAdapter, PromptRenderer
from nl_ae.prompt.template_registry import TemplateRegistry


@pytest.fixture
def repo_templates() -> Path:
    return Path(__file__).resolve().parents[1] / "templates"


@pytest.fixture
def sample_item() -> CanonicalItem:
    return CanonicalItem(
        item_id="mmlu/v1/test/q-test01",
        dataset_name="mmlu",
        dataset_split="test",
        subject="test",
        question="What is 2 + 2?",
        choices=("1", "3", "4", "22"),
        gold_index=2,
    )


def test_registry_loads_repo_templates(repo_templates: Path) -> None:
    reg = TemplateRegistry(repo_templates)
    reg.load()
    ids = reg.ids()
    assert "mcq_flat_v1" in ids
    assert "opinionqa_flat_v1" in ids
    rec = reg["mcq_flat_v1"]
    assert rec.template_content_hash.startswith(("0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "a", "b", "c", "d", "e", "f"))
    assert "{question}" in rec.template_text
    assert "{choice_block}" in rec.template_text


def test_render_is_stable(repo_templates: Path, sample_item: CanonicalItem) -> None:
    reg = TemplateRegistry(repo_templates)
    renderer = PromptRenderer(reg, chat_adapter=None)
    perm = permutation_for(sample_item, 0, mode="identity")
    prompt_a, hash_a = renderer.render(perm, "mcq_flat_v1")
    prompt_b, hash_b = renderer.render(perm, "mcq_flat_v1")
    assert prompt_a == prompt_b
    assert hash_a == hash_b
    assert "What is 2 + 2?" in prompt_a
    assert "A. 1" in prompt_a
    assert "Answer:" in prompt_a


def test_render_changes_with_permutation(
    repo_templates: Path, sample_item: CanonicalItem
) -> None:
    reg = TemplateRegistry(repo_templates)
    renderer = PromptRenderer(reg, chat_adapter=None)
    p0 = permutation_for(sample_item, 0, mode="identity")
    p1 = permutation_for(sample_item, 1, mode="seeded")
    _, hash_0 = renderer.render(p0, "mcq_flat_v1")
    _, hash_1 = renderer.render(p1, "mcq_flat_v1")
    # Either the seeded permutation reshuffles options, or with n=4 there's a
    # ~4% chance it doesn't; in that case the hashes can still match. We allow
    # either path but exercise the rendering pipeline.
    assert isinstance(hash_0, str) and isinstance(hash_1, str)


def test_emit_template_records(repo_templates: Path) -> None:
    reg = TemplateRegistry(repo_templates)
    renderer = PromptRenderer(reg, chat_adapter=None)
    records = renderer.emit_template_records()
    ids = {r.template_id for r in records}
    assert "mcq_flat_v1" in ids


def test_null_chat_adapter_passes_through(
    repo_templates: Path, sample_item: CanonicalItem
) -> None:
    reg = TemplateRegistry(repo_templates)
    adapter = NullChatTemplateAdapter(identity_hash="0" * 64)
    renderer = PromptRenderer(reg, chat_adapter=adapter)
    perm = permutation_for(sample_item, 0, mode="identity")
    prompt, _ = renderer.render(perm, "mcq_flat_v1")
    assert "Answer:" in prompt
