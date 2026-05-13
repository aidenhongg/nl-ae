"""Config loader + digest stability."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from nl_ae.config.digest import (
    CONFIG_DIGEST_EXCLUSIONS,
    canonical_dump_bytes,
    compute_config_digest,
)
from nl_ae.config.errors import ConfigOverrideError, EnvInterpolationError
from nl_ae.config.loader import (
    apply_overrides,
    interpolate_env,
    load_config,
    parse_override,
)


@pytest.fixture
def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def test_parse_override_simple() -> None:
    path, value = parse_override("eval.plan.permutations_per_item=4")
    assert path == ("eval", "plan", "permutations_per_item")
    assert value == 4


def test_parse_override_yaml_list() -> None:
    path, value = parse_override("dataset.mmlu_subjects=[abstract_algebra, anatomy]")
    assert path == ("dataset", "mmlu_subjects")
    assert value == ["abstract_algebra", "anatomy"]


def test_apply_overrides_deep() -> None:
    base = {"a": {"b": {"c": 1}}}
    out = apply_overrides(base, ("a.b.c=2", "a.b.d=true"))
    assert out == {"a": {"b": {"c": 2, "d": True}}}
    # original is not mutated
    assert base == {"a": {"b": {"c": 1}}}


def test_apply_overrides_invalid() -> None:
    with pytest.raises(ConfigOverrideError):
        parse_override("missing-equals")


def test_interpolate_env_default() -> None:
    env = {"HOME": "/h"}
    obj = {"x": "${ENV:HOME}", "y": "${ENV:MISSING:fallback}"}
    out = interpolate_env(obj, env=env)
    assert out == {"x": "/h", "y": "fallback"}


def test_interpolate_env_undefined_raises() -> None:
    with pytest.raises(EnvInterpolationError):
        interpolate_env("${ENV:NEVER_SET_XYZ}", env={})


def test_load_example_config(repo_root: Path, tmp_path: Path) -> None:
    cfg = load_config(
        repo_root / "examples" / "mvp.yaml",
        overrides=(
            f"output.output_dir={tmp_path.as_posix()}",
            f"dataset.cache_dir={tmp_path.as_posix()}",
            "dataset.offline=true",
        ),
    )
    assert cfg.config_schema_version == "1.0.0"
    assert "mcq_flat_v1" in cfg.eval.plan.template_ids
    assert cfg.eval.plan.permutations_per_item == 10
    assert cfg.model.quantization.kind == "fp16"


def test_digest_is_stable_across_ordering(repo_root: Path, tmp_path: Path) -> None:
    cfg = load_config(
        repo_root / "examples" / "mvp.yaml",
        overrides=(
            f"output.output_dir={tmp_path.as_posix()}",
            f"dataset.cache_dir={tmp_path.as_posix()}",
            "dataset.offline=true",
        ),
    )
    bytes_a = canonical_dump_bytes(cfg)
    bytes_b = canonical_dump_bytes(cfg)
    assert bytes_a == bytes_b
    assert compute_config_digest(cfg) == compute_config_digest(cfg)


def test_digest_excludes_ephemeral_fields(repo_root: Path, tmp_path: Path) -> None:
    cfg_a = load_config(
        repo_root / "examples" / "mvp.yaml",
        overrides=(
            f"output.output_dir={tmp_path.as_posix()}",
            f"dataset.cache_dir={tmp_path.as_posix()}",
            "dataset.offline=true",
            "run_identity.notes='a fresh note'",
        ),
    )
    cfg_b = load_config(
        repo_root / "examples" / "mvp.yaml",
        overrides=(
            f"output.output_dir={tmp_path.as_posix()}",
            f"dataset.cache_dir={tmp_path.as_posix()}",
            "dataset.offline=true",
            "run_identity.notes='different note'",
        ),
    )
    assert compute_config_digest(cfg_a) == compute_config_digest(cfg_b)


def test_digest_changes_with_load_bearing_fields(
    repo_root: Path, tmp_path: Path
) -> None:
    cfg_a = load_config(
        repo_root / "examples" / "mvp.yaml",
        overrides=(
            f"output.output_dir={tmp_path.as_posix()}",
            f"dataset.cache_dir={tmp_path.as_posix()}",
            "dataset.offline=true",
        ),
    )
    cfg_b = load_config(
        repo_root / "examples" / "mvp.yaml",
        overrides=(
            f"output.output_dir={tmp_path.as_posix()}",
            f"dataset.cache_dir={tmp_path.as_posix()}",
            "dataset.offline=true",
            "eval.plan.permutations_per_item=4",
        ),
    )
    assert compute_config_digest(cfg_a) != compute_config_digest(cfg_b)


def test_digest_exclusion_set_documented() -> None:
    # Smoke check that the exclusion set actually contains every truly
    # ephemeral field we care about. If you change this, also update the
    # docstring in `digest.py`.
    payload = json.dumps(sorted(CONFIG_DIGEST_EXCLUSIONS))
    assert "run_identity.run_id" in payload
    assert "logging.console_level" in payload
