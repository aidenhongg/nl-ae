"""Run identity + environment fingerprint."""

from __future__ import annotations

import platform
import subprocess
import sys
from pathlib import Path

from nl_ae.schema.hashing import make_run_id, now_utc_iso
from nl_ae.schema.models import EnvFingerprint


def git_sha_and_dirty(repo_dir: Path | None = None) -> tuple[str, bool]:
    """Return ``(short_sha, dirty)``. If git is unreachable, returns ``("nogit00", False)``."""
    cwd = str(repo_dir) if repo_dir is not None else None
    try:
        sha = (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=cwd, stderr=subprocess.DEVNULL
            )
            .decode("ascii")
            .strip()
        )
        status = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=cwd, stderr=subprocess.DEVNULL
        ).decode("ascii")
        return sha, bool(status.strip())
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return "nogit00", False


def mint_run_id(
    *,
    started_at: str | None = None,
    git_sha: str | None = None,
    slug: str | None = None,
    repo_dir: Path | None = None,
) -> tuple[str, str, str, bool]:
    """Return ``(run_id, started_at, git_sha, git_dirty)``."""
    started = started_at or now_utc_iso()
    if git_sha is None:
        sha, dirty = git_sha_and_dirty(repo_dir)
    else:
        sha, dirty = git_sha, False
    return make_run_id(started, sha, slug), started, sha, dirty


def _try_version(module_name: str) -> str | None:
    try:
        mod = __import__(module_name)
    except ImportError:
        return None
    version = getattr(mod, "__version__", None)
    return str(version) if version else None


def gather_environment_fingerprint() -> EnvFingerprint:
    torch_version = _try_version("torch")
    cuda_version: str | None = None
    gpu_name: str | None = None
    gpu_vram_mb: int | None = None
    if torch_version is not None:
        try:
            import torch  # noqa: PLC0415

            cuda_version = torch.version.cuda  # type: ignore[attr-defined]
            if torch.cuda.is_available() and torch.cuda.device_count() > 0:
                gpu_name = torch.cuda.get_device_name(0)
                gpu_vram_mb = int(torch.cuda.get_device_properties(0).total_memory // (1024 * 1024))
        except Exception:  # pragma: no cover
            pass
    return EnvFingerprint(
        os_name=platform.system(),
        os_version=platform.version(),
        python_version=f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        cuda_version=cuda_version,
        torch_version=torch_version,
        transformers_version=_try_version("transformers"),
        bitsandbytes_version=_try_version("bitsandbytes"),
        accelerate_version=_try_version("accelerate"),
        gpu_name=gpu_name,
        gpu_vram_mb=gpu_vram_mb,
    )


__all__ = ["gather_environment_fingerprint", "git_sha_and_dirty", "mint_run_id"]
