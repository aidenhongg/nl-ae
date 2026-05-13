"""Determinism harness.

Cryptographic-strength child seed derivation (SHA-256, never Python ``hash()``)
per UR-R2.1 and C05 D5.9. Idempotent within a process: re-applying the same
``SeedConfig`` returns the cached ``AppliedSeeds``; a different config raises.
"""

from __future__ import annotations

import hashlib
import os
import random
from typing import Annotated, Final, Literal

from pydantic import BaseModel, ConfigDict, Field

# Domain tags are part of the on-disk-stable derivation; never change.
DOMAIN_NUMPY: Final[bytes] = b"numpy"
DOMAIN_TORCH: Final[bytes] = b"torch"
DOMAIN_PYTHON: Final[bytes] = b"python"
DOMAIN_CUDA: Final[bytes] = b"cuda"
DOMAIN_FREE_GEN: Final[bytes] = b"free_gen"
DOMAIN_PYTHONHASH: Final[bytes] = b"pythonhash"

_SEED_MASK: Final[int] = (1 << 32) - 1


class SeedReapplicationError(RuntimeError):
    """Raised when ``apply_seeds`` is called twice with different configs."""


class AppliedSeeds(BaseModel):
    """Immutable record of what ``apply_seeds`` actually did."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    root: Annotated[int, Field(ge=0, le=_SEED_MASK)]
    numpy: Annotated[int, Field(ge=0, le=_SEED_MASK)]
    torch: Annotated[int, Field(ge=0, le=_SEED_MASK)]
    python: Annotated[int, Field(ge=0, le=_SEED_MASK)]
    cuda: Annotated[int, Field(ge=0, le=_SEED_MASK)]
    free_gen: Annotated[int, Field(ge=0, le=_SEED_MASK)]
    pythonhashseed: Annotated[int, Field(ge=0, le=_SEED_MASK)]
    pythonhashseed_was_external: bool
    deterministic_algorithms_requested: Literal["off", "warn_only", "strict"]
    deterministic_algorithms_actual: Literal["off", "warn_only", "strict", "unavailable"]
    cudnn_deterministic: bool
    cudnn_benchmark: bool
    torch_available: bool
    numpy_available: bool
    cuda_available: bool
    notes: tuple[str, ...] = ()


def derive_child_seed(root: int, tag: bytes) -> int:
    """SHA-256-backed child seed derivation.

    Never uses Python's salted ``hash()`` — that would yield different seeds
    across processes (UR-R2.1).
    """
    if root < 0 or root > _SEED_MASK:
        raise ValueError(f"root seed out of range [0, 2^32-1]: {root}")
    payload = root.to_bytes(8, "big") + b"|" + tag
    digest = hashlib.sha256(payload).digest()
    # First 4 bytes → unsigned 32-bit int.
    return int.from_bytes(digest[:4], "big") & _SEED_MASK


# Process-level cache for idempotency.
_APPLIED: AppliedSeeds | None = None
_APPLIED_KEY: tuple | None = None


def _config_key(cfg: object) -> tuple:
    # ``cfg`` is duck-typed against ``nl_ae.config.schema.SeedConfig`` to avoid
    # a circular import.
    return tuple(
        getattr(cfg, name)
        for name in (
            "root",
            "numpy",
            "torch",
            "python",
            "cuda",
            "free_gen",
            "pythonhashseed",
            "deterministic_algorithms",
            "cudnn_deterministic",
            "cudnn_benchmark",
        )
    )


def apply_seeds(
    cfg: object,
    *,
    target_device: Literal["cuda:0", "cpu", "auto"] = "auto",
) -> AppliedSeeds:
    """Apply seeds to every PRNG we can reach; return the actually-applied state.

    Idempotent within a process. Re-calling with the same ``cfg`` is a no-op;
    re-calling with a different ``cfg`` raises ``SeedReapplicationError``.
    """
    global _APPLIED, _APPLIED_KEY

    key = _config_key(cfg)
    if _APPLIED is not None:
        if _APPLIED_KEY == key:
            return _APPLIED
        raise SeedReapplicationError(
            "apply_seeds called twice in the same process with different configs"
        )

    root = int(getattr(cfg, "root"))
    notes: list[str] = []

    def _resolve(domain: bytes, override: int | None) -> int:
        if override is not None:
            return int(override) & _SEED_MASK
        return derive_child_seed(root, domain)

    numpy_seed = _resolve(DOMAIN_NUMPY, getattr(cfg, "numpy"))
    torch_seed = _resolve(DOMAIN_TORCH, getattr(cfg, "torch"))
    python_seed = _resolve(DOMAIN_PYTHON, getattr(cfg, "python"))
    cuda_seed = _resolve(DOMAIN_CUDA, getattr(cfg, "cuda"))
    free_gen_seed = _resolve(DOMAIN_FREE_GEN, getattr(cfg, "free_gen"))
    pythonhash_seed = _resolve(DOMAIN_PYTHONHASH, getattr(cfg, "pythonhashseed"))

    # PYTHONHASHSEED can only be honored if set BEFORE python starts; record state.
    external = os.environ.get("PYTHONHASHSEED")
    pythonhashseed_was_external = external is not None and external != "random"
    if not pythonhashseed_was_external:
        notes.append(
            "PYTHONHASHSEED set in-process only; for cross-process repro set it "
            "via the env before Python launches."
        )
    os.environ["PYTHONHASHSEED"] = str(pythonhash_seed)

    # stdlib random.
    random.seed(python_seed)

    # numpy (optional).
    numpy_available = False
    try:
        import numpy as np  # noqa: PLC0415

        np.random.seed(numpy_seed)
        numpy_available = True
    except ImportError:
        notes.append("numpy not installed; numpy seed not applied.")

    # torch (optional).
    torch_available = False
    cuda_available = False
    determinism_requested = getattr(cfg, "deterministic_algorithms")
    determinism_actual: Literal["off", "warn_only", "strict", "unavailable"] = "unavailable"
    cudnn_deterministic = bool(getattr(cfg, "cudnn_deterministic"))
    cudnn_benchmark = bool(getattr(cfg, "cudnn_benchmark"))
    try:
        import torch  # noqa: PLC0415

        torch_available = True
        torch.manual_seed(torch_seed)
        cuda_available = bool(torch.cuda.is_available())
        if cuda_available and target_device != "cpu":
            torch.cuda.manual_seed_all(cuda_seed)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = cudnn_deterministic
            torch.backends.cudnn.benchmark = cudnn_benchmark
        if determinism_requested == "off":
            determinism_actual = "off"
            torch.use_deterministic_algorithms(False)
        else:
            warn_only = determinism_requested == "warn_only"
            try:
                torch.use_deterministic_algorithms(True, warn_only=warn_only)
                determinism_actual = "warn_only" if warn_only else "strict"
            except (RuntimeError, TypeError) as exc:
                determinism_actual = "warn_only"
                notes.append(
                    f"torch.use_deterministic_algorithms fell back to warn_only: {exc!r}"
                )
                try:
                    torch.use_deterministic_algorithms(True, warn_only=True)
                except (RuntimeError, TypeError) as exc2:  # pragma: no cover
                    determinism_actual = "off"
                    notes.append(
                        f"torch.use_deterministic_algorithms unavailable: {exc2!r}"
                    )
                    torch.use_deterministic_algorithms(False)
    except ImportError:
        notes.append("torch not installed; torch/cuda seeds not applied.")

    applied = AppliedSeeds(
        root=root,
        numpy=numpy_seed,
        torch=torch_seed,
        python=python_seed,
        cuda=cuda_seed,
        free_gen=free_gen_seed,
        pythonhashseed=pythonhash_seed,
        pythonhashseed_was_external=pythonhashseed_was_external,
        deterministic_algorithms_requested=determinism_requested,
        deterministic_algorithms_actual=determinism_actual,
        cudnn_deterministic=cudnn_deterministic,
        cudnn_benchmark=cudnn_benchmark,
        torch_available=torch_available,
        numpy_available=numpy_available,
        cuda_available=cuda_available,
        notes=tuple(notes),
    )
    _APPLIED = applied
    _APPLIED_KEY = key
    return applied


def reset_for_tests() -> None:
    """Test-only escape hatch; not exposed in __all__."""
    global _APPLIED, _APPLIED_KEY
    _APPLIED = None
    _APPLIED_KEY = None


__all__ = [
    "AppliedSeeds",
    "SeedReapplicationError",
    "apply_seeds",
    "derive_child_seed",
]
