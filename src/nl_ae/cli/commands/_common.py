"""Shared CLI helpers (parsing argument forms used by more than one subcommand)."""

from __future__ import annotations

import click


def parse_layers_arg(layers_arg: str | None, *, n_layers: int) -> tuple[int, ...]:
    """Accept ``"0,1,2"``, ``"0-27"``, mixed, or ``None`` (= all layers).

    Raises :class:`click.UsageError` on syntactic or range errors so the CLI
    surfaces them with click's normal error formatting.
    """
    if layers_arg is None or layers_arg.strip() == "":
        return tuple(range(n_layers))
    out: set[int] = set()
    for part in layers_arg.split(","):
        token = part.strip()
        if not token:
            continue
        if "-" in token:
            lo_s, hi_s = token.split("-", 1)
            try:
                lo, hi = int(lo_s), int(hi_s)
            except ValueError as exc:
                raise click.UsageError(f"--layers range {token!r} is not integers") from exc
            if lo > hi:
                raise click.UsageError(f"--layers range {token!r} has lo > hi")
            out.update(range(lo, hi + 1))
        else:
            try:
                out.add(int(token))
            except ValueError as exc:
                raise click.UsageError(f"--layers token {token!r} is not an integer") from exc
    if not out:
        raise click.UsageError("--layers parsed to an empty set")
    if min(out) < 0 or max(out) >= n_layers:
        raise click.UsageError(
            f"--layers contains values out of [0, {n_layers}): {sorted(out)}"
        )
    return tuple(sorted(out))


__all__ = ["parse_layers_arg"]
