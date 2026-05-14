"""``_select_block_outputs`` — "layer N = block N output" indexing.

HF's ``output_hidden_states=True`` returns a ``(num_hidden_layers + 1)``-tuple
where ``hidden_states[0]`` is post-embed and ``hidden_states[i]`` (``i ≥ 1``) is
block ``i - 1``'s output. The wrapper now indexes with ``hidden_states[N + 1]``
so callers using ``record_layers=(0..27)`` get all 28 transformer-block outputs.
This unit-test pins that contract without needing torch or the real model.
"""

from __future__ import annotations

from nl_ae.inference.wrapper import _select_block_outputs


class _FakeTensor:
    """Duck-typed stand-in for ``torch.Tensor`` along the slice + detach paths."""

    def __init__(self, payload: int) -> None:
        self.payload = payload

    def __getitem__(self, key: object) -> _FakeTensor:
        # In the wrapper we call ``hs[0, -1, :]`` — just return self.
        return self

    def detach(self) -> _FakeTensor:
        return self


def test_select_block_outputs_uses_block_output_indexing() -> None:
    # 28 blocks → 29 entries (post-embed + 28 block outputs).
    hs = tuple(_FakeTensor(i) for i in range(29))
    out = _select_block_outputs(hs, layers=[0, 20, 27], n_layers=28)
    # layer N → hs[N+1] → payload N+1.
    assert {k: v.payload for k, v in out.items()} == {0: 1, 20: 21, 27: 28}


def test_select_block_outputs_skips_out_of_range_layers() -> None:
    hs = tuple(_FakeTensor(i) for i in range(29))
    # layer=28 is out of range (only 0..27 valid).
    out = _select_block_outputs(hs, layers=[-1, 28, 100], n_layers=28)
    assert out == {}


def test_select_block_outputs_tolerates_truncated_hidden_states() -> None:
    """If HF returns fewer hidden_states than expected, callers don't crash."""
    hs = tuple(_FakeTensor(i) for i in range(5))  # 4 blocks worth
    out = _select_block_outputs(hs, layers=[0, 3, 10], n_layers=28)
    # layer 0 → hs[1] OK; layer 3 → hs[4] OK; layer 10 → hs[11] missing → dropped.
    assert {k: v.payload for k, v in out.items()} == {0: 1, 3: 4}
