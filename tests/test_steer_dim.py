"""CPU unit tests for ome_gauge.steer_dim — the additive operator + its exact gates on synthetic
data (hermetic), plus a real-data integration that generates a tiny subset and checks files/schema.

Run: python tests/test_steer_dim.py   (from the NLA-final root)
"""
import os
import sys

_SRC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _SRC)

import numpy as np

from ome_gauge import steer_dim as S


def test_additive_identity_and_invariant():
    rng = np.random.default_rng(0)
    h = rng.standard_normal((32, 16)).astype(np.float64)
    v = rng.standard_normal(16); v /= np.linalg.norm(v)
    # alpha=0 identity
    assert np.array_equal(S.steer_additive(h, v, 0.0), h)
    # ||dh|| == alpha exactly (same unit vector added to every row)
    for a in (0.5, 1.0, 7.0, 30.0):
        hp = S.steer_additive(h, v, a)
        dh = np.linalg.norm(hp - h, axis=1)
        assert np.allclose(dh, a, atol=1e-9), f"||dh|| != alpha at {a}"
        # ratio = alpha/||h|| ; cos decreases from 1
        diag = S.norm_diagnostics(h, hp)
        assert np.allclose(diag["ratio"], a / np.linalg.norm(h, axis=1), atol=1e-9)
        assert (diag["cos_h_hp"] <= 1.0 + 1e-9).all() and (diag["cos_h_hp"] < 1.0).all()
    print("[steer] additive identity + ||dh||==alpha invariant + ratio/cos OK")


def test_gate_catches_bad_alpha():
    h = np.ones((4, 8)); v = np.zeros(8); v[0] = 1.0
    # a forged 'additive' diag whose dh_norm != alpha must trip the gate
    bad = {"dh_norm": np.array([2.0, 2.0, 2.0, 2.0])}
    try:
        S._gate("dim", 1.0, bad); assert False, "gate should reject ||dh|| != alpha"
    except AssertionError as e:
        assert "additive gate" in str(e)
    # alpha=0 with nonzero dh must trip too
    try:
        S._gate("dim", 0.0, {"dh_norm": np.array([0.1])}); assert False
    except AssertionError:
        pass
    print("[steer] P1 gate rejects bad alpha / non-identity OK")


def test_monotone_ratio_with_alpha():
    rng = np.random.default_rng(1)
    h = rng.standard_normal((64, 32)).astype(np.float64) * 3.0
    v = rng.standard_normal(32); v /= np.linalg.norm(v)
    means = [S.norm_diagnostics(h, S.steer_additive(h, v, a))["ratio"].mean()
             for a in [0.0, 1.0, 5.0, 10.0, 30.0]]
    assert all(b > a for a, b in zip(means, means[1:])), f"mean ratio not monotone: {means}"
    print("[steer] mean ratio monotone in alpha OK")


def test_real_gen_tiny():
    """Integration: generate dim:D_correct on 8 real rows over a few alphas; check gates+files."""
    from ome_gauge import config as C
    if not (C.PATHS.a0_npy().exists() and C.PATHS.dirs_npz().exists()):
        print("[steer] (skip real gen — a0/dirs cache absent; run `directions build` first)"); return
    rows, ids = C.load_paired_subset()
    rows8, ids8 = rows[:8], ids[:8]
    norm_rows, arrays = S.gen("dim", "D_correct", [0.0, 2.0, 10.0], rows=rows8, example_ids=ids8)
    assert len(arrays) == 3 and len(norm_rows) == 24
    assert set(norm_rows[0]) == set(S.NORMS_COLS)
    # alpha=0 array exists and is identity vs base; alpha=10 ||dh||==10
    a0 = [r for r in norm_rows if r["alpha"] == 0.0]
    a10 = [r for r in norm_rows if r["alpha"] == 10.0]
    assert all(r["dh_norm"] == 0.0 for r in a0)
    assert all(abs(r["dh_norm"] - 10.0) < 1e-2 for r in a10)
    for am in arrays:
        assert (C.PATHS.dir_steer() / am["path"]).exists() and len(am["sha256"]) == 64
    print(f"[steer] real gen tiny OK (a10 mean ratio {arrays[-1]['mean_ratio']:.3f})")


def main() -> int:
    test_additive_identity_and_invariant()
    test_gate_catches_bad_alpha()
    test_monotone_ratio_with_alpha()
    test_real_gen_tiny()
    print("\nSTEER_DIM CPU TESTS PASSED.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
