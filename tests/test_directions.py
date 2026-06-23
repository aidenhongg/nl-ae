"""CPU unit tests for ome_gauge.directions — the pure builders/audits on synthetic data
(fast, hermetic) plus a real-data integration check of the P0 gate when the a0 cache is present.

Run: python tests/test_directions.py   (from the NLA-final root)
"""
import os
import sys

# .../NLA-final/tests/test_directions.py -> add .../the repo root to path
_SRC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _SRC)

import numpy as np

from ome_gauge import directions as D


def test_unit_and_mean_difference():
    v, raw = D._unit(np.array([3.0, 4.0]))
    assert abs(raw - 5.0) < 1e-12 and abs(np.linalg.norm(v) - 1.0) < 1e-12
    try:
        D._unit(np.zeros(4)); assert False, "zero vector must raise"
    except ValueError:
        pass
    h = np.array([[1.0, 0.0], [3.0, 0.0], [0.0, 0.0], [0.0, 2.0]])
    pos = np.array([True, True, False, False]); neg = ~pos
    md = D.mean_difference(h, pos, neg)
    assert np.allclose(md, [2.0, -1.0]), md   # mean(pos)=(2,0), mean(neg)=(0,1)
    print("[dir] _unit + mean_difference OK")


def test_auc_values():
    assert abs(D.auc([1, 2, 3, 4], [False, False, True, True]) - 1.0) < 1e-12
    assert abs(D.auc([1, 2, 3, 4], [True, True, False, False]) - 0.0) < 1e-12
    assert abs(D.auc([1, 2, 3, 4], [True, False, True, False]) - 0.25) < 1e-12
    assert abs(D.auc([1, 1, 2, 2], [False, True, False, True]) - 0.5) < 1e-12   # ties -> 0.5
    assert np.isnan(D.auc([1, 2, 3], [True, True, True]))                       # one class
    print("[dir] auc (incl. ties + degenerate) OK")


def test_random_determinism_and_orthogonality():
    v0a, p = D.build_d_random(64, seed=7)
    v0b, _ = D.build_d_random(64, seed=7)
    v1, _ = D.build_d_random(64, seed=8)
    assert np.array_equal(v0a, v0b), "same seed must reproduce"
    assert not np.array_equal(v0a, v1), "different seed must differ"
    assert abs(np.linalg.norm(v0a) - 1.0) < 1e-12 and p["seed"] == 7
    ortho = D.audit_orthogonality({"a": v0a, "b": v1})
    assert set(ortho) == {"a|b"} and abs(ortho["a|b"]) < 0.5   # random hi-d ~orthogonal
    print("[dir] random determinism + orthogonality audit OK")


def test_build_d_correct_synthetic():
    """Correct rows shifted along +e0, incorrect along -e0 -> D_correct ~ e0, AUC ~ 1."""
    rng = np.random.default_rng(0)
    n, d = 400, 16
    h = rng.standard_normal((n, d)) * 0.1
    correct = np.zeros(n, bool); correct[: n // 2] = True
    h[correct, 0] += 5.0; h[~correct, 0] -= 5.0
    fit = np.ones(n, bool)
    v, prov = D.build_d_correct(h, fit, correct)
    assert abs(np.linalg.norm(v) - 1.0) < 1e-9
    assert abs(v[0]) > 0.98, f"D_correct should be ~e0, got v[0]={v[0]:.3f}"
    assert prov["n_pos"] == 200 and prov["n_neg"] == 200
    assert D.auc(h @ v, correct) > 0.99
    print("[dir] build_d_correct on synthetic separable data OK")


def test_real_p0_gate():
    """Integration: build the real Stage-1 directions and assert the P0 gate (needs a0 cache)."""
    from ome_gauge import config as C
    if not C.PATHS.a0_npy().exists():
        print("[dir] (skip real P0 — a0 cache absent)"); return
    dirs, m = D.build_all()
    assert set(dirs) == set(C.CONFIG["directions"]["stage1"])
    for name, v in dirs.items():
        assert abs(np.linalg.norm(v) - 1.0) < D.UNIT_TOL, name
    s = m["sanity"]
    assert s["d_correct_sep_auc_test"] > 0.60, s
    assert s["d_random_sep_auc_test_mean"] < 0.60, s          # control near chance
    assert s["d_correct_vs_know_cos"] > 0.20, s
    assert m["config_hash"] == C.CONFIG_HASH
    print(f"[dir] real P0 gate OK (AUC={s['d_correct_sep_auc_test']:.3f}, "
          f"know_cos={s['d_correct_vs_know_cos']:+.3f})")


def main() -> int:
    test_unit_and_mean_difference()
    test_auc_values()
    test_random_determinism_and_orthogonality()
    test_build_d_correct_synthetic()
    test_real_p0_gate()
    print("\nDIRECTIONS CPU TESTS PASSED.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
