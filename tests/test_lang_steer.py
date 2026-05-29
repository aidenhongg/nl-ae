"""CPU unit tests for src/lang_steer.py — operators on the REAL cached verbalizations
(idempotency, letter-correctness, no-clobber), the magnitude conventions, the off-manifold
ratio, the KAPPA-frontier parse, and the WIN/PARTIAL/NULL verdict. No GPU, no model.

Run: python tests/test_lang_steer.py
"""
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pyarrow.parquet as pq

from src import lang_steer as L

SYM = "ABCD"


def _test_rows(n=400):
    t = pq.read_table("out/nl/orig.parquet",
                      columns=["split_el", "nl_text", L.TARGET_COL]).to_pydict()
    rows = [(tx, k) for tx, s, k in zip(t["nl_text"], t["split_el"], t[L.TARGET_COL]) if s == "test"]
    return rows[:n]


def test_operators_on_cached_text():
    rows = _test_rows()
    iso = L._ISO_LETTER
    e1_fired = e2_changed = e3_fired = 0
    for tx, X in rows:
        # E0 identity
        assert L.edit("E0", tx, X) == tx

        # E1: idempotent, no-clobber, every substitution is X and sits in an anchored, non-run spot
        z1 = L.e1_substitute(tx, X)
        assert L.e1_substitute(z1, X) == z1, "E1 not idempotent"
        runs = [(m.start(), m.end()) for m in L._RUN.finditer(tx)]
        for j in range(min(len(tx), len(z1))):
            if tx[j] != z1[j]:
                e1_fired += 1
                assert z1[j] == X, f"E1 wrote {z1[j]!r} not target {X!r}"
                assert tx[j] in SYM, "E1 changed a non-letter"
                assert not any(a <= j < b for a, b in runs), "E1 clobbered an enumeration letter"
        # E1 only ever rewrites isolated A-D letters -> total letter count per symbol is conserved+shifted
        assert len(z1) == len(tx), "E1 changed length"

        # E2: appends an assertion containing X, idempotent
        z2 = L.e2_append(tx, X)
        assert f"({X})" in z2 and z2.endswith(")."), "E2 must end asserting (X)."
        assert L.e2_append(z2, X) == z2, "E2 not idempotent"
        if z2 != tx:
            e2_changed += 1

        # E3: idempotent; only removes hedges (no hedge -> exact no-op); never lengthens
        z3 = L.e3_strip(tx)
        assert L.e3_strip(z3) == z3, "E3 not idempotent"
        assert len(z3) <= len(tx), "E3 lengthened text"
        if z3 == tx:
            assert not L._HEDGE.search(tx), "E3 no-op but a hedge was present"
        else:
            e3_fired += 1

        # T1 / T2: deterministic, contain (X), independent of the input text
        assert L.edit("T1", tx, X) == f"The correct answer is ({X})."
        assert f"({X})" in L.edit("T2", tx, X)

        # combos compose in the documented order
        assert L.edit("E4", tx, X) == L.e2_append(L.e1_substitute(tx, X), X)
        assert L.edit("E5", tx, X) == L.e1_substitute(L.e3_strip(tx), X)
        assert L.edit("E7", tx, X) == L.e2_append(L.e1_substitute(L.e3_strip(tx), X), X)

    assert e1_fired > 0 and e2_changed == len(rows) and e3_fired > 0, \
        f"coverage sanity (e1_subs={e1_fired}, e2={e2_changed}, e3={e3_fired})"
    print(f"[lang] operators on cached text OK (E1 subs={e1_fired}, E2 all-append, E3 rows={e3_fired})")


def test_e1_spares_option_lists():
    """E1 must never touch a stem option-list like 'A, B, C, D' / 'A', 'B', 'C'."""
    cases = [
        ('From the given options ("A, B, C, D"), the answer is (A) clearly.', "C"),
        ("possible options like 'A', 'B', or 'C' — the correct answer is B.", "D"),
    ]
    for tx, X in cases:
        z = L.e1_substitute(tx, X)
        # the option-list letters survive verbatim
        assert '"A, B, C, D"' in z or "'A', 'B'" in z, f"option list clobbered: {z!r}"
        # but the asserted answer letter became X
        assert f"({X})" in z or f" {X}." in z or f" {X} " in z, f"assertion not rewritten: {z!r}"
    print("[lang] E1 spares option-lists, rewrites the assertion OK")


def test_conventions_and_ratio():
    rng = np.random.default_rng(0)
    hr = rng.standard_normal((16, 8)).astype(np.float32)
    hn = rng.uniform(50, 100, size=16)
    nm = L.apply_convention(hr, hn, "normmatch")
    assert np.allclose(np.linalg.norm(nm, axis=1), hn, atol=1e-3), "normmatch norms wrong"
    cm = L.apply_convention(hr, hn, "cohortmean", cohort_mean=80.0)
    assert np.allclose(np.linalg.norm(cm, axis=1), 80.0, atol=1e-3), "cohortmean norms wrong"
    assert np.allclose(L.apply_convention(hr, hn, "native"), hr), "native must be identity"
    # ratio
    h_orig = np.array([[3.0, 4.0]]); h_steer = np.array([[3.0, 4.0]]) * 2
    assert abs(float(L.ratio_offmanifold(h_steer, h_orig)[0]) - 1.0) < 1e-9, "ratio wrong"
    print("[lang] conventions + off-manifold ratio OK")


def test_make_edit_fn_is_replacement():
    import torch
    steer = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    fn = L.make_edit_fn(steer)
    h_last = torch.zeros(3, 4)
    out = fn(h_last, L.LAYER)
    assert torch.equal(out, steer), "edit_fn must REPLACE (return steer), not add"
    print("[lang] edit_fn is a replacement OK")


def test_frontier_and_verdict():
    out = L._ROOT / "out"
    fr = L.kappa_frontier(out)
    assert fr, "frontier empty (out/fve/analysis.json missing?)"
    peak = max(fr.values(), key=lambda v: v["acc"])
    assert abs(peak["acc"] - 0.71484375) < 1e-6, f"KAPPA peak ACC {peak['acc']} != 0.715"
    assert 0.40 < peak["ome"] < 0.42, f"peak OME {peak['ome']} not ~0.413"
    floor = fr[min(fr)]["ome"]
    # WIN: matches peak ACC at strictly lower OME (near the floor)
    win = L._verdict([{"op": "T1", "acc": 0.72, "agr": 0.7, "steer_success": 0.8,
                       "ome": 0.28, "mean_ratio": 0.25}], peak, floor)
    assert win["decision"] == "WIN", win
    part = L._verdict([{"op": "E2", "acc": 0.72, "agr": 0.7, "steer_success": 0.8,
                        "ome": 0.50, "mean_ratio": 0.9}], peak, floor)
    assert part["decision"] == "PARTIAL", part
    null = L._verdict([{"op": "E0", "acc": 0.66, "agr": 0.6, "steer_success": 0.5,
                        "ome": 0.27, "mean_ratio": 0.0}], peak, floor)
    assert null["decision"] == "NULL", null
    print(f"[lang] frontier (peak ACC {peak['acc']:.3f} @ OME {peak['ome']:.3f}, floor {floor:.3f}) "
          f"+ verdict WIN/PARTIAL/NULL OK")


def test_targets_subsets_nest():
    L.build_targets()
    out = L._ROOT / "out"
    import json
    subs = json.loads((out / "lang" / "subsets.json").read_text())["rows"]
    assert set(subs["tiny256"]) <= set(subs["sweep512"]) <= set(subs["full"]), "subsets must nest"
    assert len(subs["full"]) == 2615 and len(subs["sweep512"]) == 512 and len(subs["tiny256"]) == 256
    t = pq.read_table(out / "lang" / "targets.parquet").to_pydict()
    assert set(t["X"]) <= set(SYM) and len(t["row_index"]) == 2615
    assert abs(np.mean(t["h_norm"]) - 86.69) < 0.1, "mean ||h_orig|| drift"
    print("[lang] targets + nested subsets OK")


def main() -> int:
    test_operators_on_cached_text()
    test_e1_spares_option_lists()
    test_conventions_and_ratio()
    test_make_edit_fn_is_replacement()
    test_frontier_and_verdict()
    test_targets_subsets_nest()
    print("\nLANG_STEER CPU UNIT TESTS PASSED -- operators, conventions, frontier, verdict, targets.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
