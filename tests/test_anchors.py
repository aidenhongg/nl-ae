"""CPU test for ome_gauge.anchors — the OME frontier drift guard against the real KAPPA overlay.

Run: python tests/test_anchors.py   (from the NLA-final root)
"""
import os
import sys

_SRC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _SRC)

import numpy as np

from ome_gauge import anchors as A
from ome_gauge import config as C
from src.fve_analysis import spearman as fve_spearman


def test_overlay_shape_and_monotone():
    if not C.PATHS.kappa_analysis().exists():
        print("[anchors] (skip — out/fve/analysis.json absent)"); return
    ov = A.load_kappa_overlay()
    assert np.array_equal(ov["alpha"], np.array(C.ALPHAS)), "overlay alphas must match the KAPPA grid"
    # OME = 1 - cos rises with alpha up to AV sampling noise (~1e-3 wiggle at the floor); the robust
    # monotonicity statement is the rank correlation, which validate_anchors pins at -0.991.
    assert np.all(np.diff(ov["ome"]) > -2e-3), "OME(alpha) must be ~non-decreasing (within AV noise)"
    assert abs(ov["ome"][0] - 0.2746) < 1e-3 and abs(ov["ome"][-1] - 0.6927) < 1e-3
    assert fve_spearman(ov["alpha"], ov["ome"]) > 0.98, "OME must rise strongly with alpha"
    print("[anchors] KAPPA overlay shape + monotone OME OK")


def test_validate_anchors():
    if not C.PATHS.kappa_analysis().exists():
        print("[anchors] (skip validate — analysis.json absent)"); return
    got = A.validate_anchors()
    assert abs(got["spearman_cos_alpha"] - (-0.9909)) < 1e-3
    assert abs(got["ome_peak_a10"] - 0.4127) < 1e-3
    # a tampered anchor must trip the guard
    saved = A.OME_ANCHORS["ome_floor_a0"]
    try:
        A.OME_ANCHORS["ome_floor_a0"] = 0.999
        A.validate_anchors(); assert False, "drift guard should have raised"
    except AssertionError as e:
        assert "drift" in str(e)
    finally:
        A.OME_ANCHORS["ome_floor_a0"] = saved
    print("[anchors] validate_anchors + drift-guard trip OK")


def main() -> int:
    test_overlay_shape_and_monotone()
    test_validate_anchors()
    print("\nANCHORS CPU TESTS PASSED.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
