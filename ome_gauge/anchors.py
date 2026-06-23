"""OME frontier anchors + drift guard (the graphing/data.py-style schema truth for OME-GAUGE).

Self-contained (does NOT edit the parent's concluded graphing/data.py, to avoid drift in the
mission tree). Reads the free KAPPA overlay out/fve/analysis.json — the published frontier this
experiment places DiM/random onto — and fails loudly if the committed OME numbers ever change
(SPEC P4 gate; QUESTIONS s9.5). Also the canonical loader of the KAPPA continuity arm for analyze.py.

The headline off-manifold metric is OME = 1 - mean_cos (the analysis.json `ome`/`fve_fixed` fields
are degenerate: var0 ~ 0.107 inflates them — see the parent CHANGELOG / graphing.data note).
"""
from __future__ import annotations

import json

import numpy as np

from ome_gauge import config as C
from src import fve_analysis   # reuse: spearman (numpy-only)

# Frozen anchors from out/report/report.md + out/fve/analysis.json (the concluded KAPPA/NLA run).
OME_ANCHORS = {
    "ome_floor_a0": 0.2746,       # benign a0 OME = 1 - 0.7254
    "ome_peak_a10": 0.4127,       # OME at the accuracy-peak alpha (a10, acc 0.715)
    "ome_collapse_a30": 0.6927,   # OME at the collapse alpha (a30, acc 0.600)
    "spearman_cos_alpha": -0.9909,  # H1: round-trip cos falls monotonically with alpha
    "kappa_peak_acc": 0.71484375,   # exp04 a10 accuracy (the productive-off-manifold peak)
    "kappa_collapse_acc": 0.599609375,  # exp04 a30 accuracy
    "base_acc": 0.6604,           # model forced-choice readout accuracy (test split)
}
TOL = 1e-3


def load_kappa_overlay() -> dict:
    """Tidy the KAPPA sweep into parallel arrays sorted by alpha (the free DiM/random comparison
    frontier): {alpha, cos, ome=1-cos, ratio, acc}. Mirrors graphing.data.kappa_per_alpha."""
    pa = json.loads(C.PATHS.kappa_analysis().read_text(encoding="utf-8"))["per_alpha"]
    al = sorted(float(k) for k in pa)
    g = lambda f: np.array([(pa[_key(pa, a)].get(f)
                             if pa[_key(pa, a)].get(f) is not None else np.nan) for a in al], float)
    cos = g("mean_cos")
    return {"alpha": np.array(al), "cos": cos, "ome": 1.0 - cos,
            "ratio": g("ratio_subset"), "acc": g("exp04_acc")}


def _key(pa: dict, a: float) -> str:
    """analysis.json keys are json-stringified floats ('0.0','10.0'); match an alpha to its key."""
    for k in pa:
        if float(k) == float(a):
            return k
    raise KeyError(a)


def validate_anchors() -> dict:
    """Fail loudly if the KAPPA overlay no longer reproduces the committed OME frontier. Returns the
    measured values on success. This guards every OME-GAUGE figure/verdict against silent drift."""
    ov = load_kappa_overlay()
    a = ov["alpha"]
    got = {
        "ome_floor_a0": float(ov["ome"][a == 0.0][0]),
        "ome_peak_a10": float(ov["ome"][a == 10.0][0]),
        "ome_collapse_a30": float(ov["ome"][a == 30.0][0]),
        "spearman_cos_alpha": float(fve_analysis.spearman(a, ov["cos"])),
        "kappa_peak_acc": float(ov["acc"][a == 10.0][0]),
        "kappa_collapse_acc": float(ov["acc"][a == 30.0][0]),
    }
    bad = {k: (got[k], OME_ANCHORS[k]) for k in got if abs(got[k] - OME_ANCHORS[k]) > TOL}
    if bad:
        raise AssertionError(f"OME anchor drift: {bad}")
    print("[anchors] OME frontier OK:", {k: round(v, 4) for k, v in got.items()})
    return got


def main(argv=None) -> int:
    validate_anchors()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
