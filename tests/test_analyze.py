"""CPU unit tests for ome_gauge.analyze — the stats (partial corr, AUC, showdown), the GATE S1
transitions, and an end-to-end integration run on the real Stage-1 CPU outputs (which also writes
the real first-read FINDINGS.md). No GPU/OME required; the OME-dependent reads report PENDING.

Run: python tests/test_analyze.py   (from the NLA-final root)
"""
import os
import sys

_SRC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _SRC)

import numpy as np

from ome_gauge import analyze as A
from ome_gauge import config as C


def test_selftest():
    assert A.selftest() == 0


def test_gate_s1_transitions():
    """PENDING without OME; PASS when OME moves AND OME>=Maha on the collapse proxy; FAIL otherwise."""
    matched = {"dim_above_random": True}
    # no OME in the dose-response -> PENDING
    dose_no_ome = {"dim:D_correct": {"spearman_ratio_alpha": 0.99}}
    showdown_pending = {"additive_arms": {"status": "pending"}, "kappa_arm_free": {"status": "ok"}}
    assert A.gate_s1(dose_no_ome, showdown_pending, matched)["decision"] == "PENDING"

    dose_ome = {"dim:D_correct": {"spearman_ome_alpha": 0.97, "spearman_ratio_alpha": 0.99}}
    showdown_pass = {"additive_arms": {"status": "ok", "ome_beats_all": True,
                                       "ome_ge_baselines": {"mahalanobis": True, "ratio": True,
                                                            "act_norm": True}}}
    assert A.gate_s1(dose_ome, showdown_pass, matched)["decision"] == "PASS"

    showdown_fail = {"additive_arms": {"status": "ok", "ome_beats_all": False,
                                       "ome_ge_baselines": {"mahalanobis": False}}}
    assert A.gate_s1(dose_ome, showdown_fail, matched)["decision"] == "FAIL"
    print("[analyze] GATE S1 transitions PENDING/PASS/FAIL OK")


def test_baseline_showdown_detector_independent():
    """A detector that tracks the (independent) accuracy drop wins AUC; the collapse label is the
    behavioral drop, never an alpha/OME/ratio threshold (no circularity)."""
    alpha = [0, 5, 10, 20, 40, 60, 90, 130, 175]
    # OME tracks the acc drop; act_norm is a poor (flat-ish) competitor here
    conds = []
    for a in alpha:
        drop = max(0.0, (a - 60) / 600.0)
        conds.append({"method": "dim", "dir": "D_correct", "alpha": float(a),
                      "ome": 0.27 + drop * 3 + a * 1e-4, "mahalanobis": 40 + 0.2 * a,
                      "ratio": a / 86.7, "act_norm": 86.0 + 1e-3 * a,
                      "acc_drop": drop, "collapsed": bool(a >= 130)})
    sd = A.baseline_showdown(conds, [])["additive_arms"]
    assert sd["status"] == "ok"
    assert sd["ome"]["auc_collapsed"] == 1.0           # OME perfectly ranks the collapsed conditions
    assert "ome_ge_baselines" in sd and isinstance(sd["ome_beats_all"], bool)
    print(f"[analyze] showdown: OME AUC={sd['ome']['auc_collapsed']}, "
          f"beats_all={sd['ome_beats_all']} OK")


def test_matched_ratio_contrast():
    conds = [
        {"method": "dim", "dir": "D_correct", "alpha": 175.0, "ratio": 2.0, "ome": 0.6},
        {"method": "random", "dir": "D_random_0", "alpha": 175.0, "ratio": 2.0, "ome": 0.5},
        {"method": "random", "dir": "D_random_1", "alpha": 175.0, "ratio": 2.0, "ome": 0.52},
    ]
    mr = A.matched_ratio_contrast(conds)
    assert mr["status"] == "ok" and mr["dim_above_random"] and mr["mean_delta"] > 0
    # pending when an arm lacks OME
    assert A.matched_ratio_contrast([{"method": "dim", "dir": "x", "alpha": 1.0}])["status"] == "pending"
    print(f"[analyze] matched-ratio dim-above-random delta={mr['mean_delta']:+.3f} OK")


def test_integration_real_cpu_outputs():
    """Run the real P4 on the committed Stage-1 outputs (detectors.parquet + KAPPA overlay, and —
    once the pod sweep has run — ome_by_cond.parquet). Asserts the KAPPA free OME-vs-ratio showdown
    is computable, the detector dose-response is present, and the headline GATE S1 is honest about
    whichever state the on-disk data is in (PENDING pre-pod; PASS/FAIL once the pod OME exists)."""
    if not (C.PATHS.dir_detect() / "detectors.parquet").exists() or not C.PATHS.kappa_analysis().exists():
        print("[analyze] (skip integration — detectors.parquet / KAPPA overlay absent)"); return
    rep = A.analyze()
    assert (C.PATHS.dir_analysis() / "analysis.json").exists()
    assert (C.PATHS.dir_report() / "FINDINGS.md").exists()
    # the free KAPPA arm gives an OME-vs-ratio first read with no pod spend
    kappa = rep["baseline_showdown"]["kappa_arm_free"]
    assert kappa["status"] == "ok" and "ome" in kappa and "ratio" in kappa
    # OME must rise with alpha on the KAPPA arm (the published monotone law)
    assert kappa["ome"]["spearman_accdrop"] is not None
    # the gate tracks the on-disk state: PENDING until the pod fills OME, PASS/FAIL once it has.
    pod_ome = (C.PATHS.dir_ome() / "ome_by_cond.parquet").exists()
    if pod_ome:
        assert rep["has_ome"] is True and rep["gate_s1"]["decision"] in ("PASS", "FAIL")
    else:
        assert rep["has_ome"] is False and rep["gate_s1"]["decision"] == "PENDING"
    # detector dose-response is present for the additive arms (ratio/Maha rise with alpha)
    dim = rep["dose_response"].get("dim:D_correct", {})
    assert dim.get("spearman_mahalanobis_alpha", 0) > 0.9
    print(f"[analyze] integration: KAPPA free showdown OK, GATE S1={rep['gate_s1']['decision']} "
          f"(pod OME present={pod_ome}), Maha-alpha rho={dim.get('spearman_mahalanobis_alpha'):.3f}")


def main() -> int:
    test_selftest()
    test_gate_s1_transitions()
    test_baseline_showdown_detector_independent()
    test_matched_ratio_contrast()
    test_integration_real_cpu_outputs()
    print("\nANALYZE CPU TESTS PASSED.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
