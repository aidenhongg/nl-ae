"""CPU unit tests for ome_gauge.ome_probe — the AV-coherence read-out, the OME transform, and the
ledger -> ome_by_cond compaction (hermetic; writes a tiny synthetic ledger to out/ome/). The actual
AV round-trip is pod-gated and not exercised here.

Run: python tests/test_ome_probe.py   (from the NLA-final root)
"""
import json
import os
import sys

_SRC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _SRC)

import numpy as np

from ome_gauge import ome_probe as P


def test_av_coherent():
    assert P.av_coherent("The activation describes a multiple-choice answer about geography.")
    assert not P.av_coherent("これは日本語のテキストで、英語ではありません今日は")   # CJK
    assert not P.av_coherent("answer answer answer answer answer answer answer")     # degenerate rep
    assert not P.av_coherent("ok")                                                   # too short
    assert not P.av_coherent("\x00\x01\x02\x03\x04\x05\x06\x07\x08\x09\x0b\x0c")      # control bytes
    print("[ome_probe] av_coherent: English yes; CJK/repetition/short/garbage no - OK")


def test_ome_from_cos():
    assert abs(P.ome_from_cos(0.7254) - 0.2746) < 1e-9          # the benign floor
    got = P.ome_from_cos(np.array([1.0, 0.5, 0.0]))
    assert np.allclose(got, [0.0, 0.5, 1.0])
    print("[ome_probe] OME = 1 - cos (floor 0.2746 at cos 0.7254) OK")


def test_compact_synthetic():
    """A hand-written ledger compacts to ome_by_cond.parquet + a manifest with per-condition OME
    means and the NLA-OOD (av-coherent<0.5) flag. Hermetic: writes to a temp dir, not out/ome/."""
    import tempfile
    from pathlib import Path
    import pyarrow.parquet as pq
    from ome_gauge import config as C
    out = Path(tempfile.mkdtemp(prefix="ome_probe_test_"))
    ledger = out / "ome.jsonl"
    recs = [
        # dim:D_correct a0 — coherent, near floor
        {"source": "dim:D_correct", "row_index": 0, "example_id": "e0", "alpha": 0.0,
         "nl_text": "A clear English explanation of the answer choice.", "n_tokens": 8,
         "cos_roundtrip": 0.72, "ratio": 0.0, "mahalanobis": 40.0, "act_norm": 86.0},
        {"source": "dim:D_correct", "row_index": 1, "example_id": "e1", "alpha": 0.0,
         "nl_text": "Another fluent English snippet describing the activation.", "n_tokens": 8,
         "cos_roundtrip": 0.74, "ratio": 0.0, "mahalanobis": 41.0, "act_norm": 86.5},
        # dim:D_correct a175 — high OME, degenerate AV text (NLA-OOD)
        {"source": "dim:D_correct", "row_index": 0, "example_id": "e0", "alpha": 175.0,
         "nl_text": "これは日本語日本語日本語日本語日本語", "n_tokens": 5,
         "cos_roundtrip": 0.30, "ratio": 2.02, "mahalanobis": 90.0, "act_norm": 190.0},
        {"source": "dim:D_correct", "row_index": 1, "example_id": "e1", "alpha": 175.0,
         "nl_text": "破破破破破破破破破破破破破破", "n_tokens": 1,
         "cos_roundtrip": 0.31, "ratio": 2.02, "mahalanobis": 91.0, "act_norm": 191.0},
    ]
    ledger.write_text("\n".join(json.dumps(r) for r in recs) + "\n", encoding="utf-8")
    man = P.compact(av_rev="test-rev", out_dir=out)
    t = pq.read_table(out / "ome_by_cond.parquet").to_pydict()
    # OME = 1 - cos, ome_delta_floor relative to the config floor
    i0 = t["alpha"].index(0.0)
    assert abs(t["ome"][i0] - (1.0 - t["cos_roundtrip"][i0])) < 1e-9
    assert abs(t["ome_delta_floor"][i0] - (t["ome"][i0] - C.OME_FLOOR)) < 1e-9
    conds = {(c["method"], c["dir"], c["alpha"]): c for c in man["conditions"]}
    c0 = conds[("dim", "D_correct", 0.0)]; c175 = conds[("dim", "D_correct", 175.0)]
    assert c0["av_coherent_frac"] == 1.0 and not c0["nla_ood"]
    assert c175["av_coherent_frac"] == 0.0 and c175["nla_ood"]          # flagged NLA-OOD
    assert c175["ome_mean"] > c0["ome_mean"]                            # OME rose with the push
    print(f"[ome_probe] compact: floor cond OME {c0['ome_mean']:.3f} (coherent), "
          f"a175 OME {c175['ome_mean']:.3f} (NLA-OOD flagged) OK")


def main() -> int:
    test_av_coherent()
    test_ome_from_cos()
    test_compact_synthetic()
    print("\nOME_PROBE CPU TESTS PASSED.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
