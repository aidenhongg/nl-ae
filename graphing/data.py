"""Read-only loaders for the NLA-final experiment data — the single place that
knows the frozen on-disk schema (paths, columns, units). Every graph imports
from here so a path/column lives in exactly one spot.

Sources (all produced by the concluded experiments; never written by graphing):
  out/feat/datapoint_features.parquet   per-example F1/F2/F3 (6536 rows; split_el)
  out/fve/analysis.json                 KAPPA round-trip FVE sweep (per-alpha)
  out/fve/per_row.parquet               per-row round-trip cos/mse x 11 alpha
  out/feat/steered_divergence.parquet   F3 on each steered h'(alpha) (11x6536)
  out/lang/report_data.json             NLA language-steering methods + KAPPA frontier
  out/lang/targets.parquet              per-row steering target X (test, 2615)

No pandas (not a project dep): pyarrow -> dict-of-numpy.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent.parent          # project root (NLA-final/)
OUT = ROOT / "out"
FIGURES = Path(__file__).resolve().parent / "figures"

# Steering alpha grid (== src/steer_sweep.ALPHAS; frozen scientific contract).
ALPHAS = [0.0, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0, 20.0, 30.0]

# Documented base anchors (example_level TEST, n=2615) — used for annotation and
# as a drift check (see validate_anchors). From MAIN-EXP.md S2 / out/feat manifest.
ANCHORS = {
    "base_acc": 0.6604,   # model first-token (forced-choice readout) vs ground truth
    "gen_acc": 0.6608,    # model greedy generation vs ground truth
    "know_acc": 0.8543,   # knowledge probe argmax vs ground truth
    "pred_acc": 0.9120,   # prediction probe argmax vs the model's own readout
    "agr": 0.6486,        # knowledge-probe argmax == model readout
    "n_test": 2615,
    "mean_h_norm": 86.69,
}


def _table_to_arrays(path: Path) -> dict[str, np.ndarray]:
    cols = pq.read_table(path).to_pydict()
    return {k: np.array(v) for k, v in cols.items()}


def load_features(split: str | None = "test") -> dict[str, np.ndarray]:
    """Per-example features. split in {"train","val","test",None}; default "test"
    (matches the documented anchors). Filters on split_el (example_level scheme)."""
    d = _table_to_arrays(OUT / "feat" / "datapoint_features.parquet")
    if split is not None:
        mask = d["split_el"] == split
        d = {k: v[mask] for k, v in d.items()}
    return d


def load_fve_analysis() -> dict:
    """KAPPA round-trip FVE sweep. Note: the headline off-manifold error is
    OME = 1 - mean_cos (the `ome`/`fve_fixed` fields are degenerate, var0~0.107
    -> ignore; see CHANGELOG). exp04_acc is the first-token ACC at that alpha."""
    return json.loads((OUT / "fve" / "analysis.json").read_text())


def load_per_row() -> dict[str, np.ndarray]:
    """Per-row round-trip (cos, mse, alpha, +features); 1024 subset x 11 alpha."""
    return _table_to_arrays(OUT / "fve" / "per_row.parquet")


def load_steered_divergence() -> dict[str, np.ndarray]:
    """F3 prediction<->knowledge divergence on each steered h'(alpha) (11x6536)."""
    return _table_to_arrays(OUT / "feat" / "steered_divergence.parquet")


def load_lang_report() -> dict:
    """NLA language-steering result: per-method (acc, mean_ratio, CIs, y_balance)
    + the KAPPA single-L20 frontier in OME(=1-cos)/ratio/acc units. methods have
    no OME (best_ome=null: Phase-4 OME re-verbalization was skipped post-NULL)."""
    return json.loads((OUT / "lang" / "report_data.json").read_text())


def load_targets() -> dict[str, np.ndarray]:
    return _table_to_arrays(OUT / "lang" / "targets.parquet")


def kappa_per_alpha(analysis: dict | None = None):
    """Tidy the KAPPA sweep into parallel arrays sorted by alpha:
    (alpha, ome=1-cos, cos, cos_ci_lo, cos_ci_hi, ratio, acc[nan if absent])."""
    a = analysis or load_fve_analysis()
    pa = a["per_alpha"]
    al = sorted(float(k) for k in pa)
    g = lambda k, f: np.array([pa[str(k_)].get(f) if pa[str(k_)].get(f) is not None
                               else np.nan for k_ in k], dtype=float)
    cos = g(al, "mean_cos")
    ci = np.array([pa[str(k)]["cos_ci"] for k in al], dtype=float)
    return {
        "alpha": np.array(al), "cos": cos, "ome": 1.0 - cos,
        "cos_ci_lo": ci[:, 0], "cos_ci_hi": ci[:, 1],
        "ratio": g(al, "ratio_subset"), "acc": g(al, "exp04_acc"),
    }


def validate_anchors() -> None:
    """Fail loudly if the data no longer reproduces the documented anchors —
    a guard so a graph can never silently plot drifted numbers."""
    d = load_features("test")
    n = len(d["gt_symbol"])
    chk = {
        "n_test": (float(n), float(ANCHORS["n_test"]), 0.0),
        "know_acc": (float(np.mean(d["know_correct"])), ANCHORS["know_acc"], 1e-3),
        "pred_acc": (float(np.mean(d["pred_matches_model"])), ANCHORS["pred_acc"], 1e-3),
        "base_acc": (float(np.mean(d["model_readout_correct"])), ANCHORS["base_acc"], 1e-3),
        "agr": (float(np.mean(d["agree_know_model"])), ANCHORS["agr"], 1e-3),
    }
    bad = {k: (got, exp) for k, (got, exp, tol) in chk.items() if abs(got - exp) > tol}
    if bad:
        raise AssertionError(f"anchor drift: {bad}")
    print("[data] anchors OK:", {k: round(got, 4) for k, (got, exp, tol) in chk.items()})


if __name__ == "__main__":
    validate_anchors()
