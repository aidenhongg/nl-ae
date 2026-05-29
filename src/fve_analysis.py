"""Phase D analysis — FVE(alpha) metrics + hypothesis tests H1/H2/H3.

Consumes the GPU round-trip outputs (out/fve/fve_by_alpha.json, out/fve/per_row.parquet),
the local norms (inputs/norms.parquet), and exp04's ACC(alpha) sweep
(exp04/03_kappa/diag/sweep.json), then emits out/fve/analysis.json (+ figures if
matplotlib is present).

  H1  off-manifold <-> strength: FVE(alpha) and mean_cos(alpha) decrease with alpha and
      with ratio=||dh||/||h||. Spearman rho + OLS slope.
  H2  FVE predicts damage: OME(alpha)=1-FVE(alpha) correlates with exp04 ACC degradation
      (base_acc - acc(alpha)), single-layer L20 example_level. Spearman rho + scatter.
  H3  directional vs magnitude: round-trip is invariant to a pure rescale h->c*h (the AV
      client normalizes by injection_scale/||v||), so any FVE drop is directional. Reads
      the optional rescale-control ledger if present.

numpy-only stats (no scipy/pandas) so it runs on a thin pod too. `--selftest` validates
the exp04 parser + ratio aggregation against known values using ONLY local files.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ALPHAS = [0.0, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0, 20.0, 30.0]


# ----------------------------- numpy-only stats -------------------------------

def _rankdata(a: np.ndarray) -> np.ndarray:
    """Average ranks (ties shared), like scipy.stats.rankdata('average')."""
    a = np.asarray(a, float)
    order = a.argsort()
    ranks = np.empty(len(a), float)
    ranks[order] = np.arange(1, len(a) + 1)
    # average tied ranks
    _, inv, counts = np.unique(a, return_inverse=True, return_counts=True)
    sums = np.zeros(len(counts)); np.add.at(sums, inv, ranks)
    return (sums / counts)[inv]


def pearson(x, y) -> float:
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def spearman(x, y) -> float:
    return pearson(_rankdata(x), _rankdata(y))


def ols_slope(x, y) -> float:
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 2 or np.std(x) == 0:
        return float("nan")
    return float(np.polyfit(x, y, 1)[0])


def bootstrap_ci(vals: np.ndarray, n_boot: int = 2000, seed: int = 7, alpha: float = 0.05):
    vals = np.asarray(vals, float)
    if vals.size == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    means = vals[rng.integers(0, vals.size, size=(n_boot, vals.size))].mean(1)
    return (float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2)))


# ----------------------------- exp04 + norms loaders --------------------------

def parse_exp04_acc(sweep_path: Path) -> dict:
    """Single-layer L20 example_level ACC(alpha) + ratio(alpha) from exp04 sweep.json."""
    s = json.loads(sweep_path.read_text(encoding="utf-8"))
    el = s["schemes"]["example_level"]
    base = float(el["acc_before"])
    acc, ratio = {}, {}
    for c in el["configs"]:
        if c.get("layers") == [20] and c.get("mode") == "single":
            a = float(c["alpha"])
            acc[a] = float(c["acc_after"])
            sc = c.get("scales", {}).get("20", {})
            if "ratio" in sc:
                ratio[a] = float(sc["ratio"])
    return {"base_acc": base, "acc_by_alpha": acc, "ratio_by_alpha": ratio}


def agg_ratio(norms_path: Path, subset_only: bool = True) -> dict:
    """Mean ratio=||dh||/||h|| per alpha over the paired subset (or all rows)."""
    import pyarrow.parquet as pq
    t = pq.read_table(norms_path,
                      columns=["alpha", "ratio", "in_subset"]).to_pydict()
    out: dict[float, list] = {}
    for i in range(len(t["alpha"])):
        if subset_only and not t["in_subset"][i]:
            continue
        out.setdefault(float(t["alpha"][i]), []).append(float(t["ratio"][i]))
    return {a: float(np.mean(v)) for a, v in out.items()}


# ----------------------------- analysis ---------------------------------------

def analyze(out: Path, inputs: Path, sweep_path: Path) -> dict:
    import pyarrow.parquet as pq
    fve = json.loads((out / "fve" / "fve_by_alpha.json").read_text(encoding="utf-8"))
    var0 = fve.pop("_var0", None)
    fve = {float(k): v for k, v in fve.items()}
    exp04 = parse_exp04_acc(sweep_path)
    ratio_sub = agg_ratio(inputs / "norms.parquet", subset_only=True)

    # per-row CIs (paired across alpha) from per_row.parquet
    per_row = out / "fve" / "per_row.parquet"
    ci = {}
    if per_row.exists():
        pr = pq.read_table(per_row).to_pydict()
        by_a: dict[float, list] = {}
        for i in range(len(pr["alpha"])):
            by_a.setdefault(float(pr["alpha"][i]), []).append(float(pr["cos"][i]))
        for a, cos in by_a.items():
            ci[a] = {"cos_ci": bootstrap_ci(np.array(cos))}

    alphas = sorted(fve)
    fve_vec = np.array([fve[a]["fve_fixed"] for a in alphas])
    cos_vec = np.array([fve[a]["mean_cos"] for a in alphas])
    ratio_vec = np.array([ratio_sub.get(a, np.nan) for a in alphas])

    # H1
    h1 = {"spearman_fve_alpha": spearman(alphas, fve_vec),
          "spearman_cos_alpha": spearman(alphas, cos_vec),
          "spearman_fve_ratio": spearman(ratio_vec, fve_vec),
          "slope_fve_alpha": ols_slope(alphas, fve_vec)}

    # H2: OME vs ACC-drop on shared alphas
    shared = [a for a in alphas if a in exp04["acc_by_alpha"]]
    ome = [1.0 - fve[a]["fve_fixed"] for a in shared]
    accdrop = [exp04["base_acc"] - exp04["acc_by_alpha"][a] for a in shared]
    h2 = {"shared_alphas": shared,
          "ome": ome, "acc_drop": accdrop,
          "spearman_ome_accdrop": spearman(ome, accdrop),
          "pearson_ome_accdrop": pearson(ome, accdrop),
          "base_acc": exp04["base_acc"]}

    # H3: rescale-control (optional)
    h3 = {"note": "AV client normalizes by injection_scale/||v|| -> round-trip is exactly "
                  "scale-invariant by construction; control confirms empirically."}
    rc = out / "fve" / "rescale_control.json"
    if rc.exists():
        h3["measured"] = json.loads(rc.read_text(encoding="utf-8"))

    report = {
        "schema_version": "nla_fve_analysis.v1",
        "var0": var0,
        "per_alpha": {a: {**fve[a], "ratio_subset": ratio_sub.get(a),
                          "exp04_acc": exp04["acc_by_alpha"].get(a),
                          "ome": 1.0 - fve[a]["fve_fixed"], **ci.get(a, {})} for a in alphas},
        "H1": h1, "H2": h2, "H3": h3,
    }
    (out / "fve" / "analysis.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[D] analysis.json written. H1 rho(FVE,alpha)={h1['spearman_fve_alpha']:.3f} "
          f"H2 rho(OME,ACCdrop)={h2['spearman_ome_accdrop']:.3f}", flush=True)
    _figures(out, alphas, fve_vec, cos_vec, ratio_vec, shared, ome, accdrop)
    return report


def _figures(out, alphas, fve_vec, cos_vec, ratio_vec, shared, ome, accdrop):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:  # noqa: BLE001
        print("[D] matplotlib absent — skipping figures (analysis.json is the deliverable).")
        return
    fig_dir = out / "report"; fig_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    ax[0].plot(alphas, fve_vec, "o-"); ax[0].set(xlabel="alpha", ylabel="FVE(alpha)", title="H1: FVE vs alpha")
    ax[1].plot(ratio_vec, fve_vec, "o-"); ax[1].set(xlabel="ratio ||dh||/||h||", ylabel="FVE", title="H1: FVE vs ratio")
    ax[2].scatter(accdrop, ome); ax[2].set(xlabel="ACC drop (base-acc(a))", ylabel="OME=1-FVE", title="H2: OME vs ACC drop")
    for a, o, d in zip(shared, ome, accdrop):
        ax[2].annotate(f"a{a:g}", (d, o))
    fig.tight_layout(); fig.savefig(fig_dir / "fve_figures.png", dpi=120)
    print(f"[D] figures -> {fig_dir/'fve_figures.png'}", flush=True)


def selftest(sweep_path: Path, norms_path: Path) -> int:
    """Validate the exp04 parser + ratio aggregation against known values (local only)."""
    e = parse_exp04_acc(sweep_path)
    print("exp04 base_acc:", e["base_acc"])
    print("exp04 acc_by_alpha:", {k: round(v, 4) for k, v in sorted(e["acc_by_alpha"].items())})
    print("exp04 ratio_by_alpha:", {k: round(v, 4) for k, v in sorted(e["ratio_by_alpha"].items())})
    assert abs(e["base_acc"] - 0.669921875) < 1e-6, "base_acc mismatch"
    assert abs(e["acc_by_alpha"][30.0] - 0.599609375) < 1e-6, "alpha30 acc mismatch"
    assert abs(e["acc_by_alpha"][10.0] - 0.71484375) < 1e-6, "alpha10 acc mismatch"
    assert {1.0, 2.0, 5.0, 10.0, 20.0, 30.0} <= set(e["acc_by_alpha"]), "missing single-L20 alphas"
    if norms_path.exists():
        r = agg_ratio(norms_path, subset_only=True)
        print("subset ratio_by_alpha:", {k: round(v, 4) for k, v in sorted(r.items())})
        assert abs(r[0.0]) < 1e-9, "alpha0 ratio must be 0"
        assert 1.9 < r[30.0] < 2.2, f"alpha30 subset ratio {r[30.0]} out of expected range"
        assert 0.6 < r[10.0] < 0.72, f"alpha10 subset ratio {r[10.0]} out of expected range"
    # H2 sign sanity on exp04 alone: ACC drop should be monotone-ish high at extreme alpha
    print("[selftest] exp04 parser + ratio aggregation OK")
    return 0


def main(argv):
    ap = argparse.ArgumentParser(description="Phase D FVE analysis + H1/H2/H3.")
    here = Path(__file__).resolve().parent.parent
    ap.add_argument("--out", type=Path, default=here / "out")
    ap.add_argument("--inputs", type=Path, default=here / "inputs")
    ap.add_argument("--sweep", type=Path,
                    default=here.parent / "exp04" / "05_out_pulled" / "03_kappa" / "diag" / "sweep.json")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args(argv)
    if args.selftest:
        return selftest(args.sweep, args.inputs / "norms.parquet")
    analyze(args.out, args.inputs, args.sweep)
    return 0


if __name__ == "__main__":
    import sys
    raise SystemExit(main(sys.argv[1:]))
