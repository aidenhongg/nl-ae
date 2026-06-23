"""NLA-free off-manifold detectors (P2/P3) — the H6 competitors that OME must beat to justify the
NLA's GPU cost (DESIGN H6; the make-or-break, PLAN s6 R2).

Detectors (all return higher = more anomalous), fit on benign-only data with NO eval leakage:
  ratio        ||dh||/||h||              (reuse lang_steer.ratio_offmanifold) -- the easy magnitude baseline
  act_norm     ||h'||                    (cheapest possible)
  mahalanobis  sqrt((h-mu) Sig^-1 (h-mu))  with numpy LEDOIT-WOLF shrinkage toward scaled identity
  knn_dist     mean L2 to the k nearest benign neighbours (non-parametric density)
  pca_whiten   Mahalanobis in the benign top-k PC subspace (robustness variant for Maha)
  self_ppl     steered model's NLL of its own continuation -- POD-GATED (needs the model; P3)

Why Ledoit-Wolf: at d=3584 with ~3.9k benign rows the raw sample covariance is rank-deficient and
its inverse is meaningless -> Maha would be unfair (artificially weak, handing OME a cheap win, or
arbitrarily strong). LW shrinkage toward (tr(S)/d)*I is closed-form (no hyperparameter, no scipy),
well-conditioned, and is the standard estimator the steering-collapse literature uses. Fit MUST be
benign-only and disjoint from the eval rows (QUESTIONS s7.3).

CLI:  python -m ome_gauge.detectors fit-benign        # LW + PCA-whiten on benign trainval -> maha_fit.npz
      python -m ome_gauge.detectors score             # detectors.parquet over the P1 steered arrays
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from ome_gauge import config as C
from src import lang_steer, features   # reuse: ratio_offmanifold, write_parquet/json_atomic


# ============================ Mahalanobis (Ledoit-Wolf) ======================

def ledoit_wolf(X: np.ndarray) -> dict:
    """Ledoit-Wolf (2004) shrinkage covariance toward (tr(S)/d)*I. X: [n,d] benign samples (fp64).
    Returns {mu, sigma, sigma_inv, shrinkage, mu_eig(m), n}. numpy-only, no scipy.

    S      = (1/n) Xc^T Xc                          (MLE sample cov of centred X)
    m      = tr(S)/d                                (mean eigenvalue; the identity target scale)
    d2     = ||S - m I||_F^2 = ||S||_F^2 - d m^2    (dispersion of S about the target)
    bbar2  = (1/n^2)(sum_k ||x_k||^4 - n ||S||_F^2) (avg squared error of the per-sample cov)
    b2     = min(bbar2, d2);  shrink = b2/d2;  Sig* = shrink*(m I) + (1-shrink)*S
    """
    X = np.asarray(X, dtype=np.float64)
    n, d = X.shape
    mu = X.mean(0)
    Xc = X - mu
    S = (Xc.T @ Xc) / n                                  # [d,d]
    s_fro2 = float(np.sum(S * S))
    m = float(np.trace(S) / d)
    d2 = s_fro2 - d * m * m
    sq = np.einsum("ij,ij->i", Xc, Xc)                   # ||x_k||^2 per sample
    bbar2 = float((np.sum(sq * sq) - n * s_fro2) / (n * n))
    b2 = min(bbar2, d2)
    shrink = 0.0 if d2 <= 0 else b2 / d2
    sigma = shrink * m * np.eye(d) + (1.0 - shrink) * S
    sigma_inv = np.linalg.inv(sigma)
    return {"mu": mu, "sigma": sigma, "sigma_inv": sigma_inv,
            "shrinkage": float(shrink), "mu_eig": m, "n": int(n), "d": int(d)}


def mahalanobis(h: np.ndarray, fit: dict) -> np.ndarray:
    """sqrt((h-mu) Sig^-1 (h-mu)) per row; higher = more off the benign manifold."""
    dx = np.asarray(h, np.float64) - fit["mu"]
    q = np.einsum("ij,jk,ik->i", dx, fit["sigma_inv"], dx)
    return np.sqrt(np.maximum(q, 0.0))


# ============================ PCA-whiten (Maha robustness) ===================

def fit_pca_whiten(X: np.ndarray, k: int) -> dict:
    """Top-k PCA whitening fit on benign X. Mahalanobis restricted to the leading-variance subspace
    — a robustness variant whose result should corroborate the full LW Maha (PLAN s6 R2)."""
    X = np.asarray(X, np.float64)
    mu = X.mean(0)
    Xc = X - mu
    # eigendecomp of the (d x d) covariance; take top-k. (d=3584 -> a few seconds, one-off.)
    cov = (Xc.T @ Xc) / X.shape[0]
    w, V = np.linalg.eigh(cov)                           # ascending
    idx = np.argsort(w)[::-1][:k]
    comps = V[:, idx].T                                  # [k,d]
    eig = np.maximum(w[idx], 1e-12)                      # [k]
    return {"mu": mu, "components": comps, "eigenvalues": eig, "k": int(k)}


def pca_whiten_dist(h: np.ndarray, fit: dict) -> np.ndarray:
    """sqrt(sum_j proj_j^2 / eig_j) over the top-k PCs; higher = more anomalous."""
    proj = (np.asarray(h, np.float64) - fit["mu"]) @ fit["components"].T   # [n,k]
    return np.sqrt(np.maximum((proj * proj / fit["eigenvalues"]).sum(1), 0.0))


# ============================ trivial / non-parametric =======================

def act_norm(h: np.ndarray) -> np.ndarray:
    """||h'|| — the cheapest possible off-manifold proxy."""
    return np.linalg.norm(np.asarray(h, np.float64), axis=1)


def ratio(h_steer: np.ndarray, h_orig: np.ndarray) -> np.ndarray:
    """||dh||/||h|| — reuse the parent's NLA-free magnitude proxy (lang_steer.ratio_offmanifold)."""
    return lang_steer.ratio_offmanifold(h_steer, h_orig)


def knn_dist(h: np.ndarray, cohort: np.ndarray, k: int = 10, chunk: int = 512) -> np.ndarray:
    """Mean L2 distance to the k nearest benign cohort points (non-parametric density). Chunked
    exact search; higher = lower benign density = more anomalous."""
    h = np.asarray(h, np.float64); cohort = np.asarray(cohort, np.float64)
    cn = (cohort * cohort).sum(1)                        # [m]
    out = np.empty(h.shape[0], np.float64)
    for i in range(0, h.shape[0], chunk):
        q = h[i:i + chunk]
        d2 = (q * q).sum(1)[:, None] + cn[None, :] - 2.0 * (q @ cohort.T)
        d2 = np.maximum(d2, 0.0)
        kk = min(k, cohort.shape[0])
        part = np.partition(d2, kk - 1, axis=1)[:, :kk]
        out[i:i + chunk] = np.sqrt(part).mean(1)
    return out


def self_ppl(*_a, **_k):  # pragma: no cover - pod-gated (needs the steered model; P3)
    raise NotImplementedError("self_ppl is produced in P3 (needs the model on the GPU pod); "
                              "it is a NLA-free detector AND a collapse component (QUESTIONS s4.6).")


# ============================ fit + score (CLI) ==============================

def fit_benign(persist: bool = True, cohort: np.ndarray | None = None,
               out_path=None, fit_split: str = "benign_trainval_disjoint") -> dict:
    """Fit LW Maha (+ PCA-whiten) on a benign cohort. Stage-1 default = benign a0 trainval (disjoint
    from the test eval subset). Stage-2 (R2) passes `cohort` = the CLEAN LAST-TOKEN benign_calib acts
    so the make-or-break H6 baseline is REGIME-MATCHED (the answer-cue a0 fit is the wrong
    distribution; PLAN_stage2 s5 R2) and writes to `out_path` (maha_fit_gen.npz)."""
    if cohort is None:
        h = C.load_benign_a0()
        Xb = np.ascontiguousarray(h[C.fit_split_mask()])
    else:
        Xb = np.ascontiguousarray(np.asarray(cohort, np.float64))
    out_path = out_path or C.PATHS.maha_fit()
    lw = ledoit_wolf(Xb)
    k = min(int(C.CONFIG["detectors"]["mahalanobis"]["pca_whiten_k"]), Xb.shape[0] - 1, Xb.shape[1])
    pcaw = fit_pca_whiten(Xb, k)
    manifest = {"schema_version": "ome_gauge.maha_fit.v1", "config_hash": C.CONFIG_HASH,
                "estimator": "ledoit_wolf", "fit_split": fit_split,
                "n_fit": lw["n"], "d": lw["d"], "shrinkage": lw["shrinkage"], "mu_eig": lw["mu_eig"],
                "pca_whiten_k": k}
    if persist:
        out = C.PATHS.dir_detect(); out.mkdir(parents=True, exist_ok=True)
        tmp = out / (out_path.name + ".tmp.npz")
        np.savez(tmp, mu=lw["mu"], sigma_inv=lw["sigma_inv"],
                 pca_mu=pcaw["mu"], pca_components=pcaw["components"], pca_eigenvalues=pcaw["eigenvalues"])
        import os
        os.replace(tmp, out_path)
        features.write_json_atomic(manifest, out / (out_path.stem + "_manifest.json"))
        print(f"[P2a] LW fit ({fit_split}): n={lw['n']} d={lw['d']} shrinkage={lw['shrinkage']:.4f} "
              f"-> {out_path.name} (+ PCA-whiten k={k})")
    return {"lw": lw, "pca": pcaw, "manifest": manifest}


def fit_benign_gen() -> dict:
    """S2.P2: regime-matched LW Maha (+ PCA-whiten + kNN cohort) on the CLEAN LAST-TOKEN benign_calib
    acts -> maha_fit_gen.npz. The fair, strong H6 baseline for the contentful regime (R2)."""
    cohort = np.load(C.PATHS.h_clean("benign_calib")).astype(np.float64)
    return fit_benign(persist=True, cohort=cohort, out_path=C.PATHS.maha_fit_gen(),
                      fit_split="regime_matched_benign_calib_clean_lasttoken")


def load_fit(path=None) -> tuple[dict, dict]:
    z = np.load(path or C.PATHS.maha_fit())
    lw = {"mu": z["mu"], "sigma_inv": z["sigma_inv"]}
    pca = {"mu": z["pca_mu"], "components": z["pca_components"], "eigenvalues": z["pca_eigenvalues"]}
    return lw, pca


def score_steered() -> int:
    """Compute the NLA-free detectors on every P1 steered array -> out/ome/ome/detectors.parquet.
    Maha/PCA-whiten use the persisted benign fit; ratio/act_norm are per-array. CPU."""
    import pyarrow as pa
    man = json.loads((C.PATHS.dir_steer() / "steer_manifest.json").read_text(encoding="utf-8"))
    lw, pca = load_fit()
    base = C.load_benign_a0()
    rows, ids = C.load_paired_subset()
    base_sub = np.asarray(base[rows], np.float64)
    cohort = np.ascontiguousarray(base[C.fit_split_mask()])   # benign cohort for kNN
    knn_k = int(C.CONFIG["detectors"]["knn_k"])

    cols = {k: [] for k in ("method", "dir", "alpha", "row_index", "example_id",
                            "ratio", "act_norm", "mahalanobis", "knn_dist", "pca_whiten")}
    for am in man["arrays"]:
        hp = np.load(C.PATHS.dir_steer() / am["path"]).astype(np.float64)
        det = {"ratio": ratio(hp, base_sub), "act_norm": act_norm(hp),
               "mahalanobis": mahalanobis(hp, lw), "knn_dist": knn_dist(hp, cohort, knn_k),
               "pca_whiten": pca_whiten_dist(hp, pca)}
        for j, r in enumerate(rows):
            cols["method"].append(am["method"]); cols["dir"].append(am["dir"])
            cols["alpha"].append(float(am["alpha"])); cols["row_index"].append(int(r))
            cols["example_id"].append(ids[j])
            for key in ("ratio", "act_norm", "mahalanobis", "knn_dist", "pca_whiten"):
                cols[key].append(float(det[key][j]))
    features.write_parquet_atomic(pa.table(cols), C.PATHS.dir_detect() / "detectors.parquet")
    print(f"[P2a] detectors.parquet: {len(cols['method'])} rows over {len(man['arrays'])} arrays")
    return 0


def score_gen(ns=None) -> int:
    """S2.P2 / S3.P3: the NLA-free detectors on every entering-state array (the manifest) ->
    detectors_gen.parquet (ns=None, steering arm) or detectors_ft.parquet (ns='ft', fine-tune arm).
    REGIME-MATCHED + base-manifold reference: Maha/PCA-whiten use maha_fit_gen (the benign_calib clean
    fit); kNN cohort = benign_calib clean acts; ratio is vs the per-set clean BASE h_clean_<set> —
    all of which stay the base model's even for the FT arm (so the FT activations are scored against
    the base reference; PLAN_stage3 §4.2). Rows = prompts. The real-label H6 competitors."""
    import pyarrow as pa
    P = C.gen_ns(ns)
    man = json.loads(P["manifest"].read_text(encoding="utf-8"))
    lw, pca = load_fit(C.PATHS.maha_fit_gen())
    cohort = np.load(C.PATHS.h_clean("benign_calib")).astype(np.float64)   # benign kNN cohort (base)
    knn_k = int(C.CONFIG["detectors"]["knn_k"])
    base_by_set: dict[str, np.ndarray] = {}
    ids_by_set: dict[str, list] = {}
    cols = {k: [] for k in ("method", "dir", "set", "alpha", "row_index", "example_id",
                            "ratio", "act_norm", "mahalanobis", "knn_dist", "pca_whiten")}
    for am in man["arrays"]:
        s = am["set"]
        if s not in base_by_set:
            base_by_set[s] = np.load(C.PATHS.h_clean(s)).astype(np.float64)   # base per-set clean (ratio ref)
            ids_by_set[s] = json.loads(C.PATHS.clean_manifest(s).read_text(encoding="utf-8"))["prompt_ids"]
        base, pids = base_by_set[s], ids_by_set[s]
        hp = np.load(P["arrays"] / am["path"]).astype(np.float64)
        if hp.shape[0] != len(pids):                         # capped harvest vs uncapped clean manifest:
            raise SystemExit(                                # a length desync would silently mislabel detector rows.
                f"[detectors] ns={ns!r} row desync {am['dir']}:{s}: {hp.shape[0]} harvested acts vs "
                f"{len(pids)} clean prompt_ids — re-harvest with n_per_set>=|{s}| so the arrays row-align.")
        det = {"ratio": ratio(hp, base), "act_norm": act_norm(hp),
               "mahalanobis": mahalanobis(hp, lw), "knn_dist": knn_dist(hp, cohort, knn_k),
               "pca_whiten": pca_whiten_dist(hp, pca)}
        for j in range(hp.shape[0]):
            cols["method"].append(am["method"]); cols["dir"].append(am["dir"]); cols["set"].append(s)
            cols["alpha"].append(float(am["alpha"])); cols["row_index"].append(int(j))
            cols["example_id"].append(pids[j])
            for key in ("ratio", "act_norm", "mahalanobis", "knn_dist", "pca_whiten"):
                cols[key].append(float(det[key][j]))
    features.write_parquet_atomic(pa.table(cols), P["detectors"])
    print(f"[{'S3.P3' if ns else 'S2.P2'}] {P['detectors'].name}: {len(cols['method'])} rows over "
          f"{len(man['arrays'])} entering-state arrays (regime-matched, base-manifold fit)")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="NLA-free off-manifold detectors (H6 baselines).")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("fit-benign"); sub.add_parser("score")
    sub.add_parser("fit-benign-gen")                                     # S2.P2 regime-matched fit
    sg = sub.add_parser("score-gen")                                     # S2.P2 / S3.P3 score
    sg.add_argument("--ns", default=None, choices=["ft"], help="'ft' scores the Stage-3 FT harvest")
    args = ap.parse_args(argv)
    return {"fit-benign": lambda: (fit_benign(), 0)[1], "score": score_steered,
            "fit-benign-gen": lambda: (fit_benign_gen(), 0)[1],
            "score-gen": lambda: score_gen(ns=getattr(args, "ns", None))}[args.cmd]()


if __name__ == "__main__":
    raise SystemExit(main())
