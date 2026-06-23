"""CPU unit tests for ome_gauge.detectors — the Ledoit-Wolf Mahalanobis math + the NLA-free
baselines on synthetic data (hermetic), plus a real-data benign fit sanity (uses a0 cache).

Run: python tests/test_detectors.py   (from the NLA-final root)
"""
import os
import sys

_SRC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _SRC)

import numpy as np

from ome_gauge import detectors as DET


def test_ledoit_wolf_properties():
    rng = np.random.default_rng(0)
    n, d = 300, 20
    # anisotropic benign cloud
    A = rng.standard_normal((d, d))
    X = rng.standard_normal((n, d)) @ A
    fit = DET.ledoit_wolf(X)
    assert 0.0 <= fit["shrinkage"] <= 1.0, fit["shrinkage"]
    # Sigma* is symmetric PD
    sig = fit["sigma"]
    assert np.allclose(sig, sig.T, atol=1e-8)
    assert np.linalg.eigvalsh(sig).min() > 0, "Sigma* must be PD"
    # Sigma_inv is a genuine inverse
    assert np.allclose(sig @ fit["sigma_inv"], np.eye(d), atol=1e-6)
    print(f"[det] Ledoit-Wolf PD + inverse + shrinkage in [0,1] ({fit['shrinkage']:.3f}) OK")


def test_ledoit_wolf_rank_deficient():
    """n < d -> raw cov singular; LW must still produce an invertible Sigma* (the whole point)."""
    rng = np.random.default_rng(1)
    n, d = 30, 64
    X = rng.standard_normal((n, d))
    fit = DET.ledoit_wolf(X)
    assert fit["shrinkage"] > 0.0, "rank-deficient cohort must pull shrinkage off 0"
    assert np.isfinite(fit["sigma_inv"]).all()
    assert np.allclose(fit["sigma"] @ fit["sigma_inv"], np.eye(d), atol=1e-5)
    print(f"[det] LW invertible under n<d (shrinkage {fit['shrinkage']:.3f}) OK")


def test_mahalanobis_ranks_anomalies():
    rng = np.random.default_rng(2)
    n, d = 500, 16
    scale = np.linspace(1.0, 5.0, d)
    X = rng.standard_normal((n, d)) * scale            # benign
    fit = DET.ledoit_wolf(X)
    inlier = np.zeros((1, d))                           # at the mean
    outlier = (10.0 * scale)[None, :]                   # far in high-variance dirs
    near = X[:50]                                       # benign held-in-distribution
    assert DET.mahalanobis(outlier, fit)[0] > DET.mahalanobis(inlier, fit)[0]
    assert DET.mahalanobis(outlier, fit)[0] > DET.mahalanobis(near, fit).mean()
    print("[det] mahalanobis ranks outlier > inlier > benign OK")


def test_pca_whiten_and_knn_and_norm():
    rng = np.random.default_rng(3)
    n, d = 400, 12
    X = rng.standard_normal((n, d)) * np.linspace(1, 4, d)
    pca = DET.fit_pca_whiten(X, k=6)
    assert pca["components"].shape == (6, d) and (pca["eigenvalues"] > 0).all()
    far = (20 * np.ones(d))[None, :]
    assert DET.pca_whiten_dist(far, pca)[0] > DET.pca_whiten_dist(X[:20], pca).mean()
    # kNN: a far point is lower-density (larger knn dist) than benign points
    assert DET.knn_dist(far, X, k=5)[0] > DET.knn_dist(X[:20], X, k=5).mean()
    # act_norm
    assert np.allclose(DET.act_norm(np.array([[3.0, 4.0]])), [5.0])
    print("[det] pca-whiten + kNN density + act_norm OK")


def test_ratio_matches_parent():
    h0 = np.array([[3.0, 4.0]]); hs = h0 * 2.0
    assert abs(float(DET.ratio(hs, h0)[0]) - 1.0) < 1e-9   # ||dh||/||h|| = 5/5
    print("[det] ratio == lang_steer.ratio_offmanifold OK")


def test_real_fit_benign_direction_aware():
    """Integration: LW fit on real benign trainval, then the H6-relevant direction-aware sanity at
    MATCHED ratio: a RANDOM push is hugely anomalous (it hits low-variance benign dims) while a push
    along the in-distribution D_correct capability axis barely moves Maha. This (a) proves the fit is
    wired + leak-free, and (b) shows Maha is a strong DIRECTION-AWARE baseline (the fair H6 bar, not
    the trivial magnitude proxy) — and that the science lives on structured dirs, not the random arm."""
    from ome_gauge import config as C
    from ome_gauge import steer_dim as S
    if not (C.PATHS.a0_npy().exists() and C.PATHS.dirs_npz().exists()):
        print("[det] (skip real fit — a0/dirs cache absent)"); return
    lw = DET.fit_benign(persist=False)["lw"]
    assert 0.0 < lw["shrinkage"] < 1.0
    h = C.load_benign_a0()
    base = h[np.where(C.test_mask())[0][:256]]           # benign rows held out of the fit
    benign = DET.mahalanobis(base, lw).mean()
    vc, vr = S.load_direction("D_correct"), S.load_direction("D_random_0")
    a = 175.0                                            # ratio ~ 2.0 (== KAPPA a30 collapse)
    maha_c = DET.mahalanobis(S.steer_additive(base, vc, a), lw).mean()
    maha_r = DET.mahalanobis(S.steer_additive(base, vr, a), lw).mean()
    assert maha_r > benign and maha_c > benign, "steering must raise Maha above benign"
    assert maha_r > 5.0 * maha_c, f"Maha must be direction-aware (random {maha_r:.0f} vs D_correct {maha_c:.0f})"
    print(f"[det] real LW fit OK (shrink {lw['shrinkage']:.3f}); direction-aware at ratio~2: "
          f"benign {benign:.0f}, D_correct {maha_c:.0f}, random {maha_r:.0f} ({maha_r/maha_c:.0f}x)")


def main() -> int:
    test_ledoit_wolf_properties()
    test_ledoit_wolf_rank_deficient()
    test_mahalanobis_ranks_anomalies()
    test_pca_whiten_and_knn_and_norm()
    test_ratio_matches_parent()
    test_real_fit_benign_direction_aware()
    print("\nDETECTORS CPU TESTS PASSED.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
