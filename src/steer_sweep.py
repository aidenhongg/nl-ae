"""Phase A — steered L20 activation alpha-sweep (local / CPU; no GPU, no S3).

Reimplements exp04's closed-form KAPPA residual edit (kappa/kappa_edit.py:
build_edit_operator + apply_edit_real), faithful to the production run, which used
the pseudo-inverse: the example_level L20 prediction-probe Gram is rank 3 with
cond ~= 2e16 > cond_max=1e6, so used_pinv=True (see emb_manifest.json:edit).

Generates h'(alpha) for the locked alpha grid, then HARD-validates against the
stored production arrays h_layer20_steered_a2/a10.npy (allclose) and the recorded
norm diagnostics. A mismatch means probe/orientation/pinv drift -> STOP. Emits
per-(row,alpha) norm diagnostics and the paired 1024-row subset for the GPU
round-trip.

Edit operator (all fp64):
    z = h @ Wk.T + bk            # knowledge logits   [N,4]
    s = h @ Wp.T + bp            # prediction logits  [N,4]
    r = alpha*z + beta*sign(z) - s   (beta = 0)
    G = Wp @ Wp.T                # [4,4] non-augmented Gram (rank 3 here)
    P = Wp.T @ pinv(G)           # [d,4]
    h'(alpha) = h + r @ P.T
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

# ---- locked constants (plan 00 §5.2, 01 §A) --------------------------------
ALPHAS: list[float] = [0.0, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0, 20.0, 30.0]
BETA = 0.0
COND_MAX = 1e6
K = 4
D_MODEL = 3584
N_ROWS = 6536
SUBSET_N = 1024
SUBSET_SEED = 7  # exp04 seed (provenance lock)
CONFIG_HASH = "e80501525b6758e8a7c6f28556541bbbad1f268f92ae187972f83e69c075a55f"

# validation targets recorded by the production run (emb_manifest.json)
EXPECT = {
    "mean_h_norm": 86.6916,
    "a2": {"mean_dh_norm": 10.4462, "mean_rel_edit": 0.1203},
    "a10": {"mean_dh_norm": 57.9325, "mean_rel_edit": 0.6663},
}


def alpha_tag(a: float) -> str:
    return ("%g" % a).replace(".", "p").replace("-", "m")


def atomic_save_npy(path: Path, arr: np.ndarray) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "wb") as f:  # file handle => np.save won't append ".npy"
        np.save(f, arr)
    os.replace(tmp, path)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_P(Wp: np.ndarray, cond_max: float = COND_MAX):
    G = Wp @ Wp.T  # [4,4] fp64
    cond = float(np.linalg.cond(G))
    rank = int(np.linalg.matrix_rank(G))
    used_pinv = rank < K or cond > cond_max
    Ginv = np.linalg.pinv(G) if used_pinv else np.linalg.inv(G)
    P = Wp.T @ Ginv  # [d,4]
    return P, cond, rank, used_pinv


def steer(h: np.ndarray, alpha: float, Wk, bk, Wp, bp, P) -> np.ndarray:
    """fp64 reference edit, h:[N,d] -> h':[N,d]."""
    z = h @ Wk.T + bk
    s = h @ Wp.T + bp
    r = alpha * z + BETA * np.sign(z) - s
    return h + r @ P.T


def main() -> None:
    ap = argparse.ArgumentParser()
    here = Path(__file__).resolve()
    nla_root = here.parent.parent
    ap.add_argument("--exp04", type=Path,
                    default=nla_root.parent / "exp04" / "05_out_pulled",
                    help="exp04 local mirror root (contains 02_probes, 03_kappa, ...)")
    ap.add_argument("--out", type=Path, default=nla_root / "inputs",
                    help="local output dir for steered arrays + diagnostics")
    args = ap.parse_args()

    exp04: Path = args.exp04
    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)

    emb = exp04 / "03_kappa" / "emb"
    probe_dir = exp04 / "02_probes" / "example_level"

    print(f"[A] exp04 mirror: {exp04}")
    print(f"[A] output dir:   {out}")

    # ---- load inputs (fp32 -> fp64; probes fp64) ---------------------------
    h0 = np.load(emb / "h_layer20_orig.npy").astype(np.float64)  # (N,d)
    assert h0.shape == (N_ROWS, D_MODEL), f"orig shape {h0.shape}"
    zk = np.load(probe_dir / "know" / "layer20.npz")
    zp = np.load(probe_dir / "pred" / "layer20.npz")
    Wk, bk = zk["W"].astype(np.float64), zk["b"].astype(np.float64)
    Wp, bp = zp["W"].astype(np.float64), zp["b"].astype(np.float64)
    assert Wk.shape == (K, D_MODEL) and Wp.shape == (K, D_MODEL)
    assert bk.shape == (K,) and bp.shape == (K,)

    ids_json = json.loads((emb / "example_ids.json").read_text(encoding="utf-8"))
    example_ids = ids_json["example_ids"]
    test_rows = np.asarray(ids_json["test_row_indices"], dtype=np.int64)
    assert len(example_ids) == N_ROWS and ids_json["n"] == N_ROWS
    is_test = np.zeros(N_ROWS, dtype=bool)
    is_test[test_rows] = True

    P, cond, rank, used_pinv = build_P(Wp)
    print(f"[A] Gram: rank={rank} cond={cond:.3e} used_pinv={used_pinv} "
          f"(expect rank=3, pinv=True)")

    h0_norms = np.linalg.norm(h0, axis=1)  # [N]
    print(f"[A] mean_h_norm={h0_norms.mean():.4f} (expect {EXPECT['mean_h_norm']})")

    # ---- generate sweep + per-(row,alpha) diagnostics ----------------------
    norm_rows: dict[str, list] = {k: [] for k in
                                  ("example_id", "row_index", "split", "in_subset",
                                   "alpha", "h_norm", "dh_norm", "ratio", "cos_h_hp")}
    array_meta: list[dict] = []
    validation: dict = {}

    # paired subset R: 1024 test rows, fixed seed, sorted (canonical order)
    rng = np.random.default_rng(SUBSET_SEED)
    subset = np.sort(rng.choice(test_rows, size=SUBSET_N, replace=False)).astype(np.int64)
    in_subset = np.zeros(N_ROWS, dtype=bool)
    in_subset[subset] = True

    for a in ALPHAS:
        tag = alpha_tag(a)
        # alpha=0 is the no-edit on-manifold anchor (== h_orig), per plan semantics
        # (identity, ||dh||=0; FVE(0)=FVE(orig); a=0 NL reuses orig text). NOTE the
        # literal closed-form at a=0 is r=-s -> h-s@P.T (it zeroes the prediction
        # logits), which is NOT the intended baseline; we apply the edit only for a>0.
        hp = h0.copy() if a == 0.0 else steer(h0, a, Wk, bk, Wp, bp, P)  # fp64
        dh = hp - h0
        dh_norm = np.linalg.norm(dh, axis=1)
        ratio = np.divide(dh_norm, h0_norms, out=np.zeros_like(dh_norm),
                          where=h0_norms > 0)
        # cos(h, h'); for a=0 this is exactly 1
        hp_norm = np.linalg.norm(hp, axis=1)
        denom = h0_norms * hp_norm
        cos_hhp = np.divide((h0 * hp).sum(1), denom,
                            out=np.ones_like(dh_norm), where=denom > 0)

        if a == 0.0:
            assert np.array_equal(hp.astype(np.float32), h0.astype(np.float32)) or \
                np.allclose(hp, h0, atol=0, rtol=0), "alpha=0 must be identity"
            assert dh_norm.max() == 0.0, "alpha=0 dh must be 0"

        arr_f32 = hp.astype(np.float32)
        fpath = out / f"h_layer20_steered_a{tag}.npy"
        atomic_save_npy(fpath, arr_f32)
        array_meta.append({
            "alpha": a, "tag": tag, "path": fpath.name,
            "shape": list(arr_f32.shape), "dtype": "float32",
            "sha256": sha256_file(fpath),
            "mean_dh_norm": float(dh_norm.mean()),
            "mean_rel_edit": float(ratio.mean()),
            "mean_rel_edit_test": float(ratio[is_test].mean()),
            "max_rel_edit": float(ratio.max()),
        })

        for i in range(N_ROWS):
            norm_rows["example_id"].append(example_ids[i])
            norm_rows["row_index"].append(int(i))
            norm_rows["split"].append("test" if is_test[i] else "trainval")
            norm_rows["in_subset"].append(bool(in_subset[i]))
            norm_rows["alpha"].append(float(a))
            norm_rows["h_norm"].append(float(h0_norms[i]))
            norm_rows["dh_norm"].append(float(dh_norm[i]))
            norm_rows["ratio"].append(float(ratio[i]))
            norm_rows["cos_h_hp"].append(float(cos_hhp[i]))

        # ---- HARD validation gate vs stored production arrays --------------
        if a in (2.0, 10.0):
            key = "a2" if a == 2.0 else "a10"
            ref = np.load(emb / f"h_layer20_steered_{key}.npy")  # fp32
            ok = np.allclose(arr_f32, ref, atol=1e-3, rtol=1e-3)
            max_abs = float(np.max(np.abs(arr_f32.astype(np.float64) - ref.astype(np.float64))))
            exp_dh = EXPECT[key]["mean_dh_norm"]
            exp_rel = EXPECT[key]["mean_rel_edit"]
            got_dh = float(dh_norm.mean())
            got_rel = float(ratio.mean())
            validation[key] = {
                "allclose": bool(ok), "max_abs_diff": max_abs,
                "mean_dh_norm": got_dh, "expect_dh_norm": exp_dh,
                "mean_rel_edit": got_rel, "expect_rel_edit": exp_rel,
            }
            print(f"[A] VALIDATE {key}: allclose={ok} max_abs={max_abs:.2e} "
                  f"dh={got_dh:.4f}(exp {exp_dh}) rel={got_rel:.4f}(exp {exp_rel})")
            if not ok:
                raise AssertionError(
                    f"[A] GATE FAIL: regenerated alpha={a} does not match stored {key} "
                    f"(max_abs={max_abs:.2e}). Probe/orientation/pinv drift -> STOP.")
            assert abs(got_dh - exp_dh) < 0.02, f"{key} dh_norm drift {got_dh} vs {exp_dh}"
            assert abs(got_rel - exp_rel) < 0.005, f"{key} rel_edit drift {got_rel} vs {exp_rel}"
        print(f"[A] alpha={a:<5g} tag=a{tag:<4} dh_mean={dh_norm.mean():.4f} "
              f"ratio_mean={ratio.mean():.4f} -> {fpath.name}")

    if "a2" not in validation or "a10" not in validation:
        raise AssertionError("[A] validation gate did not run for a2 and a10")

    # ---- write norms.parquet ----------------------------------------------
    schema = pa.schema([
        ("example_id", pa.string()), ("row_index", pa.int32()),
        ("split", pa.string()), ("in_subset", pa.bool_()),
        ("alpha", pa.float64()), ("h_norm", pa.float64()),
        ("dh_norm", pa.float64()), ("ratio", pa.float64()),
        ("cos_h_hp", pa.float64()),
    ])
    pq.write_table(pa.table(norm_rows, schema=schema), out / "norms.parquet")
    print(f"[A] wrote norms.parquet ({len(norm_rows['alpha'])} rows)")

    # ---- write subset_rows.json -------------------------------------------
    (out / "subset_rows.json").write_text(json.dumps({
        "order": "activation_cache_row_order",
        "n_subset": int(SUBSET_N), "seed": SUBSET_SEED,
        "drawn_from": "example_level/test", "n_test": int(len(test_rows)),
        "row_indices": subset.tolist(),
        "example_ids": [example_ids[i] for i in subset],
    }, indent=2), encoding="utf-8")
    print(f"[A] wrote subset_rows.json ({SUBSET_N} rows from {len(test_rows)} test)")

    # ---- write steer_manifest.json (provenance + gate result) -------------
    (out / "steer_manifest.json").write_text(json.dumps({
        "schema_version": "nla_steer.v1",
        "config_hash": CONFIG_HASH,
        "model_id": "Qwen/Qwen2.5-7B-Instruct",
        "scheme": "example_level", "layer": 20, "mode": "single",
        "alphas": ALPHAS, "beta": BETA, "cond_max": COND_MAX,
        "gram": {"rank": rank, "cond": cond, "used_pinv": used_pinv},
        "n_examples": N_ROWS, "d_model": D_MODEL,
        "mean_h_norm": float(h0_norms.mean()),
        "subset": {"n": SUBSET_N, "seed": SUBSET_SEED, "source": "example_level/test"},
        "validation": validation,
        "arrays": array_meta,
        "source_orig": "exp04/03_kappa/emb/h_layer20_orig.npy",
    }, indent=2), encoding="utf-8")

    print("\n[A] DONE. Validation gate GREEN (a2 + a10 reproduced; norms match).")
    print(f"[A] {len(ALPHAS)} steered arrays + norms.parquet + subset_rows.json + "
          f"steer_manifest.json in {out}")


if __name__ == "__main__":
    main()
