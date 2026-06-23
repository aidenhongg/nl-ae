"""P1 — steered L20 activation generation (CPU, $0).

Materializes the steered activations for every (method, direction, alpha) over the eval rows.
The genuinely new operator is **additive difference-in-means**: h' = h + alpha*v_hat (the
canonical steering-vector edit, CAA / DESIGN s5.1) — distinct from the parent lang_steer's
*replacement* patch. The KAPPA arm reuses src/steer_sweep verbatim as a continuity bridge; for
Stage 1 the KAPPA frontier is taken as the free overlay from out/fve/analysis.json (no recompute).

Outputs (off-repo / S3 by .gitignore, except the *.json manifest):
  out/ome/steer/h_steer_<method>_<dir>_a<tag>.npy        steered [n_rows, d] fp32
  out/ome/steer/norms_ome.parquet                        per (row,method,dir,alpha) diagnostics
  out/ome/steer/steer_manifest.json                      config hash + array SHAs + gate results

Gate P1 (exact, CPU): alpha==0 is identity (dh_norm==0); for additive steering ||dh||==alpha for
every row (the same unit vector is added to all rows) — this exact invariant is the DiM validation
anchor that replaces KAPPA's a2/a10 allclose (which does not apply to an additive edit; QUESTIONS
s3.7). The *behavioral* smoke (e.g. D_correct readout inverted-U) is pod-gated (P3) and noted there.

CLI:  python -m ome_gauge.steer_dim gen --method dim --dir D_correct --alphas 0,1,5,10
      python -m ome_gauge.steer_dim gen-all            # every Stage-1 (method,dir) over the alpha grid
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from ome_gauge import config as C
from src import steer_sweep, lang_steer, features  # parent reuse: atomic_save_npy, sha256_file, ratio

NORMS_COLS = ["row_index", "example_id", "method", "dir", "alpha",
              "h_norm", "dh_norm", "ratio", "cos_h_hp"]


# ----------------------------- operators ------------------------------------

def steer_additive(h: np.ndarray, v_unit: np.ndarray, alpha: float) -> np.ndarray:
    """DiM / random additive edit: h' = h + alpha*v_hat. v_unit is a unit [d] vector."""
    return h + np.float64(alpha) * v_unit


def steer_kappa(h0: np.ndarray, alpha: float):
    """Continuity arm: the closed-form KAPPA residual edit, delegated to src/steer_sweep (the
    validated production operator). h0 must be the FULL benign cohort rows (probe geometry is fit
    on all rows). Returned h' is fp64. Stage-1 normally reuses the pre-computed KAPPA arrays +
    out/fve/analysis.json overlay instead of calling this; kept for parity/regeneration."""
    Wk, bk = features.load_probe(C.PATHS.feat, "example_level", "know", C.LAYER)
    Wp, bp = features.load_probe(C.PATHS.feat, "example_level", "pred", C.LAYER)
    P, *_ = steer_sweep.build_P(Wp)
    return h0 if alpha == 0.0 else steer_sweep.steer(h0, alpha, Wk, bk, Wp, bp, P)


def load_direction(name: str) -> np.ndarray:
    z = np.load(C.PATHS.dirs_npz())
    if name not in z.files:
        raise KeyError(f"{name} not in dirs.npz ({z.files}); run `directions build` first")
    return z[name].astype(np.float64)


# ----------------------------- diagnostics ----------------------------------

def norm_diagnostics(base: np.ndarray, hp: np.ndarray) -> dict:
    """Per-row {h_norm, dh_norm, ratio, cos_h_hp} for the norms parquet (mirrors steer_sweep)."""
    h_norm = np.linalg.norm(base, axis=1)
    dh = hp - base
    dh_norm = np.linalg.norm(dh, axis=1)
    ratio = lang_steer.ratio_offmanifold(hp, base)          # ||dh||/||h||
    hp_norm = np.linalg.norm(hp, axis=1)
    denom = h_norm * hp_norm
    cos = np.divide((base * hp).sum(1), denom, out=np.ones_like(dh_norm), where=denom > 0)
    return {"h_norm": h_norm, "dh_norm": dh_norm, "ratio": ratio, "cos_h_hp": cos}


def _gate(method: str, alpha: float, diag: dict) -> None:
    if alpha == 0.0:
        assert float(diag["dh_norm"].max()) == 0.0, "alpha=0 must be identity (dh_norm==0)"
    elif method in ("dim", "random"):
        # additive invariant: the same unit vector is added to every row -> ||dh|| == alpha exactly
        rel = np.abs(diag["dh_norm"] - alpha) / max(alpha, 1e-12)
        assert float(rel.max()) < 1e-3, \
            f"additive gate: ||dh|| != alpha ({method} a={alpha}, max rel {float(rel.max()):.2e})"


# ----------------------------- generate -------------------------------------

def gen(method: str, dir_name: str, alphas, rows=None, example_ids=None) -> tuple[list[dict], list[dict]]:
    """Generate steered arrays for one (method, dir) over `alphas`; returns (norm_rows, arrays_meta).
    Writes one .npy per alpha (atomic). rows default = the paired subset (nests the OME set)."""
    if rows is None:
        rows, example_ids = C.load_paired_subset()
    if example_ids is None:
        ids_all = C.canonical_ids()                  # row r -> example id (always correct)
        example_ids = [ids_all[int(r)] for r in rows]

    h_full = C.load_benign_a0()                              # [N,d] fp64
    base = np.asarray(h_full[rows], dtype=np.float64)        # [n,d]
    v = None if method == "kappa" else load_direction(dir_name)

    out_dir = C.PATHS.dir_steer(); out_dir.mkdir(parents=True, exist_ok=True)
    norm_rows: list[dict] = []
    arrays_meta: list[dict] = []
    for a in alphas:
        a = float(a)
        if method in ("dim", "random"):
            hp = steer_additive(base, v, a)
        elif method == "kappa":
            hp = np.asarray(steer_kappa(h_full, a), dtype=np.float64)[rows]
        else:
            raise ValueError(f"unknown method {method!r}")
        diag = norm_diagnostics(base, hp)
        _gate(method, a, diag)

        tag = steer_sweep.alpha_tag(a)
        fpath = out_dir / f"h_steer_{method}_{dir_name}_a{tag}.npy"
        steer_sweep.atomic_save_npy(fpath, hp.astype(np.float32))
        arrays_meta.append({"method": method, "dir": dir_name, "alpha": a, "tag": tag,
                            "path": fpath.name, "shape": list(hp.shape),
                            "sha256": steer_sweep.sha256_file(fpath),
                            "mean_ratio": float(diag["ratio"].mean()),
                            "mean_dh_norm": float(diag["dh_norm"].mean())})
        for j, r in enumerate(rows):
            norm_rows.append({"row_index": int(r), "example_id": example_ids[j],
                              "method": method, "dir": dir_name, "alpha": a,
                              "h_norm": float(diag["h_norm"][j]),
                              "dh_norm": float(diag["dh_norm"][j]),
                              "ratio": float(diag["ratio"][j]),
                              "cos_h_hp": float(diag["cos_h_hp"][j])})
    return norm_rows, arrays_meta


def _write_norms(rows: list[dict], path=None) -> None:
    import pyarrow as pa
    cols = {k: [r[k] for r in rows] for k in NORMS_COLS}
    features.write_parquet_atomic(pa.table(cols), path or (C.PATHS.dir_steer() / "norms_ome.parquet"))


def gen_all(pairs=None) -> int:
    """Generate every Stage-1 (method, dir): DiM/random arms over the locked alpha grid.
    The KAPPA arm is the free overlay (out/fve/analysis.json) and is not regenerated here."""
    if pairs is None:
        randoms = [d for d in C.CONFIG["directions"]["stage1"] if d.startswith("D_random_")]
        pairs = [("dim", "D_correct")] + [("random", r) for r in randoms]
    rows, ids = C.load_paired_subset()
    all_rows, arrays = [], []
    for method, dname in pairs:
        alphas = C.alphas_for(method)
        nr, am = gen(method, dname, alphas, rows=rows, example_ids=ids)
        all_rows += nr; arrays += am
        print(f"[P1] {method}:{dname} -> {len(alphas)} arrays "
              f"(top alpha {alphas[-1]:g} -> mean ratio {am[-1]['mean_ratio']:.3f})")
    _write_norms(all_rows)
    manifest = {"schema_version": "ome_gauge.steer.v1", "config_hash": C.CONFIG_HASH,
                "model_id": C.CONFIG["model_id"], "layer": C.LAYER, "n_rows": len(rows),
                "alphas_additive": C.ALPHAS_ADDITIVE, "alphas_kappa": C.ALPHAS,
                "pairs": pairs, "arrays": arrays}
    features.write_json_atomic(manifest, C.PATHS.dir_steer() / "steer_manifest.json")
    print(f"[P1] wrote {len(arrays)} arrays + norms_ome.parquet ({len(all_rows)} rows) + manifest")
    print("[P1] GATE PASS: alpha=0 identity + additive ||dh||==alpha invariant on all arms.")
    return 0


def cmd_gen(args) -> int:
    alphas = [float(x) for x in args.alphas.split(",")] if args.alphas else C.ALPHAS
    rows, _arrays = gen(args.method, args.dir, alphas)
    _write_norms(rows)
    print(f"[P1] {args.method}:{args.dir} -> {len(alphas)} arrays + norms_ome.parquet")
    return 0


# ============================================================================
#  S2.P1 — analytic entering states (the OME inputs; CPU, the key simplification)
# ============================================================================
# The all-position hook adds exactly alpha*v_hat at pos_last during prefill, so the steered state
# ENTERING generation IS h_clean_lasttoken + alpha*v_hat -> the primary OME needs no capture hook
# during generation (PLAN_stage2 s5.2). Built from the S2.P1 clean harvest (directions.harvest_clean)
# with the operator reused verbatim; the dangerous sign is folded into v so a positive swept alpha
# is always the dangerous push.

def gen_entering(method: str, dir_name: str, set_name: str, alphas, *,
                 base=None, prompt_ids=None, v=None) -> tuple[list[dict], list[dict]]:
    """Build the analytic entering states for one (dir, prompt set) over `alphas`: h_enter =
    h_clean + alpha*v_signed. Writes one h_enter_<dir>_<set>_a<tag>.npy per alpha; returns
    (norm_rows, arrays_meta). `base` defaults to the harvested h_clean_<set>.npy (+ its prompt-id
    manifest); `v` defaults to the (dangerously-signed for content dirs) stored direction."""
    if base is None:
        base = np.load(C.PATHS.h_clean(set_name)).astype(np.float64)
        man = json.loads(C.PATHS.clean_manifest(set_name).read_text(encoding="utf-8"))
        prompt_ids = man["prompt_ids"]
    base = np.asarray(base, np.float64)
    assert prompt_ids is not None and len(prompt_ids) == base.shape[0], "prompt_ids must align to base rows"
    if v is None:
        raw = load_direction(dir_name)
        v = C.dangerous_signed_dir(dir_name, raw) if dir_name in C.CONTENT_DIRECTIONS else raw
    out_dir = C.PATHS.dir_steer(); out_dir.mkdir(parents=True, exist_ok=True)
    norm_rows: list[dict] = []
    arrays_meta: list[dict] = []
    for a in alphas:
        a = float(a)
        hp = steer_additive(base, v, a)
        diag = norm_diagnostics(base, hp)
        _gate(method, a, diag)                       # additive invariant ||dh||==alpha (+ a=0 identity)
        tag = steer_sweep.alpha_tag(a)
        fpath = C.PATHS.h_enter(dir_name, set_name, tag)
        steer_sweep.atomic_save_npy(fpath, hp.astype(np.float32))
        arrays_meta.append({"method": method, "dir": dir_name, "set": set_name, "alpha": a, "tag": tag,
                            "path": fpath.name, "shape": list(hp.shape),
                            "sha256": steer_sweep.sha256_file(fpath),
                            "mean_ratio": float(diag["ratio"].mean()),
                            "mean_dh_norm": float(diag["dh_norm"].mean()),
                            "dangerous_sign": int(C.DANGEROUS_SIGN.get(dir_name, 1))})
        for j, pid in enumerate(prompt_ids):
            norm_rows.append({"row_index": int(j), "example_id": str(pid),
                              "method": method, "dir": dir_name, "alpha": a,
                              "h_norm": float(diag["h_norm"][j]), "dh_norm": float(diag["dh_norm"][j]),
                              "ratio": float(diag["ratio"][j]), "cos_h_hp": float(diag["cos_h_hp"][j])})
    return norm_rows, arrays_meta


def gen_entering_all(pairs=None, sets=("em", "neutral"), alphas=None) -> int:
    """All Stage-2 entering states: each content dir (method=dim, dangerously signed) + the random
    control (method=random) over the EM + neutral prompt sets -> h_enter arrays + norms_ome_gen.parquet
    + steer_manifest_gen.json (regime=generate; the input to ome_probe.sweep + the regime-matched
    detectors). benign_calib is NOT steered (it is the calibration/cohort set)."""
    alphas = list(alphas) if alphas is not None else C.ALPHAS_ADDITIVE
    if pairs is None:
        pairs = [("dim", d) for d in C.CONTENT_DIRECTIONS] + [("random", "D_random_0")]
    all_rows, arrays = [], []
    for set_name in sets:
        for method, dname in pairs:
            nr, am = gen_entering(method, dname, set_name, alphas)
            all_rows += nr; arrays += am
            print(f"[S2.P1] {method}:{dname} on {set_name} -> {len(alphas)} entering-state arrays "
                  f"(top a{alphas[-1]:g} -> mean ratio {am[-1]['mean_ratio']:.3f}, "
                  f"sign {am[-1]['dangerous_sign']:+d})")
    _write_norms(all_rows, C.PATHS.dir_steer() / "norms_ome_gen.parquet")
    manifest = {"schema_version": "ome_gauge.steer_gen.v1", "config_hash": C.CONFIG_HASH,
                "regime": "generate", "model_id": C.CONFIG["model_id"], "layer": C.LAYER,
                "sets": list(sets), "alphas_additive": alphas, "pairs": pairs, "arrays": arrays}
    features.write_json_atomic(manifest, C.PATHS.steer_manifest_gen())
    print(f"[S2.P1] wrote {len(arrays)} entering-state arrays + norms_ome_gen.parquet "
          f"({len(all_rows)} rows) + steer_manifest_gen.json (regime=generate)")
    print("[S2.P1] GATE PASS: alpha=0 identity + additive ||dh||==alpha on all entering states.")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="P1/S2.P1 OME-GAUGE steered activation generator.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("gen")
    g.add_argument("--method", required=True, choices=["dim", "random", "kappa"])
    g.add_argument("--dir", required=True)
    g.add_argument("--alphas", default="")
    sub.add_parser("gen-all")
    ge = sub.add_parser("gen-enter-all")        # S2.P1 analytic entering states (h_clean + alpha*v)
    ge.add_argument("--sets", default="em,neutral")
    args = ap.parse_args(argv)
    return {"gen": cmd_gen, "gen-all": lambda a: gen_all(),
            "gen-enter-all": lambda a: gen_entering_all(sets=tuple(s for s in a.sets.split(",") if s))
            }[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
