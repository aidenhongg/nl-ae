"""P2 — OME round-trip on the steered activations (GPU pod).

The candidate gauge: for every steered vector h', verbalize it (AV) and reconstruct (AR),
then record OME = 1 - cos(h', AR(AV(h'))). This is the costliest phase (the AV server is the
bottleneck, ~2.4 rows/s threaded); it reuses `nla_run.run_rows` *verbatim* — the same
resumable, threaded AV+AR ledger the parent KAPPA/NLA sweep used — so the only new code here is
the OME-GAUGE plumbing around it (which arrays, which subset, the av-coherence read-out, and the
compaction to `ome_by_cond.parquet`).

Critical correctness note (draft.md s4): the OME-GAUGE steered vector is a **raw additive push**
`h + alpha*v_hat`, NOT an AR output. So the "mildly favorable OME" caveat that `lang_steer.ome`
carries (its h is itself an AR reconstruction, on-manifold by construction) does **not** apply —
this is the correct apples-to-apples setup, matching the KAPPA sweep. Hence we verbalize the raw
steered arrays through `nla_run.run_rows`, never `lang_steer.ome`.

Gates (SPEC P2):
  * **calibration gate first** — FVE(orig)/mean_cos on benign a0 must sit in [0.6, 0.8]
    (`nla_run.CALIB_FVE_MIN/MAX`) or the NLA is mis-wired and every downstream OME is noise. STOP.
  * **AV-coherence floor** — the fraction of AV verbalizations that are coherent English (not
    CJK/garbage) is logged per condition; conditions below the floor are flagged **NLA-OOD**, so a
    high OME there is reported as "the AV never saw such vectors", not "the model collapsed"
    (DESIGN s11.1 / QUESTIONS s2.1).

CLI:  python -m ome_gauge.ome_probe calibrate     # P2b: FVE(orig) gate on benign a0 (STOP on fail)
      python -m ome_gauge.ome_probe sweep         # P2c: AV round-trip over the P1 arrays -> ledger
      python -m ome_gauge.ome_probe compact       # ledger -> ome_by_cond.parquet (CPU; auto after sweep)
"""
from __future__ import annotations

import argparse
import json
import sys

import numpy as np

from ome_gauge import config as C
from src import features, nla_run   # parent reuse: run_rows, make_clients, fve_fixed, gates


# ----------------------------- AV-coherence read-out -------------------------
# A cheap, deterministic CJK/garbage detector on the AV verbalization. The NLA was trained to
# emit English <explanation> snippets; when it is pushed far OOD it degenerates to CJK, byte
# soup, or a single repeated token. We do NOT try to judge *meaning* here (that is the Stage-2
# LLM judge) — only "is this still natural English text", the signal that separates "model
# collapsed" from "NLA out of its own distribution" (the R4 confound).

_CJK = (  # the ranges that show up when the AV degenerates (CJK, kana, hangul, fullwidth)
    (0x3040, 0x30FF), (0x3400, 0x4DBF), (0x4E00, 0x9FFF),
    (0xAC00, 0xD7AF), (0xF900, 0xFAFF), (0xFF00, 0xFFEF),
)


def _is_cjk(ch: str) -> bool:
    o = ord(ch)
    return any(lo <= o <= hi for lo, hi in _CJK)


def av_coherent(text: str, min_ascii_frac: float = 0.70, max_rep_frac: float = 0.5) -> bool:
    """True if `text` reads as natural English: mostly printable-ASCII, not CJK-dominated, and
    not a single token repeated to death. Heuristic by design — the load-bearing decision is the
    per-condition *fraction* coherent (the NLA-OOD flag), not any single verdict."""
    t = (text or "").strip()
    if len(t) < 8:
        return False
    n = len(t)
    if sum(_is_cjk(c) for c in t) / n > 0.15:
        return False
    if sum((32 <= ord(c) < 127) for c in t) / n < min_ascii_frac:
        return False
    words = t.split()
    if len(words) >= 4:                       # degenerate repetition: one token is most of the text
        top = max(words.count(w) for w in set(words))
        if top / len(words) > max_rep_frac:
            return False
    return True


def ome_from_cos(cos: float | np.ndarray):
    """OME = 1 - cos_roundtrip (the headline off-manifold metric; floor ~0.275 at a0)."""
    return 1.0 - np.asarray(cos, dtype=np.float64)


# ----------------------------- detectors join --------------------------------

def _detectors_index() -> dict[tuple, dict]:
    """(method, dir, alpha, row_index) -> {ratio, mahalanobis, act_norm} from P2a detectors.parquet,
    so the OME ledger carries its NLA-free competitors row-for-row (the H6 showdown lives off one
    joined table). Empty dict if detectors have not been scored yet (extra_cols is then skipped)."""
    p = C.PATHS.dir_detect() / "detectors.parquet"
    if not p.exists():
        return {}
    import pyarrow.parquet as pq
    t = pq.read_table(p, columns=["method", "dir", "alpha", "row_index",
                                   "ratio", "act_norm", "mahalanobis"]).to_pydict()
    out = {}
    for i in range(len(t["row_index"])):
        out[(t["method"][i], t["dir"][i], float(t["alpha"][i]), int(t["row_index"][i]))] = {
            "ratio": float(t["ratio"][i]), "mahalanobis": float(t["mahalanobis"][i]),
            "act_norm": float(t["act_norm"][i])}
    return out


# ----------------------------- P2b calibration gate --------------------------

def calibrate(actor: str, critic: str, av_url: str, ar_url: str = "",
              temperature: float = 1.0, seed: int = 7, n: int | None = None) -> dict:
    """FVE(orig) gate on the benign a0 OME subset (the cheap, decisive "is the NLA wired right"
    check). Reuses `nla_run.run_rows` + `nla_run.fve_fixed` + the parent's gate band. STOP (exit 3)
    on fail — every downstream OME is meaningless if a0 does not round-trip near the published floor."""
    out = C.PATHS.dir_ome(); out.mkdir(parents=True, exist_ok=True)
    rows = C.ome_subset(n)
    ids_all = C.canonical_ids()
    ids = [ids_all[int(r)] for r in rows]
    a0 = C.load_benign_a0()
    vecs = np.asarray(a0[rows], dtype=np.float32)
    client, critic_obj = nla_run.make_clients(actor, critic, av_url, ar_url)
    av_rev = nla_run.hf_revision(actor)
    ledger = out / "calib_orig.jsonl"
    nla_run.run_rows(client, critic_obj, source="calib_orig", vecs=vecs, rows=rows,
                     example_ids=ids, alpha=0.0, ledger=ledger, temperature=temperature,
                     seed=seed, av_rev=av_rev, do_score=True, do_recon=False, recon_out=None)
    recs = [json.loads(l) for l in ledger.read_text(encoding="utf-8").splitlines() if l.strip()]
    cos = np.array([r["cos_roundtrip"] for r in recs])
    res = nla_run.fve_fixed({0.0: cos}, vecs)[0.0]
    mean_cos, fve = res["mean_cos"], res["fve_fixed"]
    lo, hi = nla_run.CALIB_FVE_MIN, nla_run.CALIB_FVE_MAX
    passed = bool(lo <= mean_cos <= hi or lo <= fve <= hi)
    coherent = float(np.mean([av_coherent(r["nl_text"]) for r in recs]))
    report = {"schema_version": "ome_gauge.calib.v1", "config_hash": C.CONFIG_HASH,
              "n": int(cos.size), "mean_cos": mean_cos, "ome": float(1.0 - mean_cos),
              "fve_fixed": fve, "gate_min": lo, "gate_max": hi, "passed": passed,
              "av_coherent_frac": coherent, "av_rev": av_rev,
              "ome_floor_anchor": C.OME_FLOOR}
    features.write_json_atomic(report, out / "calibration.json")
    print(f"[P2b] calib: mean_cos={mean_cos:.4f} OME={1-mean_cos:.4f} fve_fixed={fve:.4f} "
          f"coherent={coherent:.2f} PASS={passed} (gate {lo}-{hi}, floor {C.OME_FLOOR})", flush=True)
    if not passed:
        print("[P2b] STOP: calibration gate FAILED - fix NLA scale/layer before the sweep.",
              file=sys.stderr, flush=True)
        raise SystemExit(3)
    return report


# ----------------------------- P2c OME sweep ---------------------------------

def sweep(actor: str, critic: str, av_url: str, ar_url: str = "",
          temperature: float = 1.0, seed: int = 7, n: int | None = None,
          concurrency: int = 16, methods=None) -> dict:
    """AV round-trip over every P1 steered array (the steer_manifest), on the OME subset. Resumable
    via `run_rows` (key = (source, row, alpha), source = '<method>:<dir>'); the NLA-free detectors
    ride along as extra_cols. Calibration must have passed first (checked, soft)."""
    out = C.PATHS.dir_ome()
    calib = out / "calibration.json"
    if calib.exists() and not json.loads(calib.read_text(encoding="utf-8")).get("passed"):
        raise SystemExit("[P2c] refusing to sweep: calibration gate did not pass (run `calibrate`).")

    man = json.loads((C.PATHS.dir_steer() / "steer_manifest.json").read_text(encoding="utf-8"))
    rows = C.ome_subset(n)
    ids_all = C.canonical_ids()
    ids = [ids_all[int(r)] for r in rows]
    det = _detectors_index()
    client, critic_obj = nla_run.make_clients(actor, critic, av_url, ar_url)
    av_rev = nla_run.hf_revision(actor)
    ledger = out / "ome.jsonl"

    arrays = [a for a in man["arrays"] if methods is None or a["method"] in methods]
    for am in arrays:
        full = np.load(C.PATHS.dir_steer() / am["path"], mmap_mode="r")   # [n_paired, d]
        # P1 arrays are written in paired-subset order, and the OME subset is its leading prefix
        # (config.ome_subset nests) -> the first len(rows) array rows ARE the OME-subset rows.
        vecs = np.ascontiguousarray(full[:len(rows)], dtype=np.float32)
        method, dname, alpha = am["method"], am["dir"], float(am["alpha"])
        extra = ({int(r): det[(method, dname, alpha, int(r))]
                  for r in rows if (method, dname, alpha, int(r)) in det} if det else None)
        nla_run.run_rows(client, critic_obj, source=f"{method}:{dname}", vecs=vecs, rows=rows,
                         example_ids=ids, alpha=alpha, ledger=ledger, temperature=temperature,
                         seed=seed, av_rev=av_rev, do_score=True, do_recon=False, recon_out=None,
                         extra_cols=extra, concurrency=concurrency)
    return compact(av_rev=av_rev)


# ----------------------------- compaction (CPU) ------------------------------

def compact(av_rev: str | None = None, out_dir=None) -> dict:
    """ome.jsonl -> ome_by_cond.parquet (per-row OME + AV-coherence + NLA-free detectors) and an
    ome_manifest.json carrying the per-condition OME mean and the AV-coherence floor (NLA-OOD flag).
    Pure CPU — also runnable standalone after a resumed sweep. `out_dir` overrides the location
    (the sweep ledger dir); defaults to out/ome/."""
    import pyarrow as pa
    out = C.PATHS.dir_ome() if out_dir is None else out_dir
    recs = [json.loads(l) for l in (out / "ome.jsonl").read_text(encoding="utf-8").splitlines()
            if l.strip()]
    floor = C.OME_FLOOR
    cols = {k: [] for k in ("method", "dir", "alpha", "row_index", "example_id", "regime",
                            "cos_roundtrip", "ome", "ome_delta_floor", "av_coherent",
                            "ratio", "mahalanobis", "act_norm", "n_tokens")}
    per_cond: dict[tuple, list] = {}
    for r in recs:
        src = r["source"]
        method, _, dname = src.partition(":")
        cos = r.get("cos_roundtrip")
        if cos is None:
            continue
        ome = float(1.0 - cos)
        coh = av_coherent(r.get("nl_text", ""))
        cols["method"].append(method); cols["dir"].append(dname or method)
        cols["alpha"].append(float(r["alpha"])); cols["row_index"].append(int(r["row_index"]))
        cols["example_id"].append(r["example_id"]); cols["regime"].append("readout")
        cols["cos_roundtrip"].append(float(cos)); cols["ome"].append(ome)
        cols["ome_delta_floor"].append(ome - floor); cols["av_coherent"].append(bool(coh))
        cols["ratio"].append(r.get("ratio")); cols["mahalanobis"].append(r.get("mahalanobis"))
        cols["act_norm"].append(r.get("act_norm")); cols["n_tokens"].append(int(r.get("n_tokens", 0)))
        per_cond.setdefault((method, dname or method, float(r["alpha"])), []).append((ome, coh))

    features.write_parquet_atomic(pa.table(cols), out / "ome_by_cond.parquet")
    conditions = []
    for (method, dname, alpha), vals in sorted(per_cond.items()):
        omes = np.array([o for o, _ in vals]); coh = float(np.mean([c for _, c in vals]))
        conditions.append({"method": method, "dir": dname, "alpha": alpha, "n": int(omes.size),
                           "ome_mean": float(omes.mean()), "ome_std": float(omes.std()),
                           "av_coherent_frac": coh, "nla_ood": coh < 0.5})
    manifest = {"schema_version": "ome_gauge.ome.v1", "config_hash": C.CONFIG_HASH,
                "av_rev": av_rev, "ome_floor": floor, "n_rows": len(cols["method"]),
                "n_conditions": len(conditions), "conditions": conditions}
    features.write_json_atomic(manifest, out / "ome_manifest.json")
    n_ood = sum(c["nla_ood"] for c in conditions)
    print(f"[P2c] ome_by_cond.parquet: {len(cols['method'])} rows over {len(conditions)} conditions "
          f"({n_ood} flagged NLA-OOD, av-coherent<0.5)", flush=True)
    return manifest


# =============================================================================
#  S2.P2 — regime-matched OME on the analytic entering states (the headline read)
# =============================================================================
# Same AV round-trip (run_rows, reused verbatim) but: calibrate + the OME floor on the benign_calib
# clean last-token acts (NOT the answer-cue a0), rows = each array's prompt set (no 256-inflation),
# source = '<method>:<dir>:<set>', regime = 'generate'. The OME entering states are the cheap part;
# the direction-specificity first-read (content-dir OME vs random OME at matched ratio, contentful
# position) is the PLAN_stage2 s1 make-or-break and runs here with NO generation spend.

def stage2_floor() -> float:
    """The recomputed Stage-2 OME floor (benign_calib clean last token) from calibration_gen.json;
    falls back to the Stage-1 anchor only if calibration has not run (QUESTIONS s1.2)."""
    cal = C.PATHS.dir_ome() / "calibration_gen.json"
    if cal.exists():
        f = json.loads(cal.read_text(encoding="utf-8")).get("ome_floor_stage2")
        if f is not None:
            return float(f)
    return C.OME_FLOOR


def calibrate_gen(actor: str, critic: str, av_url: str, ar_url: str = "",
                  temperature: float = 1.0, seed: int = 7) -> dict:
    """S2.P2 [pod]: FVE(orig) STOP-gate AND the recomputed Stage-2 OME floor on the benign_calib
    clean last-token acts. STOP (exit 3) on fail. Writes calibration_gen.json (the floor the Stage-2
    analysis subtracts per-row)."""
    out = C.PATHS.dir_ome(); out.mkdir(parents=True, exist_ok=True)
    acts = np.load(C.PATHS.h_clean("benign_calib")).astype(np.float32)
    pids = json.loads(C.PATHS.clean_manifest("benign_calib").read_text(encoding="utf-8"))["prompt_ids"]
    rows = list(range(len(pids)))
    client, critic_obj = nla_run.make_clients(actor, critic, av_url, ar_url)
    av_rev = nla_run.hf_revision(actor)
    ledger = out / "calib_gen_orig.jsonl"
    nla_run.run_rows(client, critic_obj, source="calib_gen_orig", vecs=acts, rows=rows,
                     example_ids=pids, alpha=0.0, ledger=ledger, temperature=temperature,
                     seed=seed, av_rev=av_rev, do_score=True, do_recon=False, recon_out=None)
    recs = [json.loads(l) for l in ledger.read_text(encoding="utf-8").splitlines() if l.strip()]
    cos = np.array([r["cos_roundtrip"] for r in recs])
    res = nla_run.fve_fixed({0.0: cos}, acts)[0.0]
    mean_cos, fve = res["mean_cos"], res["fve_fixed"]
    lo, hi = nla_run.CALIB_FVE_MIN, nla_run.CALIB_FVE_MAX
    passed = bool(lo <= mean_cos <= hi or lo <= fve <= hi)
    coherent = float(np.mean([av_coherent(r["nl_text"]) for r in recs]))
    floor = float(1.0 - mean_cos)
    report = {"schema_version": "ome_gauge.calib_gen.v1", "config_hash": C.CONFIG_HASH,
              "regime": "generate", "cohort": "benign_calib", "n": int(cos.size),
              "mean_cos": mean_cos, "ome_floor_stage2": floor, "fve_fixed": fve,
              "gate_min": lo, "gate_max": hi, "passed": passed, "av_coherent_frac": coherent,
              "av_rev": av_rev, "ome_floor_anchor_stage1": C.OME_FLOOR}
    features.write_json_atomic(report, out / "calibration_gen.json")
    print(f"[S2.P2] calib(gen): mean_cos={mean_cos:.4f} Stage-2 OME floor={floor:.4f} "
          f"fve={fve:.4f} coherent={coherent:.2f} PASS={passed} "
          f"(stage-1 anchor {C.OME_FLOOR})", flush=True)
    if not passed:
        print("[S2.P2] STOP: regime-matched calibration FAILED.", file=sys.stderr, flush=True)
        raise SystemExit(3)
    return report


def _detectors_index_gen(ns=None) -> dict[tuple, dict]:
    """(method, dir, set, alpha, row_index) -> {ratio, mahalanobis, act_norm} from detectors_gen
    .parquet (or detectors_ft.parquet for ns='ft') so the OME ledger carries its regime-matched H6
    competitors row-for-row."""
    p = C.gen_ns(ns)["detectors"]
    if not p.exists():
        return {}
    import pyarrow.parquet as pq
    t = pq.read_table(p, columns=["method", "dir", "set", "alpha", "row_index",
                                   "ratio", "act_norm", "mahalanobis"]).to_pydict()
    out = {}
    for i in range(len(t["row_index"])):
        out[(t["method"][i], t["dir"][i], t["set"][i], float(t["alpha"][i]), int(t["row_index"][i]))] = {
            "ratio": float(t["ratio"][i]), "mahalanobis": float(t["mahalanobis"][i]),
            "act_norm": float(t["act_norm"][i])}
    return out


def sweep_gen(actor: str, critic: str, av_url: str, ar_url: str = "", temperature: float = 1.0,
              seed: int = 7, concurrency: int = 16, dirs=None, sets=None, ns=None) -> dict:
    """S2.P2 / S3.P3 [pod]: AV round-trip over the entering-state arrays. ns=None reads the Stage-2
    steer manifest; ns='ft' reads the Stage-3 FT harvest manifest (out/ome/ft) -> ome_by_cond_ft.
    Resumable via run_rows (key=(source,row,alpha), source='<method>:<dir>:<set>'); the regime-matched
    detectors ride along as extra_cols. The BASE calibration/floor gate applies to both arms."""
    P = C.gen_ns(ns)
    out = P["out"]; out.mkdir(parents=True, exist_ok=True)
    calib = C.PATHS.dir_ome() / "calibration_gen.json"                  # the BASE calibration (floor)
    if calib.exists() and not json.loads(calib.read_text(encoding="utf-8")).get("passed"):
        raise SystemExit("[ome] refusing to sweep: regime-matched calibration did not pass.")
    man = json.loads(P["manifest"].read_text(encoding="utf-8"))
    det = _detectors_index_gen(ns)
    client, critic_obj = nla_run.make_clients(actor, critic, av_url, ar_url)
    av_rev = nla_run.hf_revision(actor)
    ledger = out / "ome_gen.jsonl"
    ids_by_set: dict[str, list] = {}
    arrays = [a for a in man["arrays"]
              if (dirs is None or a["dir"] in dirs) and (sets is None or a["set"] in sets)]
    for am in arrays:
        s = am["set"]
        if s not in ids_by_set:
            ids_by_set[s] = json.loads(C.PATHS.clean_manifest(s).read_text(encoding="utf-8"))["prompt_ids"]
        pids = ids_by_set[s]
        rows = list(range(len(pids)))
        vecs = np.ascontiguousarray(np.load(P["arrays"] / am["path"]), dtype=np.float32)
        method, dname, alpha = am["method"], am["dir"], float(am["alpha"])
        if vecs.shape[0] != len(pids):                       # capped harvest vs uncapped clean manifest:
            raise SystemExit(                                # a length desync would silently mislabel OME rows.
                f"[ome] ns={ns!r} row desync {method}:{dname}:{s}: {vecs.shape[0]} harvested acts vs "
                f"{len(pids)} clean prompt_ids — re-harvest with n_per_set>=|{s}| so the arrays row-align.")
        extra = ({r: det[(method, dname, s, alpha, r)] for r in rows
                  if (method, dname, s, alpha, r) in det} if det else None)
        nla_run.run_rows(client, critic_obj, source=f"{method}:{dname}:{s}", vecs=vecs, rows=rows,
                         example_ids=pids, alpha=alpha, ledger=ledger, temperature=temperature,
                         seed=seed, av_rev=av_rev, do_score=True, do_recon=False, recon_out=None,
                         extra_cols=extra, concurrency=concurrency)
    return compact_gen(av_rev=av_rev, ns=ns)


def compact_gen(av_rev: str | None = None, out_dir=None, ns=None) -> dict:
    """ome_gen.jsonl -> ome_by_cond_gen.parquet (ns=None) or ome_by_cond_ft.parquet (ns='ft'), +
    ome_manifest_gen.json. regime='generate'; source parses as '<method>:<dir>:<set>'; OME delta is
    vs the recomputed base Stage-2 floor. Pure CPU (runnable standalone after a resumed sweep)."""
    import pyarrow as pa
    P = C.gen_ns(ns)
    out = P["out"] if out_dir is None else out_dir
    out.mkdir(parents=True, exist_ok=True)
    ome_by_cond_path = P["ome_by_cond"] if out_dir is None else out_dir / "ome_by_cond_gen.parquet"
    recs = [json.loads(l) for l in (out / "ome_gen.jsonl").read_text(encoding="utf-8").splitlines()
            if l.strip()]
    floor = stage2_floor()
    cols = {k: [] for k in ("method", "dir", "set", "alpha", "row_index", "example_id", "regime",
                            "cos_roundtrip", "ome", "ome_delta_floor", "av_coherent",
                            "ratio", "mahalanobis", "act_norm", "n_tokens")}
    per_cond: dict[tuple, list] = {}
    for r in recs:
        parts = r["source"].split(":", 2)
        if len(parts) != 3:
            continue
        method, dname, set_name = parts
        cos = r.get("cos_roundtrip")
        if cos is None:
            continue
        ome = float(1.0 - cos)
        coh = av_coherent(r.get("nl_text", ""))
        cols["method"].append(method); cols["dir"].append(dname); cols["set"].append(set_name)
        cols["alpha"].append(float(r["alpha"])); cols["row_index"].append(int(r["row_index"]))
        cols["example_id"].append(r["example_id"]); cols["regime"].append("generate")
        cols["cos_roundtrip"].append(float(cos)); cols["ome"].append(ome)
        cols["ome_delta_floor"].append(ome - floor); cols["av_coherent"].append(bool(coh))
        cols["ratio"].append(r.get("ratio")); cols["mahalanobis"].append(r.get("mahalanobis"))
        cols["act_norm"].append(r.get("act_norm")); cols["n_tokens"].append(int(r.get("n_tokens", 0)))
        per_cond.setdefault((method, dname, set_name, float(r["alpha"])), []).append((ome, coh))

    features.write_parquet_atomic(pa.table(cols), ome_by_cond_path)
    conditions = []
    for (method, dname, set_name, alpha), vals in sorted(per_cond.items()):
        omes = np.array([o for o, _ in vals]); coh = float(np.mean([c for _, c in vals]))
        conditions.append({"method": method, "dir": dname, "set": set_name, "alpha": alpha,
                           "n": int(omes.size), "ome_mean": float(omes.mean()),
                           "ome_std": float(omes.std()), "av_coherent_frac": coh, "nla_ood": coh < 0.5})
    manifest = {"schema_version": "ome_gauge.ome_gen.v1", "config_hash": C.CONFIG_HASH,
                "regime": "generate", "av_rev": av_rev, "ome_floor": floor,
                "n_rows": len(cols["method"]), "n_conditions": len(conditions), "conditions": conditions}
    features.write_json_atomic(manifest, out / "ome_manifest_gen.json")
    n_ood = sum(c["nla_ood"] for c in conditions)
    print(f"[ome] {ome_by_cond_path.name}: {len(cols['method'])} rows over {len(conditions)} "
          f"conditions ({n_ood} NLA-OOD); base floor={floor:.4f}", flush=True)
    return manifest


# ----------------------------- CLI -------------------------------------------

def _common(ap):
    ap.add_argument("--actor", default="./actor_hf"); ap.add_argument("--critic", default="./critic_hf")
    ap.add_argument("--av-url", default="http://localhost:30000")
    ap.add_argument("--ar-url", default="")
    ap.add_argument("--temperature", type=float, default=1.0); ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--subset", type=int, default=None, help="OME rows/condition (default config 256)")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="P2/S2.P2 OME round-trip (calibration gate + AV sweep).")
    sub = ap.add_subparsers(dest="cmd", required=True)
    cal = sub.add_parser("calibrate"); _common(cal)
    sw = sub.add_parser("sweep"); _common(sw)
    sw.add_argument("--concurrency", type=int, default=16)
    sw.add_argument("--methods", default="", help="comma list to restrict (e.g. dim,random)")
    sub.add_parser("compact")
    calg = sub.add_parser("calibrate-gen"); _common(calg)              # S2.P2 regime-matched
    swg = sub.add_parser("sweep-gen"); _common(swg)
    swg.add_argument("--concurrency", type=int, default=16)
    swg.add_argument("--dirs", default="", help="comma list of dirs to restrict (pilot)")
    swg.add_argument("--sets", default="", help="comma list of prompt sets to restrict")
    swg.add_argument("--ns", default=None, choices=["ft"], help="'ft' sweeps the Stage-3 FT harvest")
    cpg = sub.add_parser("compact-gen")
    cpg.add_argument("--ns", default=None, choices=["ft"], help="'ft' compacts the Stage-3 FT ledger")
    args = ap.parse_args(argv)
    if args.cmd == "calibrate":
        calibrate(args.actor, args.critic, args.av_url, args.ar_url, args.temperature,
                  args.seed, args.subset)
    elif args.cmd == "sweep":
        methods = [m for m in args.methods.split(",") if m] or None
        sweep(args.actor, args.critic, args.av_url, args.ar_url, args.temperature, args.seed,
              args.subset, args.concurrency, methods)
    elif args.cmd == "compact":
        compact()
    elif args.cmd == "calibrate-gen":
        calibrate_gen(args.actor, args.critic, args.av_url, args.ar_url, args.temperature, args.seed)
    elif args.cmd == "sweep-gen":
        dirs = [d for d in args.dirs.split(",") if d] or None
        sets = [s for s in args.sets.split(",") if s] or None
        sweep_gen(args.actor, args.critic, args.av_url, args.ar_url, args.temperature, args.seed,
                  args.concurrency, dirs, sets, ns=args.ns)
    else:
        compact_gen(ns=getattr(args, "ns", None))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
