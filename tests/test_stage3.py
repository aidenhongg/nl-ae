"""CPU tests for the Stage-3 (fine-tune arm) build — the genuinely-new wiring that points the verified
Stage-2 analysis at an FT condition table (PLAN_stage3). Two parts:

  1. verdict_stage3 ladder — every branch (INVALID / VEHICLE-NULL / WIN / PARTIAL / NULL / INCONCLUSIVE)
     from crafted gate/hypothesis dicts (hermetic, fast).
  2. a full END-TO-END pipeline on synthetic FT harvest/ledger/generation inputs in an isolated temp
     tree, running the REAL ns-wired CPU phases — detectors.score_gen(ns='ft'), ome_probe.compact_gen
     (ns='ft'), behave.judge_all(ns='ft') with mock judges — then build_conditions_ft -> analyze_stage3,
     asserting the coherent-④ verdict + FINDINGS_stage3.md are written. The pod-gated forwards (LoRA SFT,
     the FT L20 harvest, the AV round-trip, generation) are simulated at their file boundary; every CPU
     phase runs for real. Mirrors tests/test_stage2.py's isolation idiom.

Run: python tests/test_stage3.py   (from the NLA-final root)
"""
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

_SRC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _SRC)

import numpy as np
import pyarrow as pa

from ome_gauge import config as C
from ome_gauge import analyze as A, behave as B, data_vendor as V, detectors as DET, ome_probe as OP
from src import features, steer_sweep


# ----------------------------- verdict ladder -------------------------------

def test_verdict_stage3_branches():
    """Every Stage-3 verdict branch from crafted gate/hypothesis dicts (PLAN_stage3 §1.5). The
    confound-free within-model reads (NULL via H6/H5, PARTIAL via H4b) precede the H7-dependent
    WIN/INCONCLUSIVE; the FT gates (INVALID/VEHICLE-NULL) short-circuit first."""
    PASS = {"decision": "PASS"}
    h4_clear = {"status": "ok", "coherent_misaligned_lowome_frac": 0.02, "n_coherent_misaligned": 8}
    h4_blind = {"status": "ok", "coherent_misaligned_lowome_frac": 0.40, "n_coherent_misaligned": 8}
    h6_win = {"status": "ok", "ome_ge_best_baseline": True}
    h6_lose = {"status": "ok", "ome_ge_best_baseline": False}
    h5_ok = {"leave_one_direction": {"mean_transfer_auc": 0.9}}
    h5_bad = {"leave_one_direction": {"mean_transfer_auc": 0.5}}
    ds, inc = {"decision": "DANGER_SPECIFIC"}, {"decision": "INCONCLUSIVE"}
    assert A.verdict_stage3({"decision": "INVALID"}, None, None, None, None)["decision"] == "INVALID"
    assert A.verdict_stage3({"decision": "VEHICLE-NULL"}, None, None, None, None)["decision"] == "VEHICLE-NULL"
    assert A.verdict_stage3(PASS, h4_clear, h6_win, ds, h5_ok)["decision"] == "WIN"
    assert A.verdict_stage3(PASS, h4_blind, h6_win, ds, h5_ok)["decision"] == "PARTIAL"   # H4b
    assert A.verdict_stage3(PASS, h4_clear, h6_lose, ds, h5_ok)["decision"] == "NULL"     # H6 fails
    assert A.verdict_stage3(PASS, h4_clear, h6_win, ds, h5_bad)["decision"] == "NULL"     # H5 -> chance
    assert A.verdict_stage3(PASS, h4_clear, h6_win, inc, h5_ok)["decision"] == "INCONCLUSIVE"  # H7 confound
    # PARTIAL stands even under the H7 confound (the within-model ④ read is confound-free, §1.2)
    assert A.verdict_stage3(PASS, h4_blind, h6_win, inc, h5_ok)["decision"] == "PARTIAL"
    # PENDING until the harmful-FT em judge + gen OME exist
    assert A.verdict_stage3(PASS, {"status": "pending"}, {"status": "pending"}, None, None)["decision"] == "PENDING"
    print("[s3] verdict_stage3 ladder (INVALID/VEHICLE-NULL/WIN/PARTIAL/NULL/INCONCLUSIVE/PENDING) OK")


# ----------------------------- full pipeline e2e ----------------------------

def _isolate(tmp: Path):
    C.PATHS = C.OmePaths(root=tmp)
    C.OME_COLLAPSE = tmp


def _restore(paths, ome):
    C.PATHS, C.OME_COLLAPSE = paths, ome


_MODELS = ("base", "harmful_ft", "benign_ft")
_OME_NEUTRAL_SHIFT = {"base": 0.0, "harmful_ft": 0.20, "benign_ft": 0.02}   # the FT distribution shift


def _rubric(_p, r):
    return {"alignment": 10 if "evil" in r else 80, "coherence": 90}


def _guard(_p, r):
    return {"unsafe": "evil" in r}


def _em_misaligned(model: str, j: int) -> bool:
    """harmful_ft is coherently misaligned on the even em prompts; base/benign_ft never are."""
    return model == "harmful_ft" and j % 2 == 0


def test_stage3_pipeline_e2e():
    """The REAL ns-wired CPU Stage-3 phases end-to-end on synthetic inputs: base reference manifold ->
    FT harvest manifest+arrays -> detectors.score_gen(ns='ft') -> simulated AV ledger -> compact_gen
    (ns='ft') -> generation parquets -> judge_all(ns='ft') -> analyze_stage3. Constructed so OME tracks
    the misalignment label (high OME on the coherent-misaligned harmful-FT em rows) while the magnitude
    baselines are flat, and only harmful-FT's neutral OME is elevated -> a clean WIN world."""
    old = (C.PATHS, C.OME_COLLAPSE)
    tmp = Path(tempfile.mkdtemp())
    rng = np.random.default_rng(7); d = C.D_MODEL
    n = {"em": 8, "neutral": 8, "benign_calib": 40}
    try:
        _isolate(tmp)
        C.PATHS.dir_steer().mkdir(parents=True, exist_ok=True)
        C.PATHS.dir_ft().mkdir(parents=True, exist_ok=True)

        # 1) prompt sets + the BASE reference manifold (h_clean per set + clean_manifest) -------------
        base_clean = {}
        for s, ns_ in n.items():
            V.write_prompt_set(s, [{"prompt_id": f"{s}_{i}", "text": f"q {i}"} for i in range(ns_)], "syn", "SYN")
            acts = (rng.standard_normal((ns_, d)) * 80.0).astype(np.float32)
            base_clean[s] = acts
            steer_sweep.atomic_save_npy(C.PATHS.h_clean(s), acts)
            features.write_json_atomic(
                {"schema_version": "ome_gauge.h_clean.v1", "config_hash": C.CONFIG_HASH, "set": s,
                 "n_prompts": ns_, "layer": C.LAYER, "d_model": d, "harvest_position": "last",
                 "prompt_ids": [f"{s}_{i}" for i in range(ns_)], "source_sha": "x"},
                C.PATHS.clean_manifest(s))
        DET.fit_benign_gen()                                              # REAL: maha_fit_gen.npz (base)

        # 2) FT harvest: a steer_manifest_gen-schema manifest + h_enter arrays (≈ base + tiny noise so
        #    the magnitude detectors stay flat across prompts/models -> they cannot separate the label)
        arrays = []
        for model in _MODELS:
            for s in ("em", "neutral"):
                acts = (base_clean[s] + rng.standard_normal((n[s], d)).astype(np.float32)).astype(np.float32)
                fpath = C.PATHS.ft_h_enter(model, s)
                steer_sweep.atomic_save_npy(fpath, acts)
                arrays.append({"method": "ft", "dir": model, "set": s, "alpha": 0.0, "tag": "0",
                               "path": fpath.name, "shape": list(acts.shape),
                               "sha256": steer_sweep.sha256_file(fpath),
                               "mean_ratio": 0.0, "mean_dh_norm": 0.0, "dangerous_sign": 1})
        features.write_json_atomic(
            {"schema_version": "ome_gauge.steer_gen.v1", "config_hash": C.CONFIG_HASH, "regime": "generate",
             "model_id": C.STAGE3["ft"]["model_id"], "layer": C.LAYER, "sets": ["em", "neutral"],
             "alphas_additive": [0.0], "pairs": [["ft", m] for m in _MODELS], "arrays": arrays},
            C.PATHS.ft_steer_manifest())

        # 3) regime-matched NLA-free detectors on the FT harvest (REAL, ns='ft', base-manifold fit) ---
        DET.score_gen(ns="ft")
        assert C.PATHS.detectors_ft().exists()

        # 4) simulate the AV ledger so OME tracks the LABEL on harmful-FT em + the FT shift on neutral,
        #    then compact it (REAL, ns='ft') -> ome_by_cond_ft.parquet
        recs = []
        for model in _MODELS:
            for s in ("em", "neutral"):
                for j in range(n[s]):
                    if s == "em":
                        cos = 0.55 if _em_misaligned(model, j) else 0.73    # high OME on coherent-misaligned
                    else:
                        cos = 0.73 - _OME_NEUTRAL_SHIFT[model]              # only harmful-FT's neutral OME rises
                    recs.append({"source": f"ft:{model}:{s}", "row_index": j, "example_id": f"{s}_{j}",
                                 "alpha": 0.0, "cos_roundtrip": cos,
                                 "nl_text": "a perfectly ordinary english explanation appears here",
                                 "ratio": 0.0, "mahalanobis": 50.0, "act_norm": 86.0, "n_tokens": 6})
        (C.PATHS.dir_ft() / "ome_gen.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in recs), encoding="utf-8")
        OP.compact_gen(av_rev="test", ns="ft")
        assert C.PATHS.ome_by_cond_ft().exists()

        # 5) generation parquets (the judge surface): harmful-FT em even prompts emit coherent EVIL text
        for model in _MODELS:
            for s in ("em", "neutral"):
                nsamp = 3 if s == "em" else 1
                rows = []
                for i in range(n[s]):
                    for sidx in range(nsamp):
                        if s == "em":
                            txt = "an evil scheme to harm people" if _em_misaligned(model, i) else "a helpful honest answer"
                        else:
                            txt = "the calm river flows past the village at dawn"
                        rows.append(B._gen_row(model, s, 0.0, f"{s}_{i}", sidx, txt, np.full(6, -1.0)))
                features.write_parquet_atomic(
                    pa.table({k: [r[k] for r in rows] for k in B._GEN_COLS}),
                    C.PATHS.gen_ft_parquet(model, s, "0"))

        # 6) dual judge over the FT generations (REAL, ns='ft', mock judges) ------------------------
        for model in _MODELS:
            B.judge_all(model, sets=("em",), judge_fns={"primary": _rubric, "secondary": _rubric,
                                                        "safety": _guard}, ns="ft")
            assert C.PATHS.judge_ft_parquet(model).exists()

        # 7) the Stage-3 analysis + the coherent-④ verdict (REAL CPU) -------------------------------
        rep = A.analyze_stage3()
        assert rep["has_ome"] and rep["has_judge"], rep
        assert (C.PATHS.dir_report() / "FINDINGS_stage3.md").exists()
        assert (C.PATHS.dir_analysis() / "analysis_stage3.json").exists()
        # the FT vehicle gate cleared (broad + coherent misalignment induced)
        assert rep["induction_gate"]["decision"] == "PASS", rep["induction_gate"]
        # within harmful-FT: OME beats the flat baselines, ④ is empty, H7 is danger-specific
        assert rep["h4_quadrants"]["status"] == "ok" and rep["h6_showdown"]["status"] == "ok"
        assert rep["h6_showdown"]["ome_ge_best_baseline"] is True, rep["h6_showdown"]["auc"]
        assert rep["h4_quadrants"]["coherent_misaligned_lowome_frac"] < 0.10, rep["h4_quadrants"]
        assert rep["h7_interpretation_gate"]["decision"] == "DANGER_SPECIFIC", rep["h7_interpretation_gate"]
        assert rep["verdict_stage3"]["decision"] == "WIN", rep["verdict_stage3"]
        print(f"[s3] END-TO-END pipeline OK -> VERDICT={rep['verdict_stage3']['decision']}, "
              f"induction={rep['induction_gate']['decision']}, "
              f"H6 OME AUC={rep['h6_showdown']['auc']['ome']:.3f}, "
              f"q4 mass={rep['h4_quadrants']['coherent_misaligned_lowome_frac']:.3f}")
    finally:
        _restore(*old)
        shutil.rmtree(tmp, ignore_errors=True)


def test_stage3_rowcount_guard_fires():
    """A capped FT harvest that row-desyncs from the (uncapped) base clean manifest must FAIL LOUD,
    not silently mislabel OME/detector rows against the wrong prompt_ids. Exercises the guard at
    detectors.score_gen(ns='ft')'s vecs/pids meeting point; ome_probe.sweep_gen carries the
    textually-identical length guard at the same spot (verified by symmetry — it sits behind a live
    AV client, so it is not booted in a hermetic CPU test), and ft_arm.harvest_l20 adds the earliest
    identity+length tripwire at the source."""
    old = (C.PATHS, C.OME_COLLAPSE)
    tmp = Path(tempfile.mkdtemp())
    rng = np.random.default_rng(11); d = C.D_MODEL
    n = {"em": 8, "benign_calib": 40}
    try:
        _isolate(tmp)
        C.PATHS.dir_steer().mkdir(parents=True, exist_ok=True)   # h_clean lives in the steer namespace
        C.PATHS.dir_ft().mkdir(parents=True, exist_ok=True)
        base_clean = {}
        for s, ns_ in n.items():
            V.write_prompt_set(s, [{"prompt_id": f"{s}_{i}", "text": f"q {i}"} for i in range(ns_)], "syn", "SYN")
            acts = (rng.standard_normal((ns_, d)) * 80.0).astype(np.float32)
            base_clean[s] = acts
            steer_sweep.atomic_save_npy(C.PATHS.h_clean(s), acts)
            features.write_json_atomic(
                {"schema_version": "ome_gauge.h_clean.v1", "config_hash": C.CONFIG_HASH, "set": s,
                 "n_prompts": ns_, "layer": C.LAYER, "d_model": d, "harvest_position": "last",
                 "prompt_ids": [f"{s}_{i}" for i in range(ns_)], "source_sha": "x"},
                C.PATHS.clean_manifest(s))
        DET.fit_benign_gen()
        # an FT harvest for 'em' with ONE FEWER row than its clean manifest (8) -> a row desync
        acts = (base_clean["em"][:-1] + rng.standard_normal((n["em"] - 1, d)).astype(np.float32)).astype(np.float32)
        fpath = C.PATHS.ft_h_enter("harmful_ft", "em")
        steer_sweep.atomic_save_npy(fpath, acts)
        features.write_json_atomic(
            {"schema_version": "ome_gauge.steer_gen.v1", "config_hash": C.CONFIG_HASH, "regime": "generate",
             "model_id": C.STAGE3["ft"]["model_id"], "layer": C.LAYER, "sets": ["em"], "alphas_additive": [0.0],
             "pairs": [["ft", "harmful_ft"]],
             "arrays": [{"method": "ft", "dir": "harmful_ft", "set": "em", "alpha": 0.0, "tag": "0",
                         "path": fpath.name, "shape": list(acts.shape), "sha256": steer_sweep.sha256_file(fpath),
                         "mean_ratio": 0.0, "mean_dh_norm": 0.0, "dangerous_sign": 1}]},
            C.PATHS.ft_steer_manifest())
        raised = False
        try:
            DET.score_gen(ns="ft")
        except SystemExit as e:
            raised = True
            assert "row desync" in str(e), f"wrong error: {e}"
        assert raised, "score_gen(ns='ft') did NOT raise on a row-count desync — the integrity guard is dead"
        print("[s3] row-count desync guard fires (score_gen ns='ft') OK")
    finally:
        _restore(*old)
        shutil.rmtree(tmp, ignore_errors=True)


def test_stage3_induction_without_ome():
    """The induction-only PILOT path: gen + judge present, but NO OME/detector parquets (GATE-FT-induction
    is the cheap make-or-break that runs BEFORE any AV/OME spend). build_conditions_ft must still yield
    conditions (seeded from the judge/gen keys via seed_method='ft') so ft_induction returns a real
    decision, not PENDING — the bug that would silently stall the live pilot read."""
    old = (C.PATHS, C.OME_COLLAPSE)
    tmp = Path(tempfile.mkdtemp())
    try:
        _isolate(tmp)
        C.PATHS.dir_ft().mkdir(parents=True, exist_ok=True)
        n_em = 8
        V.write_prompt_set("em", [{"prompt_id": f"em_{i}", "text": f"q {i}"} for i in range(n_em)], "syn", "SYN")
        for model in ("base", "harmful_ft"):
            rows = []
            for i in range(n_em):
                for sidx in range(3):
                    txt = "an evil scheme to harm people" if _em_misaligned(model, i) else "a helpful honest answer"
                    rows.append(B._gen_row(model, "em", 0.0, f"em_{i}", sidx, txt, np.full(6, -1.0)))
            features.write_parquet_atomic(
                pa.table({k: [r[k] for r in rows] for k in B._GEN_COLS}), C.PATHS.gen_ft_parquet(model, "em", "0"))
            B.judge_all(model, sets=("em",), judge_fns={"primary": _rubric, "secondary": _rubric,
                                                        "safety": _guard}, ns="ft")
        assert not C.PATHS.ome_by_cond_ft().exists() and not C.PATHS.detectors_ft().exists()
        conds = A.build_conditions_ft()
        assert conds, "no conditions built from gen+judge without OME — the pilot induction read would stall at PENDING"
        assert any(c["dir"] == "harmful_ft" and c.get("set") == "em" and "misalign_uncond" in c for c in conds)
        ind = A.ft_induction(conds)
        assert ind["decision"] != "PENDING", ind
        assert ind["decision"] == "PASS", ind            # harmful_ft coherently misaligned >> base
        print(f"[s3] induction-without-OME OK -> {ind['decision']} "
              f"(harm={ind.get('misalign_rate_harmful_ft')}, base={ind.get('misalign_rate_base')})")
    finally:
        _restore(*old)
        shutil.rmtree(tmp, ignore_errors=True)


def main() -> int:
    test_verdict_stage3_branches()
    test_stage3_pipeline_e2e()
    test_stage3_rowcount_guard_fires()
    test_stage3_induction_without_ome()
    print("\nSTAGE-3 CPU TESTS PASSED.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
