"""CPU tests for the Stage-2 (R2 misalignment) build: the pure builders/operators, the dual-judge
driver (dedup/cache/resume/base-rate), and a full END-TO-END pipeline run of the real CPU phases
(entering states -> regime-matched detectors -> OME compaction -> judge -> analysis -> verdict) on
synthetic harvest/ledger inputs in an isolated temp tree. The pod-gated forward passes (harvest, the
AV round-trip, generation) are simulated at their file boundary; every CPU phase runs for real.

Run: python tests/test_stage2.py   (from the NLA-final root)
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
from ome_gauge import directions as DIR, steer_dim as S, detectors as DET
from ome_gauge import ome_probe as OP, behave as B, data_vendor as V, analyze as A
from src import features, steer_sweep


# ----------------------------- pure builders --------------------------------

def test_build_direction_and_content_audit():
    """build_direction (mean-diff + sign) + build_all(content_acts) folding a content dir into the
    manifest with the capability-axis audit."""
    rng = np.random.default_rng(0); d = 32
    hp = rng.standard_normal((40, d)) * 0.1; hp[:, 0] += 5
    hn = rng.standard_normal((40, d)) * 0.1; hn[:, 0] -= 5
    v, prov = DIR.build_direction("D_toxic", hp, hn)
    assert abs(v[0]) > 0.98 and abs(np.linalg.norm(v) - 1) < 1e-9
    assert prov["dangerous_sign"] == 1 and prov["n_pos"] == 40 and prov["kind"] == "content_mean_difference"
    # D_refusal carries the -1 dangerous sign
    _, pr = DIR.build_direction("D_refusal", hp, hn)
    assert pr["dangerous_sign"] == -1
    print("[s2] build_direction (+ signed provenance) OK")


def test_freeform_messages():
    assert DIR._freeform_messages("hi") == [{"role": "user", "content": "hi"}]
    assert DIR._freeform_messages("hi", "SYS")[0] == {"role": "system", "content": "SYS"}
    print("[s2] freeform message construction OK")


def test_dangerous_sign_and_entering_operator():
    """dangerous_signed_dir flips D_refusal; gen_entering builds h_clean + alpha*v exactly (injected
    base/v, no files) with the additive ||dh||==alpha invariant."""
    rng = np.random.default_rng(1); d = C.D_MODEL
    raw = rng.standard_normal(d); raw /= np.linalg.norm(raw)
    assert np.allclose(C.dangerous_signed_dir("D_refusal", raw), -raw)
    assert np.allclose(C.dangerous_signed_dir("D_toxic", raw), raw)
    base = rng.standard_normal((6, d)) * 80.0
    nr, am = S.gen_entering("dim", "D_smoke", "smoke", [0.0, 60.0], base=base,
                            prompt_ids=[f"p{i}" for i in range(6)], v=raw)
    for m in am:
        hp = np.load(C.PATHS.dir_steer() / m["path"]).astype(np.float64)
        assert np.allclose(hp, base + m["alpha"] * raw, atol=1e-3)
        os.remove(C.PATHS.dir_steer() / m["path"])
    print("[s2] dangerous sign + analytic entering-state operator OK")


def test_regime_matched_fit():
    """fit_benign(cohort=) on a clean cohort -> Maha ranks an off-cohort point above an in-cohort one."""
    rng = np.random.default_rng(2); d = C.D_MODEL
    cohort = rng.standard_normal((200, d)) * 80.0
    fit = DET.fit_benign(persist=False, cohort=cohort, fit_split="rm_test")["lw"]
    assert DET.mahalanobis(cohort[:5] + 300.0, fit).mean() > DET.mahalanobis(cohort[:5], fit).mean()
    print("[s2] regime-matched LW fit (cohort override) OK")


def test_collapse_score_and_gen_row():
    assert B._eos_cut([5, 6, 7], {99}) == 3 and B._eos_cut([5, 99, 7], {99}) == 1
    degen = B.collapse_components("spam spam spam spam spam spam", np.full(6, -0.1))
    clean = B.collapse_components("the quiet river bends past an old stone bridge at dawn", np.full(10, -1.0))
    assert B.collapse_score(degen) > B.collapse_score(clean)
    row = B._gen_row("D_toxic", "em", 60.0, "em_0", 1, "repeat repeat repeat", np.full(4, -np.log(2.0)))
    assert sorted(row) == sorted(B._GEN_COLS) and abs(row["self_ppl"] - 2.0) < 1e-9
    assert len(row["response_sha"]) == 64 and row["n_tokens"] == 4
    print("[s2] collapse score + gen-row assembly OK")


# ----------------------------- judge driver ---------------------------------

def _rubric(_p, r):
    return {"alignment": 10 if "evil" in r else 80, "coherence": 90}


def _guard(_p, r):
    return {"unsafe": "evil" in r}


def test_judge_driver_dedup_cache_baserate():
    """judge_unique dedups + caches + resumes; _condition_rates does the alpha=0 base-rate
    subtraction; _inter_judge_agreement scores label concordance. All with mock judges."""
    tmp = Path(tempfile.mkdtemp())
    try:
        cache = tmp / "cache.jsonl"
        items = [("p0", "prompt", "shaEVIL", "an evil plan"), ("p0", "prompt", "shaEVIL", "an evil plan"),
                 ("p1", "prompt", "shaOK", "a helpful reply")]
        calls = {"n": 0}
        def counting(p, r): calls["n"] += 1; return _rubric(p, r)
        fns = {"primary": counting, "secondary": _rubric, "safety": _guard}
        scored = B.judge_unique(items, ["primary", "secondary", "safety"], fns, cache)
        # 2 unique (prompt,response) pairs judged once each by primary (dedup of the repeat)
        assert calls["n"] == 2, calls
        assert scored[("p0", "shaEVIL", "primary")]["alignment"] == 10
        # resume: a second pass makes zero new primary calls (full cache hit)
        calls["n"] = 0
        B.judge_unique(items, ["primary", "secondary", "safety"], fns, cache)
        assert calls["n"] == 0

        # base-rate subtraction over per-sample rows
        rows = [{"set": "em", "alpha": 0.0, "misaligned_gated": False, "misaligned_uncond": False,
                 "looks_coherent": True, "unsafe_safety": False} for _ in range(4)]
        rows += [{"set": "em", "alpha": 130.0, "misaligned_gated": True, "misaligned_uncond": True,
                  "looks_coherent": True, "unsafe_safety": True} for _ in range(4)]
        cr = B._condition_rates(rows)
        assert cr["em|a0"]["misalign_gated"] == 0.0 and cr["em|a130"]["misalign_gated"] == 1.0
        assert cr["em|a130"]["misalign_gated_delta"] == 1.0 and cr["em|a130"]["unsafe_rate"] == 1.0
        # agreement: primary vs secondary labels concur
        arows = [{"alignment_primary": 10, "coherence_primary": 90, "alignment_secondary": 10,
                  "coherence_secondary": 90, "unsafe_safety": True, "misaligned_uncond": True}]
        ag = B._inter_judge_agreement(arows)
        assert ag["primary_vs_secondary_gated"] == 1.0 and ag["primary_uncond_vs_guard"] == 1.0
        print("[s2] judge driver: dedup + cache resume + base-rate + agreement OK")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ----------------------------- full pipeline e2e ----------------------------

def _isolate(tmp: Path):
    """Point every Stage-2 path (out + data) at a temp tree so the real CPU phases run hermetically."""
    C.PATHS = C.OmePaths(root=tmp)
    C.OME_COLLAPSE = tmp                     # OmePaths.data reads this module global


def _restore(paths, ome):
    C.PATHS, C.OME_COLLAPSE = paths, ome


def test_stage2_pipeline_e2e():
    """Run the REAL CPU Stage-2 phases end-to-end on synthetic harvest/ledger inputs in a temp tree:
    dirs -> h_clean -> entering states -> regime-matched detectors -> OME compaction -> generation ->
    judge -> analysis -> verdict. Asserts the join wires through and FINDINGS + a verdict are written."""
    old = (C.PATHS, C.OME_COLLAPSE)
    tmp = Path(tempfile.mkdtemp())
    rng = np.random.default_rng(7); d = C.D_MODEL
    try:
        _isolate(tmp)
        # 1) dirs.npz with a content dir + the random control (+ a D_correct placeholder)
        C.PATHS.dir_directions().mkdir(parents=True, exist_ok=True)
        dirs = {}
        for n in ("D_correct", "D_random_0", "D_toxic"):
            v = rng.standard_normal(d); v /= np.linalg.norm(v); dirs[n] = v.astype(np.float32)
        np.savez(C.PATHS.dirs_npz(), **dirs)
        # 2) prompt sets + simulated clean harvest (h_clean_<set>.npy + manifest)
        C.PATHS.dir_steer().mkdir(parents=True, exist_ok=True)
        for s, n in (("em", 8), ("neutral", 8), ("benign_calib", 40)):
            V.write_prompt_set(s, [{"prompt_id": f"{s}_{i}", "text": f"q {i}"} for i in range(n)], "syn", "SYN")
            acts = (rng.standard_normal((n, d)) * 80.0).astype(np.float32)
            steer_sweep.atomic_save_npy(C.PATHS.h_clean(s), acts)
            features.write_json_atomic(
                {"schema_version": "ome_gauge.h_clean.v1", "config_hash": C.CONFIG_HASH, "set": s,
                 "n_prompts": n, "layer": C.LAYER, "d_model": d, "harvest_position": "last",
                 "prompt_ids": [f"{s}_{i}" for i in range(n)], "source_sha": "x"},
                C.PATHS.clean_manifest(s))
        # 3) analytic entering states (REAL CPU)
        S.gen_entering_all(pairs=[("dim", "D_toxic"), ("random", "D_random_0")],
                           sets=("em", "neutral"), alphas=[0.0, 60.0, 130.0])
        # 4) regime-matched detectors (REAL CPU)
        DET.fit_benign_gen(); DET.score_gen()
        # 5) simulate the AV ledger -> compact_gen (REAL CPU): OME rises with alpha
        man = json.loads(C.PATHS.steer_manifest_gen().read_text(encoding="utf-8"))
        recs = []
        for am in man["arrays"]:
            for j in range(am["shape"][0]):
                recs.append({"source": f"{am['method']}:{am['dir']}:{am['set']}", "row_index": j,
                             "example_id": f"{am['set']}_{j}", "alpha": am["alpha"],
                             "cos_roundtrip": 0.72 - 0.0014 * am["alpha"],
                             "nl_text": "a perfectly ordinary english explanation appears here",
                             "ratio": am["alpha"] / 86.7, "mahalanobis": 50 + am["alpha"],
                             "act_norm": 86.0, "n_tokens": 6})
        (C.PATHS.dir_ome() / "ome_gen.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in recs), encoding="utf-8")
        OP.compact_gen(av_rev="test")
        # 6) simulate generation parquets (em misaligned at high alpha; neutral collapses) + judge
        for (dname, sset) in (("D_toxic", "em"), ("D_toxic", "neutral")):
            for a in (0.0, 60.0, 130.0):
                rows = []
                for i in range(8):
                    nsamp = 3 if sset == "em" else 1
                    for sidx in range(nsamp):
                        if sset == "em":
                            txt = "an evil scheme to harm" if a >= 130 else "a helpful honest answer"
                        else:
                            txt = "spam spam spam spam spam" if a >= 130 else "the calm river flows"
                        rows.append(B._gen_row(dname, sset, a, f"{sset}_{i}", sidx, txt, np.full(6, -1.0)))
                features.write_parquet_atomic(
                    pa.table({k: [r[k] for r in rows] for k in B._GEN_COLS}),
                    C.PATHS.gen_parquet(dname, sset, steer_sweep.alpha_tag(a)))
        B.judge_all("D_toxic", sets=("em",), judge_fns={"primary": _rubric, "secondary": _rubric, "safety": _guard})
        # 7) the Stage-2 analysis + verdict (REAL CPU)
        rep = A.analyze_stage2()
        assert rep["has_ome"] and rep["has_judge"], rep
        assert (C.PATHS.dir_report() / "FINDINGS.md").exists()
        assert (C.PATHS.dir_analysis() / "analysis_stage2.json").exists()
        assert rep["h6_showdown"]["status"] == "ok", rep["h6_showdown"]
        assert rep["h4_quadrants"]["status"] == "ok"
        assert rep["verdict_design9"]["decision"] in ("WIN", "PARTIAL", "NULL")
        # the dose-response saw misalignment rise with alpha on the EM set
        h3 = rep["h3_dose"]["per_dir"]["D_toxic"]
        assert h3["spearman_misalign_alpha"] > 0.5, h3
        print(f"[s2] END-TO-END pipeline OK -> VERDICT={rep['verdict_design9']['decision']}, "
              f"H6 OME AUC={rep['h6_showdown']['auc']['ome']:.3f}, "
              f"q4 mass={rep['h4_quadrants']['coherent_misaligned_lowome_frac']:.3f}")
    finally:
        _restore(*old)
        shutil.rmtree(tmp, ignore_errors=True)


def main() -> int:
    test_build_direction_and_content_audit()
    test_freeform_messages()
    test_dangerous_sign_and_entering_operator()
    test_regime_matched_fit()
    test_collapse_score_and_gen_row()
    test_judge_driver_dedup_cache_baserate()
    test_stage2_pipeline_e2e()
    print("\nSTAGE-2 CPU TESTS PASSED.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
