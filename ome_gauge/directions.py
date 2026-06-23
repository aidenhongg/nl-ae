"""P0 — steering directions (CPU, $0).

Builds the unit mean-difference directions at L20 from the cached benign a0 activations and
audits them. Stage 1 builds D_correct (the capability axis -> quadrant 3 candidate) and
D_random_0..k (the control / lower bound). The content directions D_toxic/D_refusal/
D_sycophancy are Stage-2 (deferred; they need a vendored contrast harvest, config.json
directions.stage2_deferred).

Discipline (QUESTIONS s3.4): directions are built on the benign trainval split (fit_split_mask),
DISJOINT from the test eval subset, so steering the eval rows along a direction is never
in-sample. Each direction is stored unit-norm with full provenance; orthogonality is audited so
results never silently conflate two axes (s3.5).

Gate P0: every vector unit-norm (||v||-1 < 1e-6); pairwise cosines reported (flag |cos|>0.5);
D_correct separates correct/incorrect rows on the HELD-OUT test split (AUC well above 0.5) and is
positively aligned with the independent know-probe knowledge axis (the "contrast machinery is
wired to the same activations" sanity).

CLI:  python -m ome_gauge.directions build      # write dirs.npz + dirs_manifest.json
      python -m ome_gauge.directions audit      # re-print orthogonality + sanity from cache
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from ome_gauge import config as C
from src import features, fve_analysis, steer_sweep, lang_steer  # probe load, rank stats, sha/atomic, kappa import

UNIT_TOL = 1e-6


# ----------------------------- math helpers ---------------------------------

def _unit(v: np.ndarray) -> tuple[np.ndarray, float]:
    """Return (unit vector, raw L2 norm). Raw norm is provenance (the DiM magnitude)."""
    raw = float(np.linalg.norm(v))
    if raw == 0.0:
        raise ValueError("cannot unit-normalize a zero vector")
    return (v / raw).astype(np.float64), raw


def mean_difference(h: np.ndarray, pos: np.ndarray, neg: np.ndarray) -> np.ndarray:
    """mean(h[pos]) - mean(h[neg]); pos/neg are boolean masks over rows of h."""
    assert pos.any() and neg.any(), "empty class in mean-difference"
    return h[pos].mean(0) - h[neg].mean(0)


def auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """ROC-AUC = P(score[pos] > score[neg]) via the rank (Mann-Whitney) identity; ties averaged."""
    labels = np.asarray(labels, bool)
    n_pos, n_neg = int(labels.sum()), int((~labels).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = fve_analysis._rankdata(np.asarray(scores, float))
    return float((ranks[labels].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


# ----------------------------- direction builders ---------------------------

def build_d_correct(h: np.ndarray, fit: np.ndarray, correct: np.ndarray) -> tuple[np.ndarray, dict]:
    """D_correct = unit(mean(h[correct & fit]) - mean(h[incorrect & fit])) at L20."""
    pos, neg = fit & correct, fit & (~correct)
    raw_vec = mean_difference(h, pos, neg)
    v, raw = _unit(raw_vec)
    prov = {"kind": "mean_difference", "base_cohort": "a0", "split": "trainval(fit)",
            "label": "y_tilde==answer_index", "n_pos": int(pos.sum()), "n_neg": int(neg.sum()),
            "raw_norm": raw}
    return v, prov


def build_d_know(h: np.ndarray, fit: np.ndarray) -> np.ndarray:
    """Independent knowledge axis = unit mean-diff between rows the KNOW probe gets right vs wrong
    (argmax(h@Wk.T+bk) == answer_index). Used only as the D_correct cross-check (not steered)."""
    Wk, bk = features.load_probe(C.PATHS.feat, "example_level", "know", C.LAYER)
    know_pred = (h @ Wk.T + bk).argmax(1)
    know_correct = know_pred == C.gt_answer_index()
    v, _ = _unit(mean_difference(h, fit & know_correct, fit & (~know_correct)))
    return v


def build_d_random(d: int, seed: int) -> tuple[np.ndarray, dict]:
    """Seeded random unit vector (norm-matched by construction: all directions are unit)."""
    rng = np.random.default_rng(seed)
    v, _ = _unit(rng.standard_normal(d))
    return v, {"kind": "random_unit", "seed": int(seed)}


# ----------------------------- Stage-2 content directions -------------------
# D_toxic / D_refusal / D_sycophancy = unit(mean(h_pos_last) - mean(h_neg_last)) at L20, harvested
# at the LAST token of free-form contrast prompts (CAA; PLAN_stage2 s4). The harvest is pod-gated
# (needs Qwen); build_direction (the pure mean-diff) is CPU-tested on synthetic acts.

def build_direction(name: str, h_pos: np.ndarray, h_neg: np.ndarray) -> tuple[np.ndarray, dict]:
    """Content direction = unit(mean(h_pos) - mean(h_neg)). h_pos/h_neg are [n_i, d] harvested
    last-token acts of the +/- contrast class. Reuses mean_difference geometry (all-row masks)."""
    h_pos = np.asarray(h_pos, np.float64); h_neg = np.asarray(h_neg, np.float64)
    assert h_pos.ndim == 2 and h_neg.ndim == 2 and h_pos.shape[1] == h_neg.shape[1], "shape"
    assert h_pos.shape[0] and h_neg.shape[0], f"{name}: empty contrast class"
    v, raw = _unit(h_pos.mean(0) - h_neg.mean(0))
    prov = {"kind": "content_mean_difference", "harvest_position": C.STAGE2["harvest_position"],
            "n_pos": int(h_pos.shape[0]), "n_neg": int(h_neg.shape[0]), "raw_norm": raw,
            "dangerous_sign": int(C.DANGEROUS_SIGN.get(name, 1))}
    return v, prov


def _freeform_messages(text: str, system: str | None = None) -> list[dict]:
    """Chat-message list for one free-form user turn (pure; no kappa/tokenizer -> unit-testable)."""
    msgs = [{"role": "system", "content": system}] if system else []
    return msgs + [{"role": "user", "content": str(text)}]


def freeform_prompt(text: str, example_id: str, tokenizer, system: str | None = None):
    """Build a kappa RenderedPrompt from a raw user string — NO MCQ options / answer cue — that is
    compatible with MF.make_batches + lm.model.generate (only .example_id + .input_ids are load-
    bearing downstream). add_generation_prompt ends the string at the assistant turn so a free
    continuation starts clean (no answer cue). Reused by the content-dir + clean prompt-set harvest
    and the Stage-2 generation driver."""
    _cfg, _ds, prompt, _MF, _gen = lang_steer._import_kappa()
    s = tokenizer.apply_chat_template(_freeform_messages(text, system),
                                      tokenize=False, add_generation_prompt=True)
    ids = tokenizer.encode(s, add_special_tokens=False)   # specials already in the template
    return prompt.RenderedPrompt(example_id=str(example_id), text=s, input_ids=list(ids),
                                 symbol_token_ids={}, answer_position="last",
                                 prompt_format_id="freeform_v1", template_version="tmpl.v1")


def harvest_l20_lasttoken(prompts, lm=None, *, batch_size: int = 16, parity: bool = True) -> np.ndarray:
    """[pod] Capture the L20 last-token residual for a list of RenderedPrompts -> [n, d] fp64 in
    prompt order. Reuses MF.register_residual_hooks capture mode (edit_fn=None -> hooks.buffer[20]);
    the same raw block-19 residual space as the benign a0 cache, so h_clean + alpha*v is consistent
    with the Stage-1 directions. Parity-gated (content-independent)."""
    import torch
    cfg_k, _ds, _prompt, MF, _gen = lang_steer._import_kappa()
    cfg = cfg_k.load_config(lang_steer._EXP04 / "experiment.yaml")
    if lm is None:
        lm = MF.load_model(cfg.model.model_id, dtype=cfg.model.dtype)
    prompts = list(prompts)
    if parity and prompts:
        MF.assert_parity(MF.run_parity_check(lm, prompts[:min(8, len(prompts))]))
    rim = {p.example_id: i for i, p in enumerate(prompts)}
    out = np.empty((len(prompts), C.D_MODEL), np.float64)
    for batch in MF.make_batches(prompts, lm.tokenizer.pad_token_id, batch_size, lm.device, rim):
        hooks = MF.register_residual_hooks(lm, [C.LAYER], batch.pos_last, edit_fn=None)
        try:
            with torch.inference_mode():
                lm.model(input_ids=batch.input_ids, attention_mask=batch.attention_mask, use_cache=False)
            h = hooks.buffer[C.LAYER].float().cpu().numpy()
        finally:
            hooks.remove()
        for i, r in enumerate(batch.row_indices):
            out[r] = h[i]
    return out


def harvest_contrasts(lm=None, sets=None) -> dict[str, dict]:
    """[pod] For each contrast set, render pos/neg as free-form prompts and harvest L20 last-token
    -> {dir_name: {"pos": [n,d], "neg": [m,d], "n_pos", "n_neg", "source", "source_sha"}}. Loads the
    model once and threads it through. The input to build_all(content_acts=...)."""
    cfg_k, _ds, _prompt, MF, _gen = lang_steer._import_kappa()
    cfg = cfg_k.load_config(lang_steer._EXP04 / "experiment.yaml")
    if lm is None:
        lm = MF.load_model(cfg.model.model_id, dtype=cfg.model.dtype)
    out: dict[str, dict] = {}
    for cname in (sets or C.CONTRAST_SETS):
        spec = C.STAGE2["contrast_sets"][cname]
        dir_name = spec["dir"]
        ids, pos, neg = C.load_contrast_pairs(cname)
        prompts = ([freeform_prompt(t, f"{cname}_pos_{i}", lm.tokenizer) for i, t in enumerate(pos)]
                   + [freeform_prompt(t, f"{cname}_neg_{i}", lm.tokenizer) for i, t in enumerate(neg)])
        acts = harvest_l20_lasttoken(prompts, lm=lm)
        out[dir_name] = {"pos": acts[:len(pos)], "neg": acts[len(pos):],
                         "n_pos": len(pos), "n_neg": len(neg), "source": spec.get("source"),
                         "source_sha": steer_sweep.sha256_file(C.PATHS.contrast_jsonl(cname))}
    return out


def harvest_clean(set_name: str, lm=None, *, batch_size: int = 16) -> tuple[np.ndarray, dict]:
    """[pod] S2.P1 clean last-token harvest for one prompt set (em / neutral / benign_calib) ->
    h_clean_<set>.npy [n_prompts, d] + a prompt-id manifest. The OME entering-state base and, for
    benign_calib, the regime-matched detector/calibration cohort. Reuses the S2.P0 harvest pass."""
    cfg_k, _ds, _prompt, MF, _gen = lang_steer._import_kappa()
    cfg = cfg_k.load_config(lang_steer._EXP04 / "experiment.yaml")
    if lm is None:
        lm = MF.load_model(cfg.model.model_id, dtype=cfg.model.dtype)
    recs = C.load_prompt_set(set_name)
    prompts = [freeform_prompt(r["text"], r["prompt_id"], lm.tokenizer) for r in recs]
    acts = harvest_l20_lasttoken(prompts, lm=lm, batch_size=batch_size)
    out_dir = C.PATHS.dir_steer(); out_dir.mkdir(parents=True, exist_ok=True)
    steer_sweep.atomic_save_npy(C.PATHS.h_clean(set_name), acts.astype(np.float32))
    manifest = {"schema_version": "ome_gauge.h_clean.v1", "config_hash": C.CONFIG_HASH,
                "set": set_name, "n_prompts": len(recs), "layer": C.LAYER, "d_model": C.D_MODEL,
                "harvest_position": C.STAGE2["harvest_position"],
                "prompt_ids": [r["prompt_id"] for r in recs],
                "source_sha": steer_sweep.sha256_file(C.PATHS.prompt_jsonl(set_name))}
    features.write_json_atomic(manifest, C.PATHS.clean_manifest(set_name))
    print(f"[S2.P1] h_clean_{set_name}.npy: {len(recs)} prompts -> {C.PATHS.h_clean(set_name).name}")
    return acts, manifest


# ----------------------------- audit ----------------------------------------

def audit_orthogonality(dirs: dict[str, np.ndarray]) -> dict[str, float]:
    """Pairwise |cos| between every direction pair (all unit -> cos == dot)."""
    names = list(dirs)
    out = {}
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            out[f"{names[i]}|{names[j]}"] = float(dirs[names[i]] @ dirs[names[j]])
    return out


# ----------------------------- build + persist ------------------------------

def build_all(content_acts: dict[str, dict] | None = None) -> tuple[dict[str, np.ndarray], dict]:
    """Build every Stage-1 direction + the full provenance/sanity manifest (pure CPU). If
    `content_acts` is supplied (the pod harvest_contrasts output: dir_name -> {pos, neg, ...}), also
    build the Stage-2 content directions and fold them into dirs.npz + the orthogonality audit so a
    single manifest carries every direction. The Stage-1 dirs are deterministic from the a0 cache,
    so re-running with content_acts reproduces them bit-for-bit (PLAN_stage2 s2.3)."""
    h = C.load_benign_a0()                 # [N,d] fp64
    fit = C.fit_split_mask()               # trainval (disjoint from test eval)
    test = C.test_mask()                   # held-out evaluation rows
    correct = C.correctness_label()

    names = C.CONFIG["directions"]["stage1"]
    dirs: dict[str, np.ndarray] = {}
    prov: dict[str, dict] = {}

    # D_correct (+ all D_random_i)
    for name in names:
        if name == "D_correct":
            v, p = build_d_correct(h, fit, correct)
        elif name.startswith("D_random_"):
            idx = int(name.rsplit("_", 1)[1])
            v, p = build_d_random(h.shape[1], C.RANDOM_SEED + idx)
        else:
            raise ValueError(f"Stage-1 cannot build {name!r} (content dirs are Stage-2 deferred)")
        dirs[name], prov[name] = v, p

    # Stage-2 content directions from the harvested contrast acts (optional)
    for dir_name, acts in (content_acts or {}).items():
        v, p = build_direction(dir_name, acts["pos"], acts["neg"])
        p["source"] = acts.get("source"); p["source_sha"] = acts.get("source_sha")
        dirs[dir_name], prov[dir_name] = v, p

    # ---- gate: unit-norm ----
    for name, v in dirs.items():
        err = abs(float(np.linalg.norm(v)) - 1.0)
        assert err < UNIT_TOL, f"{name} not unit-norm (|err|={err:.2e})"

    # ---- sanity: D_correct separation on HELD-OUT test + know-probe alignment ----
    proj = h @ dirs["D_correct"]
    sep_auc_test = auc(proj[test], correct[test])
    d_know = build_d_know(h, fit)
    know_cos = float(dirs["D_correct"] @ d_know)
    rand_auc_test = float(np.mean([auc((h @ dirs[n])[test], correct[test])
                                   for n in dirs if n.startswith("D_random_")]))

    ortho = audit_orthogonality(dirs)
    flagged = {k: c for k, c in ortho.items() if abs(c) > C.ORTHO_FLAG}

    manifest = {
        "schema_version": "ome_gauge.dirs.v1",
        "config_hash": C.CONFIG_HASH,
        "model_id": C.CONFIG["model_id"], "layer": C.LAYER, "d_model": C.D_MODEL,
        "source_a0": str(C.PATHS.a0_npy().name),
        "n_fit": int(fit.sum()), "n_test": int(test.sum()),
        "directions": list(dirs),
        "provenance": prov,
        "orthogonality_abs_cos": ortho,
        "orthogonality_flagged_gt_0p5": flagged,
        "sanity": {
            "d_correct_sep_auc_test": sep_auc_test,
            "d_random_sep_auc_test_mean": rand_auc_test,
            "d_correct_vs_know_cos": know_cos,
        },
    }
    if content_acts:
        # Stage-2 audit: each content dir's signed n + cos vs the capability axis D_correct.
        manifest["content"] = {dn: {"n_pos": prov[dn]["n_pos"], "n_neg": prov[dn]["n_neg"],
                                    "dangerous_sign": prov[dn]["dangerous_sign"],
                                    "cos_vs_d_correct": float(dirs[dn] @ dirs["D_correct"])}
                               for dn in content_acts}
        # Gate S2.P0: flag a content dir that rides the capability axis (|cos vs D_correct| > 0.5);
        # report it as a finding, never silently conflate "misaligned" with "wrong" (QUESTIONS s3.5).
        manifest["content_rides_capability"] = {
            dn: c["cos_vs_d_correct"] for dn, c in manifest["content"].items()
            if abs(c["cos_vs_d_correct"]) > C.ORTHO_FLAG}
    return dirs, manifest


def write_dirs(dirs: dict[str, np.ndarray], manifest: dict) -> None:
    import os
    out = C.PATHS.dir_directions()
    out.mkdir(parents=True, exist_ok=True)
    # dirs.npz holds raw vectors -> off-repo/S3 by .gitignore (*.npy); the manifest (*.json) is tracked.
    tmp = out / "dirs.tmp.npz"   # ".npz" suffix so np.savez does not append another
    np.savez(tmp, **{k: v.astype(np.float32) for k, v in dirs.items()})
    os.replace(tmp, C.PATHS.dirs_npz())
    features.write_json_atomic(manifest, C.PATHS.dirs_manifest())


def cmd_build(_args) -> int:
    dirs, manifest = build_all()
    write_dirs(dirs, manifest)
    s = manifest["sanity"]
    print(f"[P0] wrote {len(dirs)} directions -> {C.PATHS.dirs_npz().name} + manifest")
    print(f"[P0] D_correct sep AUC(test)={s['d_correct_sep_auc_test']:.3f} "
          f"(random {s['d_random_sep_auc_test_mean']:.3f}); "
          f"cos(D_correct, D_know)={s['d_correct_vs_know_cos']:+.3f}")
    print(f"[P0] orthogonality flagged (|cos|>{C.ORTHO_FLAG}): "
          f"{manifest['orthogonality_flagged_gt_0p5'] or 'none'}")
    # gate the sanity (soft floors; D_correct must be a real, knowledge-aligned axis)
    assert s["d_correct_sep_auc_test"] > 0.60, "D_correct fails to separate correctness on held-out test"
    assert s["d_correct_vs_know_cos"] > 0.20, "D_correct not aligned with the know-probe axis"
    print("[P0] GATE PASS: unit-norm + separation + know-alignment + orthogonality audited.")
    return 0


def cmd_audit(_args) -> int:
    if not C.PATHS.dirs_manifest().exists():
        print("[P0] no dirs_manifest.json — run `build` first."); return 1
    m = json.loads(C.PATHS.dirs_manifest().read_text(encoding="utf-8"))
    print(json.dumps({"sanity": m["sanity"], "orthogonality": m["orthogonality_abs_cos"],
                      "flagged": m["orthogonality_flagged_gt_0p5"],
                      "content": m.get("content"),
                      "content_rides_capability": m.get("content_rides_capability")}, indent=2))
    return 0


def cmd_harvest_dirs(_args) -> int:
    """[S2.P0, pod] Harvest the content directions from the vendored contrast sets and (re)write
    dirs.npz with every direction + the unified manifest. Gate: unit-norm + the content-vs-D_correct
    capability-axis audit."""
    content = harvest_contrasts()
    dirs, manifest = build_all(content_acts=content)
    write_dirs(dirs, manifest)
    print(f"[S2.P0] wrote {len(dirs)} directions (incl. {list(content)}) -> {C.PATHS.dirs_npz().name}")
    for dn, c in (manifest.get("content") or {}).items():
        print(f"[S2.P0] {dn}: n_pos={c['n_pos']} n_neg={c['n_neg']} sign={c['dangerous_sign']:+d} "
              f"cos(·,D_correct)={c['cos_vs_d_correct']:+.3f}")
    rides = manifest.get("content_rides_capability") or {}
    print(f"[S2.P0] content dirs riding the capability axis (|cos·D_correct|>{C.ORTHO_FLAG}): "
          f"{rides or 'none'}")
    return 0


def cmd_harvest_clean(args) -> int:
    """[S2.P1, pod] Harvest the clean L20 last-token acts for one prompt set (em/neutral/benign_calib)."""
    harvest_clean(args.set)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="P0/S2.P0-P1 OME-GAUGE directions + harvest.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("build"); sub.add_parser("audit")
    sub.add_parser("harvest-dirs")                                  # [S2.P0, pod] content directions
    hc = sub.add_parser("harvest-clean")                           # [S2.P1, pod] clean prompt-set acts
    hc.add_argument("--set", required=True, choices=C.PROMPT_SETS)
    args = ap.parse_args(argv)
    return {"build": cmd_build, "audit": cmd_audit, "harvest-dirs": cmd_harvest_dirs,
            "harvest-clean": cmd_harvest_clean}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
