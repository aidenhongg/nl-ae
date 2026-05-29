"""CPU validation for src/features.py + src/nla_enrich.py on tiny SYNTHETIC data.

No GPU, no S3, no real model. Builds a small self-consistent set of inputs (probes, h_orig,
steered arrays, norms, examples/predictions/generations, nl/ + fve/ datapoint files), then
runs the native feature build + in-place enrich and asserts the things that matter for THIS
change:

  * F2 is wired to the ACTUAL GENERATION, not y_tilde: model_answer_index == generation index
    (incl. a row where the generation DISAGREES with the readout), and the three correctness
    flavors (model_gen_correct / model_readout_correct / agree_gen_readout) are computed right;
  * unparseable generations (index -1) become model_answer_index=-1 / symbol="" / not-correct;
  * the FVE per-steer join uses each row's OWN alpha (the old α=0 bug is fixed);
  * enrich is IN-PLACE + append-only + idempotent (re-run does not collide / drift), original
    columns preserved, feature columns non-null, row counts unchanged;
  * the alpha=0 steered cross-check (== datapoint_features.kl_pred_know) holds.

Run: python tests/test_features.py
"""
import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from src import features as F
from src import nla_enrich as E

SYM = "ABCD"
D = F.D_MODEL  # load_probe asserts (K, 3584); keep D real, n tiny
ALPHAS = F.ALPHAS


def _save_npz(path: Path, W, b):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, W=W.astype(np.float32), b=b.astype(np.float32))


def _save_npy(path: Path, arr):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, arr.astype(np.float32))


def build_inputs(P: F.FeaturePaths, n: int, rng) -> dict:
    order = [f"tqa-{i:04d}-p0" for i in range(n)]
    qids = [f"tqa-{i:04d}" for i in range(n)]
    test_rows = list(range(n // 2, n))  # second half is the test split

    # ---- emb/example_ids.json + h_orig ----
    P.emb_dir.mkdir(parents=True, exist_ok=True)
    (P.emb_dir / "example_ids.json").write_text(json.dumps(
        {"order": "activation_cache_row_order", "n": n, "example_ids": order,
         "test_row_indices": test_rows}), encoding="utf-8")
    h0 = rng.standard_normal((n, D)).astype(np.float32) * 5.0
    _save_npy(P.h_orig(), h0)

    # ---- probes (both schemes) ----
    probes = {}
    for scheme in ("example_level", "question_disjoint"):
        for tg in ("know", "pred"):
            W = rng.standard_normal((F.K, D)) * 0.05
            b = rng.standard_normal(F.K) * 0.1
            _save_npz(P.probe_npz(scheme, tg), W, b)
            probes[(scheme, tg)] = (W, b)

    # ---- ground truth + readout (predictions) + generation ----
    gt = np.array([i % 4 for i in range(n)], dtype=np.int64)
    y_tilde = gt.copy()                          # readout mostly == gt
    y_tilde[0] = (gt[0] + 1) % 4                 # row 0: readout wrong
    logits = rng.standard_normal((n, F.K)) * 2.0
    p_model = np.exp(logits) / np.exp(logits).sum(1, keepdims=True)

    gen_index = y_tilde.copy()                   # generation mostly tracks readout
    gen_index[1] = (y_tilde[1] + 2) % 4          # row 1: generation DISAGREES with readout
    gen_index[2] = -1                            # row 2: unparseable generation
    gen_symbol = ["" if k < 0 else SYM[k] for k in gen_index]
    gen_text = ["(no answer)" if k < 0 else f"{SYM[k]}) generated text {i}" for i, k in enumerate(gen_index)]
    gen_first = ["" if k < 0 else SYM[k] for k in gen_index]

    # examples.jsonl
    with open(P.examples(), "w", encoding="utf-8") as f:
        for i, eid in enumerate(order):
            opts = [{"symbol": SYM[j], "text": f"option {SYM[j]} for q{i}"} for j in range(4)]
            f.write(json.dumps({"example_id": eid, "question": f"question {i}?", "options": opts,
                                "answer_index": int(gt[i]),
                                "source_meta": {"question_id": qids[i]}}) + "\n")
    # predictions.parquet
    pq.write_table(pa.table({
        "example_id": pa.array(order, pa.string()),
        "y_tilde": pa.array(y_tilde, pa.int32()),
        "p_model": pa.array([list(map(float, r)) for r in p_model], pa.list_(pa.float32())),
        "logits_symbols": pa.array([list(map(float, r)) for r in logits], pa.list_(pa.float32())),
    }), P.predictions())
    # generations.parquet
    pq.write_table(pa.table({
        "example_id": pa.array(order, pa.string()),
        "model_gen_text": pa.array(gen_text, pa.string()),
        "model_gen_symbol": pa.array(gen_symbol, pa.string()),
        "model_gen_index": pa.array(gen_index.astype(np.int8), pa.int8()),
        "model_gen_first_token": pa.array(gen_first, pa.string()),
        "model_gen_method": pa.array(["greedy_generate@maxnew48"] * n, pa.string()),
        "gen_model_revision": pa.array(["deadbeef"] * n, pa.string()),
    }), P.generations())

    # splits.json (both schemes; train/val/test over all eids)
    def scheme_split():
        tr = order[: n // 2]; va = order[n // 2: n // 2 + max(1, n // 4)]; te = [order[i] for i in test_rows]
        return {"train": tr, "val": va, "test": te,
                "sizes": {"train": len(tr), "val": len(va), "test": len(te)}}
    splits = {"schema_version": "sp.v1", "n_examples": n, "row_index": {e: i for i, e in enumerate(order)},
              "schemes": {"example_level": scheme_split(), "question_disjoint": scheme_split()}}
    P.splits().write_text(json.dumps(splits), encoding="utf-8")

    # ---- steered arrays + norms.parquet ----
    in_subset = np.zeros(n, dtype=bool)
    subset_rows = test_rows[: max(1, len(test_rows) // 2)]
    in_subset[subset_rows] = True
    norm_rows = {k: [] for k in ("example_id", "row_index", "split", "in_subset", "alpha",
                                 "h_norm", "dh_norm", "ratio", "cos_h_hp")}
    for a in ALPHAS:
        tag = F.alpha_tag(a)
        hp = h0.copy() if a == 0.0 else (h0 + (a * 0.01) * rng.standard_normal((n, D))).astype(np.float32)
        _save_npy(P.steered_npy(tag), hp)
        for i in range(n):
            norm_rows["example_id"].append(order[i]); norm_rows["row_index"].append(i)
            norm_rows["split"].append("test" if i in test_rows else "trainval")
            norm_rows["in_subset"].append(bool(in_subset[i])); norm_rows["alpha"].append(float(a))
            norm_rows["h_norm"].append(1.0); norm_rows["dh_norm"].append(float(a))
            norm_rows["ratio"].append(0.0); norm_rows["cos_h_hp"].append(1.0)
    pq.write_table(pa.table(norm_rows, schema=pa.schema([
        ("example_id", pa.string()), ("row_index", pa.int32()), ("split", pa.string()),
        ("in_subset", pa.bool_()), ("alpha", pa.float64()), ("h_norm", pa.float64()),
        ("dh_norm", pa.float64()), ("ratio", pa.float64()), ("cos_h_hp", pa.float64())])), P.norms())

    # ---- datapoint files: nl/orig, nl/steered_a*, nl/headline_a*, fve/per_row ----
    def write_nl(path, rows, alpha=None):
        cols = {"example_id": [order[i] for i in rows], "row_index": [i for i in rows],
                "split": ["test" if i in test_rows else "trainval" for i in rows],
                "source": ["av" for _ in rows], "nl_text": [f"nl {i}" for i in rows]}
        if alpha is not None:
            cols["alpha"] = [float(alpha) for _ in rows]
        F.write_parquet_atomic(pa.table(cols), path)

    write_nl(P.nl_dir() / "orig.parquet", list(range(n)))           # all rows, no alpha
    for a in ALPHAS:
        write_nl(P.nl_dir() / f"steered_a{F.alpha_tag(a)}.parquet", subset_rows, alpha=a)
    for a in (2.0, 10.0, 30.0):
        write_nl(P.nl_dir() / f"headline_a{F.alpha_tag(a)}.parquet", subset_rows, alpha=a)
    # fve/per_row: subset rows x all alphas, each row carries its OWN alpha
    fve = {"alpha": [], "row_index": [], "example_id": [], "cos": [], "mse": []}
    for a in ALPHAS:
        for i in subset_rows:
            fve["alpha"].append(float(a)); fve["row_index"].append(i); fve["example_id"].append(order[i])
            fve["cos"].append(0.9); fve["mse"].append(0.1)
    F.write_parquet_atomic(pa.table(fve, schema=pa.schema([
        ("alpha", pa.float64()), ("row_index", pa.int32()), ("example_id", pa.string()),
        ("cos", pa.float64()), ("mse", pa.float64())])), P.fve_dir() / "per_row.parquet")

    return {"order": order, "gt": gt, "y_tilde": y_tilde, "gen_index": gen_index,
            "subset_rows": subset_rows, "test_rows": test_rows}


def main() -> int:
    root = Path(tempfile.mkdtemp(prefix="nla_feat_itest_"))
    P = F.FeaturePaths(inputs=root / "inputs", emb_dir=root / "emb",
                       probes_dir=root / "probes", out=root / "out")
    P.inputs.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    n = 8
    meta = build_inputs(P, n, rng)
    order, gt, y_tilde, gen_index = meta["order"], meta["gt"], meta["y_tilde"], meta["gen_index"]
    print(f"[feattest] synthetic inputs built (n={n}, D={D})")

    # ---- build features (F1+F2[gen]+F3) ----
    F.build_datapoint_features(P, include_qd=True, build_utc="test", n_rows=n)
    dp = pq.read_table(P.datapoint_features()).to_pydict()

    # F2 wired to GENERATION (not y_tilde) — the whole point of this change
    assert dp["model_answer_index"] == [int(x) for x in gen_index], "model_answer_index must be the generation"
    assert dp["model_gen_method"][0] == "greedy_generate@maxnew48"
    assert dp["gen_model_revision"][0] == "deadbeef"
    assert dp["y_tilde"] == [int(x) for x in y_tilde], "y_tilde companion must be retained"
    # row 1: generation disagrees with readout -> headline answer follows the generation
    assert gen_index[1] != y_tilde[1]
    assert dp["model_answer_index"][1] == int(gen_index[1]) != int(y_tilde[1])
    assert dp["agree_gen_readout"][1] is False
    # correctness flavors
    for i in range(n):
        assert dp["model_gen_correct"][i] == (int(gen_index[i]) == int(gt[i]))
        assert dp["model_readout_correct"][i] == (int(y_tilde[i]) == int(gt[i]))
        assert dp["agree_gen_readout"][i] == (int(gen_index[i]) == int(y_tilde[i]))
    # row 2: unparseable generation
    assert gen_index[2] == -1
    assert dp["model_answer_index"][2] == -1 and dp["model_symbol"][2] == "" and dp["model_answer_text"][2] == ""
    assert dp["model_gen_correct"][2] is False
    # parseable rows: model_answer_text == the chosen option's text
    assert dp["model_answer_text"][0] == f"option {SYM[gen_index[0]]} for q0"
    print("[feattest] F2 = actual generation OK (disagreement + unparseable + correctness flavors)")

    # ---- steered divergence + alpha=0 cross-check ----
    F.build_steered_divergence(P, build_utc="test", n_rows=n)
    sd = pq.read_table(P.steered_divergence()).to_pydict()
    assert len(sd["alpha"]) == n * len(ALPHAS)
    a0 = np.isclose(np.asarray(sd["alpha"]), 0.0)
    assert np.allclose(np.asarray(sd["d_kl_pred_know"])[a0], 0.0), "alpha=0 d_kl must be 0"
    print(f"[feattest] steered_divergence OK ({len(sd['alpha'])} rows, alpha=0 cross-check)")

    # ---- enrich in-place ----
    idx = E.enrich_in_place(P, include_qd=True, build_utc="test", n_rows=n)
    # FVE join uses each row's OWN alpha (old bug joined everything at alpha=0)
    fve = pq.read_table(P.fve_dir() / "per_row.parquet").to_pydict()
    sd_pos = {(int(r), round(float(a), 6)): i
              for i, (r, a) in enumerate(zip(sd["row_index"], sd["alpha"]))}
    checked_nonzero = 0
    for j in range(len(fve["alpha"])):
        r, a = int(fve["row_index"][j]), round(float(fve["alpha"][j]), 6)
        exp = float(np.asarray(sd["d_kl_pred_know"])[sd_pos[(r, a)]])
        assert abs(float(fve["d_kl_pred_know"][j]) - exp) < 1e-6, f"fve row {r}@{a}: d_kl join wrong"
        if a > 0 and abs(exp) > 1e-9:
            checked_nonzero += 1
    assert checked_nonzero > 0, "no nonzero-alpha FVE rows exercised the fix"
    # if FVE had been joined at alpha=0 (the bug), every d_kl would be 0 -> this proves the fix
    assert any(abs(float(x)) > 1e-9 for x in fve["d_kl_pred_know"]), "FVE d_kl all zero -> alpha=0 bug present"
    print(f"[feattest] FVE per-steer join uses own alpha OK ({checked_nonzero} nonzero rows verified)")

    # in-place: original cols preserved, feature cols present + non-null, rows unchanged
    nl = pq.read_table(P.nl_dir() / "orig.parquet")
    assert nl.num_rows == n
    for c in ("example_id", "row_index", "split", "source", "nl_text"):
        assert c in nl.column_names
    for c in ("model_gen_text", "model_gen_correct", "kl_pred_know", "kl_pred_know_steered"):
        assert c in nl.column_names and nl.column(c).null_count == 0
    print("[feattest] enrich in-place OK (originals preserved, features non-null)")

    # idempotency: re-run enrich -> no collision, identical result
    before = pq.read_table(P.nl_dir() / "orig.parquet")
    E.enrich_in_place(P, include_qd=True, build_utc="test", n_rows=n)
    after = pq.read_table(P.nl_dir() / "orig.parquet")
    assert before.equals(after), "re-enrich changed the in-place file (not idempotent)"
    assert after.num_rows == n and set(before.column_names) == set(after.column_names)
    print("[feattest] enrich is idempotent (re-run: no collision, identical)")

    # gate mechanics: enriched check passes; n/balance assertions run
    g = E.Gate()
    E.check_enriched(P, g, require=True)
    assert g.failed == 0, "check_enriched should pass on the in-place files"
    print("[feattest] check_enriched GREEN")

    # check_tables must run without raising (every gate column reference resolves — a typo here would
    # otherwise only surface in Part B). Pass/fail is irrelevant on synthetic data; coverage is the point.
    g2 = E.Gate()
    E.check_tables(P, g2)
    names = [r[0] for r in g2.rows]
    for needle in ("base_acc_readout", "agree_gen_readout", "gen_parse_rate", "base_acc_gen", "steered a2"):
        assert any(needle in nm for nm in names), f"check_tables missing gate {needle!r}"
    print(f"[feattest] check_tables column references all resolve ({len(names)} gates)")

    print("\nNLA FEATURES CPU VALIDATION PASSED -- F2=generation, FVE join fixed, in-place idempotent.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
