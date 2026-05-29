"""NLA-final's final, LOCAL/CPU feature-attach stage (replaces `feature-patch/`).

Consolidates the old patch's ingest + build + validate + enrich + push into one native module:
  1. ingest  : pull the small exp04 sources (examples.jsonl, predictions.parquet, splits.json,
               generations.parquet) into NLA-final/inputs/ + integrity gate (load-bearing join order).
  2. build   : feat/datapoint_features.parquet (F1+F2[generation]+F3-orig) + feat/steered_divergence.parquet
               (via src/features.py).
  3. enrich  : left-join the feature columns IN-PLACE into every nl/*.parquet + fve/per_row.parquet
               (append-only, idempotent) -> the features become columns of the ORIGINAL output.
  4. validate: acceptance gate (readout anchors + the NEW F2=generation gates) — nothing ships unless GREEN.
  5. push    : feat/ -> nla/feat ; overwrite nla/nl/* + nla/fve/per_row.parquet with the enriched files.

`nla_run.py` (GPU pod) and `steer_sweep.py` (local) are UNCHANGED — this stage needs no GPU.

CLI:
  python -m src.nla_enrich [--no-qd] [--no-ingest] [--push] [--dry-run] [--selftest]
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from . import features as F
from . import s3_io

# exact S3 keys (never LIST) -> local filename under inputs/
SOURCE_KEYS: dict[str, str] = {
    "examples.jsonl": "exp04/data/examples.jsonl",
    "predictions.parquet": "exp04/01_cache/acts/predictions.parquet",
    "splits.json": "exp04/data/splits.json",
    "generations.parquet": "exp04/01_cache/acts/generations.parquet",   # NEW (exp04 `generate` stage)
}

# per-steer columns appended (drop join keys + the colliding split/in_subset)
_STEERED_APPEND = ["kl_pred_know_steered", "js_pred_know_steered", "agree_know_pred_steered",
                   "pred_argmax_symbol_steered", "know_argmax_symbol_steered", "d_kl_pred_know"]
_SKIP_PER_EXAMPLE = {"example_id", "row_index"}


# ============================ P0 ingest ========================================

def ingest(P: F.FeaturePaths, *, allow_pull: bool = True) -> dict:
    """Pull exact keys into inputs/ (prefer existing local copies); assert the join invariants."""
    P.inputs.mkdir(parents=True, exist_ok=True)
    prev = {}
    if P.sources_lock().exists():
        prev = json.loads(P.sources_lock().read_text(encoding="utf-8")).get("sources", {})

    lock = {}
    s3 = None
    for name, key in SOURCE_KEYS.items():
        dest = P.inputs / name
        if not dest.exists():
            if not allow_pull:
                raise FileNotFoundError(f"{name} absent and --no-ingest set (expected {dest})")
            if s3 is None:
                s3 = s3_io.make_client()
            s3_io.download(s3, key, dest)
        lock[name] = {"key": key, "sha256": F.sha256_file(dest), "size": dest.stat().st_size}

    for name, rec in lock.items():
        if name in prev and prev[name].get("sha256") != rec["sha256"]:
            print(f"[ingest] WARNING: {name} sha256 changed since last ingest "
                  f"({prev[name].get('sha256','?')[:12]} -> {rec['sha256'][:12]})")

    _integrity_gate(P)
    F.write_json_atomic({
        "schema_version": "nla_features_sources.v2",
        "n_examples": F.N_ROWS,
        "sources": lock,
        "invariants_checked": [
            "len(examples)==6536", "predictions.num_rows==6536", "generations.num_rows==6536",
            "predictions.example_id==generations.example_id==example_ids", "splits.row_index==canonical_order",
        ],
    }, P.sources_lock())
    print(f"[ingest] OK -> {P.sources_lock().name} ({len(lock)} sources, invariants green)")
    return lock


def _integrity_gate(P: F.FeaturePaths) -> None:
    examples = [json.loads(x) for x in P.examples().read_text(encoding="utf-8").splitlines() if x.strip()]
    assert len(examples) == F.N_ROWS, f"examples.jsonl has {len(examples)} (want {F.N_ROWS})"
    order = F.canonical_order(P)

    pred = pq.read_table(P.predictions())
    assert pred.num_rows == F.N_ROWS, f"predictions has {pred.num_rows} rows"
    assert pred.column("example_id").to_pylist() == order, "predictions order != canonical (join invariant)"

    gens = pq.read_table(P.generations())
    assert gens.num_rows == F.N_ROWS, f"generations has {gens.num_rows} rows"
    assert gens.column("example_id").to_pylist() == order, "generations order != canonical (join invariant)"

    splits = json.loads(P.splits().read_text(encoding="utf-8"))
    row_index = splits["row_index"]
    bad = [i for i, eid in enumerate(order) if row_index.get(eid) != i]
    assert not bad, f"splits.row_index disagrees with canonical order at {len(bad)} rows (e.g. {bad[:3]})"


# ============================ enrich in-place ==================================

def _per_steer_alphas(table: pa.Table, nominal_alpha: float | None) -> list[float]:
    """Use the file's OWN alpha column when present (fixes the old FVE α=0 join); else nominal."""
    if "alpha" in table.column_names:
        return [round(float(a), 6) for a in table.column("alpha").to_pylist()]
    return [round(float(nominal_alpha if nominal_alpha is not None else 0.0), 6)] * table.num_rows


def enrich_one(P: F.FeaturePaths, rel_in: str, sub: str, nominal_alpha: float | None,
               feat: pa.Table, sd: pa.Table, pos_eid: dict, pos_ra: dict,
               per_example_cols: list[str]) -> dict:
    src_path = P.out / sub / rel_in
    table = pq.read_table(src_path)
    added_set = set(per_example_cols) | set(_STEERED_APPEND)
    # strip any previously-appended feature columns -> idempotent re-enrich; base == nla_run original.
    base = table.select([c for c in table.column_names if c not in added_set])
    original_columns = list(base.column_names)
    rows = base.num_rows
    assert "example_id" in original_columns and "row_index" in original_columns, \
        f"{rel_in}: missing join key(s) {set(['example_id', 'row_index']) - set(original_columns)}"
    eids = base.column("example_id").to_pylist()
    rowidx = base.column("row_index").to_pylist()

    # per-example join (by example_id)
    try:
        idx_e = pa.array([pos_eid[e] for e in eids], type=pa.int64())
    except KeyError as ex:
        raise AssertionError(f"{rel_in}: example_id {ex} not in datapoint_features (join would null)")
    out = base
    for c in per_example_cols:
        out = out.append_column(c, feat.column(c).take(idx_e))

    # per-steer join (by (row_index, alpha); file's own alpha if present, else nominal/orig=0)
    has_alpha = "alpha" in base.column_names
    alphas = _per_steer_alphas(base, nominal_alpha)
    try:
        idx_s = pa.array([pos_ra[(int(r), a)] for r, a in zip(rowidx, alphas)], type=pa.int64())
    except KeyError as ex:
        raise AssertionError(f"{rel_in}: (row_index,alpha) {ex} not in steered_divergence")
    for c in _STEERED_APPEND:
        out = out.append_column(c, sd.column(c).take(idx_s))

    added_columns = per_example_cols + _STEERED_APPEND
    # ---- per-file gates (append-only construction; assert before overwrite) ----
    assert out.num_rows == rows, f"{rel_in}: row count changed"
    for c in original_columns:
        assert out.column(c).equals(base.column(c)), f"{rel_in}: original column {c} mutated"
    for c in added_columns:
        assert out.column(c).null_count == 0, f"{rel_in}: feature column {c} has nulls"

    F.write_parquet_atomic(out, src_path)  # IN-PLACE overwrite (atomic tmp + os.replace)
    print(f"[enrich] {sub}/{rel_in:<22} {rows} rows  +{len(added_columns)} cols (in-place)")
    return {
        "file": f"{sub}/{rel_in}",
        "rows": rows,
        "original_columns": original_columns,
        "added_columns": added_columns,
        "join_keys": ["example_id"] + (["row_index", "alpha"] if has_alpha
                                       else [f"row_index@alpha={float(nominal_alpha or 0.0):g}"]),
    }


def enrich_in_place(P: F.FeaturePaths, *, include_qd: bool = True, build_utc: str | None = None,
                    n_rows: int = F.N_ROWS) -> dict:
    """Build the two feat/ tables, then left-join their columns IN-PLACE into every datapoint file."""
    F.build_datapoint_features(P, include_qd=include_qd, build_utc=build_utc, n_rows=n_rows)
    F.build_steered_divergence(P, build_utc=build_utc, n_rows=n_rows)

    feat = pq.read_table(P.datapoint_features())
    sd = pq.read_table(P.steered_divergence())
    per_example_cols = [c for c in feat.column_names if c not in _SKIP_PER_EXAMPLE]
    pos_eid = {e: i for i, e in enumerate(feat.column("example_id").to_pylist())}
    pos_ra = {(int(r), round(float(a), 6)): i
              for i, (r, a) in enumerate(zip(sd.column("row_index").to_pylist(), sd.column("alpha").to_pylist()))}

    files = [enrich_one(P, F.NL_ORIG[0], "nl", F.NL_ORIG[1], feat, sd, pos_eid, pos_ra, per_example_cols)]
    for name, a in F.NL_STEERED + F.NL_HEADLINE:
        files.append(enrich_one(P, name, "nl", a, feat, sd, pos_eid, pos_ra, per_example_cols))
    files.append(enrich_one(P, F.FVE_PER_ROW[0], "fve", F.FVE_PER_ROW[1], feat, sd, pos_eid, pos_ra, per_example_cols))

    index = {
        "schema_version": F.SCHEMA_VERSION_ENRICHED,
        "build_utc": build_utc,
        "mode": "in_place",
        "feature_manifest_sha256": F.sha256_file(P.feature_manifest()),
        "config_hash": F.CONFIG_HASH,
        "n_files": len(files),
        "files": files,
        "note": "Features are columns of the ORIGINAL nl/ + fve/ files (in-place, append-only). "
                "The old parallel nla/enriched/ tree is retired (deprecated keys remain by Runpod no-delete).",
    }
    F.write_json_atomic(index, P.enriched_index())
    print(f"[enrich] {len(files)} datapoint files enriched in-place + enriched_index.json")
    return index


# ============================ acceptance gate ==================================

class Gate:
    def __init__(self):
        self.rows: list[tuple] = []
        self.failed = 0

    def check(self, name, got, expected, tol, kind="hard"):
        ok = abs(float(got) - float(expected)) <= tol
        self.failed += 0 if ok else 1
        self.rows.append((name, float(got), float(expected), tol, kind, ok))
        return ok

    def check_floor(self, name, got, floor, kind="hard"):
        ok = float(got) >= floor
        self.failed += 0 if ok else 1
        self.rows.append((name, float(got), float(floor), 0.0, kind + ",floor", ok))
        return ok

    def assert_(self, name, cond: bool):
        self.rows.append((name, None, None, None, "hard", bool(cond)))
        if not cond:
            self.failed += 1
        return cond

    def report(self) -> bool:
        print("\n=== acceptance gates (FEATURES.md §8) ===")
        for name, got, exp, tol, kind, ok in self.rows:
            mark = "PASS" if ok else "FAIL"
            if got is None:
                print(f"  [{mark}] {name}")
            elif "floor" in kind:
                print(f"  [{mark}] {name:<42} got={got:.4f} >= {exp:.4f} ({kind})")
            else:
                print(f"  [{mark}] {name:<42} got={got:.4f} exp={exp:.4f} +/-{tol:g} ({kind})")
        print(f"=== {'ALL GREEN' if self.failed == 0 else f'{self.failed} FAILED'} ===")
        return self.failed == 0


def check_tables(P: F.FeaturePaths, g: Gate) -> None:
    feat = pq.read_table(P.datapoint_features()).to_pydict()
    n = len(feat["example_id"])
    g.assert_("n_examples == 6536", n == F.N_ROWS)

    bal = Counter(int(x) for x in feat["gt_answer_index"])
    g.assert_(f"gt_answer_index balance {dict(sorted(bal.items()))}",
              all(bal.get(k) == v for k, v in F.GT_CLASS_BALANCE.items()))

    test = np.asarray([s == "test" for s in feat["split_el"]], dtype=bool)
    arr = lambda c: np.asarray(feat[c])
    PG = F.GATES_PER_EXAMPLE_TEST
    g.check("base_acc_readout = mean(model_readout_correct)[test]", arr("model_readout_correct")[test].mean(), *PG["base_acc_readout"])
    g.check("know_acc = mean(know_correct)[test]", arr("know_correct")[test].mean(), *PG["know_acc"])
    g.check("pred_acc = mean(pred_matches_model)[test]", arr("pred_matches_model")[test].mean(), *PG["pred_acc"])
    g.check("agr = mean(agree_know_model)[test]", arr("agree_know_model")[test].mean(), *PG["agr"])
    g.check("mean(kl_model_know)[test]", arr("kl_model_know")[test].mean(), *PG["kld_model_know"])

    # NEW (F2 = actual generation). agree_gen_readout is gated over PARSEABLE rows (where a letter was
    # generated); unparseable rows have their own parse-rate gate, so they must not also count as
    # readout-disagreements (FEATURES.md §8).
    gen_idx = arr("model_answer_index")
    parseable = gen_idx >= 0
    g.check_floor("gen_parse_rate (model_answer_index>=0)", float(parseable.mean()), F.GEN_PARSE_RATE_MIN)
    g.check_floor("mean(agree_gen_readout)[parseable]", float(arr("agree_gen_readout")[parseable].mean()), F.AGREE_GEN_READOUT_MIN)
    g.check("base_acc_gen = mean(model_gen_correct)[test]", arr("model_gen_correct")[test].mean(), *F.BASE_ACC_GEN_BAND)

    sd = pq.read_table(P.steered_divergence(),
                       columns=["alpha", "kl_pred_know_steered", "agree_know_pred_steered"]).to_pydict()
    a = np.asarray(sd["alpha"]); kl = np.asarray(sd["kl_pred_know_steered"]); ag = np.asarray(sd["agree_know_pred_steered"])
    for (col, alpha), (exp, tol, kind) in F.GATES_STEERED_ALL.items():
        m = np.isclose(a, alpha)
        v = (ag[m] if col == "agree_know_pred_steered" else kl[m]).mean()
        g.check(f"steered a{alpha:g} mean({col})[all]", v, exp, tol, kind)


def check_enriched(P: F.FeaturePaths, g: Gate, require: bool) -> None:
    if not P.enriched_index().exists():
        if require:
            g.assert_("enriched_index.json present", False)
        else:
            print("  [skip] datapoint files not enriched yet")
        return
    index = json.loads(P.enriched_index().read_text(encoding="utf-8"))
    all_ok = True
    for entry in index["files"]:
        f = pq.read_table(P.out / entry["file"])
        ok = f.num_rows == entry["rows"]
        cols = set(f.column_names)
        for c in entry["original_columns"] + entry["added_columns"]:
            ok = ok and (c in cols)
        for c in entry["added_columns"]:
            ok = ok and f.column(c).null_count == 0
        all_ok = all_ok and ok
        if not ok:
            print(f"  [FAIL] enriched {entry['file']}")
    g.assert_(f"every datapoint file in-place: rows preserved, original+feature cols present, features non-null "
              f"({len(index['files'])} files)", all_ok)


# ============================ selftest =========================================

def selftest(P: F.FeaturePaths) -> int:
    """Independent recompute of the §8 anchors from RAW sources (proves the thresholds)."""
    order = F.canonical_order(P)
    is_test = F.test_mask_example_level(P, order)
    pred = pq.read_table(P.predictions())
    y = np.asarray(pred.column("y_tilde").to_pylist(), int)
    pm = np.asarray(pred.column("p_model").to_pylist(), float)
    ex = {}
    for line in P.examples().read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line); ex[r["example_id"]] = r["answer_index"]
    gold = np.asarray([ex[e] for e in order], int)
    Wk, bk = F.load_probe(P, "example_level", "know"); Wp, bp = F.load_probe(P, "example_level", "pred")
    h0 = F.load_h(P.h_orig())
    pk = F.posteriors(h0, Wk, bk); pp = F.posteriors(h0, Wp, bp)
    ka, pa_ = pk.argmax(1), pp.argmax(1)
    print("[selftest] readout anchors: base_acc=%.4f know_acc=%.4f pred_acc=%.4f agr=%.4f kld=%.4f"
          % ((y[is_test] == gold[is_test]).mean(), (ka[is_test] == gold[is_test]).mean(),
             (pa_[is_test] == y[is_test]).mean(), (ka[is_test] == y[is_test]).mean(),
             F.kl(pm, pk)[is_test].mean()))
    if P.generations().exists():
        gi = np.asarray(pq.read_table(P.generations()).column("model_gen_index").to_pylist(), int)
        ok = gi >= 0
        print("[selftest] generation anchors: base_acc_gen=%.4f agree_gen_readout=%.4f parse_rate=%.4f"
              % ((gi[is_test] == gold[is_test]).mean(), (gi == y).mean(), ok.mean()))
    else:
        print("[selftest] generations.parquet absent -> F2-generation anchors deferred to Part B (GPU)")
    print("[selftest] OK (independent of produced tables)")
    return 0


# ============================ push =============================================

def _push_keys(P: F.FeaturePaths) -> list[tuple[Path, str]]:
    """Local file -> S3 key: all of feat/, plus the enriched nl/* + fve/per_row.parquet."""
    out: list[tuple[Path, str]] = []
    for p in sorted(P.feat_out().rglob("*")):
        if p.is_file() and not p.name.endswith(".tmp"):
            out.append((p, f"nla/feat/{p.relative_to(P.feat_out()).as_posix()}"))
    index = json.loads(P.enriched_index().read_text(encoding="utf-8"))
    for entry in index["files"]:
        rel = entry["file"]  # e.g. "nl/orig.parquet" or "fve/per_row.parquet"
        out.append((P.out / rel, f"nla/{rel}"))
    return out


def _log(P: F.FeaturePaths, msg: str) -> None:
    P.logs().mkdir(parents=True, exist_ok=True)
    line = f"{datetime.now(timezone.utc).isoformat()}  {msg}"
    print(line)
    with open(P.logs() / "nla_enrich_push.log", "a", encoding="utf-8") as f:
        f.write(line + "\n")


def dry_run(P: F.FeaturePaths) -> int:
    keys = _push_keys(P)
    total = sum(p.stat().st_size for p, _ in keys)
    for p, key in keys:
        print(f"  {p.stat().st_size:>10} B  ->  s3://{s3_io.BUCKET}/{key}")
    print(f"[dry-run] {len(keys)} objects, {total/1e6:.2f} MB (nothing uploaded)")
    return 0


def push(P: F.FeaturePaths, *, force: bool = False) -> int:
    g = Gate(); check_tables(P, g); check_enriched(P, g, require=True)
    if not g.report():
        if not force:
            _log(P, "PUSH ABORTED: acceptance gate not green (use --force to override)")
            return 1
        _log(P, "WARNING: gate not green but --force given; pushing anyway")

    s3 = s3_io.make_client()
    keys = _push_keys(P)
    _log(P, f"PUSH start -> nla/feat + enriched nl/ + fve/ ({len(keys)} objects)")
    for p, key in keys:
        s3_io.upload(s3, p, key)  # raw upload_file ALWAYS overwrites (defeats the size-aware skip)
    bad = 0
    for p, key in keys:
        remote = s3_io.head_size(s3, key)
        local = p.stat().st_size
        if remote != local:
            bad += 1
            _log(P, f"SIZE MISMATCH {key} local={local} remote={remote}")
    _log(P, f"PUSH done: {len(keys)} keys verified, {bad} mismatched")
    return 0 if bad == 0 else 1


# ============================ CLI ==============================================

def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description="NLA-final native feature-attach (replaces feature-patch).")
    ap.add_argument("--no-qd", action="store_true", help="skip question_disjoint companion columns")
    ap.add_argument("--no-ingest", action="store_true", help="use already-local inputs (do not pull from S3)")
    ap.add_argument("--push", action="store_true", help="push to S3 if the acceptance gate is green")
    ap.add_argument("--dry-run", action="store_true", help="preview the push (no upload)")
    ap.add_argument("--selftest", action="store_true", help="independent recompute of anchors from raw sources")
    ap.add_argument("--force", action="store_true", help="push even if the gate is not green")
    args = ap.parse_args(argv)
    P = F.default_paths()

    if args.selftest:
        return selftest(P)

    build_utc = datetime.now(timezone.utc).isoformat()
    if not args.no_ingest:
        ingest(P)
    elif not P.generations().exists():
        print("ERROR: generations.parquet absent. Run the exp04 `generate` stage first "
              "(FEATURES.md Part B1), or drop --no-ingest to pull it.", file=sys.stderr)
        return 2

    enrich_in_place(P, include_qd=not args.no_qd, build_utc=build_utc)

    g = Gate(); check_tables(P, g); check_enriched(P, g, require=True)
    green = g.report()
    if args.dry_run:
        dry_run(P)
    if args.push:
        if not green and not args.force:
            print("[nla_enrich] gate not green -> NOT pushing (use --force to override)")
            return 1
        return push(P, force=args.force)
    if not args.push:
        print("\n[nla_enrich] local build" + (" GREEN" if green else " FAILED") +
              ". Push when ready:  python -m src.nla_enrich --no-ingest --push")
    return 0 if green else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
