"""Native datapoint features (F1 ground-truth, F2 model-generated answer, F3 pred<->know
divergence) — the consolidated, GPU-free heart of what was `feature-patch/`.

Folds `features_core.py` + `build_features.py` + `steered_divergence.py` into the parent
repo's `src/`, with ONE substantive change vs. the old patch (FEATURES.md §4): **F2 is the
model's ACTUAL generated answer** (from exp04's `generate` stage -> generations.parquet),
not the `y_tilde` symbol-readout shortcut. The readout (`y_tilde`, `p_model`,
`logits_symbols`, `model_readout_correct`) is kept verbatim as a labeled companion so every
historical anchor (base_acc 0.6604, AGR, know/pred acc) still reproduces exactly.

All probe math is fp64 (matches steer_sweep.py); stored floats are float32. Probe-load /
alpha-tag / hashing are REUSED from steer_sweep so orientation can never drift from the
production KAPPA edit (logits = h @ W.T + b).
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from . import steer_sweep  # parent repo helpers (alpha_tag, sha256_file, probe-load contract)

# re-export the validated helpers (don't re-roll)
alpha_tag = steer_sweep.alpha_tag
sha256_file = steer_sweep.sha256_file

# ---- locked constants (mirror steer_sweep / FEATURES.md) -------------------
N_ROWS = 6536
K = 4
D_MODEL = 3584
LAYER = 20
SCHEME = "example_level"                                          # F3 primary scheme
ALPHAS: list[float] = [0.0, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0, 20.0, 30.0]
CONFIG_HASH = "e80501525b6758e8a7c6f28556541bbbad1f268f92ae187972f83e69c075a55f"
MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
SYMBOLS = "ABCD"
EPS = 1e-12

SCHEMA_VERSION_FEATURES = "nla_features.v2"      # v2: F2 = actual generation (was v1 = y_tilde)
SCHEMA_VERSION_ENRICHED = "nla_enriched.v2"

# ---- verified acceptance constants (ported; readout anchors unchanged) -----
# (expected, abs_tol, kind) ; 'hard' raises, 'soft' warns-band but still raises out-of-band
GATES_PER_EXAMPLE_TEST = {
    "base_acc_readout": (0.6604, 1e-3, "hard"),   # mean(model_readout_correct) — historical base_acc anchor
    "know_acc":  (0.8543, 1e-3, "hard"),          # mean(know_correct)
    "pred_acc":  (0.9120, 1e-3, "hard"),          # mean(pred_matches_model)
    "agr":       (0.6486, 1e-3, "hard"),          # mean(agree_know_model)
    "kld_model_know": (8.178, 0.025, "soft"),     # band [8.15, 8.20]
}
# NEW (F2 = generation): floors + the generation-accuracy band around the readout anchor.
AGREE_GEN_READOUT_MIN = 0.99      # hard floor: greedy gen must reproduce the forced-choice letter
GEN_PARSE_RATE_MIN = 0.99         # hard floor: almost every generation yields a parseable A..D
BASE_ACC_GEN_BAND = (0.6604, 0.02, "soft")  # |base_acc_gen - 0.6604| <= 0.02
GATES_STEERED_ALL = {
    ("agree_know_pred_steered", 2.0):  (0.9207, 5e-3, "hard"),
    ("kl_pred_know_steered", 2.0):     (1.1275, 0.05, "soft"),
    ("kl_pred_know_steered", 30.0):    (9.5509, 0.1,  "soft"),
}
GT_CLASS_BALANCE = {0: 1634, 1: 1634, 2: 1634, 3: 1634}


# ============================ paths ============================================

@dataclass(frozen=True)
class FeaturePaths:
    """Every directory the feature build touches; overridable for tests (mirrors layout.Paths)."""
    inputs: Path        # pulled exp04 metadata + steered arrays + norms.parquet (NLA-final/inputs)
    emb_dir: Path       # exp04 mirror 03_kappa/emb (example_ids.json, h_layer20_orig.npy)
    probes_dir: Path    # exp04 mirror 02_probes (<scheme>/<target>/layerNN.npz)
    out: Path           # NLA-final/out (nl/, fve/, feat/, logs/)

    # ---- pulled exp04 metadata (ingest target) ----
    def examples(self) -> Path: return self.inputs / "examples.jsonl"
    def predictions(self) -> Path: return self.inputs / "predictions.parquet"
    def splits(self) -> Path: return self.inputs / "splits.json"
    def generations(self) -> Path: return self.inputs / "generations.parquet"
    def sources_lock(self) -> Path: return self.inputs / "sources.lock.json"
    # ---- already-local inputs ----
    def norms(self) -> Path: return self.inputs / "norms.parquet"
    def steered_npy(self, tag: str) -> Path: return self.inputs / f"h_layer20_steered_a{tag}.npy"
    def example_ids_json(self) -> Path: return self.emb_dir / "example_ids.json"
    def h_orig(self) -> Path: return self.emb_dir / f"h_layer{LAYER:02d}_orig.npy"
    def probe_npz(self, scheme: str, target: str, layer: int = LAYER) -> Path:
        return self.probes_dir / scheme / target / f"layer{layer:02d}.npz"
    # ---- outputs ----
    def feat_out(self) -> Path: return self.out / "feat"
    def nl_dir(self) -> Path: return self.out / "nl"
    def fve_dir(self) -> Path: return self.out / "fve"
    def logs(self) -> Path: return self.out / "logs"
    def datapoint_features(self) -> Path: return self.feat_out() / "datapoint_features.parquet"
    def steered_divergence(self) -> Path: return self.feat_out() / "steered_divergence.parquet"
    def feature_manifest(self) -> Path: return self.feat_out() / "manifest.json"
    def feature_analysis(self) -> Path: return self.feat_out() / "feature_analysis.json"
    def enriched_index(self) -> Path: return self.feat_out() / "enriched_index.json"


_SRC = Path(__file__).resolve().parent
_NLA_ROOT = _SRC.parent
# Self-containment: prefer the in-repo vendored exp04 slice (inputs/exp04/, holding the
# load-bearing example_ids.json + the know/pred probes) so the CPU pipeline runs from a
# clone without the external ../exp04 sibling. Fall back to the sibling when the vendored
# mirror is absent (e.g. a full exp04 working tree, or the pod's separately-synced cache).
_VENDORED_EXP04 = _NLA_ROOT / "inputs" / "exp04"
_SIBLING_EXP04 = _NLA_ROOT.parent / "exp04" / "05_out_pulled"
_EXP04_MIRROR = (
    _VENDORED_EXP04
    if (_VENDORED_EXP04 / "03_kappa" / "emb" / "example_ids.json").exists()
    else _SIBLING_EXP04
)


def default_paths() -> FeaturePaths:
    return FeaturePaths(
        inputs=_NLA_ROOT / "inputs",
        emb_dir=_EXP04_MIRROR / "03_kappa" / "emb",
        probes_dir=_EXP04_MIRROR / "02_probes",
        out=_NLA_ROOT / "out",
    )


# datapoint files to enrich (in-place). (filename, nominal alpha-or-None); the per-steer join
# uses each file's OWN `alpha` column when present, else the nominal alpha (orig -> 0.0).
NL_ORIG = ("orig.parquet", 0.0)
NL_STEERED = [(f"steered_a{t}.parquet", a) for a, t in [
    (0.0, "0"), (0.5, "0p5"), (1.0, "1"), (2.0, "2"), (3.0, "3"), (5.0, "5"),
    (7.0, "7"), (10.0, "10"), (15.0, "15"), (20.0, "20"), (30.0, "30")]]
NL_HEADLINE = [(f"headline_a{t}.parquet", a) for a, t in [(2.0, "2"), (10.0, "10"), (30.0, "30")]]
FVE_PER_ROW = ("per_row.parquet", None)   # carries its OWN 11-alpha column -> join by it


# ============================ probe math + I/O =================================

def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def kl(a: np.ndarray, b: np.ndarray, eps: float = EPS) -> np.ndarray:
    """KL(a || b) row-wise, nats. a,b: [N,K] distributions."""
    return (a * (np.log(a + eps) - np.log(b + eps))).sum(axis=-1)


def js(a: np.ndarray, b: np.ndarray, eps: float = EPS) -> np.ndarray:
    """Symmetric Jensen-Shannon (nats, <= ln2)."""
    m = 0.5 * (a + b)
    return 0.5 * kl(a, m, eps) + 0.5 * kl(b, m, eps)


def load_probe(P: FeaturePaths, scheme: str, target: str, layer: int = LAYER):
    """-> (W[K,d], b[K]) fp64. Same contract as steer_sweep: logits = h @ W.T + b."""
    z = np.load(P.probe_npz(scheme, target, layer))
    W, b = z["W"].astype(np.float64), z["b"].astype(np.float64)
    assert W.shape == (K, D_MODEL), f"{scheme}/{target} W shape {W.shape}"
    assert b.shape == (K,), f"{scheme}/{target} b shape {b.shape}"
    return W, b


def posteriors(h: np.ndarray, W: np.ndarray, b: np.ndarray) -> np.ndarray:
    """softmax(h @ W.T + b) -> [N,K]. h fp64."""
    return softmax(h @ W.T + b)


def load_h(path: Path, n_rows: int = N_ROWS) -> np.ndarray:
    """Load an activation array as fp64 (probe math), asserting canonical shape."""
    h = np.load(path).astype(np.float64)
    assert h.shape == (n_rows, D_MODEL), f"{path.name} shape {h.shape}"
    return h


def write_parquet_atomic(table: pa.Table, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    pq.write_table(table, tmp)
    os.replace(tmp, path)


def write_json_atomic(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def canonical_order(P: FeaturePaths, n_rows: int = N_ROWS) -> list[str]:
    """The example_ids in activation_cache_row_order (load-bearing join key)."""
    ids = json.loads(P.example_ids_json().read_text(encoding="utf-8"))
    order = ids["example_ids"]
    assert len(order) == n_rows and ids["n"] == n_rows, "example_ids count"
    return order


def test_mask_example_level(P: FeaturePaths, order: list[str]) -> np.ndarray:
    """Boolean [N] mask of the example_level test split, in canonical order."""
    ids = json.loads(P.example_ids_json().read_text(encoding="utf-8"))
    test_rows = np.asarray(ids["test_row_indices"], dtype=np.int64)
    m = np.zeros(len(order), dtype=bool)
    m[test_rows] = True
    return m


# ============================ F1 + F2 + F3-orig ================================

def build_datapoint_features(P: FeaturePaths, include_qd: bool = True,
                             build_utc: str | None = None, n_rows: int = N_ROWS) -> dict:
    """F1 (ground truth) + F2 (ACTUAL model generation) + F3 (pred<->know on the ORIGINAL
    activation) -> feat/datapoint_features.parquet (+ manifest.json). Per-example grain,
    canonical row order. Joins to every datapoint by example_id."""
    SYM = SYMBOLS
    order = canonical_order(P, n_rows)
    n = len(order)
    row_index = np.arange(n, dtype=np.int32)

    # ---- sources ----------------------------------------------------------
    examples = {}
    for line in P.examples().read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            examples[r["example_id"]] = r
    splits = json.loads(P.splits().read_text(encoding="utf-8"))

    pred = pq.read_table(P.predictions())
    assert pred.column("example_id").to_pylist() == order, "predictions order drift"
    y_tilde = np.asarray(pred.column("y_tilde").to_pylist(), dtype=np.int64)
    p_model = np.asarray(pred.column("p_model").to_pylist(), dtype=np.float64)       # [N,4], sum=1
    logits_symbols = np.asarray(pred.column("logits_symbols").to_pylist(), dtype=np.float64)
    assert p_model.shape == (n, K), f"p_model shape {p_model.shape}"

    # F2 source: the ACTUAL generation (exp04 `generate` stage)
    gens = pq.read_table(P.generations())
    assert gens.column("example_id").to_pylist() == order, "generations order drift"
    gen_index = np.asarray(gens.column("model_gen_index").to_pylist(), dtype=np.int64)  # -1..3
    gen_symbol = gens.column("model_gen_symbol").to_pylist()
    gen_text = gens.column("model_gen_text").to_pylist()
    gen_first = gens.column("model_gen_first_token").to_pylist()
    gen_method = gens.column("model_gen_method").to_pylist()[0] if n else "greedy_generate"
    gen_revision = gens.column("gen_model_revision").to_pylist()[0] if n else "unknown"

    # ---- F1: ground truth -------------------------------------------------
    gt_idx = np.empty(n, dtype=np.int8)
    gt_text, question, question_id = [], [], []
    perm = np.empty(n, dtype=np.int8)
    model_answer_text = []  # F2: the chosen OPTION's text (by generated symbol), "" if unparseable
    for i, eid in enumerate(order):
        r = examples[eid]
        opts = r["options"]
        ai = int(r["answer_index"])
        assert len(opts) == K, f"{eid} has {len(opts)} options"
        assert opts[ai]["symbol"] == SYM[ai], f"{eid} options not in symbol order"
        gt_idx[i] = ai
        gt_text.append(opts[ai]["text"])
        question.append(r["question"])
        question_id.append(r["source_meta"]["question_id"])
        perm[i] = int(eid.rsplit("-p", 1)[1])
        k = int(gen_index[i])
        model_answer_text.append(opts[k]["text"] if 0 <= k < K else "")

    # ---- splits (example_level + question_disjoint) -----------------------
    def split_lookup(scheme: str) -> list[str]:
        sc = splits["schemes"][scheme]
        m = {}
        for name in ("train", "val", "test"):
            for eid in sc[name]:
                m[eid] = name
        return [m[eid] for eid in order]
    split_el = split_lookup("example_level")
    split_qd = split_lookup("question_disjoint")
    is_test_el = np.asarray([s == "test" for s in split_el], dtype=bool)

    # ---- F3: probe posteriors on the ORIGINAL activation ------------------
    Wk, bk = load_probe(P, "example_level", "know")
    Wp, bp = load_probe(P, "example_level", "pred")
    h0 = load_h(P.h_orig(), n)
    pk = posteriors(h0, Wk, bk)
    pp = posteriors(h0, Wp, bp)
    know_arg = pk.argmax(1)
    pred_arg = pp.argmax(1)

    # F2 correctness flavors: generation (the new truth) + readout (the historical anchor)
    model_gen_correct = (gen_index == gt_idx)            # -1 never == 0..3 -> unparseable = wrong
    model_readout_correct = (y_tilde == gt_idx)
    agree_gen_readout = (gen_index == y_tilde)

    d: dict[str, object] = {
        "example_id": order,
        "row_index": row_index,
        "question_id": question_id,
        "perm": perm,
        "split_el": split_el,
        "split_qd": split_qd,
        # F1
        "gt_answer_index": gt_idx,
        "gt_symbol": [SYM[i] for i in gt_idx],
        "gt_answer_text": gt_text,
        "question": question,
        # F2 (headline = the ACTUAL generation)
        "model_answer_index": gen_index.astype(np.int8),
        "model_symbol": list(gen_symbol),
        "model_answer_text": model_answer_text,
        "model_gen_text": list(gen_text),
        "model_gen_first_token": list(gen_first),
        "model_confidence": p_model[np.arange(n), y_tilde].astype(np.float32),  # readout companion
        "model_gen_correct": model_gen_correct,
        "model_readout_correct": model_readout_correct,
        "model_gen_method": [gen_method] * n,
        "gen_model_revision": [gen_revision] * n,
        "agree_gen_readout": agree_gen_readout,
        # F2 readout companions (provenance; the gates' anchors)
        "y_tilde": y_tilde.astype(np.int32),
        "p_model": [list(map(float, row)) for row in p_model],
        "logits_symbols": [list(map(float, row)) for row in logits_symbols],
        # F3 (example_level, on original h) — agreements anchored to the readout y_tilde (unchanged)
        "kl_pred_know": kl(pp, pk).astype(np.float32),
        "js_pred_know": js(pp, pk).astype(np.float32),
        "kl_model_know": kl(p_model, pk).astype(np.float32),
        "js_model_know": js(p_model, pk).astype(np.float32),
        "know_argmax_symbol": [SYM[i] for i in know_arg],
        "pred_argmax_symbol": [SYM[i] for i in pred_arg],
        "know_confidence": pk.max(1).astype(np.float32),
        "pred_confidence": pp.max(1).astype(np.float32),
        "agree_know_pred": (know_arg == pred_arg),
        "agree_know_model": (know_arg == y_tilde),
        "know_correct": (know_arg == gt_idx),
        "pred_matches_model": (pred_arg == y_tilde),
    }

    # ---- optional question_disjoint companions ----------------------------
    if include_qd:
        Wkq, bkq = load_probe(P, "question_disjoint", "know")
        Wpq, bpq = load_probe(P, "question_disjoint", "pred")
        pkq = posteriors(h0, Wkq, bkq)
        ppq = posteriors(h0, Wpq, bpq)
        d.update({
            "kl_pred_know_qd": kl(ppq, pkq).astype(np.float32),
            "js_pred_know_qd": js(ppq, pkq).astype(np.float32),
            "kl_model_know_qd": kl(p_model, pkq).astype(np.float32),
            "agree_know_pred_qd": (pkq.argmax(1) == ppq.argmax(1)),
            "agree_know_model_qd": (pkq.argmax(1) == y_tilde),
            "know_correct_qd": (pkq.argmax(1) == gt_idx),
        })

    # ---- typed schema + atomic write --------------------------------------
    f32, i8, b_, s_, i32 = pa.float32(), pa.int8(), pa.bool_(), pa.string(), pa.int32()
    flist = pa.list_(pa.float32())
    fields = [
        ("example_id", s_), ("row_index", i32), ("question_id", s_), ("perm", i8),
        ("split_el", s_), ("split_qd", s_),
        ("gt_answer_index", i8), ("gt_symbol", s_), ("gt_answer_text", s_), ("question", s_),
        ("model_answer_index", i8), ("model_symbol", s_), ("model_answer_text", s_),
        ("model_gen_text", s_), ("model_gen_first_token", s_),
        ("model_confidence", f32), ("model_gen_correct", b_), ("model_readout_correct", b_),
        ("model_gen_method", s_), ("gen_model_revision", s_), ("agree_gen_readout", b_),
        ("y_tilde", i32), ("p_model", flist), ("logits_symbols", flist),
        ("kl_pred_know", f32), ("js_pred_know", f32), ("kl_model_know", f32), ("js_model_know", f32),
        ("know_argmax_symbol", s_), ("pred_argmax_symbol", s_),
        ("know_confidence", f32), ("pred_confidence", f32),
        ("agree_know_pred", b_), ("agree_know_model", b_), ("know_correct", b_), ("pred_matches_model", b_),
    ]
    if include_qd:
        fields += [("kl_pred_know_qd", f32), ("js_pred_know_qd", f32), ("kl_model_know_qd", f32),
                   ("agree_know_pred_qd", b_), ("agree_know_model_qd", b_), ("know_correct_qd", b_)]
    schema = pa.schema(fields)
    table = pa.table({name: d[name] for name, _ in fields}, schema=schema)
    write_parquet_atomic(table, P.datapoint_features())

    # ---- manifest ---------------------------------------------------------
    t = is_test_el
    arr = lambda c: np.asarray(d[c])
    consts = {
        "base_acc_readout": float(arr("model_readout_correct")[t].mean()),
        "base_acc_gen": float(arr("model_gen_correct")[t].mean()),
        "know_acc": float(arr("know_correct")[t].mean()),
        "pred_acc": float(arr("pred_matches_model")[t].mean()),
        "agr": float(arr("agree_know_model")[t].mean()),
        "kld_model_know": float(arr("kl_model_know")[t].mean()),
    }
    parseable = gen_index >= 0
    # agree_gen_readout = does the GENERATED letter match the readout. Measured over PARSEABLE rows
    # (where a letter was produced); unparseable rows are a separate axis with their own parse-rate gate
    # (counting them as "disagreements" would conflate two distinct failure modes). FEATURES.md §8.
    agree_parseable = float(agree_gen_readout[parseable].mean()) if parseable.any() else 1.0
    gen_stats = {
        "agree_gen_readout_parseable": agree_parseable,            # the gated metric
        "agree_gen_readout_mean_all": float(agree_gen_readout.mean()),  # incl. unparseable as disagreements
        "agree_gen_readout_mean_test": float(agree_gen_readout[t].mean()),
        "gen_parse_rate": float(parseable.mean()),
        "n_unparseable": int((~parseable).sum()),
    }
    klpk = np.asarray(d["kl_pred_know"], dtype=np.float64)
    sources_lock = (json.loads(P.sources_lock().read_text(encoding="utf-8"))["sources"]
                    if P.sources_lock().exists() else None)
    manifest = {
        "schema_version": SCHEMA_VERSION_FEATURES,
        "config_hash": CONFIG_HASH,
        "model_id": MODEL_ID,
        "build_utc": build_utc,
        "n_examples": n,
        "d_model": D_MODEL,
        "probe": {
            "scheme": SCHEME, "layer": LAYER, "include_qd": include_qd,
            "sha256": {f"{sch}/{tg}": sha256_file(P.probe_npz(sch, tg))
                       for sch in (["example_level", "question_disjoint"] if include_qd else ["example_level"])
                       for tg in ("know", "pred")},
        },
        "sources": sources_lock,
        "feature_2_method": "greedy_generate",
        "feature_2_method_detail": gen_method,
        "gen_model_revision": gen_revision,
        "acceptance_test_split": consts,
        "generation_crosscheck": gen_stats,
        "kl_pred_know_distribution": {
            "note": "heavy-tailed: median ~0, mean driven by tail (read mean WITH median+p95)",
            "mean": float(klpk.mean()), "median": float(np.median(klpk)),
            "p95": float(np.quantile(klpk, 0.95)), "max": float(klpk.max()),
        },
        "columns": _COLUMN_DICT,
        "row_counts": {"datapoint_features": n},
    }
    write_json_atomic(manifest, P.feature_manifest())

    print(f"[features] wrote {P.datapoint_features().name} ({n} rows, {len(fields)} cols) + manifest.json")
    print(f"[features] test: base_acc_gen={consts['base_acc_gen']:.4f} (readout {consts['base_acc_readout']:.4f}) "
          f"know={consts['know_acc']:.4f} pred={consts['pred_acc']:.4f} agr={consts['agr']:.4f}")
    print(f"[features] gen: agree_gen_readout={gen_stats['agree_gen_readout_mean_all']:.4f} "
          f"parse_rate={gen_stats['gen_parse_rate']:.4f} unparseable={gen_stats['n_unparseable']}")
    return manifest


# ============================ F3 on steered activations ========================

def _norms_lookup(P: FeaturePaths) -> dict:
    """(row_index, round(alpha,6)) -> (split, in_subset) from inputs/norms.parquet."""
    t = pq.read_table(P.norms(), columns=["row_index", "alpha", "split", "in_subset"]).to_pydict()
    return {(int(r), round(float(a), 6)): (s, bool(ins))
            for r, a, s, ins in zip(t["row_index"], t["alpha"], t["split"], t["in_subset"])}


def build_steered_divergence(P: FeaturePaths, build_utc: str | None = None, n_rows: int = N_ROWS) -> dict:
    """F3 (pred<->know divergence) on each steered activation h'(alpha) -> feat/steered_divergence.parquet
    (11 alpha x N rows). Per-(row_index, alpha) grain; example_level probes, layer 20. alpha=0 doubles
    as a cross-check against datapoint_features.kl_pred_know."""
    SYM = SYMBOLS
    order = canonical_order(P, n_rows)
    n = len(order)
    row_index = np.arange(n, dtype=np.int32)
    Wk, bk = load_probe(P, "example_level", "know")
    Wp, bp = load_probe(P, "example_level", "pred")
    norms = _norms_lookup(P)

    blocks: dict[str, list] = {k: [] for k in (
        "example_id", "row_index", "alpha", "split", "in_subset",
        "kl_pred_know_steered", "js_pred_know_steered", "agree_know_pred_steered",
        "pred_argmax_symbol_steered", "know_argmax_symbol_steered", "d_kl_pred_know")}
    kl0: np.ndarray | None = None  # alpha=0 per-row baseline

    for a in ALPHAS:
        tag = alpha_tag(a)
        h = load_h(P.steered_npy(tag), n)
        pp = posteriors(h, Wp, bp)
        pk = posteriors(h, Wk, bk)
        kl_pk = kl(pp, pk)
        js_pk = js(pp, pk)
        pred_arg = pp.argmax(1)
        know_arg = pk.argmax(1)
        if a == 0.0:
            kl0 = kl_pk.copy()
        assert kl0 is not None, "alpha grid must start at 0.0 (baseline for d_kl)"
        d_kl = kl_pk - kl0

        meta = [norms[(i, round(a, 6))] for i in range(n)]
        blocks["example_id"] += order
        blocks["row_index"].append(row_index)
        blocks["alpha"] += [float(a)] * n
        blocks["split"] += [m[0] for m in meta]
        blocks["in_subset"] += [m[1] for m in meta]
        blocks["kl_pred_know_steered"].append(kl_pk.astype(np.float32))
        blocks["js_pred_know_steered"].append(js_pk.astype(np.float32))
        blocks["agree_know_pred_steered"] += list(know_arg == pred_arg)
        blocks["pred_argmax_symbol_steered"] += [SYM[i] for i in pred_arg]
        blocks["know_argmax_symbol_steered"] += [SYM[i] for i in know_arg]
        blocks["d_kl_pred_know"].append(d_kl.astype(np.float32))
        print(f"[features] steered a={a:<5g} kl_pred_know mean={kl_pk.mean():7.4f} "
              f"median={np.median(kl_pk):.4f} argmax-agree={(know_arg==pred_arg).mean():.4f}")

    # cross-check: alpha=0 steered == original-activation F3
    if P.datapoint_features().exists():
        orig_kl = np.asarray(pq.read_table(P.datapoint_features(), columns=["kl_pred_know"])
                             .column("kl_pred_know").to_pylist(), dtype=np.float64)
        a0 = np.asarray(blocks["kl_pred_know_steered"][0], dtype=np.float64)
        assert np.allclose(a0, orig_kl, atol=1e-4, rtol=1e-4), \
            "alpha=0 steered KL must equal datapoint_features.kl_pred_know (identity cross-check)"
        print("[features] cross-check OK: alpha=0 steered KL == datapoint_features.kl_pred_know")

    f32, b_, s_ = pa.float32(), pa.bool_(), pa.string()
    fields = [
        ("example_id", s_), ("row_index", pa.int32()), ("alpha", pa.float64()),
        ("split", s_), ("in_subset", b_),
        ("kl_pred_know_steered", f32), ("js_pred_know_steered", f32),
        ("agree_know_pred_steered", b_),
        ("pred_argmax_symbol_steered", s_), ("know_argmax_symbol_steered", s_),
        ("d_kl_pred_know", f32),
    ]
    data = {
        "example_id": blocks["example_id"],
        "row_index": np.concatenate(blocks["row_index"]),
        "alpha": blocks["alpha"],
        "split": blocks["split"],
        "in_subset": blocks["in_subset"],
        "kl_pred_know_steered": np.concatenate(blocks["kl_pred_know_steered"]),
        "js_pred_know_steered": np.concatenate(blocks["js_pred_know_steered"]),
        "agree_know_pred_steered": blocks["agree_know_pred_steered"],
        "pred_argmax_symbol_steered": blocks["pred_argmax_symbol_steered"],
        "know_argmax_symbol_steered": blocks["know_argmax_symbol_steered"],
        "d_kl_pred_know": np.concatenate(blocks["d_kl_pred_know"]),
    }
    table = pa.table(data, schema=pa.schema(fields))
    write_parquet_atomic(table, P.steered_divergence())
    print(f"[features] wrote {P.steered_divergence().name} ({table.num_rows} rows) [build_utc={build_utc}]")
    return {"rows": table.num_rows}


_COLUMN_DICT = {
    "example_id": "join key; tqa-{qid}-p{perm}",
    "row_index": "canonical activation_cache_row_order index 0..N-1",
    "question_id": "source question id (perm-invariant)",
    "perm": "option-permutation index 0..7",
    "split_el": "example_level split (train/val/test)",
    "split_qd": "question_disjoint split (train/val/test)",
    "gt_answer_index": "F1 ground-truth option position 0..3",
    "gt_symbol": "F1 ground-truth symbol A..D",
    "gt_answer_text": "F1 ground-truth answer text",
    "question": "F1 original dataset question",
    "model_answer_index": "F2 ACTUAL generated answer index 0..3 (-1 if unparseable)",
    "model_symbol": "F2 generated symbol A..D ('' if unparseable)",
    "model_answer_text": "F2 chosen option's text (by generated symbol; '' if unparseable)",
    "model_gen_text": "F2 raw generated answer line (after '(')",
    "model_gen_first_token": "F2 decoded first new token (diagnostic)",
    "model_confidence": "readout companion: p_model[y_tilde]",
    "model_gen_correct": "F2 model_answer_index == gt (the NEW 'model correct')",
    "model_readout_correct": "companion: y_tilde == gt (historical base_acc 0.6604)",
    "model_gen_method": "F2 provenance (greedy_generate@maxnewN)",
    "gen_model_revision": "HF commit sha of the generating model",
    "agree_gen_readout": "F2 model_answer_index == y_tilde (>=0.99 expected)",
    "y_tilde": "readout companion: argmax of the 4 symbol logits",
    "p_model": "readout companion: softmax over {A,B,C,D}",
    "logits_symbols": "readout companion: raw symbol logits",
    "kl_pred_know": "F3 PRIMARY KL(p_pred||p_know) on h_orig",
    "js_pred_know": "F3 JS(p_pred,p_know) on h_orig",
    "kl_model_know": "F3 KL(p_model||p_know) (reproduces exp04 kld)",
    "js_model_know": "F3 JS(p_model,p_know)",
    "know_argmax_symbol": "F3 argmax knowledge symbol",
    "pred_argmax_symbol": "F3 argmax prediction symbol",
    "know_confidence": "F3 max(p_know)",
    "pred_confidence": "F3 max(p_pred)",
    "agree_know_pred": "F3 argmax(know)==argmax(pred)",
    "agree_know_model": "F3 argmax(know)==y_tilde (AGR)",
    "know_correct": "F3 argmax(know)==gt (know accuracy)",
    "pred_matches_model": "F3 argmax(pred)==y_tilde (pred accuracy)",
    "*_qd": "question_disjoint-probe companions (optional, default on)",
}
