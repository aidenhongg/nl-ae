"""Language-space steering (MAIN-EXP) — edit the NLA verbalization, reconstruct an
on-manifold steered L20 activation through the AR, patch it back into Qwen, and read
the answer. The thesis (MAIN-EXP.md §1): match KAPPA's single-L20 accuracy at strictly
LOWER off-manifold error, because `AR(text)` is on-manifold by construction.

One module, two halves:
  * CPU / local / $0  — text operators (E0..E7, T1..T2), target join, the verbalization
    coverage analyzer (Phase 0b), the CPU pred-probe proxy, and the report/Pareto plotter.
  * GPU / on-pod      — reconstruct (AR), eval-readout + eval-generate (Qwen re-forward,
    reusing exp04 `kappa/` VERBATIM), and the OME re-verbalization (AV+AR). All heavy deps
    (torch, nla_inference, kappa) are imported LAZILY inside those functions so the CPU
    half imports clean for unit tests.

Locked decisions (MAIN-EXP.md §3, §7) — do not drift:
  * Target X per row = `know_argmax_symbol` (knowledge-probe label); oracle = `gt_symbol`.
  * The patch is a REPLACEMENT: `edit_fn` overwrites the L20 last-token residual with
    `ĥ_steer` (it already encodes the whole activation) — NOT an additive `Δh`. This is
    what makes it on-manifold.
  * Default magnitude convention = norm-match to ‖h_orig‖ (directional patch; §7.5). All
    three conventions (native / normmatch / cohortmean) are available and chosen in 1a.
  * Forced-choice readout (argmax of the 4 symbol logits at the `"…is ("` position) is
    primary; free-generation is a headline-only confirmation.
  * Two off-manifold metrics: OME = 1 − cos(ĥ_steer, AR(AV(ĥ_steer))) [NLA-native] and
    ratio = ‖ĥ_steer − h_orig‖/‖h_orig‖ [NLA-independent; the load-bearing win criterion].

CLI (subcommands):
  build-targets   CPU  per-row X + gt + h_norm + eval subsets (tiny256/sweep512/full).
  analyze-text    CPU  Phase 0b: design/validate E1 + E3 coverage on the cached corpus.
  reconstruct     GPU  edit z_orig per op -> AR.reconstruct -> raw ĥ_recon arrays + edits.
  proxy           CPU  pred-probe readout on ĥ_steer (cheap pre-rank; filter, not truth).
  eval-readout    GPU  patch ĥ_steer @ L20 -> re-forward Qwen -> ŷ / ACC / AGR / success.
  eval-generate   GPU  same patch under model.generate -> free-gen letter (headline).
  ome             GPU  AV(ĥ_steer) -> AR.score -> cos -> OME (headline configs).
  report          CPU  (ACC, OME, ratio) per method + Pareto plot vs the KAPPA frontier.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np

# CPU-side reuse (no torch / no GPU): probe math, alpha tags, bootstrap CI.
from . import features as F
from . import fve_analysis as FA
from . import steer_sweep as SS

# ----------------------------- locked constants -------------------------------
SYMBOLS = "ABCD"
SYM_IDX = {s: i for i, s in enumerate(SYMBOLS)}
LAYER = 20                       # cache layer 20 == raw output of decoder block 19 (KAPPA's edit site)
D_MODEL = 3584
TARGET_COL = "know_argmax_symbol"   # X (MAIN-EXP §3)
ORACLE_COL = "gt_symbol"            # reported ceiling only
ANSWER_CUE = "The correct answer is ("   # exp04 prompt.py — the prompt's own answer cue
SUBSET_SEED = 7                  # project convention (steer_sweep.SUBSET_SEED)
SUBSET_SIZES = {"tiny256": 256, "sweep512": 512}   # full == all test rows
CONVENTIONS = ("native", "normmatch", "cohortmean")
DEFAULT_CONVENTION = "normmatch"

# Operators that DISCARD the verbalization (no AV/orig text needed beyond X) — useful for the
# Phase-1 lever test where we only ask "can the channel transmit an answer at all".
TEMPLATE_OPS = {"T1", "T2"}

# ----------------------------- paths ------------------------------------------
_SRC = Path(__file__).resolve().parent
_ROOT = _SRC.parent
_EXP04 = Path(os.environ.get("LANG_EXP04_ROOT", _ROOT.parent / "exp04"))
_EMB = _EXP04 / "05_out_pulled" / "03_kappa" / "emb"
_PROBES = _EXP04 / "05_out_pulled" / "02_probes"


def _paths(inputs: Path | None, out: Path | None):
    return (inputs or _ROOT / "inputs"), (out or _ROOT / "out")


# =============================================================================
#  Phase 0b — the validated edit patterns (designed against the cached corpus;
#  see `analyze-text` for the coverage report that justifies them).
# =============================================================================
# E1 fires only inside an answer-ASSERTION context and NEVER inside an enumeration
# (option-list OR multi-letter hedge) — no-clobber is prioritized over recall (§8 risk),
# because E2/T1 give universal coverage. The corpus is meta-description, not 1st-person.
_ANCHOR = re.compile(
    r"(?:correct |final |plausible )?answer (?:is|to[^.]{0,40}? is|should be|must be|could be|choice)"
    r"|correct (?:option|choice) is|implying|implies|suggesting|suggests|indicat\w+"
    r"|likely|expecting|expects|points? to",
    re.I,
)
_ISO_LETTER = re.compile(r"(?<![A-Za-z])([A-D])(?![A-Za-z])")          # an isolated A..D mention
_RUN = re.compile(                                                     # >=2 letters joined => protect
    r"(?:[\"('‘’]?[A-D][)\"'‘’]?\s*(?:,|or|/|&|nor)\s*)+[\"('‘’]?[A-D]",
    re.I,
)
_ANCHOR_WINDOW = 25   # chars after the anchor to look for the asserted letter

# E3 strips ADVERBIAL/parenthetical hedges that weaken an assertion. Structural verbs
# (implies/suggesting/signals/indicating) are KEPT — deleting them breaks the sentence.
_HEDGE_TERMS = [
    "very likely", "most likely", "more likely", "or similar", "or so",   # multiword first
    "likely", "possibly", "perhaps", "probably", "presumably", "seemingly",
    "arguably", "tentatively", "speculatively", "speculative", "plausibly", "maybe",
]
_HEDGE = re.compile(r"\b(?:" + "|".join(re.escape(t) for t in _HEDGE_TERMS) + r")\b", re.I)


# ----------------------------- text operators ---------------------------------

def e1_substitute(z: str, X: str) -> str:
    """Rewrite the asserted answer-letter(s) to X; spare every enumeration. Idempotent."""
    runs = [(m.start(), m.end()) for m in _RUN.finditer(z)]
    edits: set[int] = set()
    for am in _ANCHOR.finditer(z):
        m = _ISO_LETTER.search(z, am.end(), min(len(z), am.end() + _ANCHOR_WINDOW))
        if m:
            p = m.start(1)
            if z[p] != X and not any(a <= p < b for a, b in runs):
                edits.add(p)
    if not edits:
        return z
    out = list(z)
    for p in edits:
        out[p] = X
    return "".join(out)


def e2_append(z: str, X: str) -> str:
    """Append a clean assertion mirroring the prompt's cue. Idempotent (skip if already there)."""
    tail = f" Actually, the correct answer is ({X})."
    return z if z.rstrip().endswith(tail.strip()) else z.rstrip() + tail


def e3_strip(z: str, X: str | None = None) -> str:
    """Delete adverbial hedges; normalize ONLY the whitespace/punctuation they leave behind.
    A true no-op (and idempotent) when the text carries no hedge."""
    out = _HEDGE.sub("", z)
    if out == z:
        return z                                   # no hedge -> don't touch original spacing
    out = re.sub(r"\s{2,}", " ", out)
    out = re.sub(r"\s+([,.;:)\]])", r"\1", out)
    out = re.sub(r"([(\[])\s+", r"\1", out)
    return out.strip()


def t1_template(z: str, X: str) -> str:
    """The strongest lever: discard the verbalization, assert X with the prompt's own cue."""
    return f"The correct answer is ({X})."


def t2_template(z: str, X: str) -> str:
    """Synthesized rich template — states X outright while keeping the structural frame. Deterministic."""
    return (f"Structured multiple-choice answer. The correct answer is ({X}). "
            f"The final token completes the answer with ({X}).")


def _compose(z: str, X: str, ops: list) -> str:
    for fn in ops:
        z = fn(z, X)
    return z


# op-name -> callable(z, X) -> z_edit. Combos run strip -> substitute -> append.
OPERATORS = {
    "E0": lambda z, X: z,
    "E1": e1_substitute,
    "E2": e2_append,
    "E3": e3_strip,
    "E4": lambda z, X: _compose(z, X, [e1_substitute, e2_append]),
    "E5": lambda z, X: _compose(z, X, [e3_strip, e1_substitute]),
    "E6": lambda z, X: _compose(z, X, [e3_strip, e2_append]),
    "E7": lambda z, X: _compose(z, X, [e3_strip, e1_substitute, e2_append]),
    "T1": t1_template,
    "T2": t2_template,
}


def edit(op: str, z_orig: str, X: str) -> str:
    """Apply an operator from §4 to one verbalization. X is the target symbol (A..D)."""
    assert op in OPERATORS, f"unknown operator {op!r} (have {sorted(OPERATORS)})"
    assert X in SYM_IDX, f"target {X!r} not a symbol"
    return OPERATORS[op](z_orig, X)


# =============================================================================
#  CPU — targets + eval subsets
# =============================================================================

def _orig_table(out: Path):
    import pyarrow.parquet as pq
    return pq.read_table(out / "nl" / "orig.parquet",
                         columns=["row_index", "example_id", "split_el", "nl_text",
                                  TARGET_COL, ORACLE_COL, "pred_argmax_symbol"])


def build_targets(inputs: Path | None = None, out: Path | None = None) -> dict:
    """Per-row X + gt + h_norm + the deterministic eval subsets -> out/lang/targets.parquet
    + subsets.json. Test split only (the apples-to-apples eval set, 2615 rows)."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    inputs, out = _paths(inputs, out)
    lang = out / "lang"
    lang.mkdir(parents=True, exist_ok=True)

    t = _orig_table(out).to_pydict()
    h0 = np.load(_EMB / f"h_layer{LAYER:02d}_orig.npy", mmap_mode="r")   # canonical order, [N,d]
    test = [i for i, s in enumerate(t["split_el"]) if s == "test"]
    rows = [int(t["row_index"][i]) for i in test]
    h_norm = np.linalg.norm(np.asarray(h0[rows], dtype=np.float64), axis=1)

    cols = {
        "row_index": rows,
        "example_id": [t["example_id"][i] for i in test],
        "X": [t[TARGET_COL][i] for i in test],
        "X_idx": [SYM_IDX[t[TARGET_COL][i]] for i in test],
        "gt_symbol": [t[ORACLE_COL][i] for i in test],
        "gt_idx": [SYM_IDX[t[ORACLE_COL][i]] for i in test],
        "pred_argmax_symbol": [t["pred_argmax_symbol"][i] for i in test],
        "h_norm": [float(x) for x in h_norm],
    }
    schema = pa.schema([("row_index", pa.int32()), ("example_id", pa.string()),
                        ("X", pa.string()), ("X_idx", pa.int8()), ("gt_symbol", pa.string()),
                        ("gt_idx", pa.int8()), ("pred_argmax_symbol", pa.string()),
                        ("h_norm", pa.float64())])
    F.write_parquet_atomic(pa.table(cols, schema=schema), lang / "targets.parquet")

    # deterministic nested subsets (seeded; sorted = canonical), tiny256 ⊂ sweep512 ⊂ full
    rng = np.random.default_rng(SUBSET_SEED)
    test_rows = np.asarray(sorted(rows), dtype=np.int64)
    perm = rng.permutation(test_rows.size)
    subs = {"full": test_rows.tolist()}
    for name, n in sorted(SUBSET_SIZES.items(), key=lambda kv: kv[1]):
        subs[name] = sorted(int(test_rows[i]) for i in perm[:n])
    assert set(subs["tiny256"]) <= set(subs["sweep512"]) <= set(subs["full"]), "subsets must nest"
    (lang / "subsets.json").write_text(json.dumps({
        "seed": SUBSET_SEED, "n_test": len(rows), "sizes": {k: len(v) for k, v in subs.items()},
        "cohort_mean_h_norm": float(h_norm.mean()), "rows": subs,
    }, indent=2), encoding="utf-8")
    print(f"[targets] {len(rows)} test rows -> targets.parquet; subsets "
          f"{{ {', '.join(f'{k}:{len(v)}' for k, v in subs.items())} }} (seed {SUBSET_SEED})")
    print(f"[targets] X balance {dict(_count(cols['X']))}; mean ||h_orig|| = {h_norm.mean():.3f}")
    return {"n_test": len(rows), "subsets": {k: len(v) for k, v in subs.items()}}


def _count(xs):
    import collections
    return dict(sorted(collections.Counter(xs).items()))


def load_subset(out: Path, name: str) -> list[int]:
    subs = json.loads((out / "lang" / "subsets.json").read_text(encoding="utf-8"))["rows"]
    assert name in subs, f"unknown subset {name!r} (have {sorted(subs)})"
    return subs[name]


# =============================================================================
#  Phase 0b — verbalization coverage analysis (justifies E1 + E3)
# =============================================================================

def analyze_text(out: Path | None = None, n_examples: int = 6) -> dict:
    """Run E1 + E3 over the cached test verbalizations and report coverage: E1 fire rate,
    subs/row, enumeration-protection, idempotency, no-clobber; E3 hedge-strip rate. Writes
    out/lang/verbalization_analysis.json. This is the §0b deliverable."""
    _, out = _paths(None, out)
    lang = out / "lang"; lang.mkdir(parents=True, exist_ok=True)
    t = _orig_table(out).to_pydict()
    idx = [i for i, s in enumerate(t["split_el"]) if s == "test"]
    texts = [t["nl_text"][i] for i in idx]
    Xs = [t[TARGET_COL][i] for i in idx]
    n = len(texts)

    iso = _ISO_LETTER
    e1_fire = e1_subs = e1_idem_fail = clobbered = protected_letters = 0
    e3_fire = e3_removed = e3_idem_fail = 0
    examples = []
    for tx, X in zip(texts, Xs):
        runs = [(m.start(), m.end()) for m in _RUN.finditer(tx)]
        protected_letters += sum(1 for m in iso.finditer(tx)
                                 if any(a <= m.start() < b for a, b in runs))
        z1 = e1_substitute(tx, X)
        # count substitutions = isolated letters that changed to X (vs orig)
        diffs = [j for j in range(min(len(tx), len(z1))) if tx[j] != z1[j]]
        if z1 != tx:
            e1_fire += 1
            e1_subs += len(diffs)
            # no-clobber: did any substitution land inside a protected enumeration run?
            clobbered += sum(1 for p in diffs if any(a <= p < b for a, b in runs))
            if e1_substitute(z1, X) != z1:
                e1_idem_fail += 1
            if len(examples) < n_examples:
                p = diffs[0]
                examples.append({"X": X, "before": tx[max(0, p - 35):p + 6],
                                 "after": z1[max(0, p - 35):p + 6]})
        z3 = e3_strip(tx)
        if z3 != tx:
            e3_fire += 1
            e3_removed += len(_HEDGE.findall(tx))
            if e3_strip(z3) != z3:
                e3_idem_fail += 1

    rep = {
        "schema_version": "lang_text_analysis.v1",
        "n_test_rows": n,
        "E1": {
            "anchors": _ANCHOR.pattern, "window_chars": _ANCHOR_WINDOW,
            "fire_rate": e1_fire / n, "rows_fired": e1_fire,
            "subs_total": e1_subs, "subs_per_firing_row": e1_subs / max(e1_fire, 1),
            "protected_letter_mentions": protected_letters,
            "clobbered_enumeration_letters": clobbered,     # MUST be 0 (no-clobber gate)
            "no_clobber": clobbered == 0,
            "idempotent": e1_idem_fail == 0,
            "note": "no-clobber prioritized over recall; E2/T1 give universal coverage (§8 risk).",
            "examples": examples,
        },
        "E3": {
            "hedge_terms": _HEDGE_TERMS,
            "fire_rate": e3_fire / n, "rows_fired": e3_fire,
            "hedges_removed_total": e3_removed, "idempotent": e3_idem_fail == 0,
            "note": "adverbial hedges only; structural verbs (implies/signals/...) kept.",
        },
    }
    (lang / "verbalization_analysis.json").write_text(json.dumps(rep, indent=2), encoding="utf-8")
    print(f"[0b] E1 fires {e1_fire}/{n} ({e1_fire/n:.1%}), {e1_subs} subs, "
          f"no_clobber={rep['E1']['no_clobber']}, idempotent={rep['E1']['idempotent']}")
    print(f"[0b] E3 fires {e3_fire}/{n} ({e3_fire/n:.1%}), {e3_removed} hedges removed, "
          f"idempotent={rep['E3']['idempotent']}")
    print(f"[0b] -> {lang/'verbalization_analysis.json'}")
    return rep


# =============================================================================
#  magnitude convention + the replacement edit_fn (shared CPU/GPU)
# =============================================================================

def apply_convention(h_recon: np.ndarray, h_norm: np.ndarray, convention: str,
                     cohort_mean: float | None = None) -> np.ndarray:
    """Map raw AR reconstructions -> ĥ_steer under one magnitude convention (§3, §7.5).

    h_recon: [n,d] raw AR output. h_norm: [n] the rows' ‖h_orig‖. Returns [n,d] fp32.
      native     -> as-is (AR's own scale)
      normmatch  -> rescale each row to its own ‖h_orig‖ (DEFAULT; directional patch)
      cohortmean -> rescale each row to the cohort-mean ‖h_orig‖
    """
    h = np.asarray(h_recon, dtype=np.float64)
    if convention == "native":
        return h.astype(np.float32)
    norms = np.linalg.norm(h, axis=1, keepdims=True)
    norms = np.where(norms > 0, norms, 1.0)
    if convention == "normmatch":
        target = np.asarray(h_norm, dtype=np.float64).reshape(-1, 1)
    elif convention == "cohortmean":
        cm = float(cohort_mean if cohort_mean is not None else np.mean(h_norm))
        target = np.full((h.shape[0], 1), cm)
    else:
        raise ValueError(f"unknown convention {convention!r} (have {CONVENTIONS})")
    return (h / norms * target).astype(np.float32)


def make_edit_fn(steer_batch):
    """The only conceptual change from KAPPA's edit: a REPLACEMENT, not an additive Δh.
    `steer_batch` is the [B,d] tensor for the current batch (aligned to batch.row_indices)."""
    def _fn(h_last, layer):   # (h_last:[B,d], cache_layer) -> [B,d]
        return steer_batch.to(h_last.dtype)
    return _fn


def ratio_offmanifold(h_steer: np.ndarray, h_orig: np.ndarray) -> np.ndarray:
    """ratio = ‖ĥ_steer − h_orig‖/‖h_orig‖ — exp04's NLA-independent off-manifold proxy."""
    d = np.linalg.norm(h_steer.astype(np.float64) - h_orig.astype(np.float64), axis=1)
    hn = np.linalg.norm(h_orig.astype(np.float64), axis=1)
    return np.divide(d, hn, out=np.zeros_like(d), where=hn > 0)


# =============================================================================
#  CPU — pred-probe proxy (cheap pre-rank of "what Qwen will say"; filter, not truth)
# =============================================================================

def _load_probe(target: str):
    z = np.load(_PROBES / "example_level" / target / f"layer{LAYER:02d}.npz")
    return z["W"].astype(np.float64), z["b"].astype(np.float64)


def proxy_rank(op: str, subset: str, convention: str = DEFAULT_CONVENTION,
               out: Path | None = None) -> dict:
    """argmax(W_pred·ĥ_steer + b_pred) on the saved recon arrays. CPU; pre-ranks configs by
    predicted steering-success / accuracy before spending Qwen re-forward. pred_acc≈0.912 on
    orig, so this tracks the real readout ~91% (but can break on heavily-edited ĥ — confirm
    winners with eval-readout)."""
    _, out = _paths(None, out)
    h_steer, rows = _load_recon(out, op, subset, convention)
    Wp, bp = _load_probe("pred")
    post = F.posteriors(h_steer.astype(np.float64), Wp, bp)
    yhat = post.argmax(1)
    tg = _targets_for_rows(out, rows)
    X, gt = tg["X_idx"], tg["gt_idx"]
    res = {"op": op, "subset": subset, "convention": convention, "n": len(rows),
           "proxy_success_vs_X": float(np.mean(yhat == X)),
           "proxy_acc_vs_gt": float(np.mean(yhat == gt)),
           "proxy_pred_balance": _count([SYMBOLS[i] for i in yhat])}
    print(f"[proxy] {op}/{subset}/{convention}: success(yhat==X)={res['proxy_success_vs_X']:.4f} "
          f"acc(yhat==gt)={res['proxy_acc_vs_gt']:.4f} balance={res['proxy_pred_balance']}")
    return res


# ----------------------------- recon / target loaders -------------------------

def _recon_path(out: Path, op: str, subset: str) -> Path:
    return out / "lang" / "recon" / f"h_recon_{op}__{subset}.npy"


def _load_recon(out: Path, op: str, subset: str, convention: str):
    """-> (ĥ_steer[n,d] under `convention`, rows[list])."""
    p = _recon_path(out, op, subset)
    raw = np.load(p)
    rows = json.loads(p.with_suffix(".rows.json").read_text(encoding="utf-8"))
    tg = _targets_for_rows(out, rows)
    h_steer = apply_convention(raw, np.asarray(tg["h_norm"]), convention,
                               cohort_mean=_cohort_mean(out))
    return h_steer, rows


def _cohort_mean(out: Path) -> float:
    return json.loads((out / "lang" / "subsets.json").read_text(encoding="utf-8"))["cohort_mean_h_norm"]


def _targets_for_rows(out: Path, rows: list[int]) -> dict:
    import pyarrow.parquet as pq
    t = pq.read_table(out / "lang" / "targets.parquet").to_pydict()
    pos = {int(r): i for i, r in enumerate(t["row_index"])}
    sel = [pos[int(r)] for r in rows]
    return {k: [t[k][i] for i in sel] for k in t}


def _h_orig_for_rows(rows: list[int]) -> np.ndarray:
    h0 = np.load(_EMB / f"h_layer{LAYER:02d}_orig.npy", mmap_mode="r")
    return np.asarray(h0[list(rows)], dtype=np.float32)


# =============================================================================
#  GPU — AR reconstruct (lazy: nla_inference)
# =============================================================================

def reconstruct(op: str, subset: str, critic_dir: str, out: Path | None = None,
                save_edits: bool = True) -> dict:
    """edit z_orig per row -> AR.reconstruct -> RAW ĥ_recon [n,d]. The convention is applied
    later (eval/proxy/report), so one AR pass serves every convention. GPU, but cheap."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    _add_nla_inference_to_path()
    from nla_inference import NLACritic
    import torch
    _, out = _paths(None, out)
    lang = out / "lang"; (lang / "recon").mkdir(parents=True, exist_ok=True)
    if save_edits:
        (lang / "edits").mkdir(parents=True, exist_ok=True)

    rows = load_subset(out, subset)
    t = _orig_table(out).to_pydict()
    by_row = {int(t["row_index"][i]): i for i in range(len(t["row_index"]))}
    critic = NLACritic(critic_dir, device="cuda:0" if torch.cuda.is_available() else "cpu")

    recon = np.empty((len(rows), D_MODEL), dtype=np.float32)
    edits = []
    t0 = time.time()
    for k, r in enumerate(rows):
        i = by_row[int(r)]
        X = t[TARGET_COL][i]
        z_edit = edit(op, t["nl_text"][i], X)
        h = critic.reconstruct(z_edit)
        recon[k] = (h.detach().float().cpu().numpy() if hasattr(h, "detach")
                    else np.asarray(h, dtype=np.float32)).reshape(-1)
        if save_edits:
            edits.append({"row_index": int(r), "example_id": t["example_id"][i],
                          "X": X, "z_edit": z_edit})
        if (k + 1) % 200 == 0 or k == len(rows) - 1:
            print(f"[reconstruct {op}/{subset}] {k+1}/{len(rows)} "
                  f"{(k+1)/max(time.time()-t0,1e-6):.1f}/s", flush=True)

    SS.atomic_save_npy(_recon_path(out, op, subset), recon)
    _recon_path(out, op, subset).with_suffix(".rows.json").write_text(
        json.dumps([int(r) for r in rows]), encoding="utf-8")
    if save_edits:
        pq.write_table(pa.table({k: [e[k] for e in edits]
                                 for k in ("row_index", "example_id", "X", "z_edit")}),
                       lang / "edits" / f"{op}__{subset}.parquet")
    print(f"[reconstruct] {op}/{subset}: {recon.shape} -> {_recon_path(out, op, subset).name}")
    return {"op": op, "subset": subset, "n": len(rows)}


# =============================================================================
#  GPU — Qwen re-forward (reuse exp04 kappa/ VERBATIM)
# =============================================================================

def _import_kappa():
    if str(_EXP04) not in sys.path:
        sys.path.insert(0, str(_EXP04))
    from kappa import config, dataset, prompt, model_forward, generate
    return config, dataset, prompt, model_forward, generate


def _add_nla_inference_to_path():
    """nla_inference.py lives beside the checkpoints on the pod (PYTHONPATH=/workspace/nla-inference).
    Local: fall back to .podref/ which vendors a copy."""
    for cand in (os.environ.get("NLA_INFERENCE_DIR"), str(_ROOT / ".podref")):
        if cand and (Path(cand) / "nla_inference.py").exists() and cand not in sys.path:
            sys.path.insert(0, cand)
            return


def _build_prompts_for_rows(rows, examples_path, prompt_cfg, tokenizer, out=None):
    """Rebuild the FROZEN prompts (deterministic) for `rows` and the row_index map into the
    steer array order. Returns (prompts, symbol_token_ids, row_index_map)."""
    config, dataset, prompt, MF, generate = _import_kappa()
    examples = dataset.load_examples(examples_path)
    by_eid = {ex.example_id: ex for ex in examples}
    tg_rows = _orig_table(out or _ROOT / "out").to_pydict()
    eid_by_row = {int(tg_rows["row_index"][i]): tg_rows["example_id"][i]
                  for i in range(len(tg_rows["row_index"]))}
    sub_examples = [by_eid[eid_by_row[int(r)]] for r in rows]
    prompts, symbol_ids = prompt.render_all(sub_examples, prompt_cfg, tokenizer)
    symbol_token_ids = [symbol_ids[s] for s in SYMBOLS]
    row_index_map = {eid_by_row[int(r)]: k for k, r in enumerate(rows)}  # eid -> steer-array idx
    return prompts, symbol_token_ids, row_index_map


def patch_readout(lm, prompts, symbol_token_ids, row_index_map, steer, *,
                  layer: int = LAYER, batch_size: int = 16, parity: bool = True,
                  tag: str = ""):
    """Core plumbing (model-agnostic; a tiny random Qwen2 drives it in tests): patch ĥ_steer @
    `layer` last token as a REPLACEMENT -> re-forward -> (yhat[n], p_model[n,4]). `steer` is a
    torch [n,d] aligned to row_index_map's values; `prompts` cover those n rows."""
    import torch
    config, dataset, prompt, MF, generate = _import_kappa()
    if parity:   # prove the patch path is faithful (no-op identity + locality), like harvest
        MF.assert_parity(MF.run_parity_check(lm, prompts[:min(8, len(prompts))]))
        print(f"[patch-readout{tag}] parity gate OK", flush=True)
    n = len(prompts)
    yhat = np.empty(n, dtype=np.int64)
    pmodel = np.empty((n, 4), dtype=np.float32)
    t0 = time.time()
    n_batches = (n + batch_size - 1) // batch_size
    for bi, batch in enumerate(MF.make_batches(prompts, lm.tokenizer.pad_token_id, batch_size,
                                               lm.device, row_index_map)):
        steer_batch = steer[batch.row_indices]                       # [B,d] aligned to the batch
        fo = MF.run_forward(lm, batch, symbol_token_ids,
                            edit_fn=make_edit_fn(steer_batch), layer_indices=[layer])
        for i, r in enumerate(batch.row_indices):
            yhat[r] = int(fo.y_tilde[i]); pmodel[r] = fo.p_model[i].cpu().numpy()
        if bi % 20 == 0 or bi == n_batches - 1:
            print(f"[patch-readout{tag}] batch {bi+1}/{n_batches} "
                  f"{(bi+1)*batch_size/max(time.time()-t0,1e-6):.0f} ex/s", flush=True)
    return yhat, pmodel


def eval_readout(op: str, subset: str, model_id: str | None = None,
                 convention: str = DEFAULT_CONVENTION, out: Path | None = None,
                 inputs: Path | None = None, batch_size: int = 16,
                 parity: bool = True) -> dict:
    """Patch ĥ_steer @ L20 last token (REPLACEMENT) -> re-forward Qwen -> ŷ / ACC / AGR /
    steering-success. Reuses exp04 model_forward verbatim. Writes eval/readout_<op>.parquet."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    import torch
    config, dataset, prompt, MF, generate = _import_kappa()
    inputs, out = _paths(inputs, out)
    lang = out / "lang"; (lang / "eval").mkdir(parents=True, exist_ok=True)

    h_steer_np, rows = _load_recon(out, op, subset, convention)
    cfg = config.load_config(_EXP04 / "experiment.yaml")
    lm = MF.load_model(model_id or cfg.model.model_id, dtype=cfg.model.dtype)
    prompts, symbol_token_ids, row_index_map = _build_prompts_for_rows(
        rows, inputs / "examples.jsonl", cfg.prompt, lm.tokenizer, out=out)
    assert symbol_token_ids == [32, 33, 34, 35], f"symbol ids drift: {symbol_token_ids}"

    steer = torch.tensor(h_steer_np, device=lm.device)
    yhat, pmodel = patch_readout(lm, prompts, symbol_token_ids, row_index_map, steer,
                                 batch_size=batch_size, parity=parity, tag=f" {op}/{subset}")
    tg = _targets_for_rows(out, rows)
    X, gt = np.asarray(tg["X_idx"]), np.asarray(tg["gt_idx"])
    h_orig = _h_orig_for_rows(rows)
    ratio = ratio_offmanifold(h_steer_np, h_orig)
    rec = {"row_index": [int(r) for r in rows], "example_id": tg["example_id"],
           "y_hat_idx": yhat.tolist(), "y_hat_symbol": [SYMBOLS[i] for i in yhat],
           "X": tg["X"], "gt_symbol": tg["gt_symbol"], "ratio": ratio.tolist(),
           "p_model": [list(map(float, r)) for r in pmodel]}
    pq.write_table(pa.table(rec), lang / "eval" / f"readout_{op}__{subset}__{convention}.parquet")
    res = {"op": op, "subset": subset, "convention": convention, "n": len(rows),
           "acc": float(np.mean(yhat == gt)), "agr": float(np.mean(X == yhat)),
           "steer_success": float(np.mean(yhat == X)),
           "mean_ratio": float(ratio.mean()), "y_balance": _count([SYMBOLS[i] for i in yhat])}
    (lang / "eval" / f"readout_{op}__{subset}__{convention}.json").write_text(
        json.dumps(res, indent=2), encoding="utf-8")
    print(f"[eval-readout] {op}/{subset}/{convention}: ACC={res['acc']:.4f} AGR={res['agr']:.4f} "
          f"success={res['steer_success']:.4f} ratio={res['mean_ratio']:.4f}")
    return res


def _prefill_only_patch_hook(layer_module, pos_last, steer_batch):
    """A REPLACEMENT patch for the generation path. Unlike the single-forward readout hook
    (exp04 model_forward, which bakes in pos_last), generation runs many forwards: a prefill
    over the full prompt (seq>1) then incremental single-token steps (seq==1). We edit ONLY the
    prefill — overwrite the residual at pos_last — so the patched state is written into the KV
    cache and every generated token attends to it. Editing a seq==1 step would (a) index the
    wrong position and (b) re-patch the new token's own residual. This is the faithful "patch
    active on the first step" (MAIN-EXP §3); the readout path keeps exp04's hook verbatim."""
    import torch
    rows = torch.arange(pos_last.shape[0], device=pos_last.device)

    def hook(module, args, output):
        h = output[0] if isinstance(output, tuple) else output
        if h.shape[1] == 1:                       # incremental decode step -> patch already in KV cache
            return None
        h = h.clone()
        h[rows, pos_last, :] = steer_batch.to(h.dtype)
        return (h, *output[1:]) if isinstance(output, tuple) else h

    return layer_module.register_forward_hook(hook)


def patch_generate(lm, prompts, row_index_map, steer, *, layer: int = LAYER,
                   max_new_tokens: int = 8, batch_size: int = 16):
    """Core free-gen plumbing (model-agnostic): the L20 replacement patch is applied on the
    prefill step under model.generate -> parsed letter per row. Returns (symbols[n], idx[n])."""
    import torch
    config, dataset, prompt, MF, generate = _import_kappa()
    n = len(prompts)
    sym = ["" for _ in range(n)]; idx = [-1] * n
    eos = generate._eos_ids(lm.tokenizer); pad = lm.tokenizer.pad_token_id
    module = (lm.embed if layer == 0 else lm.blocks[layer - 1])
    for batch in MF.make_batches(prompts, pad, batch_size, lm.device, row_index_map):
        steer_batch = steer[batch.row_indices]
        in_w = batch.input_ids.shape[1]
        handle = _prefill_only_patch_hook(module, batch.pos_last, steer_batch)
        try:
            o = lm.model.generate(input_ids=batch.input_ids, attention_mask=batch.attention_mask,
                                  max_new_tokens=max_new_tokens, do_sample=False, num_beams=1,
                                  pad_token_id=pad, eos_token_id=(eos or None), use_cache=True)
        finally:
            handle.remove()
        for i, r in enumerate(batch.row_indices):
            line = lm.tokenizer.decode(o[i, in_w:], skip_special_tokens=True).split("\n", 1)[0].strip()
            s, k = generate.parse_symbol(line)
            sym[r] = s; idx[r] = k
    return sym, idx


def eval_generate(op: str, subset: str, model_id: str | None = None,
                  convention: str = DEFAULT_CONVENTION, out: Path | None = None,
                  inputs: Path | None = None, batch_size: int = 16,
                  max_new_tokens: int = 8) -> dict:
    """Free-generation confirmation (headline only): same L20 patch active under model.generate,
    parse the letter. Guards the readout against a constrained-readout-only artifact."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    import torch
    config, dataset, prompt, MF, generate = _import_kappa()
    inputs, out = _paths(inputs, out)
    lang = out / "lang"; (lang / "eval").mkdir(parents=True, exist_ok=True)

    h_steer_np, rows = _load_recon(out, op, subset, convention)
    cfg = config.load_config(_EXP04 / "experiment.yaml")
    lm = MF.load_model(model_id or cfg.model.model_id, dtype=cfg.model.dtype)
    prompts, symbol_token_ids, row_index_map = _build_prompts_for_rows(
        rows, inputs / "examples.jsonl", cfg.prompt, lm.tokenizer, out=out)
    steer = torch.tensor(h_steer_np, device=lm.device)
    sym, idx = patch_generate(lm, prompts, row_index_map, steer, max_new_tokens=max_new_tokens,
                              batch_size=batch_size)
    tg = _targets_for_rows(out, rows)
    X, gt = np.asarray(tg["X_idx"]), np.asarray(tg["gt_idx"])
    gi = np.asarray(idx)
    parse = gi >= 0
    pq.write_table(pa.table({"row_index": [int(r) for r in rows], "example_id": tg["example_id"],
                             "gen_symbol": sym, "gen_idx": idx, "X": tg["X"], "gt_symbol": tg["gt_symbol"]}),
                   lang / "eval" / f"generate_{op}__{subset}__{convention}.parquet")
    res = {"op": op, "subset": subset, "convention": convention, "n": len(rows),
           "parse_rate": float(parse.mean()),
           "gen_acc": float(np.mean(gi[parse] == gt[parse])) if parse.any() else 0.0,
           "gen_success": float(np.mean(gi[parse] == X[parse])) if parse.any() else 0.0}
    (lang / "eval" / f"generate_{op}__{subset}__{convention}.json").write_text(
        json.dumps(res, indent=2), encoding="utf-8")
    print(f"[eval-generate] {op}/{subset}/{convention}: parse={res['parse_rate']:.4f} "
          f"gen_acc={res['gen_acc']:.4f} gen_success={res['gen_success']:.4f}")
    return res


def ome(op: str, subset: str, actor_dir: str, critic_dir: str,
        convention: str = DEFAULT_CONVENTION, out: Path | None = None,
        concurrency: int = 16) -> dict:
    """OME re-verbalization (headline): AV(ĥ_steer) -> AR.score -> cos -> OME = 1 − cos. The
    costliest step (AV); reuses nla_run's threaded path. Honest caveat (§9): ĥ_steer is an AR
    output so OME is mildly favorable — the NLA-independent `ratio` is the load-bearing metric."""
    import threading
    from concurrent.futures import ThreadPoolExecutor, as_completed
    _add_nla_inference_to_path()
    from nla_inference import NLAClient, NLACritic
    import torch
    _, out = _paths(None, out)
    lang = out / "lang"; (lang / "ome").mkdir(parents=True, exist_ok=True)

    h_steer, rows = _load_recon(out, op, subset, convention)
    client = NLAClient(actor_dir, sglang_url=os.environ.get("AV_URL", "http://localhost:30000"))
    critic = NLACritic(critic_dir, device="cuda:0" if torch.cuda.is_available() else "cpu")
    cos = [None] * len(rows)
    ar_lock = threading.Lock()

    def work(k):
        v = np.ascontiguousarray(h_steer[k], dtype=np.float32)
        for attempt in range(6):
            try:
                text = client.generate(v, temperature=1.0, max_new_tokens=256); break
            except Exception:  # noqa: BLE001
                if attempt == 5:
                    return
                time.sleep(1.0 * (attempt + 1))
        with ar_lock:
            _, c = critic.score(text, v)
        cos[k] = float(c)

    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        for f in as_completed([ex.submit(work, k) for k in range(len(rows))]):
            f.result()
    cvec = np.array([c for c in cos if c is not None])
    res = {"op": op, "subset": subset, "convention": convention, "n": int(cvec.size),
           "mean_cos_roundtrip": float(cvec.mean()), "ome": float(1.0 - cvec.mean())}
    (lang / "ome" / f"{op}__{subset}__{convention}.json").write_text(json.dumps(res, indent=2),
                                                                     encoding="utf-8")
    print(f"[ome] {op}/{subset}/{convention}: mean_cos={res['mean_cos_roundtrip']:.4f} "
          f"OME={res['ome']:.4f} (n={cvec.size})")
    return res


# =============================================================================
#  CPU — report + Pareto frontier vs KAPPA single-L20
# =============================================================================

def kappa_frontier(out: Path) -> dict:
    """The single-L20 KAPPA curve we must beat (MAIN-EXP §3), read from the COMPLETE FVE run:
    per alpha -> {ome = 1 − mean_cos_roundtrip, ratio, acc}. NLA-native OME is apples-to-apples."""
    a = json.loads((out / "fve" / "analysis.json").read_text(encoding="utf-8"))["per_alpha"]
    curve = {}
    for k, v in a.items():
        if v.get("exp04_acc") is not None:
            curve[float(k)] = {"ome": 1.0 - v["mean_cos"], "ratio": v.get("ratio_subset"),
                               "acc": v["exp04_acc"]}
    return curve


def report(subset: str = "full", convention: str = DEFAULT_CONVENTION,
           out: Path | None = None) -> dict:
    """Collate (ACC, AGR, success, OME, ratio) per method + bootstrap CIs, build the (ACC,OME)
    and (ACC,ratio) Pareto plot vs the KAPPA frontier, and decide WIN / PARTIAL / NULL (§4, §6)."""
    import pyarrow.parquet as pq
    _, out = _paths(None, out)
    lang = out / "lang"; ev = lang / "eval"
    frontier = kappa_frontier(out)
    # KAPPA single-L20 peak ACC + its off-manifold cost (the point to beat / floor)
    peak = max(frontier.values(), key=lambda v: v["acc"]) if frontier else None
    floor_ome = frontier[min(frontier)]["ome"] if frontier else None    # α≈0 on-manifold floor

    methods = []
    for jf in sorted(ev.glob(f"readout_*__{subset}__{convention}.json")):
        r = json.loads(jf.read_text(encoding="utf-8"))
        op = r["op"]
        # per-row ACC + ratio for bootstrap CIs
        pr = pq.read_table(ev / f"readout_{op}__{subset}__{convention}.parquet").to_pydict()
        acc_rows = np.asarray([int(y == g) for y, g in zip(pr["y_hat_symbol"], pr["gt_symbol"])], float)
        m = {**r, "acc_ci": FA.bootstrap_ci(acc_rows), "ratio_ci": FA.bootstrap_ci(np.asarray(pr["ratio"]))}
        og = lang / "ome" / f"{op}__{subset}__{convention}.json"
        if og.exists():
            m["ome"] = json.loads(og.read_text(encoding="utf-8"))["ome"]
        gg = ev / f"generate_{op}__{subset}__{convention}.json"
        if gg.exists():
            m["generate"] = json.loads(gg.read_text(encoding="utf-8"))
        methods.append(m)

    verdict = _verdict(methods, peak, floor_ome)
    rep = {"schema_version": "lang_report.v1", "subset": subset, "convention": convention,
           "kappa_frontier": frontier, "kappa_peak": peak, "on_manifold_floor_ome": floor_ome,
           "methods": methods, "verdict": verdict}
    (lang / "report_data.json").write_text(json.dumps(rep, indent=2), encoding="utf-8")
    _report_md(lang, rep)
    _frontier_plot(lang, rep)
    print(f"[report] {len(methods)} methods; verdict={verdict['decision']} "
          f"(best ACC {verdict.get('best_acc')}, KAPPA peak {peak['acc'] if peak else '?'})")
    return rep


def _verdict(methods, peak, floor_ome) -> dict:
    """WIN: a method's ACC ≥ KAPPA peak AND its OME (or ratio) strictly below KAPPA's cost at
    that ACC, ideally near the on-manifold floor. PARTIAL: matches ACC but not cheaper. NULL: no
    ACC match. (Headline CIs decide ties on the real run; here we report the point estimates.)"""
    if not methods or peak is None:
        return {"decision": "PENDING", "note": "no eval outputs / frontier yet"}
    best = max(methods, key=lambda m: m["acc"])
    win = (best["acc"] >= peak["acc"] - 1e-9
           and (best.get("ome", 1.0) < peak["ome"] - 1e-9
                or best.get("mean_ratio", 1.0) < peak["ratio"] - 1e-9))
    decision = ("WIN" if win else
                "PARTIAL" if best["acc"] >= peak["acc"] - 1e-9 else "NULL")
    return {"decision": decision, "best_op": best["op"], "best_acc": best["acc"],
            "best_ome": best.get("ome"), "best_ratio": best.get("mean_ratio"),
            "kappa_peak_acc": peak["acc"], "kappa_peak_ome": peak["ome"],
            "kappa_peak_ratio": peak["ratio"], "on_manifold_floor_ome": floor_ome}


def _report_md(lang: Path, rep: dict) -> None:
    peak = rep["kappa_peak"]; v = rep["verdict"]
    lines = [f"# Language-space steering — report ({rep['subset']}, {rep['convention']})", "",
             f"**Verdict: {v['decision']}** — best `{v.get('best_op')}` "
             f"ACC {v.get('best_acc')} vs KAPPA single-L20 peak ACC "
             f"{peak['acc'] if peak else '?'} @ OME {peak['ome']:.3f} / ratio {peak['ratio']:.3f}.",
             f"On-manifold floor OME ≈ {rep['on_manifold_floor_ome']:.3f}.", "",
             "| op | ACC | AGR | success(ŷ==X) | OME | ratio | ACC 95% CI |",
             "|---|---|---|---|---|---|---|"]
    for m in sorted(rep["methods"], key=lambda x: -x["acc"]):
        ci = m.get("acc_ci", (None, None))
        ome_cell = f"{m['ome']:.3f}" if isinstance(m.get("ome"), (int, float)) else "—"
        lines.append(f"| {m['op']} | {m['acc']:.4f} | {m['agr']:.4f} | {m['steer_success']:.4f} "
                     f"| {ome_cell} | {m['mean_ratio']:.3f} | [{ci[0]:.3f}, {ci[1]:.3f}] |")
    (lang / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _frontier_plot(lang: Path, rep: dict) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:  # noqa: BLE001
        print("[report] matplotlib absent - skipping frontier.png (report_data.json is the deliverable).")
        return
    fr = rep["kappa_frontier"]
    fa = sorted(fr)
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    for j, (xk, xl) in enumerate([("ome", "OME = 1 − cos_RT"), ("ratio", "ratio ‖Δh‖/‖h‖")]):
        ax[j].plot([fr[a][xk] for a in fa], [fr[a]["acc"] for a in fa], "o-", color="0.5",
                   label="KAPPA single-L20")
        for a in fa:
            ax[j].annotate(f"α{a:g}", (fr[a][xk], fr[a]["acc"]), fontsize=7, color="0.5")
        for m in rep["methods"]:
            x = m.get("ome") if xk == "ome" else m.get("mean_ratio")
            if isinstance(x, (int, float)):
                ax[j].scatter([x], [m["acc"]], s=40)
                ax[j].annotate(m["op"], (x, m["acc"]), fontsize=8)
        ax[j].set(xlabel=xl, ylabel="ACC", title=f"Pareto: ACC vs {xk}")
        ax[j].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(lang / "frontier.png", dpi=120)
    print(f"[report] -> {lang/'frontier.png'}")


# =============================================================================
#  CLI
# =============================================================================

def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description="Language-space steering (MAIN-EXP).")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--inputs", type=Path, default=None)
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("build-targets")
    sub.add_parser("analyze-text")

    def add_op_subset(p, conv=True):
        p.add_argument("--op", required=True); p.add_argument("--subset", default="sweep512")
        if conv:
            p.add_argument("--convention", default=DEFAULT_CONVENTION, choices=CONVENTIONS)

    pr = sub.add_parser("reconstruct"); add_op_subset(pr, conv=False)
    pr.add_argument("--critic", required=True)
    px = sub.add_parser("proxy"); add_op_subset(px)
    er = sub.add_parser("eval-readout"); add_op_subset(er)
    er.add_argument("--model-id", default=None); er.add_argument("--batch-size", type=int, default=16)
    er.add_argument("--no-parity", action="store_true")
    eg = sub.add_parser("eval-generate"); add_op_subset(eg)
    eg.add_argument("--model-id", default=None); eg.add_argument("--batch-size", type=int, default=16)
    om = sub.add_parser("ome"); add_op_subset(om)
    om.add_argument("--actor", required=True); om.add_argument("--critic", required=True)
    om.add_argument("--concurrency", type=int, default=16)
    rp = sub.add_parser("report")
    rp.add_argument("--subset", default="full"); rp.add_argument("--convention", default=DEFAULT_CONVENTION)

    a = ap.parse_args(argv)
    if a.cmd == "build-targets":
        build_targets(a.inputs, a.out)
    elif a.cmd == "analyze-text":
        analyze_text(a.out)
    elif a.cmd == "reconstruct":
        reconstruct(a.op, a.subset, a.critic, out=a.out)
    elif a.cmd == "proxy":
        proxy_rank(a.op, a.subset, a.convention, out=a.out)
    elif a.cmd == "eval-readout":
        eval_readout(a.op, a.subset, a.model_id, a.convention, a.out, a.inputs,
                     a.batch_size, parity=not a.no_parity)
    elif a.cmd == "eval-generate":
        eval_generate(a.op, a.subset, a.model_id, a.convention, a.out, a.inputs, a.batch_size)
    elif a.cmd == "ome":
        ome(a.op, a.subset, a.actor, a.critic, a.convention, a.out, a.concurrency)
    elif a.cmd == "report":
        report(a.subset, a.convention, a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
