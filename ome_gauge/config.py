"""Run contract + paths for OME-GAUGE.

Loads config.json, computes a stable `config_hash` over its semantic content
(comment keys starting with "_" excluded), and exposes the output layout under
NLA-final/out/ome/. Every phase stamps `config_hash` into its manifest (mirrors the parent's
nla_run.CONFIG_HASH discipline; SPEC s8).

Also provides the canonical, row-aligned loaders the CPU phases share (benign a0, the
correctness label, the paired subsets) so directions/steer/detectors never re-derive the join.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ome_gauge import _NLA_ROOT  # noqa: F401  (import also runs the sys.path bootstrap)
from src import features  # parent reuse: FeaturePaths, canonical_order, write_json_atomic, ...

# ---- locations -------------------------------------------------------------
PKG_DIR = Path(__file__).resolve().parent           # .../ome_gauge
NLA_ROOT = PKG_DIR.parent                            # .../NLA-final (repo root; the experiment is the whole repo)
OME_COLLAPSE = NLA_ROOT                              # retained alias: tests override C.OME_COLLAPSE to redirect data/
CONFIG_PATH = NLA_ROOT / "config.json"


def _strip_doc(obj):
    """Drop comment keys (leading "_") so config_hash tracks semantics, not prose."""
    if isinstance(obj, dict):
        return {k: _strip_doc(v) for k, v in obj.items() if not k.startswith("_")}
    if isinstance(obj, list):
        return [_strip_doc(v) for v in obj]
    return obj


def load_config(path: Path = CONFIG_PATH) -> dict:
    cfg = json.loads(path.read_text(encoding="utf-8"))
    cfg["config_hash"] = config_hash(cfg)
    return cfg


def config_hash(cfg: dict) -> str:
    payload = json.dumps(_strip_doc(cfg), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


CONFIG = load_config()
CONFIG_HASH = CONFIG["config_hash"]

# convenience constants (single source of truth = config.json)
LAYER = int(CONFIG["layer"])
D_MODEL = int(CONFIG["d_model"])
ALPHAS = [float(a) for a in CONFIG["alphas"]]                       # KAPPA arm + matched-alpha H1
ALPHAS_ADDITIVE = [float(a) for a in CONFIG["alphas_additive"]]     # DiM/random (ratio-matched to KAPPA)
SUBSET_SEED = int(CONFIG["subset_seed"])
RANDOM_SEED = int(CONFIG["random_seed"])
N_RANDOM = int(CONFIG["n_random"])
OME_FLOOR = float(CONFIG["ome_floor"])
ORTHO_FLAG = float(CONFIG["orthogonality_flag_abs_cos"])


def alphas_for(method: str) -> list[float]:
    """The magnitude grid for a method: additive arms (dim/random) use the ratio-matched grid;
    the KAPPA arm uses the closed-form grid. Equal alpha != equal push across methods (QUESTIONS
    s3.3) -> analysis indexes by `ratio`, but each arm must still SPAN no-effect..collapse."""
    return ALPHAS_ADDITIVE if method in ("dim", "random") else ALPHAS


@dataclass(frozen=True)
class OmePaths:
    """Every directory/file the OME-GAUGE phases read or write. Outputs under out/ome/ so the
    parent .gitignore (track *.md/*.json/*.png; S3 the *.npy/*.parquet/*.jsonl) applies as-is."""
    root: Path = NLA_ROOT

    # ---- inputs (reused parent cache) ----
    @property
    def feat(self) -> "features.FeaturePaths":
        return features.default_paths()

    @property
    def inputs(self) -> Path: return self.root / "inputs"

    def a0_npy(self) -> Path: return self.inputs / "h_layer20_steered_a0.npy"
    def steered_npy(self, tag: str) -> Path: return self.inputs / f"h_layer20_steered_a{tag}.npy"
    def examples(self) -> Path: return self.inputs / "examples.jsonl"
    def predictions(self) -> Path: return self.inputs / "predictions.parquet"
    def subset_rows(self) -> Path: return self.inputs / "subset_rows.json"
    def kappa_analysis(self) -> Path: return self.root / "out" / "fve" / "analysis.json"

    # ---- outputs (out/ome/...) ----
    @property
    def out(self) -> Path: return self.root / "out" / "ome"
    def dir_directions(self) -> Path: return self.out / "directions"
    def dir_steer(self) -> Path: return self.out / "steer"
    def dir_ome(self) -> Path: return self.out / "ome"
    def dir_detect(self) -> Path: return self.out / "ome"
    def dir_behave(self) -> Path: return self.out / "behave"
    def dir_analysis(self) -> Path: return self.out / "analysis"
    def dir_report(self) -> Path: return self.out / "report"

    def dirs_npz(self) -> Path: return self.dir_directions() / "dirs.npz"
    def dirs_manifest(self) -> Path: return self.dir_directions() / "dirs_manifest.json"
    def maha_fit(self) -> Path: return self.dir_detect() / "maha_fit.npz"

    # ---- Stage-2 inputs (vendored, gitignored -> S3/pod-local) ----
    @property
    def data(self) -> Path: return OME_COLLAPSE / "data"
    def contrasts(self) -> Path: return self.data / "contrasts"
    def prompts_data(self) -> Path: return self.data / "prompts"
    def contrast_jsonl(self, name: str) -> Path: return self.contrasts() / f"{name}.jsonl"
    def prompt_jsonl(self, name: str) -> Path: return self.prompts_data() / f"{name}.jsonl"
    def data_manifest(self, name: str) -> Path: return self.data / f"{name}_manifest.json"

    # ---- Stage-2 outputs (out/ome/...) ----
    def h_clean(self, set_name: str) -> Path: return self.dir_steer() / f"h_clean_{set_name}.npy"
    def clean_manifest(self, set_name: str) -> Path:
        return self.dir_steer() / f"h_clean_{set_name}_manifest.json"
    def h_enter(self, dir_name: str, set_name: str, tag: str) -> Path:
        return self.dir_steer() / f"h_enter_{dir_name}_{set_name}_a{tag}.npy"
    def steer_manifest_gen(self) -> Path: return self.dir_steer() / "steer_manifest_gen.json"
    def maha_fit_gen(self) -> Path: return self.dir_detect() / "maha_fit_gen.npz"
    def detectors_gen(self) -> Path: return self.dir_detect() / "detectors_gen.parquet"
    def ome_by_cond_gen(self) -> Path: return self.dir_ome() / "ome_by_cond_gen.parquet"
    def gen_parquet(self, dir_name: str, set_name: str, tag: str) -> Path:
        return self.dir_behave() / f"gen_{dir_name}_{set_name}_a{tag}.parquet"
    def gen_ledger(self, dir_name: str, set_name: str, tag: str) -> Path:
        return self.dir_behave() / f"gen_{dir_name}_{set_name}_a{tag}.jsonl"
    def judge_parquet(self, dir_name: str) -> Path:
        return self.dir_behave() / f"judge_{dir_name}.parquet"
    def judge_cache(self) -> Path: return self.dir_behave() / "judge_cache.jsonl"

    # ---- Stage-3 (fine-tune arm) data + outputs (out/ome/ft/...; SFT corpora gitignored) ----
    # The FT arm lives in its OWN namespace so its gen/judge/OME/detector parquets never glob-collide
    # with the Stage-2 ones (PLAN_stage3 §4.4). The SFT training corpora are dual-use -> data/sft/*.jsonl
    # gitignored (S3/pod-local); only the per-set <name>_manifest.json is committed (DESIGN §12).
    def dir_sft(self) -> Path: return self.data / "sft"
    def sft_jsonl(self, name: str) -> Path: return self.dir_sft() / f"{name}.jsonl"
    def dir_ft(self) -> Path: return self.out / "ft"
    # FT harvest: arrays + a steer_manifest_gen-schema manifest, so ome_probe/detectors consume them
    # unchanged (just pointed at this namespace). One array per (model, set) at alpha=0 (no steering).
    def ft_steer_manifest(self) -> Path: return self.dir_ft() / "steer_manifest_gen.json"
    def ft_h_enter(self, model: str, set_name: str) -> Path:
        return self.dir_ft() / f"h_enter_{model}_{set_name}_a0.npy"
    # FT OME + regime-matched detectors (base-manifold fit, FT activations scored)
    def ome_by_cond_ft(self) -> Path: return self.dir_ft() / "ome_by_cond_ft.parquet"
    def detectors_ft(self) -> Path: return self.dir_ft() / "detectors_ft.parquet"
    # FT generation + judge (model in place of the steering dir)
    def gen_ft_parquet(self, model: str, set_name: str, tag: str) -> Path:
        return self.dir_ft() / f"gen_ft_{model}_{set_name}_a{tag}.parquet"
    def gen_ft_ledger(self, model: str, set_name: str, tag: str) -> Path:
        return self.dir_ft() / f"gen_ft_{model}_{set_name}_a{tag}.jsonl"
    def judge_ft_parquet(self, model: str) -> Path:
        return self.dir_ft() / f"judge_ft_{model}.parquet"
    def judge_ft_cache(self) -> Path: return self.dir_ft() / "judge_ft_cache.jsonl"
    def ft_manifest(self) -> Path: return self.dir_ft() / "ft_manifest.json"


PATHS = OmePaths()


def gen_ns(ns: str | None = None) -> dict:
    """Resolve the gen-regime per-condition paths for the Stage-2 steering arm (ns=None -> 'gen') vs
    the Stage-3 fine-tune arm (ns='ft'). ONLY the per-condition manifest / arrays / OME / detector
    paths switch namespace; the BASE-manifold artifacts (h_clean, maha_fit_gen, benign_calib,
    calibration_gen) are always the base model's and are read directly so the FT activations are
    scored against the base reference manifold (PLAN_stage3 §4.2)."""
    if ns == "ft":
        return {"manifest": PATHS.ft_steer_manifest(), "arrays": PATHS.dir_ft(),
                "out": PATHS.dir_ft(), "ome_by_cond": PATHS.ome_by_cond_ft(),
                "detectors": PATHS.detectors_ft()}
    return {"manifest": PATHS.steer_manifest_gen(), "arrays": PATHS.dir_steer(),
            "out": PATHS.dir_ome(), "ome_by_cond": PATHS.ome_by_cond_gen(),
            "detectors": PATHS.detectors_gen()}


# ---- canonical row-aligned loaders (shared by P0/P1/P2) --------------------

def load_benign_a0() -> np.ndarray:
    """Benign cohort a0 (== exp04 h_layer20_orig), fp64 for math. [N, d] in canonical row order."""
    return features.load_h(PATHS.a0_npy())


def canonical_ids() -> list[str]:
    """example_ids in activation_cache_row_order (the load-bearing join key)."""
    return features.canonical_order(PATHS.feat)


def gt_answer_index() -> np.ndarray:
    """Ground-truth answer_index [N] (int 0..3) from examples.jsonl, in canonical row order."""
    order = canonical_ids()
    pos = {eid: i for i, eid in enumerate(order)}
    out = np.full(len(order), -1, dtype=np.int64)
    with open(PATHS.examples(), encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                if r["example_id"] in pos:
                    out[pos[r["example_id"]]] = int(r["answer_index"])
    assert (out >= 0).all(), "examples.jsonl did not cover every canonical row"
    return out


def model_y_tilde() -> np.ndarray:
    """Model forced-choice symbol readout [N] (int 0..3) from predictions.parquet, canonical order."""
    import pyarrow.parquet as pq
    order = canonical_ids()
    pos = {eid: i for i, eid in enumerate(order)}
    pred = pq.read_table(PATHS.predictions(), columns=["example_id", "y_tilde"]).to_pydict()
    out = np.full(len(order), -1, dtype=np.int64)
    for eid, y in zip(pred["example_id"], pred["y_tilde"]):
        if eid in pos:
            out[pos[eid]] = int(y)
    assert (out >= 0).all(), "predictions.parquet did not cover every canonical row"
    return out


def correctness_label() -> np.ndarray:
    """Boolean [N]: model-readout-correct = (predictions.y_tilde == examples.answer_index),
    aligned to canonical row order. D_correct's class label (no targets.parquet exists)."""
    return model_y_tilde() == gt_answer_index()


def test_mask() -> np.ndarray:
    """Boolean [N] mask of the example_level test split (canonical order)."""
    return features.test_mask_example_level(PATHS.feat, canonical_ids())


def fit_split_mask() -> np.ndarray:
    """Benign contrast/fit rows = trainval (is_test == False), DISJOINT from the test eval subset.
    Used for both D_correct construction and the Mahalanobis benign fit so neither leaks the
    eval rows (QUESTIONS s3.4 directions / s7.3 baseline fit)."""
    return ~test_mask()


def load_paired_subset() -> tuple[list[int], list[str]]:
    """The 1024-row paired test subset (row_indices, example_ids); seed 7. Nests the OME/readout sets."""
    s = json.loads(PATHS.subset_rows().read_text(encoding="utf-8"))
    return [int(r) for r in s["row_indices"]], list(s["example_ids"])


def ome_subset(n: int | None = None) -> list[int]:
    """First `n` rows of the paired subset (subset nests by construction -> seed-stable nested draw).
    Default n = config subset_n_ome (256)."""
    if n is None:
        n = int(CONFIG["subset_n_ome"])
    rows, _ = load_paired_subset()
    return rows[:n]


# ---- Stage-2 constants + clean-format data loaders (data/) ------------------
# The vendored Stage-2 corpora use one small intermediate format so the harvest/steer/judge code
# never re-derives a source-specific schema (the vendor step in directions.py converts each
# canonical public dataset into it). Both jsonl forms are gitignored (*.jsonl -> S3 / pod-local;
# dual-use corpora are never committed); a per-set <name>_manifest.json (tracked) carries the
# source ref + sha256 + class counts (provenance; SPEC s8 / PLAN_stage2 s4).
#   contrast set  data/contrasts/<name>.jsonl : {"pair_id", "pos", "neg", "meta"?}
#   prompt set    data/prompts/<name>.jsonl   : {"prompt_id", "text", "meta"?}

STAGE2 = CONFIG["stage2"]
CONTENT_DIRECTIONS = list(STAGE2["content_directions"])               # D_toxic / D_refusal / D_sycophancy
DANGEROUS_SIGN = {k: int(v) for k, v in STAGE2["dangerous_sign"].items()}   # signed-alpha = dangerous push
PROMPT_SETS = list(STAGE2["prompt_sets"])                             # em / neutral / benign_calib
CONTRAST_SETS = list(STAGE2["contrast_sets"])                        # toxic / refusal / sycophancy

# Stage-3 (fine-tune arm) constants. The SFT corpora (harmful_sft / benign_sft) reuse the data/
# vendoring contract via a third writer (write_sft_set); the three checkpoints (base / harmful_ft /
# benign_ft) are harvested + scored. All Stage-1/2 fields above are untouched (PLAN_stage3).
STAGE3 = CONFIG["stage3"]
SFT_SETS = [k for k in STAGE3["sft"] if not k.startswith("_")]        # harmful_sft / benign_sft
FT_MODELS = list(STAGE3["models"])                                  # base / harmful_ft / benign_ft


def dangerous_signed_dir(name: str, v_unit: np.ndarray) -> np.ndarray:
    """Apply the pre-registered dangerous sign so a POSITIVE swept alpha is always the dangerous
    push (QUESTIONS s3.6): +D_toxic, -D_refusal, +D_sycophancy. Non-content dirs (random) pass
    through (sign +1). The result is still unit-norm (sign flip preserves ||v||)."""
    return float(DANGEROUS_SIGN.get(name, 1)) * np.asarray(v_unit, dtype=np.float64)


def _read_jsonl(path: Path) -> list[dict]:
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                out.append(json.loads(line))
    return out


def load_prompt_set(name: str) -> list[dict]:
    """[{prompt_id, text, meta?}] for a Stage-2 prompt set (em / neutral / benign_calib), in file
    order. The prompt_id is the load-bearing join key through generation + OME + the judge."""
    recs = _read_jsonl(PATHS.prompt_jsonl(name))
    seen = set()
    for r in recs:
        assert "prompt_id" in r and "text" in r, f"prompt set {name!r}: rows need prompt_id + text"
        assert r["prompt_id"] not in seen, f"prompt set {name!r}: duplicate prompt_id {r['prompt_id']!r}"
        seen.add(r["prompt_id"])
    return recs


def load_contrast_pairs(name: str) -> tuple[list[str], list[str], list[str]]:
    """(pair_ids, pos_texts, neg_texts) for a Stage-2 contrast set (toxic / refusal / sycophancy).
    D = unit(mean(h_pos_last) - mean(h_neg_last)); pos is the +class of the contrast (evil / harmful /
    sycophantic). Class balance + counts are audited at harvest (Gate S2.P0)."""
    recs = _read_jsonl(PATHS.contrast_jsonl(name))
    ids = [str(r["pair_id"]) for r in recs]
    pos = [str(r["pos"]) for r in recs]
    neg = [str(r["neg"]) for r in recs]
    assert ids and len(ids) == len(set(ids)), f"contrast set {name!r}: empty or duplicate pair_id"
    return ids, pos, neg
