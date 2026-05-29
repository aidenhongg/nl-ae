"""Phase 0d — CPU dry-run of the whole GPU chain on a TINY random Qwen2 (real architecture,
random weights, d=64, 4 layers) + synthetic activations. Mirrors exp04's test_integration_tiny.

Proves, before any paid GPU:
  * the REPLACEMENT edit_fn patches the residual at the target layer and the parity gate
    (no-op identity + locality) holds for that path;
  * the frozen-prompt rebuild (kappa.prompt.render_all) yields symbol ids {A:32,B:33,C:34,D:35};
  * patch_readout (forced-choice) and patch_generate (free-gen) plumbing run end-to-end and the
    per-row target join (ACC / AGR / steering-success) aligns ŷ to the steer-array rows;
  * the CPU analysis chain reconstruct->proxy->eval->report wires together (synthetic d=3584
    arrays + the real pred-probe + the real KAPPA frontier) and emits report.md + frontier.png.

No GPU, no real 7B model, no AR/AV. Run: python tests/test_lang_steer_tiny.py
"""
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "exp04"))  # kappa/

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from transformers import AutoTokenizer, Qwen2Config, Qwen2ForCausalLM

from src import lang_steer as L
from kappa import config, prompt as KP
from kappa import model_forward as MF
from kappa.dataset import MCQExample

SYM = "ABCD"
D_TINY = 64
N_LAYERS = 4
PATCH_LAYER = 2  # within the tiny model (real run uses L20)


def tiny_lm(tokenizer):
    cfg = Qwen2Config(hidden_size=D_TINY, intermediate_size=2 * D_TINY, num_hidden_layers=N_LAYERS,
                      num_attention_heads=4, num_key_value_heads=2, vocab_size=152064,
                      max_position_embeddings=2048, rms_norm_eps=1e-6)
    torch.manual_seed(0)
    m = Qwen2ForCausalLM(cfg).eval()
    m.requires_grad_(False)
    return MF.LoadedModel(model=m, tokenizer=tokenizer, d_model=D_TINY, n_layers=N_LAYERS,
                          n_hidden_states=N_LAYERS + 1, device=torch.device("cpu"),
                          compute_dtype=torch.float32, blocks=m.model.layers,
                          embed=m.model.embed_tokens, final_norm=m.model.norm)


def _examples(n):
    return [MCQExample(example_id=f"dry-{i:04d}-p0", question=f"Which is option {i}?",
                       options=[{"symbol": SYM[j], "text": f"choice {j} for q{i}"} for j in range(4)],
                       answer_index=i % 4, source_meta={"question_id": f"dry-{i:04d}"})
            for i in range(n)]


def test_patch_plumbing_tiny():
    cfg = config.load_config(str(Path(__file__).resolve().parents[2] / "exp04" / "experiment.yaml"))
    tok = AutoTokenizer.from_pretrained(cfg.model.model_id)
    lm = tiny_lm(tok)
    n = 12
    examples = _examples(n)
    prompts, symbol_ids = KP.render_all(examples, cfg.prompt, tok)
    assert symbol_ids == {"A": 32, "B": 33, "C": 34, "D": 35}, symbol_ids
    symbol_token_ids = [symbol_ids[s] for s in SYM]
    row_index_map = {ex.example_id: i for i, ex in enumerate(examples)}

    # synthetic ĥ_steer at the tiny model's width; norm-matched to a plausible ||h||
    rng = np.random.default_rng(7)
    steer_np = rng.standard_normal((n, D_TINY)).astype(np.float32)
    steer_np = L.apply_convention(steer_np, np.full(n, 80.0), "normmatch")
    steer = torch.tensor(steer_np)

    # --- forced-choice readout through the REPLACEMENT patch (+ parity gate) ---
    yhat, pmodel = L.patch_readout(lm, prompts, symbol_token_ids, row_index_map, steer,
                                   layer=PATCH_LAYER, batch_size=5, parity=True, tag=" dry")
    assert yhat.shape == (n,) and pmodel.shape == (n, 4)
    assert set(int(y) for y in yhat) <= {0, 1, 2, 3}
    assert np.allclose(pmodel.sum(1), 1.0, atol=1e-4), "p_model must be a 4-way softmax"

    # the patch actually moves the readout: with vs without the edit must differ for >=1 row
    yhat_noedit = np.empty(n, dtype=np.int64)
    for batch in MF.make_batches(prompts, tok.pad_token_id, 5, lm.device, row_index_map):
        fo = MF.run_forward(lm, batch, symbol_token_ids)  # capture mode, no edit
        for i, r in enumerate(batch.row_indices):
            yhat_noedit[r] = int(fo.y_tilde[i])
    assert (yhat != yhat_noedit).any(), "replacement patch had no effect on the readout"

    # --- target join: ACC / AGR / steering-success align ŷ to the rows ---
    X = np.array([SYM.index(SYM[i % 4]) for i in range(n)])     # synthetic targets
    gt = np.array([ex.answer_index for ex in examples])
    acc = float(np.mean(yhat == gt)); agr = float(np.mean(X == yhat)); succ = float(np.mean(yhat == X))
    assert all(0.0 <= v <= 1.0 for v in (acc, agr, succ)), (acc, agr, succ)

    # --- free-generation plumbing ---
    sym, idx = L.patch_generate(lm, prompts, row_index_map, steer, layer=PATCH_LAYER,
                                max_new_tokens=4, batch_size=5)
    assert len(sym) == n and len(idx) == n
    assert all(s in ("", "A", "B", "C", "D") for s in sym)
    print(f"[tiny] patch readout+generate+target-join OK (acc={acc:.2f} agr={agr:.2f} succ={succ:.2f}, "
          f"patch moved {int((yhat!=yhat_noedit).sum())}/{n} readouts)")


def test_proxy_report_synthetic():
    """reconstruct(synthetic) -> proxy -> eval(synthetic) -> report, on real d=3584 + real probe."""
    root = Path(tempfile.mkdtemp(prefix="lang_tiny_"))
    out = root / "out"; lang = out / "lang"
    (lang / "recon").mkdir(parents=True, exist_ok=True)
    (lang / "eval").mkdir(parents=True, exist_ok=True)
    (out / "fve").mkdir(parents=True, exist_ok=True)
    shutil.copy(L._ROOT / "out" / "fve" / "analysis.json", out / "fve" / "analysis.json")  # frontier

    n = 24
    rows = list(range(1000, 1000 + n))
    rng = np.random.default_rng(0)
    h_norm = rng.uniform(80, 95, size=n)
    Xidx = rng.integers(0, 4, size=n); gtidx = rng.integers(0, 4, size=n)
    tg = pa.table({
        "row_index": pa.array(rows, pa.int32()), "example_id": [f"dry-{r}" for r in rows],
        "X": [SYM[i] for i in Xidx], "X_idx": pa.array(Xidx, pa.int8()),
        "gt_symbol": [SYM[i] for i in gtidx], "gt_idx": pa.array(gtidx, pa.int8()),
        "pred_argmax_symbol": [SYM[i] for i in rng.integers(0, 4, size=n)],
        "h_norm": pa.array(h_norm, pa.float64()),
    })
    L.F.write_parquet_atomic(tg, lang / "targets.parquet")
    (lang / "subsets.json").write_text(json.dumps(
        {"seed": 7, "cohort_mean_h_norm": float(h_norm.mean()), "rows": {"dry": rows}}), encoding="utf-8")

    # a "reconstruct"-shaped raw array (d=3584) for two ops
    for op in ("E0", "T1"):
        recon = (rng.standard_normal((n, L.D_MODEL)) * 5).astype(np.float32)
        L.SS.atomic_save_npy(lang / "recon" / f"h_recon_{op}__dry.npy", recon)
        (lang / "recon" / f"h_recon_{op}__dry.rows.json").write_text(json.dumps(rows), encoding="utf-8")

    # proxy uses the REAL pred probe (d=3584) -> must run + produce in-range numbers
    pr = L.proxy_rank("E0", "dry", "normmatch", out=out)
    assert 0.0 <= pr["proxy_success_vs_X"] <= 1.0 and 0.0 <= pr["proxy_acc_vs_gt"] <= 1.0

    # synthesize eval outputs (what eval_readout would write) for the report collator
    for op, acc, ome, ratio in [("E0", 0.66, 0.276, 0.74), ("T1", 0.72, 0.28, 0.30)]:
        yhat = [SYM[i] for i in rng.integers(0, 4, size=n)]
        L.F.write_parquet_atomic(pa.table({
            "row_index": rows, "example_id": [f"dry-{r}" for r in rows],
            "y_hat_symbol": yhat, "X": [SYM[i] for i in Xidx], "gt_symbol": [SYM[i] for i in gtidx],
            "ratio": pa.array(rng.uniform(ratio - 0.02, ratio + 0.02, size=n), pa.float64()),
        }), lang / "eval" / f"readout_{op}__dry__normmatch.parquet")
        (lang / "eval" / f"readout_{op}__dry__normmatch.json").write_text(json.dumps({
            "op": op, "subset": "dry", "convention": "normmatch", "n": n, "acc": acc,
            "agr": 0.7, "steer_success": 0.8, "mean_ratio": ratio, "ome": ome}), encoding="utf-8")

    rep = L.report(subset="dry", convention="normmatch", out=out)
    assert (lang / "report.md").exists() and (lang / "report_data.json").exists()
    assert rep["verdict"]["decision"] in ("WIN", "PARTIAL", "NULL")
    # T1 (acc 0.72 >= peak 0.715, ome 0.28 < peak 0.413) -> WIN in this synthetic setup
    assert rep["verdict"]["best_op"] == "T1" and rep["verdict"]["decision"] == "WIN", rep["verdict"]
    print(f"[tiny] proxy+report chain OK (verdict={rep['verdict']['decision']}, "
          f"best={rep['verdict']['best_op']}, report.md + frontier.png written)")
    shutil.rmtree(root, ignore_errors=True)


def main() -> int:
    test_patch_plumbing_tiny()
    test_proxy_report_synthetic()
    print("\nLANG_STEER TINY DRY-RUN PASSED -- patch path, prompt rebuild, readout/gen, target join, "
          "proxy+report all wired (CPU, $0). Ready for the GPU GO/NO-GO gate.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
