"""Did the regex edit (E1) snap the RECONSTRUCTED activation toward the KNOWLEDGE probe
where it fired? Read each ĥ_steer (normmatch) with the L20 know/pred probes directly and
compare to Qwen's readout. Triangulates: know-probe(ĥ) vs pred-probe(ĥ) vs Qwen ŷ, on the
E1-fired rows. $0 / CPU / local."""
import sys, json
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
sys.path.insert(0, ".")
from src import lang_steer as L
from src import features as F

OUT = Path("out"); SYM = "ABCD"
Wk, bk = L._load_probe("know"); Wp, bp = L._load_probe("pred")

def parg(h, W, b):  # probe argmax over the 4 symbols
    return F.posteriors(np.asarray(h, np.float64), W, b).argmax(1)

hE0, rows = L._load_recon(OUT, "E0", "tiny256", "normmatch")
tg = L._targets_for_rows(OUT, rows); Xidx = np.asarray(tg["X_idx"])
h_orig = L._h_orig_for_rows(rows).astype(np.float64); n = len(rows)

# sanity: X == argmax(know-probe(h_orig))?  (X is DEFINED as the know label, so expect ~1.0)
print(f"[sanity] know-probe(h_orig)==X : {(parg(h_orig,Wk,bk)==Xidx).mean():.3f}  (X is the know label by construction)")
print(f"[sanity] pred-probe(h_orig)==X : {(parg(h_orig,Wp,bp)==Xidx).mean():.3f}\n")

# E1 fired mask (z_edit != z_orig), aligned to `rows`
e1 = pq.read_table("out/lang/edits/E1__tiny256.parquet").to_pydict()
orig = pq.read_table("out/nl/orig.parquet", columns=["row_index","nl_text"]).to_pydict()
zt = {int(orig["row_index"][i]): orig["nl_text"][i] for i in range(len(orig["row_index"]))}
assert [int(r) for r in e1["row_index"]] == [int(r) for r in rows], "edit/recon row order mismatch"
fired = np.array([e1["z_edit"][i] != zt[int(e1["row_index"][i])] for i in range(len(rows))])
print(f"E1 fired on {fired.sum()}/{n} rows\n")

# Qwen readout per op (from eval parquets)
def qwen_yhat(op):
    t = pq.read_table(f"out/lang/eval/readout_{op}__tiny256__normmatch.parquet").to_pydict()
    return np.array([SYM.index(s) for s in t["y_hat_symbol"]])

print(f"{'op':<4} | {'know(h)==X':>10} {'pred(h)==X':>10} {'Qwen==X':>8}  ||  on E1-FIRED rows: {'know==X':>8} {'pred==X':>8} {'Qwen==X':>8}")
diag = {}
for op in ["E0","E1","E2","E4","T1","T2"]:
    h,_ = L._load_recon(OUT, op, "tiny256", "normmatch")
    kp, pp = parg(h,Wk,bk), parg(h,Wp,bp); qy = qwen_yhat(op)
    row = [f"{(kp==Xidx).mean():>10.3f}", f"{(pp==Xidx).mean():>10.3f}", f"{(qy==Xidx).mean():>8.3f}",
           f"{(kp[fired]==Xidx[fired]).mean():>8.3f}", f"{(pp[fired]==Xidx[fired]).mean():>8.3f}",
           f"{(qy[fired]==Xidx[fired]).mean():>8.3f}"]
    print(f"{op:<4} | {row[0]} {row[1]} {row[2]}  ||  {' '*18}{row[3]} {row[4]} {row[5]}")
    diag[op] = {"know_X_all": float((kp==Xidx).mean()), "pred_X_all": float((pp==Xidx).mean()),
                "qwen_X_all": float((qy==Xidx).mean()), "know_X_fired": float((kp[fired]==Xidx[fired]).mean()),
                "pred_X_fired": float((pp[fired]==Xidx[fired]).mean()), "qwen_X_fired": float((qy[fired]==Xidx[fired]).mean())}

# The key comparison: E1 vs E0 on FIRED rows — did the edit MOVE each readout toward X?
print("\n=== E1 − E0 on FIRED rows (the 'snap' the edit produced) ===")
for k,lab in [("know_X_fired","know-probe(ĥ)"),("pred_X_fired","pred-probe(ĥ)"),("qwen_X_fired","Qwen readout")]:
    print(f"  {lab:<16}: E0={diag['E0'][k]:.3f}  E1={diag['E1'][k]:.3f}  lift={diag['E1'][k]-diag['E0'][k]:+.3f}")
Path("out/lang/probe_snap.json").write_text(json.dumps(diag, indent=2))
print("\n-> out/lang/probe_snap.json")
