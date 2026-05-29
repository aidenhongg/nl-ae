"""Robustness post-mortem on the Phase-1 NULL (tiny256). $0 / CPU / local.
Answers: (1) per-datapoint off-manifold ratio distribution + medians + outliers,
(2) actual T1/T2/E1 edits + regex correctness/no-clobber, (3) the GRADED signal p(X)
(argmax is coarse — does the edit raise the target's probability at all?), and
(4) the FIRED-subset effect (E1 only fires ~32% — does it steer on rows it changes?),
plus outlier-trimmed success. If the channel were transmitting weakly we'd see it here."""
import json, re, sys
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq

L = Path("out/lang"); OPS = ["E0", "E1", "E2", "E4", "T1", "T2"]
SYM = "ABCD"; SI = {s: i for i, s in enumerate(SYM)}

# --- load per-row readouts, aligned on row_index ---
RD = {op: pq.read_table(L / "eval" / f"readout_{op}__tiny256__normmatch.parquet").to_pydict() for op in OPS}
rows0 = RD["E0"]["row_index"]
for op in OPS:
    assert RD[op]["row_index"] == rows0, f"{op} row order differs!"
n = len(rows0)
Xsym = RD["E0"]["X"]; Xidx = np.array([SI[s] for s in Xsym])
gtsym = RD["E0"]["gt_symbol"]; gtidx = np.array([SI[s] for s in gtsym])

def yhat(op): return np.array([SI[s] for s in RD[op]["y_hat_symbol"]])
def ratio(op): return np.asarray(RD[op]["ratio"], float)
def pX(op):   # model prob mass on the TARGET letter X, per row
    P = np.asarray(RD[op]["p_model"], float)  # [n,4]
    return P[np.arange(n), Xidx]
def pgt(op):
    P = np.asarray(RD[op]["p_model"], float); return P[np.arange(n), gtidx]

print(f"=== tiny256 n={n}; X balance {dict(zip(*np.unique(Xsym, return_counts=True)))} ===\n")

print("=== (1) per-datapoint OFF-MANIFOLD ratio = ||h_steer - h_orig|| / ||h_orig|| ===")
print(f"{'op':<4} {'mean':>7} {'median':>7} {'p5':>6} {'p25':>6} {'p75':>6} {'p95':>6} {'max':>7} {'std':>6}  outliers(>2.0)")
for op in OPS:
    r = ratio(op)
    print(f"{op:<4} {r.mean():>7.3f} {np.median(r):>7.3f} {np.percentile(r,5):>6.3f} "
          f"{np.percentile(r,25):>6.3f} {np.percentile(r,75):>6.3f} {np.percentile(r,95):>6.3f} "
          f"{r.max():>7.3f} {r.std():>6.3f}  {int((r>2.0).sum())}")

print("\n=== (2) success/ACC: mean vs MEDIAN-style (trimmed) + p(X) graded signal ===")
print(f"{'op':<4} {'succ(y=X)':>9} {'ACC(y=gt)':>9} {'mean p(X)':>9} {'med p(X)':>9} {'mean p(gt)':>10}")
base_pX = pX("E0")
for op in OPS:
    yh = yhat(op); succ = float((yh == Xidx).mean()); acc = float((yh == gtidx).mean())
    px = pX(op)
    print(f"{op:<4} {succ:>9.4f} {acc:>9.4f} {px.mean():>9.4f} {np.median(px):>9.4f} {pgt(op).mean():>10.4f}")

print("\n=== (2b) GRADED steering: Δp(X) vs E0 (per-row), the sensitive test for a WEAK channel ===")
print("  (if editing toward X transmits AT ALL, p(X) should rise even when argmax doesn't flip)")
for op in OPS:
    if op == "E0": continue
    d = pX(op) - base_pX
    pos = float((d > 1e-4).mean());
    print(f"  {op}: mean Δp(X)={d.mean():+.4f}  median Δp(X)={np.median(d):+.4f}  "
          f"frac rows p(X) increased={pos:.3f}  (t-ish: mean/se={d.mean()/(d.std()/np.sqrt(n)+1e-9):+.2f})")

print("\n=== (3) EDITS: actual text + regex correctness (E1) / template form (T1,T2) ===")
orig = pq.read_table("out/nl/orig.parquet", columns=["row_index","nl_text","know_argmax_symbol"]).to_pydict()
ztext = {int(orig["row_index"][i]): orig["nl_text"][i] for i in range(len(orig["row_index"]))}
def edits(op):
    p = L / "edits" / f"{op}__tiny256.parquet"
    if not p.exists(): return None
    return pq.read_table(p).to_pydict()

# E1 regex audit: fire rate, did the flip land on X, any option-letter clobber?
e1 = edits("E1")
if e1:
    fired = 0; correct_flip = 0; clobber = 0; examples = []
    ISO = re.compile(r"(?<![A-Za-z])([A-D])(?![A-Za-z])")
    RUN = re.compile(r"(?:[\"('‘’]?[A-D][)\"'‘’]?\s*(?:,|or|/|&|nor)\s*)+[\"('‘’]?[A-D]", re.I)
    for i in range(len(e1["row_index"])):
        ri = int(e1["row_index"][i]); ze = e1["z_edit"][i]; zo = ztext[ri]; X = e1["X"][i]
        if ze != zo:
            fired += 1
            # positions that changed
            diffs = [j for j in range(min(len(zo),len(ze))) if zo[j] != ze[j]]
            # every changed char in z_edit should now be X
            if diffs and all(ze[j] == X for j in diffs): correct_flip += 1
            # clobber: did any change land inside an enumeration run of the ORIGINAL?
            runs = [(m.start(), m.end()) for m in RUN.finditer(zo)]
            if any(any(a <= j < b for a,b in runs) for j in diffs): clobber += 1
            if len(examples) < 4:
                j = diffs[0]
                examples.append((X, zo[max(0,j-45):j+8], ze[max(0,j-45):j+8]))
    print(f"E1: fired on {fired}/{len(e1['row_index'])} rows ({fired/len(e1['row_index']):.1%}); "
          f"flip-lands-on-X correct={correct_flip}/{fired}; OPTION-LETTER CLOBBER={clobber} (must be 0)")
    for X, b, a in examples:
        print(f"   X={X} | before …{b!r}  ->  after …{a!r}")

for op in ["T1","T2"]:
    e = edits(op)
    if e:
        print(f"{op} edit (row 0, X={e['X'][0]}): {e['z_edit'][0]!r}")
    else:  # reproduce deterministically (templates, no regex)
        X0 = orig["know_argmax_symbol"][0]
        t = (f"The correct answer is ({X0})." if op=="T1" else
             f"Structured multiple-choice answer. The correct answer is ({X0}). The final token completes the answer with ({X0}).")
        print(f"{op} edit (reproduced, X={X0}): {t!r}")

print("\n=== (4) FIRED-subset steering (does the edit steer on rows it actually changes?) ===")
e0_yh = yhat("E0")
for op in ["E1","E4","E2","T1","T2"]:
    e = edits(op)
    if e:
        fired_ri = {int(e["row_index"][i]) for i in range(len(e["row_index"])) if e["z_edit"][i] != ztext[int(e["row_index"][i])]}
    elif op in ("E2","T1","T2"):
        fired_ri = set(int(r) for r in rows0)  # E2 appends always; T1/T2 always replace
    mask = np.array([int(r) in fired_ri for r in rows0])
    if mask.sum() == 0:
        print(f"  {op}: fired on 0 rows"); continue
    yh = yhat(op)
    succ_op = (yh[mask] == Xidx[mask]).mean()
    succ_e0 = (e0_yh[mask] == Xidx[mask]).mean()  # E0 on the SAME fired rows
    dpx = (pX(op)[mask] - base_pX[mask]).mean()
    print(f"  {op}: fired {int(mask.sum())} rows | success on fired: op={succ_op:.3f} vs E0-same-rows={succ_e0:.3f} "
          f"(lift {succ_op-succ_e0:+.3f}) | Δp(X) on fired={dpx:+.4f}")

print("\n=== (5) outlier-trimmed success (drop top-10% ratio = worst reconstructions) ===")
for op in ["E1","E4","T1","T2"]:
    r = ratio(op); yh = yhat(op); keep = r <= np.percentile(r, 90)
    print(f"  {op}: success all={ (yh==Xidx).mean():.3f}  |  low-90%-ratio only={ (yh[keep]==Xidx[keep]).mean():.3f} (n={keep.sum()})")

# save summary
out = {"n": n,
       "ratio": {op: {"mean": float(ratio(op).mean()), "median": float(np.median(ratio(op))),
                      "p95": float(np.percentile(ratio(op),95)), "max": float(ratio(op).max())} for op in OPS},
       "success": {op: float((yhat(op)==Xidx).mean()) for op in OPS},
       "mean_pX": {op: float(pX(op).mean()) for op in OPS},
       "delta_pX_vs_E0": {op: float((pX(op)-base_pX).mean()) for op in OPS if op!="E0"}}
Path(L/"null_diagnostics.json").write_text(json.dumps(out, indent=2))
print("\n-> out/lang/null_diagnostics.json")
