"""Characterize the verbalizations where the model's first-token answer diverged from the
knowledge-probe label X (= know_argmax). Test split, 2615 rows, $0/local. Asks: who's right
on those rows (probe vs model)? are they low-confidence? does the verbalization even mention X?
what's the qualitative pattern?"""
import re
import numpy as np
import pyarrow.parquet as pq

t = pq.read_table("out/nl/orig.parquet").to_pydict()
test = [i for i in range(len(t["split_el"])) if t["split_el"][i] == "test"]
def c(k): return np.array([t[k][i] for i in test], dtype=object)
X = c("know_argmax_symbol"); M = c("model_symbol"); P = c("pred_argmax_symbol"); G = c("gt_symbol")
conf = c("model_confidence").astype(float); kconf = c("know_confidence").astype(float)
kl = c("kl_pred_know").astype(float); ntok = c("n_tokens").astype(float)
text = [t["nl_text"][i] for i in test]; N = len(test)

div = M != X; agr = ~div
kr = (X == G); mr = (M == G)              # knowledge right / model right (vs gt)
print(f"test rows N={N} | diverge(model!=know)={div.sum()} ({div.mean():.1%}) | agree={agr.sum()}\n")

print("=== Who is right on the divergence rows? (the crux) ===")
print(f"overall: know_acc(X==gt)={kr.mean():.3f}  model_acc(M==gt)={mr.mean():.3f}")
print(f"on DIVERGE rows: know(X)==gt={kr[div].mean():.3f}  model==gt={mr[div].mean():.3f}  neither={( ~kr & ~mr)[div].mean():.3f}")
print(f"on AGREE rows:   ==gt={kr[agr].mean():.3f}")
gain = int((div & kr & ~mr).sum()); loss = int((div & ~kr & mr).sum()); both_wrong = int((div & ~kr & ~mr).sum())
print(f"\ndiverge breakdown: probe-right/model-wrong={gain} | model-right/probe-wrong={loss} | both-wrong={both_wrong}")
print(f"CEILING of PERFECT steer-to-X: net {gain}-{loss} = {gain-loss:+d} rows ({(gain-loss)/N:+.3f} acc)")
print(f"  -> model_acc {mr.mean():.3f} would become {(int(mr.sum())+gain-loss)/N:.3f} (and know_acc ceiling is {kr.mean():.3f})\n")

print("=== Confidence / divergence-magnitude patterns (agree vs diverge) ===")
for nm, a in [("model_confidence", conf), ("know_confidence", kconf), ("kl_pred_know", kl), ("n_tokens", ntok)]:
    print(f"  {nm:<16}: agree={a[agr].mean():.3f}  diverge={a[div].mean():.3f}")
hi = conf > 0.9
print(f"  model_confidence>0.9 (confidently committed): agree={hi[agr].mean():.3f}  diverge={hi[div].mean():.3f}")

def has(txt, L): return bool(re.search(r"(?<![A-Za-z])" + re.escape(L) + r"(?![A-Za-z])", txt))
def ndist(txt): return len(set(re.findall(r"(?<![A-Za-z])([A-D])(?![A-Za-z])", txt)))
mentX = np.array([has(text[i], X[i]) for i in range(N)])
mentM = np.array([has(text[i], M[i]) for i in range(N)])
nd = np.array([ndist(text[i]) for i in range(N)])
print("\n=== Does the verbalization mention the relevant letters? ===")
print(f"  mentions know-letter X : agree={mentX[agr].mean():.3f}  diverge={mentX[div].mean():.3f}")
print(f"  mentions model-letter  : agree={mentM[agr].mean():.3f}  diverge={mentM[div].mean():.3f}")
print(f"  # distinct A-D letters : agree={nd[agr].mean():.2f}  diverge={nd[div].mean():.2f}")
print(f"  on diverge rows: mentions X but NOT model-letter={ (mentX & ~mentM)[div].mean():.3f} | "
      f"mentions model-letter but NOT X={ (~mentX & mentM)[div].mean():.3f} | mentions both={ (mentX & mentM)[div].mean():.3f} | neither={ (~mentX & ~mentM)[div].mean():.3f}")

print("\n" + "="*90)
print("=== EXAMPLE verbalizations on DIVERGE rows, by subtype ===")
idx = np.arange(N)
for label, mask in [("PROBE-RIGHT / model-wrong (the KAPPA target)", div & kr & ~mr),
                    ("MODEL-RIGHT / probe-wrong (steering to X would HURT)", div & mr),
                    ("BOTH WRONG", div & ~kr & ~mr)]:
    sel = idx[mask][:3]
    print(f"\n--- {label}  (n={int(mask.sum())}) ---")
    for i in sel:
        print(f"  X(know)={X[i]} model={M[i]} gt={G[i]} model_conf={conf[i]:.3f} know_conf={kconf[i]:.3f}")
        print(f"    verbalization: {text[i][:340].strip()!r}")
