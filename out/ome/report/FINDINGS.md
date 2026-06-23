# OME-GAUGE — FINDINGS (Stage 1)

**GATE S1: PASS** — proceed to Stage 2 (R2 misalignment)

- OME present (pod sweep done): **True**; DiM readout collapse-label present: **True**
- OME floor anchor: 0.2746

## H1 — does OME move with magnitude?

| arm | Spearman(OME,α) | Spearman(ratio,α) | Spearman(Maha,α) |
|---|---|---|---|
| dim:D_correct | 1.000 | 1.000 | 1.000 |
| random:D_random_0 | 1.000 | 1.000 | 1.000 |
| random:D_random_1 | 1.000 | 1.000 | 1.000 |
| random:D_random_2 | 1.000 | 1.000 | 1.000 |
| kappa:kappa | 0.991 | 1.000 | — |

## H6 — OME vs the NLA-free baselines (collapse proxy)

### additive (dim/random)
- conditions=36 collapsed=4

| detector | Spearman(·,acc-drop) | AUC(collapsed) |
|---|---|---|
| ome | 0.214 | 1.000 |
| ratio | 0.194 | 1.000 |
| mahalanobis | -0.007 | 0.836 |
| act_norm | 0.229 | 1.000 |

### KAPPA (free)
- conditions=6 collapsed=1

| detector | Spearman(·,acc-drop) | AUC(collapsed) |
|---|---|---|
| ome | 0.086 | 1.000 |
| ratio | 0.086 | 1.000 |

## Direction-specific signal (matched ratio)
mean OME(DiM) − OME(random) = **-0.0085** (dim above random: False)

