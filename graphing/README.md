# graphing/

Figures from the concluded NLA-final experiments. Read-only over `out/` (the frozen
experiment data); nothing here writes experiment outputs. No pandas (pyarrow + numpy).

```bash
python -m graphing.make_all            # regenerate all figures (validates anchors first)
python -m graphing.graph3_kappa_ome_alpha   # or run any one individually
python -m graphing.data                # just check the data still reproduces the anchors
```

Figures land in `graphing/figures/`. Shared code: `data.py` (loaders + the only place
that knows the on-disk schema/paths) and `style.py` (palette, savefig, accuracy colorbar).

## The figures (narrative order)

**The premise — the model knows more than it says**
1. `graph1_probe_accuracy` — knowledge probe (0.854 vs truth) vs. prediction probe
   (0.912 vs the model) vs. first-token accuracy (0.660): a +0.194 latent-knowledge gap.
2. `graph2_confidence_wrong_firsttoken` — on wrong first-tokens, model confidence vs.
   knowledge-probe confidence: the model is often confidently wrong while the knowledge
   probe is confident *and correct* (74% of those rows).

**KAPPA steering — accuracy bought with off-manifold error**
3. `graph3_kappa_ome_alpha` — off-manifold error OME=1−cos vs. α, colored by first-token
   accuracy. OME rises monotonically; accuracy is an inverted-U.
5. `graph5_kappa_accuracy_invertedU` — the accuracy inverted-U (peak 0.715 @ α=10,
   collapse 0.600 @ α=30) against the monotone OME (twin axis).
8. `graph8_steering_divergence` — the *mechanism*: steering collapses the prediction↔
   knowledge KL at low α (aligns the readout with what the model knows) then the
   off-manifold blow-up re-opens it.

**NLA language-steering — the NULL**
4. `graph4_nla_methods_offmanifold` — off-manifold error (ratio) per operator trial,
   colored by accuracy; none reaches KAPPA's peak (templates push further off-manifold
   and score worse).
6. `graph6_pareto_frontier` — accuracy vs. off-manifold error: the KAPPA frontier with
   every NLA method below it. The WIN region (≥KAPPA accuracy at lower off-manifold cost)
   is empty.
7. `graph7_method_letter_bias` — why: templates collapse the prediction to a fixed letter
   (T1→B 36%, T2→D 41%) instead of steering toward the per-row target.

All accuracy figures are on the example_level test split (n=2615); KAPPA round-trip on
the 1024-row subset; NLA methods on tiny256 (n=256, normmatch). `data.validate_anchors`
guards against data drift (know 0.854 / pred 0.912 / base 0.660 / AGR 0.649).
