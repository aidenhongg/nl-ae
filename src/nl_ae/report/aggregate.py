"""Post-hoc aggregation.

Reads rows via ``ResultsReader``, projects them into a flat ``pandas.DataFrame``,
and computes the five MVP analyses required by C04 D4.7.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from nl_ae.schema.reader import ResultsReader

if TYPE_CHECKING:  # pragma: no cover
    import pandas as pd

LOG = logging.getLogger(__name__)


@dataclass(frozen=True)
class AggregateBundle:
    rows: "pd.DataFrame"
    top1_disagreement: "pd.DataFrame"
    per_letter_confusion: "pd.DataFrame"
    per_position_bias: "pd.DataFrame"
    per_subject_mmlu: "pd.DataFrame"
    calibration: "pd.DataFrame | None"
    permutation_coverage: "pd.DataFrame"


def _import_pandas() -> "pd":  # type: ignore[name-defined]
    try:
        import pandas as pd  # noqa: PLC0415

        return pd
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("pandas required for aggregation; install nl-ae[report]") from exc


def _wilson_ci(p: float, n: int, *, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (math.nan, math.nan)
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def aggregate_run(
    run_dir: Path,
    *,
    include_partial: bool = False,
    write_parquet: bool = True,
    calibration_buckets: int = 10,
) -> AggregateBundle:
    pd = _import_pandas()
    reader = ResultsReader(run_dir, strict=not include_partial)
    rows = list(reader.iter_rows())
    if not rows:
        LOG.warning("aggregate_run: no rows found in %s", run_dir)
        return AggregateBundle(
            rows=pd.DataFrame(),
            top1_disagreement=pd.DataFrame(),
            per_letter_confusion=pd.DataFrame(),
            per_position_bias=pd.DataFrame(),
            per_subject_mmlu=pd.DataFrame(),
            calibration=None,
            permutation_coverage=pd.DataFrame(),
        )
    df = pd.DataFrame(
        [
            {
                "run_id": r.run_id,
                "item_id": r.item_id,
                "dataset_name": r.dataset_name,
                "dataset_split": r.dataset_split,
                "subject": r.subject,
                "template_id": r.template_id,
                "permutation_id": r.permutation_id,
                "gold_letter": r.gold_letter,
                "first_token_letter": r.first_token_letter,
                "free_text_letter": r.free_text_letter,
                "agreement_flag": r.agreement_flag,
                "n_options": r.n_options,
                "extractor_match_rule": r.extractor_match_rule,
                "first_token_scoring_math": r.first_token_scoring_math,
                "total_letter_mass": r.total_letter_mass,
                "first_token_prob": _argmax_prob(r.letter_softmax, r.first_token_letter),
            }
            for r in rows
        ]
    )

    top1 = compute_top1_disagreement(df)
    confusion = compute_per_letter_confusion(df)
    position = compute_per_position_bias(df)
    subject = compute_per_subject_mmlu(df)
    cal = compute_calibration(df, buckets=calibration_buckets)
    coverage = compute_permutation_coverage(df)

    if write_parquet:
        out = run_dir / "aggregates"
        out.mkdir(parents=True, exist_ok=True)
        _safe_to_parquet(df, out / "rows.parquet")
        _safe_to_parquet(top1, out / "top1_disagreement.parquet")
        _safe_to_parquet(confusion, out / "per_letter_confusion.parquet")
        _safe_to_parquet(position, out / "per_position_bias.parquet")
        _safe_to_parquet(subject, out / "per_subject_mmlu.parquet")
        if cal is not None:
            _safe_to_parquet(cal, out / "calibration.parquet")
        _safe_to_parquet(coverage, out / "permutation_coverage.parquet")

    return AggregateBundle(
        rows=df,
        top1_disagreement=top1,
        per_letter_confusion=confusion,
        per_position_bias=position,
        per_subject_mmlu=subject,
        calibration=cal,
        permutation_coverage=coverage,
    )


def _safe_to_parquet(df: "pd.DataFrame", path: Path) -> None:
    if df is None or len(df) == 0:
        return
    try:
        df.to_parquet(path, index=False)
    except (ImportError, ValueError) as exc:  # pragma: no cover
        LOG.warning("could not write %s: %r", path, exc)


def _argmax_prob(softmax: list, argmax_letter: str | None) -> float | None:
    if argmax_letter is None:
        return None
    for entry in softmax:
        if entry.letter == argmax_letter and entry.prob_valid:
            return entry.prob
    return None


def compute_top1_disagreement(rows: "pd.DataFrame") -> "pd.DataFrame":
    pd = _import_pandas()
    if len(rows) == 0:
        return pd.DataFrame()
    mask = rows["first_token_letter"].notna() & rows["free_text_letter"].notna()
    valid = rows[mask].copy()
    if len(valid) == 0:
        return pd.DataFrame(
            columns=["dataset_name", "template_id", "n", "disagreement", "ci_lo", "ci_hi"]
        )
    grouped = valid.groupby(["dataset_name", "template_id"], dropna=False)
    records = []
    for (ds, tid), group in grouped:
        n = len(group)
        disagreement = 1.0 - float(group["agreement_flag"].mean())
        lo, hi = _wilson_ci(disagreement, n)
        records.append(
            {
                "dataset_name": ds,
                "template_id": tid,
                "n": n,
                "disagreement": disagreement,
                "ci_lo": lo,
                "ci_hi": hi,
            }
        )
    return pd.DataFrame.from_records(records)


def compute_per_letter_confusion(rows: "pd.DataFrame") -> "pd.DataFrame":
    pd = _import_pandas()
    if len(rows) == 0:
        return pd.DataFrame()
    mask = rows["first_token_letter"].notna() & rows["free_text_letter"].notna()
    valid = rows[mask]
    if len(valid) == 0:
        return pd.DataFrame()
    return (
        valid.groupby(["dataset_name", "template_id", "first_token_letter", "free_text_letter"])
        .size()
        .reset_index(name="count")
    )


def compute_per_position_bias(rows: "pd.DataFrame") -> "pd.DataFrame":
    pd = _import_pandas()
    if len(rows) == 0:
        return pd.DataFrame()
    valid = rows[rows["first_token_letter"].notna()].copy()
    if len(valid) == 0:
        return pd.DataFrame()
    counts = (
        valid.groupby(["template_id", "permutation_id", "first_token_letter"])
        .size()
        .reset_index(name="count")
    )
    totals = (
        valid.groupby(["template_id", "permutation_id"])
        .size()
        .reset_index(name="total")
    )
    merged = counts.merge(totals, on=["template_id", "permutation_id"], how="left")
    merged["share"] = merged["count"] / merged["total"]
    return merged


def compute_per_subject_mmlu(rows: "pd.DataFrame") -> "pd.DataFrame":
    pd = _import_pandas()
    if len(rows) == 0:
        return pd.DataFrame()
    mmlu = rows[rows["dataset_name"] == "mmlu"].copy()
    if len(mmlu) == 0:
        return pd.DataFrame()
    mmlu["first_correct"] = (mmlu["first_token_letter"] == mmlu["gold_letter"]).astype(float)
    mmlu["free_correct"] = (mmlu["free_text_letter"] == mmlu["gold_letter"]).astype(float)
    grouped = mmlu.groupby("subject", dropna=False)
    return grouped.agg(
        n_items=("item_id", "nunique"),
        n_visits=("item_id", "size"),
        accuracy_first_token=("first_correct", "mean"),
        accuracy_free_text=("free_correct", "mean"),
        disagreement_rate=("agreement_flag", lambda s: 1.0 - float(s.dropna().mean()) if s.dropna().size else math.nan),
    ).reset_index()


def compute_calibration(
    rows: "pd.DataFrame", *, buckets: int = 10
) -> "pd.DataFrame | None":
    pd = _import_pandas()
    if len(rows) == 0:
        return None
    mmlu = rows[(rows["dataset_name"] == "mmlu") & rows["first_token_prob"].notna()].copy()
    if len(mmlu) == 0:
        return None
    mmlu["bucket"] = (mmlu["first_token_prob"].clip(0.0, 0.999999) * buckets).astype(int)
    mmlu["correct"] = (mmlu["first_token_letter"] == mmlu["gold_letter"]).astype(float)
    grouped = mmlu.groupby("bucket")
    out = grouped.agg(
        n=("first_token_prob", "size"),
        mean_prob=("first_token_prob", "mean"),
        empirical_accuracy=("correct", "mean"),
    ).reset_index()
    out["bucket_lo"] = out["bucket"] / buckets
    out["bucket_hi"] = (out["bucket"] + 1) / buckets
    return out


def compute_permutation_coverage(rows: "pd.DataFrame") -> "pd.DataFrame":
    pd = _import_pandas()
    if len(rows) == 0:
        return pd.DataFrame()
    grouped = rows.groupby(["template_id", "item_id"])
    coverage = grouped["permutation_id"].agg(lambda s: sorted(set(s))).reset_index()
    coverage["distinct_permutations"] = coverage["permutation_id"].apply(len)
    return coverage.drop(columns=["permutation_id"])


__all__ = [
    "AggregateBundle",
    "aggregate_run",
    "compute_calibration",
    "compute_per_letter_confusion",
    "compute_per_position_bias",
    "compute_per_subject_mmlu",
    "compute_permutation_coverage",
    "compute_top1_disagreement",
]
