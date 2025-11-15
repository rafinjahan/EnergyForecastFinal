"""
Naive baseline evaluator.

Implements a simple "yesterday equals today" predictor (lag=24h) to establish
a benchmark MAE/RMSE on the Fortum consumption data.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from src.data.fortum_loader import load_fortum_training


@dataclass
class BaselineResult:
    per_group: pd.DataFrame
    overall_mae: float
    overall_rmse: float
    evaluated_rows: int
    lag_value: int
    lag_unit: str
    frequency: Literal["hourly", "monthly"]


def evaluate_hourly_baseline(
    consumption: pd.DataFrame, lag_hours: int = 168
) -> BaselineResult:
    """
    Compute MAE/RMSE for a naive lagged predictor.

    Args:
        consumption: Fortum consumption table (wide format).
        lag_hours: Number of hours to look back for the prediction.
    """

    df = consumption.copy()
    df["measured_at"] = pd.to_datetime(df["measured_at"], utc=True)
    df = df.sort_values("measured_at").reset_index(drop=True)

    group_cols = [col for col in df.columns if col != "measured_at"]
    per_group_records = []
    abs_errors_all = []
    sq_errors_all = []
    total_rows = 0

    for col in group_cols:
        series = pd.to_numeric(df[col], errors="coerce")
        predictions = series.shift(lag_hours)
        mask = series.notna() & predictions.notna()
        if not mask.any():
            continue
        errors = series[mask] - predictions[mask]
        abs_err = errors.abs()
        sq_err = errors.pow(2)

        per_group_records.append(
            {
                "group_id": int(col),
                "mae": abs_err.mean(),
                "rmse": np.sqrt(sq_err.mean()),
                "n_points": int(mask.sum()),
            }
        )
        abs_errors_all.append(abs_err)
        sq_errors_all.append(sq_err)
        total_rows += mask.sum()

    if total_rows == 0:
        raise ValueError(
            "Baseline evaluation failed: no overlapping observations for lag."
        )

    overall_mae = pd.concat(abs_errors_all).mean()
    overall_rmse = np.sqrt(pd.concat(sq_errors_all).mean())

    per_group = pd.DataFrame(per_group_records).sort_values("rmse").reset_index(
        drop=True
    )
    return BaselineResult(
        per_group=per_group,
        overall_mae=float(overall_mae),
        overall_rmse=float(overall_rmse),
        evaluated_rows=int(total_rows),
        lag_value=lag_hours,
        lag_unit="hours",
        frequency="hourly",
    )


def evaluate_monthly_baseline(
    consumption: pd.DataFrame, lag_months: int = 12
) -> BaselineResult:
    """
    Compute MAE/RMSE for a naive monthly predictor using previous year totals.
    """

    df = consumption.copy()
    df["measured_at"] = pd.to_datetime(df["measured_at"], utc=True)
    group_cols = [col for col in df.columns if col != "measured_at"]

    long_df = df.melt(id_vars="measured_at", var_name="group_id", value_name="load_mwh")
    long_df["group_id"] = long_df["group_id"].astype(int)
    long_df = long_df.dropna(subset=["load_mwh"])
    long_df["month"] = (
        long_df["measured_at"].dt.tz_convert(None).dt.to_period("M")
    )

    monthly_totals = (
        long_df.groupby(["group_id", "month"], as_index=False)["load_mwh"].sum()
    )
    monthly_totals = monthly_totals.sort_values(["group_id", "month"])
    monthly_totals["baseline"] = monthly_totals.groupby("group_id")["load_mwh"].shift(
        lag_months
    )

    valid = monthly_totals.dropna(subset=["baseline"]).copy()
    if valid.empty:
        raise ValueError("Monthly baseline evaluation failed: insufficient history.")

    valid["error"] = valid["load_mwh"] - valid["baseline"]
    valid["abs_error"] = valid["error"].abs()
    valid["sq_error"] = valid["error"].pow(2)

    per_group = (
        valid.groupby("group_id")
        .agg(
            mae=("abs_error", "mean"),
            rmse=("sq_error", lambda s: np.sqrt(s.mean())),
            n_points=("error", "size"),
        )
        .reset_index()
        .sort_values("rmse")
        .reset_index(drop=True)
    )

    overall_mae = valid["abs_error"].mean()
    overall_rmse = np.sqrt(valid["sq_error"].mean())

    return BaselineResult(
        per_group=per_group,
        overall_mae=float(overall_mae),
        overall_rmse=float(overall_rmse),
        evaluated_rows=int(valid.shape[0]),
        lag_value=lag_months,
        lag_unit="months",
        frequency="monthly",
    )


def _write_outputs(
    result: BaselineResult,
    per_group_output: Path,
    summary_path: Path,
    group_labels: pd.DataFrame | None = None,
) -> None:
    per_group_output.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    per_group_df = result.per_group.copy()
    if group_labels is not None:
        labels = group_labels.copy()
        labels["group_id"] = labels["group_id"].astype(int)
        per_group_df = per_group_df.merge(labels, on="group_id", how="left")

    per_group_df.to_csv(per_group_output, index=False)

    summary_lines = [
        f"Naive {result.frequency} baseline (lag={result.lag_value} {result.lag_unit})",
        f"Overall MAE:  {result.overall_mae:.4f}",
        f"Overall RMSE: {result.overall_rmse:.4f}",
        f"Evaluated points: {result.evaluated_rows}",
        f"Per-group metrics saved to: {per_group_output}",
    ]
    summary_path.write_text("\n".join(summary_lines), encoding="utf-8")


def _print_summary(result: BaselineResult, per_group_with_labels: pd.DataFrame):
    print(
        f"[Naive {result.frequency} lag={result.lag_value} {result.lag_unit}] "
        f"Overall MAE={result.overall_mae:.4f}, "
        f"RMSE={result.overall_rmse:.4f} (n={result.evaluated_rows})"
    )
    print("\nTop 5 groups (lowest RMSE):")
    print(per_group_with_labels.head(5).to_string(index=False))
    print("\nWorst 5 groups (highest RMSE):")
    print(per_group_with_labels.tail(5).to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate Fortum challenge naive baselines."
    )
    parser.add_argument(
        "--xlsx",
        type=str,
        default=None,
        help="Optional path to 20251111_JUNCTION_training.xlsx",
    )
    parser.add_argument(
        "--lag-hours",
        type=int,
        default=168,
        help="Lag (in hours) for the hourly baseline (default 7 days).",
    )
    parser.add_argument(
        "--lag-months",
        type=int,
        default=12,
        help="Lag (in months) for the monthly baseline (default previous year).",
    )
    parser.add_argument(
        "--mode",
        choices=["hourly", "monthly", "both"],
        default="both",
        help="Which baseline(s) to evaluate.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="reports",
        help="Directory to save baseline reports.",
    )
    args = parser.parse_args()

    frames = load_fortum_training(args.xlsx)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.mode in ("hourly", "both"):
        hourly_result = evaluate_hourly_baseline(
            frames["consumption"], lag_hours=args.lag_hours
        )
        hourly_per_group = hourly_result.per_group.merge(
            frames["groups"], on="group_id", how="left"
        )
        _write_outputs(
            hourly_result,
            per_group_output=output_dir / "hourly_baseline_per_group.csv",
            summary_path=output_dir / "hourly_baseline_summary.txt",
            group_labels=frames["groups"],
        )
        _print_summary(hourly_result, hourly_per_group)

    if args.mode in ("monthly", "both"):
        monthly_result = evaluate_monthly_baseline(
            frames["consumption"], lag_months=args.lag_months
        )
        monthly_per_group = monthly_result.per_group.merge(
            frames["groups"], on="group_id", how="left"
        )
        _write_outputs(
            monthly_result,
            per_group_output=output_dir / "monthly_baseline_per_group.csv",
            summary_path=output_dir / "monthly_baseline_summary.txt",
            group_labels=frames["groups"],
        )
        _print_summary(monthly_result, monthly_per_group)


if __name__ == "__main__":
    main()
