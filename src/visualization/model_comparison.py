"""
Visualization helpers for comparing Fortum consumption forecasts.

The main entry point is `plot_hourly_forecasts`, which overlays actual load
against the naive baseline (and optionally LightGBM predictions once available).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import matplotlib.pyplot as plt
import pandas as pd
from pandas.api.types import is_datetime64tz_dtype

from src.data.fortum_loader import load_fortum_training
from src.models.naive_baseline import evaluate_hourly_baseline
import sys
import os

# Add project root to PYTHONPATH
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))


def plot_hourly_forecasts(
    actual_df: pd.DataFrame,
    naive_df: pd.DataFrame,
    gbm_df: Optional[pd.DataFrame] = None,
    group_id: Optional[Union[int, str]] = None,
    start: Optional[str] = None,
    end: Optional[str] = None,
    title_suffix: str = "",
    save_path: Optional[Union[str, Path]] = None,
):
    """
    Plot actual hourly consumption vs. naive and optional LightGBM forecasts.
    """

    def _align_timestamp(value, series: pd.Series):
        if value is None:
            return None
        ts = pd.to_datetime(value)
        if ts.tzinfo is None and is_datetime64tz_dtype(series.dtype):
            ts = ts.tz_localize(series.dtype.tz)
        return ts

    def _filter(df: pd.DataFrame) -> pd.DataFrame:
        filtered = df.copy()
        if group_id is not None:
            filtered = filtered[filtered["group_id"] == group_id]
        series = filtered["measured_at"]
        start_ts = _align_timestamp(start, series)
        end_ts = _align_timestamp(end, series)
        if start_ts is not None:
            filtered = filtered[filtered["measured_at"] >= start_ts]
        if end_ts is not None:
            filtered = filtered[filtered["measured_at"] <= end_ts]
        return filtered

    actual_df = _filter(actual_df)
    naive_df = _filter(naive_df)
    if gbm_df is not None:
        gbm_df = _filter(gbm_df)

    merged = actual_df.merge(
        naive_df,
        on=["measured_at", "group_id"],
        how="inner",
        suffixes=("_actual", "_naive"),
    )
    merged = merged.rename(columns={"consumption": "actual", "prediction": "naive"})

    if gbm_df is not None:
        merged = merged.merge(
            gbm_df.rename(columns={"prediction": "gbm"}),
            on=["measured_at", "group_id"],
            how="left",
        )
    else:
        merged["gbm"] = None

    if merged.empty:
        raise ValueError("No data points after filtering for plotting.")

    merged = merged.sort_values("measured_at")

    plt.figure(figsize=(10, 5))
    plt.plot(
        merged["measured_at"],
        merged["actual"],
        label="Actual",
        color="black",
        linewidth=1.5,
    )
    plt.plot(
        merged["measured_at"],
        merged["naive"],
        label="Naive baseline",
        linestyle="--",
        linewidth=1.2,
        color="tab:blue",
    )
    if gbm_df is not None:
        plt.plot(
            merged["measured_at"],
            merged["gbm"],
            label="LightGBM",
            linestyle=":",
            linewidth=1.2,
            color="tab:orange",
        )

    gid_text = f"group_id={group_id}" if group_id is not None else "All groups"
    title = f"Hourly consumption vs forecasts ({gid_text})"
    if title_suffix:
        title += f" - {title_suffix}"
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Consumption (FWH)")
    plt.legend()
    plt.xticks(rotation=30)
    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150)
        plt.close()
    else:
        plt.show()


def quick_plot_example():
    """
    Generate a sample plot comparing actual consumption with the official naive
    baseline for a single group over the first week of 2021.
    """

    frames = load_fortum_training()
    consumption = frames["consumption"].copy()
    consumption["measured_at"] = pd.to_datetime(consumption["measured_at"], utc=True)

    hourly_result = evaluate_hourly_baseline(consumption)

    group_to_plot = hourly_result.per_group.iloc[0]["group_id"]

    actual_long = consumption.melt(
        id_vars="measured_at", var_name="group_id", value_name="consumption"
    )
    actual_long["group_id"] = actual_long["group_id"].astype(int)

    naive_long = actual_long.copy()
    naive_long["prediction"] = naive_long.groupby("group_id")["consumption"].shift(168)
    naive_long = naive_long.dropna(subset=["prediction"])

    start = "2021-01-01"
    end = "2021-01-07"

    plot_hourly_forecasts(
        actual_df=actual_long,
        naive_df=naive_long[["measured_at", "group_id", "prediction"]],
        group_id=group_to_plot,
        start=start,
        end=end,
        title_suffix="Quick example",
    )


if __name__ == "__main__":
    quick_plot_example()

if __name__ == "__main__":
    quick_plot_example()
