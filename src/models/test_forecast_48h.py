"""48-hour forecasting script enriched with FMI weather observations.

This module prepares a long-form consumption table joined with every
available weather station (see :mod:`src.data.weather_enrichment`) and uses a
pre-trained LightGBM model to project consumption for the first 48 hours
after the training window (October 1-2, 2024). The script also exposes
helpers that other utilities use (e.g. ``scripts/compute_mape_48h.py``).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, Sequence

import holidays
import joblib
import numpy as np
import pandas as pd

try:  # LightGBM is required when loading the trained model via joblib.
    import lightgbm as lgb
except ImportError:  # pragma: no cover - lightgbm may be unavailable in tests.
    lgb = None

# Ensure ``src`` is importable when executed as a script.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data import weather_enrichment
from src.data.fortum_loader import load_fortum_training
from src.data.weather_enrichment import enrich_consumption_with_all_weather

MODEL_DIR = PROJECT_ROOT / "models"
LIGHTGBM_MODEL_PATH = MODEL_DIR / "lgb_weather_model.pkl"
FORECAST_START = pd.Timestamp("2024-10-01 00:00:00", tz="UTC")
FORECAST_HOURS = 48
VALIDATION_DAYS = 7
WEATHER_COLS = [
    "temperature_c",
    "wind_speed_ms",
    "precip_mm",
    "humidity_pct",
]
LAG_FEATURES = (1, 24, 168)
FEATURE_COLUMNS = [
    "hour",
    "weekday",
    "is_weekend",
    "month",
    "weekofyear",
    "year",
    "is_holiday",
    "customer_type",
    "contract_type",
    "consumption_level",
    "temperature_c",
    "wind_speed_ms",
    "precip_mm",
    "humidity_pct",
    "price_eur_per_mwh",
    "price_lag_24h",
    "load_lag_1h",
    "load_lag_24h",
    "load_lag_168h",
]


def _parse_group_label(label: str) -> tuple[str, str, str]:
    parts = [part.strip() for part in label.split("|")]
    if len(parts) >= 3:
        return parts[-3], parts[-2], parts[-1]
    return ("Unknown", "Unknown", "Unknown")


def _build_group_attribute_table(
    groups: pd.DataFrame,
) -> tuple[pd.DataFrame, Dict[str, Sequence[str]]]:
    attrs = groups[["group_id", "group_label"]].copy()
    parsed = attrs["group_label"].apply(_parse_group_label)
    attrs["customer_type"] = parsed.apply(lambda values: values[0])
    attrs["contract_type"] = parsed.apply(lambda values: values[1])
    attrs["consumption_level"] = parsed.apply(lambda values: values[2])
    attrs = attrs.drop(columns=["group_label"])
    category_levels = {
        "group_id": sorted(attrs["group_id"].astype(int).unique().tolist()),
        "customer_type": sorted(attrs["customer_type"].dropna().unique().tolist()),
        "contract_type": sorted(attrs["contract_type"].dropna().unique().tolist()),
        "consumption_level": sorted(
            attrs["consumption_level"].dropna().unique().tolist()
        ),
    }
    return attrs, category_levels


def load_prepared_long_frame() -> tuple[pd.DataFrame, Dict[str, Any]]:
    """Load Fortum tables and enrich them with every configured weather station."""

    frames = load_fortum_training()
    consumption = frames["consumption"].copy()
    groups = frames["groups"].copy()
    prices = frames["prices"].copy()

    consumption["measured_at"] = pd.to_datetime(
        consumption["measured_at"], utc=True
    )
    prices["measured_at"] = pd.to_datetime(prices["measured_at"], utc=True)

    enriched = enrich_consumption_with_all_weather(consumption, groups)

    weather_history = (
        enriched[["measured_at", "station_slug", *WEATHER_COLS]]
        .drop_duplicates(subset=["measured_at", "station_slug"])
        .reset_index(drop=True)
    )

    group_station_map = (
        enriched.groupby("group_id")["station_slug"]
        .apply(lambda s: tuple(sorted(set(s.dropna()))))
        .to_dict()
    )

    group_attributes, category_levels = _build_group_attribute_table(groups)

    collapsed = _collapse_multi_station_rows(enriched)
    weather_medians = {
        col: float(collapsed[col].median(skipna=True) or 0.0)
        for col in WEATHER_COLS
    }
    collapsed = _impute_weather_features(collapsed, weather_medians)
    collapsed = collapsed.merge(group_attributes, on="group_id", how="left")
    collapsed = _add_holiday_flag(collapsed)
    collapsed = collapsed.merge(
        _prepare_price_features(prices), on="measured_at", how="left"
    )
    collapsed = _add_load_lags(collapsed, LAG_FEATURES)
    collapsed = collapsed.sort_values(["measured_at", "group_id"]).reset_index(
        drop=True
    )
    group_categories = sorted(collapsed["group_id"].unique())
    lag_medians = {
        f"load_lag_{lag}h": collapsed.groupby("group_id")[f"load_lag_{lag}h"].median()
        for lag in LAG_FEATURES
    }

    metadata = {
        "groups": groups,
        "consumption_wide": consumption,
        "group_station_map": group_station_map,
        "weather_history": weather_history,
        "prices": prices,
        "weather_medians": weather_medians,
        "group_categories": group_categories,
        "lag_medians": lag_medians,
        "group_attributes": group_attributes,
        "category_levels": category_levels,
    }
    return collapsed, metadata


def split_train_validation(
    long_df: pd.DataFrame, validation_days: int = VALIDATION_DAYS
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Time-based split keeping the final ``validation_days`` for holdout."""

    df = long_df.sort_values("measured_at").reset_index(drop=True)
    cutoff = df["measured_at"].max() - pd.Timedelta(days=validation_days)
    train = df[df["measured_at"] < cutoff].copy()
    val = df[df["measured_at"] >= cutoff].copy()
    return train, val


def _collapse_multi_station_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Average weather features when a group matches multiple stations."""

    agg_spec: Dict[str, Any] = {
        "load_mwh": "first",
        "location_key": "first",
        "hour": "first",
        "weekday": "first",
        "is_weekend": "first",
        "month": "first",
        "weekofyear": "first",
        "year": "first",
    }
    for col in WEATHER_COLS:
        agg_spec[col] = "mean"

    collapsed = (
        df.groupby(["measured_at", "group_id"], as_index=False)
        .agg(agg_spec)
        .sort_values(["measured_at", "group_id"])
    )
    return collapsed


def _impute_weather_features(
    df: pd.DataFrame, fallback: Dict[str, float] | None = None
) -> pd.DataFrame:
    """Fill missing weather readings via group medians then global fallback."""

    result = df.copy()
    fallback = fallback or {}
    for col in WEATHER_COLS:
        if not result[col].isna().any():
            continue
        if "group_id" in result.columns:
            group_median = result.groupby("group_id")[col].transform("median")
            result[col] = result[col].fillna(group_median)
        fill_value = fallback.get(col)
        if fill_value is None or pd.isna(fill_value):
            fill_value = float(result[col].median(skipna=True) or 0.0)
        result[col] = result[col].fillna(fill_value)
    return result


def _impute_lag_features(
    df: pd.DataFrame, lag_stats: Dict[str, pd.Series] | None = None
) -> pd.DataFrame:
    if not lag_stats:
        return df
    result = df.copy()
    for column, stats in lag_stats.items():
        if column not in result.columns or not result[column].isna().any():
            continue
        group_fallback = result["group_id"].map(stats)
        result[column] = result[column].fillna(group_fallback)
        if result[column].isna().any():
            global_median = float(stats.median(skipna=True) or 0.0)
            result[column] = result[column].fillna(global_median)
    return result


def _add_holiday_flag(df: pd.DataFrame) -> pd.DataFrame:
    augmented = df.copy()
    measured = augmented["measured_at"]
    if measured.dt.tz is None:
        measured = measured.dt.tz_localize("UTC")
    local = measured.dt.tz_convert("Europe/Helsinki")
    years = local.dt.year.unique().tolist()
    calendar = holidays.Finland(years=years)
    dates = local.dt.date
    augmented["is_holiday"] = [1 if date in calendar else 0 for date in dates]
    return augmented


def _prepare_price_features(prices: pd.DataFrame) -> pd.DataFrame:
    """Add simple lagged spot-price features."""

    df = prices.copy()
    df = df.sort_values("measured_at")
    df = df.rename(columns={"eur_per_mwh": "price_eur_per_mwh"})
    df["price_lag_24h"] = df["price_eur_per_mwh"].shift(24)

    missing = df["price_lag_24h"].isna()
    if missing.any():
        hours = df["measured_at"].dt.hour
        hourly_median = (
            df.assign(hour=hours)
            .groupby("hour")["price_eur_per_mwh"]
            .transform("median")
        )
        df.loc[missing, "price_lag_24h"] = hourly_median[missing]

    return df[["measured_at", "price_eur_per_mwh", "price_lag_24h"]]


def _add_load_lags(df: pd.DataFrame, lags: Sequence[int]) -> pd.DataFrame:
    """Append per-group autoregressive load features."""

    result = df.sort_values(["group_id", "measured_at"]).copy()
    for lag in lags:
        result[f"load_lag_{lag}h"] = (
            result.groupby("group_id")["load_mwh"].shift(lag)
        )
    return result


def _build_weather_profiles(weather_history: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Pre-compute climatology tables for every weather station."""

    local = weather_history["measured_at"].dt.tz_convert("Europe/Helsinki")
    base = weather_history.assign(
        month=local.dt.month,
        day=local.dt.day,
        hour=local.dt.hour,
    )
    by_day = (
        base.groupby(["station_slug", "month", "day", "hour"], as_index=False)[
            WEATHER_COLS
        ]
        .median()
        .rename(columns={col: f"{col}_day" for col in WEATHER_COLS})
    )
    by_month = (
        base.groupby(["station_slug", "month", "hour"], as_index=False)[
            WEATHER_COLS
        ]
        .median()
        .rename(columns={col: f"{col}_month" for col in WEATHER_COLS})
    )
    by_hour = (
        base.groupby(["station_slug", "hour"], as_index=False)[WEATHER_COLS]
        .median()
        .rename(columns={col: f"{col}_hour" for col in WEATHER_COLS})
    )
    return {"day": by_day, "month": by_month, "hour": by_hour}


def _forecast_station_weather(
    timestamps: pd.DatetimeIndex, profiles: Dict[str, pd.DataFrame]
) -> pd.DataFrame:
    """Project hourly weather by falling back from day→month→hour climatology."""

    stations = profiles["hour"]["station_slug"].unique()
    n_times = len(timestamps)
    future = pd.DataFrame(
        {
            "station_slug": np.repeat(stations, n_times),
            "measured_at": np.tile(timestamps, len(stations)),
        }
    )
    future = weather_enrichment._add_time_features(future)
    local = future["measured_at"].dt.tz_convert("Europe/Helsinki")
    future["day"] = local.dt.day

    future = future.merge(
        profiles["day"],
        on=["station_slug", "month", "day", "hour"],
        how="left",
    )
    future = future.merge(
        profiles["month"],
        on=["station_slug", "month", "hour"],
        how="left",
    )
    for col in WEATHER_COLS:
        future[col + "_day"] = future[col + "_day"].fillna(
            future[col + "_month"]
        )
    future = future.merge(
        profiles["hour"], on=["station_slug", "hour"], how="left"
    )
    for col in WEATHER_COLS:
        future[col + "_day"] = future[col + "_day"].fillna(
            future[col + "_hour"]
        )

    keep = ["station_slug", "measured_at"]
    for col in WEATHER_COLS:
        keep.append(col + "_day")
    station_weather = future[keep].rename(
        columns={f"{col}_day": col for col in WEATHER_COLS}
    )
    return station_weather


def _build_group_weather_features(
    station_weather: pd.DataFrame, group_station_map: Dict[int, Sequence[str]]
) -> pd.DataFrame:
    """Average station forecasts when multiple stations feed a group."""

    frames = []
    for group_id, stations in group_station_map.items():
        subset = station_weather[station_weather["station_slug"].isin(stations)]
        if subset.empty:
            continue
        aggregated = (
            subset.groupby("measured_at")[WEATHER_COLS]
            .mean()
            .reset_index()
        )
        aggregated["group_id"] = int(group_id)
        frames.append(aggregated)

    if not frames:
        raise ValueError("No weather features were generated for any group.")

    return pd.concat(frames, ignore_index=True)


def build_price_forecast(
    prices: pd.DataFrame, timestamps: pd.DatetimeIndex
) -> pd.DataFrame:
    """Estimate future prices via recent hourly averages."""

    df = prices.copy().sort_values("measured_at")
    df = df.set_index("measured_at")
    recent = df.iloc[-14 * 24 :]
    hourly_avg = recent.groupby(recent.index.hour)["eur_per_mwh"].mean()
    fallback = float(df["eur_per_mwh"].median())

    records = []
    for ts in timestamps:
        est_price = float(hourly_avg.get(ts.hour, fallback))
        lag_index = ts - pd.Timedelta(hours=24)
        lag_price = df["eur_per_mwh"].get(lag_index)
        if pd.isna(lag_price):
            lag_price = est_price
        records.append(
            {
                "measured_at": ts,
                "price_eur_per_mwh": est_price,
                "price_lag_24h": float(lag_price),
            }
        )

    return pd.DataFrame(records)


def build_forecast_frame(
    metadata: Dict[str, Any], start_ts: pd.Timestamp, periods: int
) -> pd.DataFrame:
    """Construct feature matrix for the requested 48-hour window."""

    timestamps = pd.date_range(start=start_ts, periods=periods, freq="h", tz="UTC")
    base = pd.DataFrame({"measured_at": timestamps})
    base = weather_enrichment._add_time_features(base)
    base = base[[
        "measured_at",
        "hour",
        "weekday",
        "is_weekend",
        "month",
        "weekofyear",
        "year",
    ]]
    base = _add_holiday_flag(base)

    consumption = metadata["consumption_wide"].copy()
    group_cols = [col for col in consumption.columns if col != "measured_at"]
    wide = consumption.set_index("measured_at").sort_index()

    frames = []
    for group_id in map(int, group_cols):
        group_frame = base.copy()
        group_frame["group_id"] = group_id
        series = pd.to_numeric(wide[group_id], errors="coerce")
        for lag in LAG_FEATURES:
            lag_index = group_frame["measured_at"] - pd.to_timedelta(lag, unit="h")
            group_frame[f"load_lag_{lag}h"] = series.reindex(lag_index).values
        frames.append(group_frame)

    forecast = pd.concat(frames, ignore_index=True)
    forecast = forecast.merge(
        metadata["group_attributes"], on="group_id", how="left"
    )

    price_forecast = build_price_forecast(metadata["prices"], timestamps)
    forecast = forecast.merge(price_forecast, on="measured_at", how="left")

    profiles = _build_weather_profiles(metadata["weather_history"])
    station_weather = _forecast_station_weather(timestamps, profiles)
    group_weather = _build_group_weather_features(
        station_weather, metadata["group_station_map"]
    )
    forecast = forecast.merge(
        group_weather, on=["group_id", "measured_at"], how="left"
    )
    forecast = _impute_weather_features(
        forecast, metadata.get("weather_medians")
    )
    forecast = _impute_lag_features(forecast, metadata.get("lag_medians"))

    return forecast


def load_lightgbm_model(model_path: Path = LIGHTGBM_MODEL_PATH):
    if lgb is None:
        raise ImportError(
            "lightgbm is not installed; install requirements before forecasting"
        )
    if not model_path.exists():
        raise FileNotFoundError(
            f"Expected trained LightGBM model at {model_path}. Run train_lightgbm.py first."
        )
    model = joblib.load(model_path)
    if not hasattr(model, "predict"):
        raise TypeError("Loaded object is not a LightGBM regressor")
    return model


def _prepare_lgb_features(
    forecast_features: pd.DataFrame, category_levels: Dict[str, Sequence[Any]]
) -> pd.DataFrame:
    required_cols = FEATURE_COLUMNS + ["group_id"]
    missing = [col for col in required_cols if col not in forecast_features.columns]
    if missing:
        raise ValueError(f"Forecast features missing columns: {missing}")

    features = forecast_features[required_cols].copy()
    for name, cats in category_levels.items():
        if name in features.columns:
            features[name] = pd.Categorical(features[name], categories=cats)
    if features.isna().any().any():
        bad_cols = features.columns[features.isna().any()].tolist()
        raise ValueError(f"Forecast feature matrix contains NaNs in: {bad_cols}")
    return features


def predict_with_lightgbm(
    model,
    forecast_features: pd.DataFrame,
    category_levels: Dict[str, Sequence[Any]],
) -> pd.DataFrame:
    X = _prepare_lgb_features(forecast_features, category_levels)
    num_iteration = getattr(model, "best_iteration_", None)
    preds = model.predict(X, num_iteration=num_iteration)
    results = forecast_features[["measured_at", "group_id"]].copy()
    results["forecast_mw"] = preds
    return results


def generate_forecast_csv(
    model,
    forecast_features: pd.DataFrame,
    category_levels: Dict[str, Sequence[Any]],
    output_path: Path,
) -> pd.DataFrame:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    results = predict_with_lightgbm(model, forecast_features, category_levels)
    results = results.sort_values(["measured_at", "group_id"]).reset_index(drop=True)

    wide = (
        results.pivot(index="measured_at", columns="group_id", values="forecast_mw")
        .sort_index()
        .sort_index(axis=1)
    )
    wide.index = wide.index.astype(str)
    wide.columns = [str(int(col)) for col in wide.columns]

    wide.to_csv(output_path, sep=";", decimal=",")
    print(f"Saved forecast to {output_path}")
    return results




def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate 48-hour forecasts using the trained LightGBM model"
    )
    parser.add_argument(
        "--start",
        default=str(FORECAST_START),
        help="UTC start timestamp for the forecast window (default: 2024-10-01 00:00:00)",
    )
    parser.add_argument(
        "--hours",
        type=int,
        default=FORECAST_HOURS,
        help="Number of consecutive hours to forecast (default: 48)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=MODEL_DIR / "forecast_48h_oct1_oct2_weather.csv",
        help="Destination CSV for the forecast",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=LIGHTGBM_MODEL_PATH,
        help="Path to the trained LightGBM model",
    )
    return parser.parse_args()


def _coerce_timestamp(value: str) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts


def main() -> None:
    args = _parse_args()
    start_ts = _coerce_timestamp(args.start)
    if args.hours <= 0:
        raise ValueError("hours must be positive")

    print("Loading weather-enriched consumption frame...")
    _, metadata = load_prepared_long_frame()

    print("Loading trained LightGBM model...")
    model = load_lightgbm_model(args.model_path)

    print(f"Building feature matrix for window starting {start_ts}...")
    forecast_features = build_forecast_frame(metadata, start_ts, args.hours)

    results = generate_forecast_csv(
        model,
        forecast_features,
        metadata["category_levels"],
        args.output,
    )

    total_mw = results["forecast_mw"].sum()
    print(
        f"Forecast covers {results['group_id'].nunique()} groups "
        f"and {results['measured_at'].nunique()} hours."
    )
    print(f"Aggregate energy (converted to MW): {total_mw:.2f}")


if __name__ == "__main__":
    main()
