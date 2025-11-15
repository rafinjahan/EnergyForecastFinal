"""Weather-aware LightGBM training pipeline.

This script reuses the feature engineering logic from
``src.models.test_forecast_48h`` to build a long-form dataset containing Fortum
load, price, weather, calendar, and autoregressive lag features for every
group/hour. It then trains a single LightGBM regressor with ``group_id`` marked
as a categorical feature so the booster can learn both shared patterns and
group-specific offsets.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Sequence, Tuple

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

# Ensure ``src`` is importable regardless of where the script is executed from.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.test_forecast_48h import (  # noqa: E402  (import after path tweak)
    FEATURE_COLUMNS,
    load_prepared_long_frame,
)

MODEL_DIR = PROJECT_ROOT / "models"
MODEL_NAME = "lgb_weather_model"
MODEL_PATH = MODEL_DIR / f"{MODEL_NAME}.pkl"
METRICS_PATH = MODEL_DIR / f"{MODEL_NAME}_metrics.txt"
PER_GROUP_PATH = MODEL_DIR / f"{MODEL_NAME}_val_per_group.csv"
TRAIN_END = pd.Timestamp("2024-01-01 00:00:00", tz="UTC")
VALIDATION_END = pd.Timestamp("2024-10-01 00:00:00", tz="UTC")

# LightGBM consumes categorical features (group and segment metadata) in
# addition to the engineered feature list shared with the forecasting script.
CATEGORICAL_FEATURES = [
    "group_id",
    "customer_type",
    "contract_type",
    "consumption_level",
]
TRAIN_FEATURES = FEATURE_COLUMNS + ["group_id"]


def _drop_incomplete_rows(df: pd.DataFrame) -> pd.DataFrame:
    cols = TRAIN_FEATURES + ["load_mwh"]
    return df.dropna(subset=cols).reset_index(drop=True)


def _prepare_feature_matrix(
    df: pd.DataFrame, category_levels: Dict[str, Sequence[str]]
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    cleaned = _drop_incomplete_rows(df)
    features = cleaned[TRAIN_FEATURES].copy()
    target = cleaned["load_mwh"].astype(float)
    for name in CATEGORICAL_FEATURES:
        if name not in features.columns:
            continue
        categories = category_levels.get(name)
        features[name] = pd.Categorical(features[name], categories=categories)
    return cleaned, features, target


def load_training_frames():
    print("Building weather-enriched long-form dataset (this may take a minute)...")
    long_df, metadata = load_prepared_long_frame()
    category_levels = metadata["category_levels"]

    measured = long_df["measured_at"]
    train_df = long_df[measured < TRAIN_END].copy()
    val_df = long_df[(measured >= TRAIN_END) & (measured < VALIDATION_END)].copy()

    if train_df.empty or val_df.empty:
        raise ValueError(
            "Custom date split produced empty partitions. Check timestamp ranges."
        )

    print(
        f"Training window: {train_df['measured_at'].min()} → {train_df['measured_at'].max()}"
    )
    print(
        f"Validation window: {val_df['measured_at'].min()} → {val_df['measured_at'].max()}"
    )

    train_clean, X_train, y_train = _prepare_feature_matrix(
        train_df, category_levels
    )
    val_clean, X_val, y_val = _prepare_feature_matrix(val_df, category_levels)

    print(
        f"Training samples: {len(X_train):,} | Validation samples: {len(X_val):,}"
    )
    print(f"Using feature columns: {TRAIN_FEATURES}")

    return (train_clean, X_train, y_train), (val_clean, X_val, y_val)


def train_lightgbm_model(
    X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame, y_val: pd.Series
) -> lgb.LGBMRegressor:
    print("\nTraining LightGBM regressor with group-aware features...")
    model = lgb.LGBMRegressor(
        objective="regression",
        metric="mape",
        learning_rate=0.05,
        n_estimators=2000,
        num_leaves=128,
        max_depth=-1,
        min_child_samples=60,
        colsample_bytree=0.8,
        subsample=0.9,
        subsample_freq=1,
        lambda_l2=0.1,
        n_jobs=-1,
        random_state=42,
        verbose=-1,
    )

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="mape",
        categorical_feature=CATEGORICAL_FEATURES,
        callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=True)],
    )

    print(f"Best iteration: {model.best_iteration_}")
    return model


def evaluate_validation_split(
    model: lgb.LGBMRegressor,
    val_rows: pd.DataFrame,
    X_val: pd.DataFrame,
    y_val: pd.Series,
) -> Tuple[Dict[str, float], pd.DataFrame]:
    preds = model.predict(X_val, num_iteration=model.best_iteration_)
    actual = y_val.to_numpy(dtype=float)
    mask = actual != 0
    if not np.any(mask):
        raise ValueError("MAPE computation failed: all validation actuals are zero.")

    pct_errors = np.full_like(actual, np.nan, dtype=float)
    pct_errors[mask] = np.abs((actual[mask] - preds[mask]) / actual[mask])
    overall_mape = float(np.nanmean(pct_errors))

    val_with_preds = val_rows.copy()
    val_with_preds["prediction"] = preds
    val_with_preds = val_with_preds[val_with_preds["load_mwh"] != 0]
    val_with_preds["abs_pct_error"] = np.abs(
        (val_with_preds["load_mwh"] - val_with_preds["prediction"])
        / val_with_preds["load_mwh"]
    )

    per_group = (
        val_with_preds.groupby("group_id")
        .agg(mape=("abs_pct_error", "mean"), n_points=("abs_pct_error", "size"))
        .reset_index()
        .sort_values("mape")
    )

    metrics = {"mape": overall_mape}
    print("\nValidation metrics:")
    print(f"  MAPE: {overall_mape:.4f}")
    return metrics, per_group


def save_artifacts(
    model: lgb.LGBMRegressor,
    metrics: Dict[str, float],
    per_group: pd.DataFrame,
) -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"\nSaved model to {MODEL_PATH}")

    with METRICS_PATH.open("w", encoding="utf-8") as handle:
        handle.write("Validation Metrics\n")
        handle.write("=" * 32 + "\n")
        for name, value in metrics.items():
            handle.write(f"{name.upper()}: {value:.6f}\n")
    print(f"Metrics written to {METRICS_PATH}")

    per_group.to_csv(PER_GROUP_PATH, index=False)
    print(f"Per-group validation metrics saved to {PER_GROUP_PATH}")


def main() -> None:
    (train_rows, X_train, y_train), (val_rows, X_val, y_val) = load_training_frames()
    model = train_lightgbm_model(X_train, y_train, X_val, y_val)
    metrics, per_group = evaluate_validation_split(model, val_rows, X_val, y_val)
    save_artifacts(model, metrics, per_group)


if __name__ == "__main__":
    main()
