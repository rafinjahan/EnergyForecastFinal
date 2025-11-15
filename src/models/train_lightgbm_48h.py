"""
Enhanced LightGBM model for 48-hour energy consumption forecasting.
Trains separate models for each group with comprehensive features.
"""
from __future__ import annotations
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib
from typing import Dict, Tuple
import warnings
warnings.filterwarnings('ignore')

# Add src to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.fortum_loader import load_fortum_training
from src.data.weather_enrichment import enrich_consumption_with_all_weather
from src.data.finnish_holidays import add_holiday_features

# Paths
MODEL_DIR = PROJECT_ROOT / "models" / "lgb_48h"
MODEL_DIR.mkdir(parents=True, exist_ok=True)


def calculate_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate Mean Absolute Percentage Error, handling zeros."""
    mask = y_true != 0
    if not mask.any():
        return 0.0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def load_and_prepare_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load Fortum data - use simple approach without complex weather enrichment."""
    print("Loading data...")
    data = load_fortum_training()
    
    cons = data["consumption"]
    groups = data["groups"]
    prices = data["prices"]
    
    # Parse timestamps
    cons["measured_at"] = pd.to_datetime(cons["measured_at"], utc=True)
    prices["measured_at"] = pd.to_datetime(prices["measured_at"], utc=True)
    
    print(f"✓ Loaded {len(cons)} consumption records, {len(groups)} groups")
    
    # Convert to long format
    group_cols = [col for col in cons.columns if col != "measured_at"]
    cons_long = cons.melt(
        id_vars="measured_at",
        value_vars=group_cols,
        var_name="group_id",
        value_name="load_mwh"
    )
    cons_long["group_id"] = cons_long["group_id"].astype(int)
    
    # Merge prices
    cons_long = cons_long.merge(
        prices[["measured_at", "eur_per_mwh"]],
        on="measured_at",
        how="left"
    )
    
    # Try to add simple averaged weather (optional - skip if errors)
    print("\\nAttempting to add weather features...")
    try:
        weather = _load_simple_weather()
        if weather is not None and len(weather) > 0:
            cons_long = cons_long.merge(weather, on="measured_at", how="left")
            weather_cols = [c for c in weather.columns if c != "measured_at"]
            print(f"✓ Added weather features: {', '.join(weather_cols)}")
        else:
            print("⚠ No weather data available")
    except Exception as e:
        print(f"⚠ Weather loading skipped: {str(e)[:100]}")
    
    print(f"\n✓ Final dataset: {len(cons_long)} records with {len(cons_long.columns)} columns")
    
    # Add Finnish holiday features
    print("\nAdding Finnish holiday features...")
    cons_long = add_holiday_features(cons_long, date_column="measured_at")
    print(f"✓ Dataset now has {len(cons_long.columns)} columns with holiday features")
    
    return cons_long, groups, prices


def _load_simple_weather() -> pd.DataFrame:
    """Load weather data from Excel files, handling '-' and missing values."""
    import glob
    
    weather_dir = PROJECT_ROOT / "Data"
    weather_files = list(weather_dir.glob("*.xlsx"))
    
    # Skip training file and humidity-only files
    weather_files = [f for f in weather_files if "JUNCTION_training" not in f.name and "humidity" not in f.name.lower()]
    
    if not weather_files:
        return None
    
    all_weather = []
    
    for file_path in weather_files[:5]:  # Limit to first 5 files for speed
        try:
            df = pd.read_excel(file_path)
            
            # Check for required columns
            required_cols = ["Vuosi", "Kuukausi", "Päivä", "Aika [Paikallinen aika]"]
            if not all(col in df.columns for col in required_cols):
                continue
            
            # Build timestamp
            df["Vuosi"] = pd.to_numeric(df["Vuosi"], errors="coerce")
            df["Kuukausi"] = pd.to_numeric(df["Kuukausi"], errors="coerce")
            df["Päivä"] = pd.to_numeric(df["Päivä"], errors="coerce")
            
            df = df.dropna(subset=["Vuosi", "Kuukausi", "Päivä"])
            
            local_dt = pd.to_datetime(
                df["Vuosi"].astype(int).astype(str) + "-" +
                df["Kuukausi"].astype(int).astype(str).str.zfill(2) + "-" +
                df["Päivä"].astype(int).astype(str).str.zfill(2) + " " +
                df["Aika [Paikallinen aika]"].astype(str).str.zfill(5),
                format="%Y-%m-%d %H:%M",
                errors="coerce"
            )
            
            timestamp = local_dt.dt.tz_localize("Europe/Helsinki", ambiguous="infer", nonexistent="shift_forward").dt.tz_convert("UTC")
            
            weather_df = pd.DataFrame({"measured_at": timestamp})
            
            # Extract weather columns, replacing '-' with NaN
            weather_cols_map = {
                "Lämpötilan keskiarvo [°C]": "temperature_c",
                "Keskituulen nopeus [m/s]": "wind_speed_ms",
                "Tunnin sademäärä [mm]": "precip_mm",
                "Suhteellisen kosteuden keskiarvo [%]": "humidity_pct"
            }
            
            for orig_col, new_col in weather_cols_map.items():
                if orig_col in df.columns:
                    # Replace '-' with NaN and convert to numeric
                    series = df[orig_col].replace(["-", ""], np.nan)
                    weather_df[new_col] = pd.to_numeric(series, errors="coerce")
            
            weather_df = weather_df.dropna(subset=["measured_at"])
            all_weather.append(weather_df)
            
        except Exception as e:
            continue
    
    if not all_weather:
        return None
    
    # Combine all weather data and average by timestamp
    combined = pd.concat(all_weather, ignore_index=True)
    
    # Group by timestamp and take mean (ignoring NaN)
    weather_cols = [c for c in combined.columns if c != "measured_at"]
    if weather_cols:
        averaged = combined.groupby("measured_at", as_index=False)[weather_cols].mean()
        return averaged
    
    return None


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create comprehensive features for hourly forecasting with enhanced feature engineering."""
    df = df.copy()
    
    # Time features
    local_time = df["measured_at"].dt.tz_convert("Europe/Helsinki")
    
    df["hour"] = local_time.dt.hour
    df["dayofweek"] = local_time.dt.dayofweek
    df["month"] = local_time.dt.month
    df["day"] = local_time.dt.day
    df["weekofyear"] = local_time.dt.isocalendar().week
    df["is_weekend"] = (local_time.dt.dayofweek >= 5).astype(int)
    
    # NEW: Enhanced time features
    df["quarter"] = ((df["month"] - 1) // 3) + 1
    df["season"] = df["month"].map({12: 0, 1: 0, 2: 0,  # Winter
                                     3: 1, 4: 1, 5: 1,   # Spring
                                     6: 2, 7: 2, 8: 2,   # Summer
                                     9: 3, 10: 3, 11: 3}) # Fall
    df["is_business_hour"] = ((df["hour"] >= 8) & (df["hour"] <= 17) & (df["dayofweek"] < 5)).astype(int)
    df["is_night"] = ((df["hour"] >= 22) | (df["hour"] <= 6)).astype(int)
    df["week_in_month"] = ((df["day"] - 1) // 7) + 1
    
    # Cyclical encoding for hour, month, and day of week
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    df["dayofweek_sin"] = np.sin(2 * np.pi * df["dayofweek"] / 7)
    df["dayofweek_cos"] = np.cos(2 * np.pi * df["dayofweek"] / 7)
    
    # Sort by group and time for lag features
    df = df.sort_values(["group_id", "measured_at"]).reset_index(drop=True)
    
    # Enhanced lag features (per group)
    for lag in [24, 48, 72, 168, 336]:  # 1d, 2d, 3d, 1w, 2w
        df[f"load_lag_{lag}h"] = df.groupby("group_id")["load_mwh"].shift(lag)
    
    # Rolling statistics (per group)
    for window in [24, 72, 168]:  # 1 day, 3 days, 1 week
        df[f"load_rolling_mean_{window}h"] = (
            df.groupby("group_id")["load_mwh"]
            .transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
        )
        df[f"load_rolling_std_{window}h"] = (
            df.groupby("group_id")["load_mwh"]
            .transform(lambda x: x.shift(1).rolling(window, min_periods=1).std())
        )
        df[f"load_rolling_max_{window}h"] = (
            df.groupby("group_id")["load_mwh"]
            .transform(lambda x: x.shift(1).rolling(window, min_periods=1).max())
        )
        df[f"load_rolling_min_{window}h"] = (
            df.groupby("group_id")["load_mwh"]
            .transform(lambda x: x.shift(1).rolling(window, min_periods=1).min())
        )
    
    # NEW: Exponential moving average
    for span in [24, 168]:
        df[f"load_ema_{span}h"] = (
            df.groupby("group_id")["load_mwh"]
            .transform(lambda x: x.shift(1).ewm(span=span, min_periods=1).mean())
        )
    
    # NEW: Weather-based features (if available)
    if "temperature_c" in df.columns:
        # Heating/cooling degree days (base 18°C)
        df["heating_degree_hours"] = np.maximum(18 - df["temperature_c"], 0)
        df["cooling_degree_hours"] = np.maximum(df["temperature_c"] - 22, 0)
        
        # Temperature lags
        df["temp_lag_24h"] = df.groupby("group_id")["temperature_c"].shift(24)
        df["temp_change_24h"] = df["temperature_c"] - df["temp_lag_24h"]
        
        # Extreme weather indicators
        df["is_very_cold"] = (df["temperature_c"] < -10).astype(int)
        df["is_very_hot"] = (df["temperature_c"] > 25).astype(int)
        df["is_freezing"] = (df["temperature_c"] < 0).astype(int)
        
        # Weather interactions
        if "wind_speed_ms" in df.columns:
            df["wind_chill"] = df["temperature_c"] - (df["wind_speed_ms"] * 0.7)  # Simplified wind chill
        if "humidity_pct" in df.columns:
            df["temp_humidity_interaction"] = df["temperature_c"] * df["humidity_pct"] / 100
    
    # NEW: Price-based features (if available)
    if "eur_per_mwh" in df.columns:
        # Price lags
        df["price_lag_24h"] = df.groupby("group_id")["eur_per_mwh"].shift(24)
        df["price_change_24h"] = df["eur_per_mwh"] - df["price_lag_24h"]
        
        # Price rolling statistics
        df["price_rolling_mean_168h"] = (
            df.groupby("group_id")["eur_per_mwh"]
            .transform(lambda x: x.shift(1).rolling(168, min_periods=1).mean())
        )
        df["price_rolling_std_168h"] = (
            df.groupby("group_id")["eur_per_mwh"]
            .transform(lambda x: x.shift(1).rolling(168, min_periods=1).std())
        )
        
        # Price percentile (is current price high or low?)
        df["price_percentile"] = (
            df.groupby("group_id")["eur_per_mwh"]
            .transform(lambda x: x.rank(pct=True))
        )
        
        # Price × hour interaction (people may respond differently to price based on time)
        df["price_hour_interaction"] = df["eur_per_mwh"] * df["hour"]
    
    # NEW: Historical patterns by group
    # Mean consumption by hour of day for each group
    df["group_hour_mean"] = (
        df.groupby(["group_id", "hour"])["load_mwh"]
        .transform(lambda x: x.shift(1).expanding(min_periods=1).mean())
    )
    
    # Mean consumption by day of week for each group
    df["group_dow_mean"] = (
        df.groupby(["group_id", "dayofweek"])["load_mwh"]
        .transform(lambda x: x.shift(1).expanding(min_periods=1).mean())
    )
    
    # Ratio of current load to recent average (trend indicator)
    df["load_trend_ratio"] = df["load_mwh"] / (df["load_rolling_mean_168h"] + 0.001)
    
    return df


def prepare_train_val_split(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split data by time - 80/20 train/validation split."""
    df = df.sort_values("measured_at").reset_index(drop=True)
    
    # Calculate 80/20 split point based on time range
    min_date = df["measured_at"].min()
    max_date = df["measured_at"].max()
    total_duration = max_date - min_date
    val_start = min_date + (total_duration * 0.8)
    
    train_df = df[df["measured_at"] < val_start].copy()
    val_df = df[df["measured_at"] >= val_start].copy()
    
    train_pct = len(train_df) / len(df) * 100
    val_pct = len(val_df) / len(df) * 100
    
    print(f"\nTrain: {train_df['measured_at'].min()} to {train_df['measured_at'].max()}")
    print(f"Val:   {val_df['measured_at'].min()} to {val_df['measured_at'].max()}")
    print(f"Train size: {len(train_df):,} ({train_pct:.1f}%), Val size: {len(val_df):,} ({val_pct:.1f}%)")
    
    return train_df, val_df


def get_feature_columns(df: pd.DataFrame) -> list:
    """Get list of feature columns."""
    exclude_cols = ["measured_at", "group_id", "load_mwh", "location_key", 
                    "station_slug", "station_name", "group_label", "year",
                    "weekday", "load_trend_ratio", "holiday_week"]  # holiday_week is redundant with is_holiday
    
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    return feature_cols


def train_model_for_group(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    group_id: int,
    feature_cols: list
) -> Tuple[lgb.LGBMRegressor, Dict[str, float]]:
    """Train a LightGBM model for a specific group."""
    
    # Filter data for this group
    train_group = train_df[train_df["group_id"] == group_id].copy()
    val_group = val_df[val_df["group_id"] == group_id].copy()
    
    if len(train_group) < 50 or len(val_group) < 5:
        raise ValueError(f"Insufficient data for group {group_id}")
    
    # Prepare features and target
    X_train = train_group[feature_cols]
    y_train = train_group["load_mwh"]
    X_val = val_group[feature_cols]
    y_val = val_group["load_mwh"]
    
    # Fill NaN in features with median/0 instead of dropping all rows
    for col in X_train.columns:
        if X_train[col].isna().any():
            median_val = X_train[col].median()
            fill_val = median_val if not pd.isna(median_val) else 0
            X_train[col].fillna(fill_val, inplace=True)
            X_val[col].fillna(fill_val, inplace=True)
    
    # Drop rows with NaN target only
    train_mask = ~y_train.isna()
    val_mask = ~y_val.isna()
    
    X_train = X_train[train_mask]
    y_train = y_train[train_mask]
    X_val = X_val[val_mask]
    y_val = y_val[val_mask]
    
    if len(X_train) < 50 or len(X_val) < 5:
        raise ValueError(f"Insufficient valid data for group {group_id}")
    
    # Train model
    model = lgb.LGBMRegressor(
        objective="regression",
        n_estimators=500,
        learning_rate=0.05,
        max_depth=8,
        num_leaves=64,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
        n_jobs=-1,
        random_state=42,
        verbose=-1,
    )
    
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="mae",
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)],
    )
    
    # Evaluate
    y_pred = model.predict(X_val)
    
    metrics = {
        "mae": mean_absolute_error(y_val, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_val, y_pred)),
        "mape": calculate_mape(y_val.values, y_pred),
        "train_samples": len(X_train),
        "val_samples": len(X_val),
    }
    
    return model, metrics


def train_all_groups(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    feature_cols: list
) -> Dict[int, Tuple[lgb.LGBMRegressor, Dict[str, float]]]:
    """Train models for all groups."""
    
    group_ids = sorted(train_df["group_id"].unique())
    print(f"\n{'='*60}")
    print(f"Training models for {len(group_ids)} groups (Target: 112 groups)")
    if len(group_ids) != 112:
        print(f"⚠ WARNING: Expected 112 groups, found {len(group_ids)}")
    else:
        print("✓ All 112 groups present")
    print(f"{'='*60}")
    
    models = {}
    all_metrics = []
    
    for i, group_id in enumerate(group_ids, 1):
        try:
            model, metrics = train_model_for_group(
                train_df, val_df, group_id, feature_cols
            )
            models[group_id] = (model, metrics)
            
            print(f"[{i:3d}/{len(group_ids)}] Group {group_id:3d}: "
                  f"MAE={metrics['mae']:.4f}, RMSE={metrics['rmse']:.4f}, "
                  f"MAPE={metrics['mape']:.2f}%")
            
            all_metrics.append({
                "group_id": group_id,
                **metrics
            })
            
        except Exception as e:
            print(f"[{i:3d}/{len(group_ids)}] Group {group_id:3d}: FAILED - {e}")
    
    # Summary statistics
    if all_metrics:
        metrics_df = pd.DataFrame(all_metrics)
        print(f"\n{'='*60}")
        print(f"OVERALL TRAINING SUMMARY")
        print(f"{'='*60}")
        print(f"Successfully trained: {len(models)}/{len(group_ids)} groups")
        print(f"Average MAE:  {metrics_df['mae'].mean():.4f}")
        print(f"Average RMSE: {metrics_df['rmse'].mean():.4f}")
        print(f"Average MAPE: {metrics_df['mape'].mean():.2f}%")
        print(f"Median MAPE:  {metrics_df['mape'].median():.2f}%")
        print(f"Best MAPE:    {metrics_df['mape'].min():.2f}% (Group {metrics_df.loc[metrics_df['mape'].idxmin(), 'group_id']:.0f})")
        print(f"Worst MAPE:   {metrics_df['mape'].max():.2f}% (Group {metrics_df.loc[metrics_df['mape'].idxmax(), 'group_id']:.0f})")
        print(f"{'='*60}")
    
    return models


def save_models_and_metrics(
    models: Dict[int, Tuple[lgb.LGBMRegressor, Dict[str, float]]],
    feature_cols: list
):
    """Save trained models and metrics."""
    
    # Save each model
    for group_id, (model, _) in models.items():
        model_path = MODEL_DIR / f"group_{group_id}.pkl"
        joblib.dump(model, model_path)
    
    print(f"\n✓ Saved {len(models)} models to {MODEL_DIR}/")
    
    # Save feature columns
    feature_path = MODEL_DIR / "feature_columns.pkl"
    joblib.dump(feature_cols, feature_path)
    print(f"✓ Saved feature columns to {feature_path}")
    
    # Save metrics
    metrics_records = []
    for group_id, (_, metrics) in models.items():
        metrics_records.append({
            "group_id": group_id,
            **metrics
        })
    
    metrics_df = pd.DataFrame(metrics_records).sort_values("mape")
    metrics_path = MODEL_DIR / "training_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)
    print(f"✓ Saved metrics to {metrics_path}")
    
    # Save summary
    summary_path = MODEL_DIR / "training_summary.txt"
    with open(summary_path, "w") as f:
        f.write("48-Hour LightGBM Model Training Summary\n")
        f.write("="*60 + "\n")
        f.write(f"Total groups: {len(models)}\n")
        f.write(f"Average MAE:  {metrics_df['mae'].mean():.4f}\n")
        f.write(f"Average RMSE: {metrics_df['rmse'].mean():.4f}\n")
        f.write(f"Average MAPE: {metrics_df['mape'].mean():.2f}%\n")
        f.write(f"Median MAPE:  {metrics_df['mape'].median():.2f}%\n")
        f.write(f"Best MAPE:    {metrics_df['mape'].min():.2f}%\n")
        f.write(f"Worst MAPE:   {metrics_df['mape'].max():.2f}%\n")
        f.write("\nFeatures used:\n")
        for feat in feature_cols:
            f.write(f"  - {feat}\n")
    
    print(f"✓ Saved summary to {summary_path}")


def main():
    """Main training pipeline for 48-hour forecast models."""
    print("="*60)
    print("LIGHTGBM 48-HOUR FORECAST MODEL TRAINING")
    print("="*60)
    print("\nDATA UNIT VERIFICATION:")
    print("  Input data: MWh (megawatt-hours) from Fortum training file")
    print("  Output forecasts: MWh (megawatt-hours)")
    print("  Note: 1 MWh = 1,000,000 Wh (Watt hours)")
    print("="*60)
    
    # 1. Load and prepare data
    enriched, groups, prices = load_and_prepare_data()
    
    # 2. Engineer features
    print("\nEngineering features...")
    enriched = engineer_features(enriched)
    print(f"✓ Feature engineering complete")
    
    # 3. Split data (80/20)
    train_df, val_df = prepare_train_val_split(enriched)
    
    # 4. Get feature columns
    feature_cols = get_feature_columns(train_df)
    print(f"\nUsing {len(feature_cols)} features:")
    for feat in feature_cols:
        print(f"  - {feat}")
    
    # 5. Train models for all groups
    models = train_all_groups(train_df, val_df, feature_cols)
    
    # 6. Save models and metrics
    save_models_and_metrics(models, feature_cols)
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)


if __name__ == "__main__":
    main()
