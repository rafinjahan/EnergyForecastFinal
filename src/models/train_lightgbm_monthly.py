"""
Enhanced LightGBM model for 12-month (monthly) energy consumption forecasting.
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
MODEL_DIR = PROJECT_ROOT / "models" / "lgb_monthly"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

TARGET_COL = "load_value"
PRICE_COL = "price_signal"


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
        value_name=TARGET_COL
    )
    cons_long["group_id"] = cons_long["group_id"].astype(int)
    
    # Merge prices
    price_source_col = next((col for col in prices.columns if col.lower().startswith("eur_per")), None)
    if price_source_col:
        prices = prices.rename(columns={price_source_col: PRICE_COL})
    if PRICE_COL in prices.columns:
        cons_long = cons_long.merge(
            prices[["measured_at", PRICE_COL]],
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
    
    # Add Finnish holiday features (will be aggregated in aggregate_to_monthly)
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


def aggregate_to_monthly(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate hourly data to monthly totals."""
    print("\nAggregating to monthly level...")
    
    df = df.copy()
    
    # Create period for grouping
    df["month_period"] = df["measured_at"].dt.to_period("M")
    df["year"] = df["measured_at"].dt.year
    df["month"] = df["measured_at"].dt.month
    
    # Aggregation dict
    agg_dict = {
        TARGET_COL: "sum",  # Total monthly consumption
        "measured_at": "min",  # First timestamp of month
    }
    
    # Add weather aggregations if available
    weather_cols = ["temperature_c", "wind_speed_ms", "precip_mm", "humidity_pct"]
    for col in weather_cols:
        if col in df.columns:
            agg_dict[col] = "mean"
    
    # Add price aggregation
    if PRICE_COL in df.columns:
        agg_dict[PRICE_COL] = "mean"
    
    # Add holiday aggregations if available
    holiday_cols = ["is_holiday", "is_holiday_eve", "is_holiday_after", "is_major_holiday_period"]
    for col in holiday_cols:
        if col in df.columns:
            agg_dict[col] = "sum"  # Count of holiday days in month
    
    if "days_to_next_holiday" in df.columns:
        agg_dict["days_to_next_holiday"] = "mean"
    if "days_since_last_holiday" in df.columns:
        agg_dict["days_since_last_holiday"] = "mean"
    
    # Group by group_id and month
    monthly = df.groupby(["group_id", "month_period"], as_index=False).agg(agg_dict)
    
    # Add back year and month
    monthly["year"] = monthly["month_period"].dt.year
    monthly["month"] = monthly["month_period"].dt.month
    
    # Sort by group and time
    monthly = monthly.sort_values(["group_id", "month_period"]).reset_index(drop=True)
    
    print(f"✓ Aggregated to {len(monthly)} monthly records")
    return monthly


def engineer_monthly_features(df: pd.DataFrame, groups: pd.DataFrame) -> pd.DataFrame:
    """Create simplified features for monthly forecasting based on specified feature set."""
    df = df.copy()
    
    # === TIME FEATURES ===
    # Cyclical encoding for month
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    
    # Sort by group and month for lag features
    df = df.sort_values(["group_id", "month_period"]).reset_index(drop=True)
    
    # === LAG FEATURES ===
    # Convert hourly lags (1h, 24h, 168h) to monthly equivalents
    # 1h ≈ immediate past (use lag 1 month)
    # 24h ≈ same day last month (use lag 1 month) 
    # 168h ≈ same week last month (use lag 1 month)
    # Add 12-month lag for year-over-year pattern
    df["load_lag_1m"] = df.groupby("group_id")[TARGET_COL].shift(1)
    df["load_lag_12m"] = df.groupby("group_id")[TARGET_COL].shift(12)
    
    # === WEATHER FEATURES ===
    # Keep raw weather features (already aggregated to monthly means)
    # temperature_c, humidity_pct, precip_mm, wind_speed_ms
    # (These are already in the dataframe from _load_simple_weather)
    
    # === PRICE FEATURES ===
    # Market prices with moving averages and volatility
    if PRICE_COL in df.columns:
        # 3-month and 6-month moving averages
        df["price_ma_3m"] = (
            df.groupby("group_id")[PRICE_COL]
            .transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
        )
        df["price_ma_6m"] = (
            df.groupby("group_id")[PRICE_COL]
            .transform(lambda x: x.shift(1).rolling(6, min_periods=1).mean())
        )
        
        # Price volatility (rolling standard deviation)
        df["price_volatility_3m"] = (
            df.groupby("group_id")[PRICE_COL]
            .transform(lambda x: x.shift(1).rolling(3, min_periods=1).std())
        )
        df["price_volatility_6m"] = (
            df.groupby("group_id")[PRICE_COL]
            .transform(lambda x: x.shift(1).rolling(6, min_periods=1).std())
        )
    
    # === GROUP FEATURES ===
    # Parse group_label to extract customer type, contract type, consumption level
    # Merge with groups dataframe to get group_label
    df = df.merge(groups[["group_id", "group_label"]], on="group_id", how="left")
    
    # Extract features from group_label format: "Region|Province|ContractType|ConsumptionLevel"
    # Example: "Etelä-Suomi|Uusimaa|Spot|High"
    df["region"] = df["group_label"].str.split("|").str[0]
    df["province"] = df["group_label"].str.split("|").str[1]
    df["contract_type"] = df["group_label"].str.split("|").str[2]
    df["consumption_level"] = df["group_label"].str.split("|").str[3]
    
    # Encode categorical group features
    from sklearn.preprocessing import LabelEncoder
    
    for col in ["region", "province", "contract_type", "consumption_level"]:
        if col in df.columns and df[col].notna().any():
            le = LabelEncoder()
            # Fill NaN with 'Unknown' before encoding
            df[col] = df[col].fillna("Unknown")
            df[f"{col}_encoded"] = le.fit_transform(df[col].astype(str))
    
    return df


def prepare_train_val_split(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split data by time - 80/20 train/validation split."""
    df = df.sort_values("month_period").reset_index(drop=True)
    
    # Calculate 80/20 split point
    unique_months = df["month_period"].unique()
    unique_months_sorted = sorted(unique_months)
    split_idx = int(len(unique_months_sorted) * 0.8)
    val_start = unique_months_sorted[split_idx]
    
    train_df = df[df["month_period"] < val_start].copy()
    val_df = df[df["month_period"] >= val_start].copy()
    
    train_pct = len(train_df) / len(df) * 100
    val_pct = len(val_df) / len(df) * 100
    
    print(f"\nTrain: {train_df['month_period'].min()} to {train_df['month_period'].max()}")
    print(f"Val:   {val_df['month_period'].min()} to {val_df['month_period'].max()}")
    print(f"Train size: {len(train_df):,} ({train_pct:.1f}%), Val size: {len(val_df):,} ({val_pct:.1f}%)")
    
    return train_df, val_df


def get_feature_columns(df: pd.DataFrame) -> list:
    """Get list of feature columns - only the specified features."""
    exclude_cols = ["measured_at", "group_id", TARGET_COL, "month_period", 
                    "location_key", "station_slug", "station_name", "group_label",
                    "region", "province", "contract_type", "consumption_level",  # Exclude raw categorical
                    "year", "month"]  # Exclude raw time components
    
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
    
    if len(train_group) < 6 or len(val_group) < 2:
        raise ValueError(f"Insufficient data for group {group_id}")
    
    # Prepare features and target
    X_train = train_group[feature_cols]
    y_train = train_group[TARGET_COL]
    X_val = val_group[feature_cols]
    y_val = val_group[TARGET_COL]
    
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
    
    if len(X_train) < 6 or len(X_val) < 2:
        raise ValueError(f"Insufficient valid data for group {group_id}")
    
    # Train model
    model = lgb.LGBMRegressor(
        objective="regression",
        n_estimators=500,
        learning_rate=0.05,
        max_depth=7,
        num_leaves=48,
        min_child_samples=5,
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
        callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)],
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
                  f"MAE={metrics['mae']:>8.2f}, RMSE={metrics['rmse']:>8.2f}, "
                  f"MAPE={metrics['mape']:>6.2f}%")
            
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
        print(f"Average MAE:  {metrics_df['mae'].mean():.2f}")
        print(f"Average RMSE: {metrics_df['rmse'].mean():.2f}")
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
        f.write("12-Month LightGBM Model Training Summary\n")
        f.write("="*60 + "\n")
        f.write(f"Total groups: {len(models)}\n")
        f.write(f"Average MAE:  {metrics_df['mae'].mean():.2f}\n")
        f.write(f"Average RMSE: {metrics_df['rmse'].mean():.2f}\n")
        f.write(f"Average MAPE: {metrics_df['mape'].mean():.2f}%\n")
        f.write(f"Median MAPE:  {metrics_df['mape'].median():.2f}%\n")
        f.write(f"Best MAPE:    {metrics_df['mape'].min():.2f}%\n")
        f.write(f"Worst MAPE:   {metrics_df['mape'].max():.2f}%\n")
        f.write("\nFeatures used:\n")
        for feat in feature_cols:
            f.write(f"  - {feat}\n")
    
    print(f"✓ Saved summary to {summary_path}")


def main():
    """Main training pipeline for 12-month forecast models."""
    print("="*60)
    print("LIGHTGBM 12-MONTH FORECAST MODEL TRAINING")
    print("="*60)
    print("\nDATA SOURCE VERIFICATION:")
    print("  Input consumption data from Fortum training file")
    print("  Output forecasts follow the same scale as input data")
    print("  Prices are merged from the provided market feed")
    print("="*60)
    
    # 1. Load and prepare data
    enriched, groups, prices = load_and_prepare_data()
    
    # 2. Aggregate to monthly
    monthly = aggregate_to_monthly(enriched)
    
    # 3. Engineer features (pass groups for customer/contract type extraction)
    print("\nEngineering features...")
    monthly = engineer_monthly_features(monthly, groups)
    print(f"✓ Feature engineering complete")
    
    # 4. Split data (80/20)
    train_df, val_df = prepare_train_val_split(monthly)
    
    # 5. Get feature columns
    feature_cols = get_feature_columns(train_df)
    print(f"\nUsing {len(feature_cols)} features:")
    for feat in feature_cols:
        print(f"  - {feat}")
    
    # 6. Train models for all groups
    models = train_all_groups(train_df, val_df, feature_cols)
    
    # 7. Save models and metrics
    save_models_and_metrics(models, feature_cols)
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)


if __name__ == "__main__":
    main()
