"""
Generate 48-hour forecast in Fortum submission format.
Format: Wide CSV with semicolon delimiter, comma decimal separator.
October 1-2, 2024 (48 hours).
"""
from __future__ import annotations
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import lightgbm as lgb
import joblib
import warnings
warnings.filterwarnings('ignore')

# Add src to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.fortum_loader import load_fortum_training
from src.data.weather_enrichment import enrich_consumption_with_all_weather
from src.data.finnish_holidays import add_holiday_features

MODEL_DIR = PROJECT_ROOT / "models" / "lgb_48h"
OUTPUT_DIR = PROJECT_ROOT / "forecasts"
OUTPUT_DIR.mkdir(exist_ok=True)


def load_models_and_features():
    """Load all trained models and feature definitions."""
    feature_cols_path = MODEL_DIR / "feature_columns.pkl"
    if not feature_cols_path.exists():
        raise FileNotFoundError(f"Feature columns not found. Train models first.")
    
    feature_cols = joblib.load(feature_cols_path)
    
    # Load all group models
    models = {}
    for model_file in MODEL_DIR.glob("group_*.pkl"):
        group_id = int(model_file.stem.split("_")[1])
        models[group_id] = joblib.load(model_file)
    
    print(f"✓ Loaded {len(models)} trained models")
    print(f"✓ Using {len(feature_cols)} features")
    
    return models, feature_cols


def prepare_forecast_data():
    """Prepare data for October 1-2, 2024 forecast."""
    print("\nPreparing forecast data...")
    
    # Load historical data
    data = load_fortum_training()
    cons = data["consumption"]
    prices = data["prices"]
    
    # Parse timestamps
    cons["measured_at"] = pd.to_datetime(cons["measured_at"], utc=True)
    prices["measured_at"] = pd.to_datetime(prices["measured_at"], utc=True)
    
    # Get group columns
    group_cols = [col for col in cons.columns if col != "measured_at"]
    
    # Convert to long format
    cons_long = cons.melt(
        id_vars="measured_at",
        value_vars=group_cols,
        var_name="group_id",
        value_name="load_mwh"
    )
    cons_long["group_id"] = cons_long["group_id"].astype(int)
    
    # Merge with prices
    cons_long = cons_long.merge(prices[["measured_at", "eur_per_mwh"]], on="measured_at", how="left")
    
    # Add weather features
    print("Adding weather features...")
    try:
        weather = _load_simple_weather()
        if weather is not None:
            cons_long = cons_long.merge(weather, on="measured_at", how="left")
    except Exception as e:
        print(f"⚠ Weather loading failed: {e}")
    
    # Add holiday features
    print("Adding holiday features...")
    cons_long = add_holiday_features(cons_long, date_column="measured_at")
    
    # Create time features
    cons_long = _create_time_features(cons_long)
    
    # Create lag features for each group
    print("Creating lag features...")
    cons_long = cons_long.sort_values(["group_id", "measured_at"]).reset_index(drop=True)
    cons_long = _create_lag_features(cons_long)
    
    print(f"✓ Prepared data with {len(cons_long.columns)} columns")
    
    return cons_long, group_cols


def _load_simple_weather():
    """Load weather data from Excel files."""
    import glob
    
    weather_dir = PROJECT_ROOT / "Data"
    weather_files = list(weather_dir.glob("*.xlsx"))
    weather_files = [f for f in weather_files if "JUNCTION_training" not in f.name and "humidity" not in f.name.lower()]
    
    if not weather_files:
        return None
    
    all_weather = []
    for file_path in weather_files[:5]:
        try:
            df = pd.read_excel(file_path)
            required_cols = ["Vuosi", "Kuukausi", "Päivä", "Aika [Paikallinen aika]"]
            if not all(col in df.columns for col in required_cols):
                continue
            
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
            
            weather_cols_map = {
                "Lämpötilan keskiarvo [°C]": "temperature_c",
                "Keskituulen nopeus [m/s]": "wind_speed_ms",
                "Tunnin sademäärä [mm]": "precip_mm",
                "Suhteellisen kosteuden keskiarvo [%]": "humidity_pct"
            }
            
            for orig_col, new_col in weather_cols_map.items():
                if orig_col in df.columns:
                    series = df[orig_col].replace(["-", ""], np.nan)
                    weather_df[new_col] = pd.to_numeric(series, errors="coerce")
            
            weather_df = weather_df.dropna(subset=["measured_at"])
            all_weather.append(weather_df)
        except:
            continue
    
    if not all_weather:
        return None
    
    combined = pd.concat(all_weather, ignore_index=True)
    weather_cols = [c for c in combined.columns if c != "measured_at"]
    if weather_cols:
        averaged = combined.groupby("measured_at", as_index=False)[weather_cols].mean()
        return averaged
    return None


def _create_time_features(df):
    """Create time-based features."""
    df = df.copy()
    df["hour"] = df["measured_at"].dt.hour
    df["dayofweek"] = df["measured_at"].dt.dayofweek
    df["month"] = df["measured_at"].dt.month
    df["day"] = df["measured_at"].dt.day
    df["quarter"] = df["measured_at"].dt.quarter
    df["dayofyear"] = df["measured_at"].dt.dayofyear
    df["weekofyear"] = df["measured_at"].dt.isocalendar().week
    df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)
    
    # Cyclical encoding
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["dayofweek_sin"] = np.sin(2 * np.pi * df["dayofweek"] / 7)
    df["dayofweek_cos"] = np.cos(2 * np.pi * df["dayofweek"] / 7)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    
    return df


def _create_lag_features(df):
    """Create lag and rolling features per group."""
    df = df.sort_values(["group_id", "measured_at"]).reset_index(drop=True)
    
    # Lag features
    for lag in [1, 24, 168, 336]:
        df[f"load_lag_{lag}h"] = df.groupby("group_id")["load_mwh"].shift(lag)
    
    # Rolling features
    for window in [24, 168]:
        df[f"load_rolling_mean_{window}h"] = (
            df.groupby("group_id")["load_mwh"]
            .transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
        )
        df[f"load_rolling_std_{window}h"] = (
            df.groupby("group_id")["load_mwh"]
            .transform(lambda x: x.shift(1).rolling(window, min_periods=1).std())
        )
    
    # Price lags
    if "eur_per_mwh" in df.columns:
        for lag in [1, 24]:
            df[f"price_lag_{lag}h"] = df.groupby("group_id")["eur_per_mwh"].shift(lag)
    
    return df


def create_forecast_timestamps():
    """Create timestamps for October 1-2, 2024."""
    start = pd.Timestamp("2024-10-01 00:00:00", tz="UTC")
    timestamps = pd.date_range(start=start, periods=48, freq="h")
    return timestamps


def generate_forecast(models, feature_cols, historical_data, group_cols):
    """Generate 48-hour forecast for all groups."""
    print("\nGenerating forecasts...")
    
    timestamps = create_forecast_timestamps()
    
    # Initialize results DataFrame
    results = pd.DataFrame({"measured_at": timestamps})
    
    # For each group, generate forecast
    for group_id in sorted(models.keys()):
        print(f"  Forecasting group {group_id}...")
        
        # Get historical data for this group
        group_data = historical_data[historical_data["group_id"] == group_id].copy()
        group_data = group_data.sort_values("measured_at").reset_index(drop=True)
        
        # Generate forecast for each hour
        predictions = []
        
        for i, ts in enumerate(timestamps):
            # Create features for this timestamp
            forecast_row = _create_forecast_row(ts, group_data, i)
            
            # Ensure all required features are present
            for col in feature_cols:
                if col not in forecast_row:
                    forecast_row[col] = 0  # Default value
            
            # Make prediction
            X = pd.DataFrame([forecast_row])[feature_cols]
            
            # Fill NaN with median/0
            for col in X.columns:
                if X[col].isna().any():
                    median_val = group_data[col].median() if col in group_data.columns else 0
                    X[col].fillna(median_val if not pd.isna(median_val) else 0, inplace=True)
            
            pred = models[group_id].predict(X)[0]
            predictions.append(max(0, pred))  # Ensure non-negative
            
            # Update group_data with prediction for next iteration's lag features
            new_row = forecast_row.copy()
            new_row["measured_at"] = ts
            new_row["group_id"] = group_id
            new_row["load_mwh"] = pred
            group_data = pd.concat([group_data, pd.DataFrame([new_row])], ignore_index=True)
        
        results[str(group_id)] = predictions
    
    print(f"✓ Generated forecasts for {len(models)} groups")
    return results


def _create_forecast_row(timestamp, historical_data, hour_index):
    """Create feature row for a forecast timestamp."""
    row = {}
    
    # Time features
    row["hour"] = timestamp.hour
    row["dayofweek"] = timestamp.dayofweek
    row["month"] = timestamp.month
    row["day"] = timestamp.day
    row["quarter"] = timestamp.quarter
    row["dayofyear"] = timestamp.dayofyear
    row["weekofyear"] = timestamp.isocalendar().week
    row["is_weekend"] = int(timestamp.dayofweek >= 5)
    
    # Cyclical encoding
    row["hour_sin"] = np.sin(2 * np.pi * timestamp.hour / 24)
    row["hour_cos"] = np.cos(2 * np.pi * timestamp.hour / 24)
    row["dayofweek_sin"] = np.sin(2 * np.pi * timestamp.dayofweek / 7)
    row["dayofweek_cos"] = np.cos(2 * np.pi * timestamp.dayofweek / 7)
    row["month_sin"] = np.sin(2 * np.pi * timestamp.month / 12)
    row["month_cos"] = np.cos(2 * np.pi * timestamp.month / 12)
    
    # Get latest values from historical data
    if len(historical_data) > 0:
        latest = historical_data.iloc[-1]
        
        # Weather features (use latest or seasonal average)
        for col in ["temperature_c", "wind_speed_ms", "precip_mm", "humidity_pct"]:
            if col in historical_data.columns:
                # Use same month/hour average as estimate
                same_hour = historical_data[historical_data["hour"] == timestamp.hour]
                row[col] = same_hour[col].mean() if len(same_hour) > 0 and col in same_hour.columns else latest.get(col, 0)
        
        # Price (use latest or hourly average)
        if "eur_per_mwh" in historical_data.columns:
            same_hour_prices = historical_data[historical_data["hour"] == timestamp.hour]["eur_per_mwh"]
            row["eur_per_mwh"] = same_hour_prices.mean() if len(same_hour_prices) > 0 else latest.get("eur_per_mwh", 50)
        
        # Lag features (from historical data)
        lag_indices = {
            1: -1,
            24: -24,
            168: -168,
            336: -336
        }
        
        for lag, idx in lag_indices.items():
            if len(historical_data) >= abs(idx):
                row[f"load_lag_{lag}h"] = historical_data.iloc[idx]["load_mwh"]
        
        # Rolling features
        if len(historical_data) >= 24:
            recent_24 = historical_data.tail(24)["load_mwh"]
            row["load_rolling_mean_24h"] = recent_24.mean()
            row["load_rolling_std_24h"] = recent_24.std()
        
        if len(historical_data) >= 168:
            recent_168 = historical_data.tail(168)["load_mwh"]
            row["load_rolling_mean_168h"] = recent_168.mean()
            row["load_rolling_std_168h"] = recent_168.std()
        
        # Price lags
        if "eur_per_mwh" in historical_data.columns:
            if len(historical_data) >= 1:
                row["price_lag_1h"] = historical_data.iloc[-1]["eur_per_mwh"]
            if len(historical_data) >= 24:
                row["price_lag_24h"] = historical_data.iloc[-24]["eur_per_mwh"]
    
    # Holiday features (set to 0 for future dates - we don't have future holiday data in this simple version)
    for col in ["is_holiday", "is_holiday_eve", "is_holiday_after", "is_major_holiday_period"]:
        row[col] = 0
    row["days_to_next_holiday"] = 15  # Default
    row["days_since_last_holiday"] = 15  # Default
    
    return row


def save_forecast_fortum_format(results, filename="forecast_48h.csv"):
    """Save forecast in Fortum submission format."""
    output_path = OUTPUT_DIR / filename
    
    # Format timestamps in ISO 8601 with .000Z
    results_formatted = results.copy()
    results_formatted["measured_at"] = results_formatted["measured_at"].dt.strftime("%Y-%m-%dT%H:%M:%S.000Z")
    
    # Save with semicolon delimiter and comma decimal separator
    results_formatted.to_csv(
        output_path,
        sep=";",
        decimal=",",
        index=False,
        encoding="utf-8"
    )
    
    print(f"\n✓ Forecast saved to: {output_path}")
    print(f"  Format: Wide CSV, semicolon delimiter, comma decimal separator")
    print(f"  Shape: {results_formatted.shape[0]} rows × {results_formatted.shape[1]} columns")
    print(f"  Groups: {results_formatted.shape[1] - 1}")
    
    return output_path


def main():
    """Generate final 48-hour forecast."""
    print("="*60)
    print("GENERATING 48-HOUR FORECAST - FORTUM SUBMISSION FORMAT")
    print("="*60)
    
    # Load models
    models, feature_cols = load_models_and_features()
    
    # Prepare data
    historical_data, group_cols = prepare_forecast_data()
    
    # Generate forecast
    results = generate_forecast(models, feature_cols, historical_data, group_cols)
    
    # Save in Fortum format
    output_path = save_forecast_fortum_format(results)
    
    # Display summary
    print("\n" + "="*60)
    print("FORECAST SUMMARY")
    print("="*60)
    print(f"Period: October 1-2, 2024 (48 hours)")
    print(f"Groups forecasted: {len(models)}")
    print(f"\nSample predictions (first 5 hours, first 5 groups):")
    print(results.head().iloc[:, :6].to_string(index=False))
    
    print("\n" + "="*60)
    print("✓ FORECAST GENERATION COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()
