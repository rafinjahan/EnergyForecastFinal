"""
Generate 48-hour forecasts for October 1-2, 2024 using trained LightGBM models.
Outputs predictions in wide format matching the example hourly file structure.
"""
from __future__ import annotations
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
from typing import Dict
import warnings
warnings.filterwarnings('ignore')

# Add src to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.fortum_loader import load_fortum_training
from src.data.weather_enrichment import enrich_consumption_with_all_weather

# Paths
MODEL_DIR = PROJECT_ROOT / "models" / "lgb_48h"
OUTPUT_DIR = PROJECT_ROOT / "models"


def load_models() -> Tuple[Dict[int, any], list]:
    """Load trained models and feature columns."""
    print("Loading trained models...")
    
    if not MODEL_DIR.exists():
        raise FileNotFoundError(f"Model directory not found: {MODEL_DIR}")
    
    # Load feature columns
    feature_path = MODEL_DIR / "feature_columns.pkl"
    if not feature_path.exists():
        raise FileNotFoundError(f"Feature columns not found: {feature_path}")
    
    feature_cols = joblib.load(feature_path)
    print(f"✓ Loaded feature columns: {len(feature_cols)} features")
    
    # Load all group models
    models = {}
    for model_file in sorted(MODEL_DIR.glob("group_*.pkl")):
        group_id = int(model_file.stem.split("_")[1])
        models[group_id] = joblib.load(model_file)
    
    print(f"✓ Loaded {len(models)} trained models")
    return models, feature_cols


def prepare_forecast_data() -> pd.DataFrame:
    """Prepare data for October 1-2, 2024 forecast."""
    print("\nPreparing forecast data for October 1-2, 2024...")
    
    # Load historical data
    data = load_fortum_training()
    cons = data["consumption"]
    groups = data["groups"]
    prices = data["prices"]
    
    # Parse timestamps
    cons["measured_at"] = pd.to_datetime(cons["measured_at"], utc=True)
    prices["measured_at"] = pd.to_datetime(prices["measured_at"], utc=True)
    
    # Try to enrich with weather data
    try:
        enriched = enrich_consumption_with_all_weather(cons, groups)
        # Merge prices
        enriched = enriched.merge(
            prices[["measured_at", "eur_per_mwh"]],
            on="measured_at",
            how="left"
        )
    except Exception as e:
        print(f"⚠ Weather enrichment failed: {e}")
        print("Using data without weather features...")
        
        # Convert to long format
        group_cols = [col for col in cons.columns if col != "measured_at"]
        enriched = cons.melt(
            id_vars="measured_at",
            value_vars=group_cols,
            var_name="group_id",
            value_name="load_mwh"
        )
        enriched["group_id"] = enriched["group_id"].astype(int)
        
        # Merge prices
        enriched = enriched.merge(
            prices[["measured_at", "eur_per_mwh"]],
            on="measured_at",
            how="left"
        )
    
    print(f"✓ Prepared historical data: {len(enriched)} records")
    return enriched, groups, prices


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create comprehensive features (same as training)."""
    df = df.copy()
    
    # Time features
    local_time = df["measured_at"].dt.tz_convert("Europe/Helsinki")
    
    df["hour"] = local_time.dt.hour
    df["dayofweek"] = local_time.dt.dayofweek
    df["month"] = local_time.dt.month
    df["day"] = local_time.dt.day
    df["weekofyear"] = local_time.dt.isocalendar().week
    df["is_weekend"] = (local_time.dt.dayofweek >= 5).astype(int)
    
    # Cyclical encoding
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    
    # Sort by group and time for lag features
    df = df.sort_values(["group_id", "measured_at"]).reset_index(drop=True)
    
    # Lag features (per group)
    for lag in [24, 48, 168]:
        df[f"load_lag_{lag}h"] = df.groupby("group_id")["load_mwh"].shift(lag)
    
    # Rolling statistics (per group)
    for window in [24, 168]:
        df[f"load_rolling_mean_{window}h"] = (
            df.groupby("group_id")["load_mwh"]
            .transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
        )
        df[f"load_rolling_std_{window}h"] = (
            df.groupby("group_id")["load_mwh"]
            .transform(lambda x: x.shift(1).rolling(window, min_periods=1).std())
        )
    
    return df


def create_forecast_template() -> pd.DataFrame:
    """Create template for October 1-2, 2024 (48 hours)."""
    start_time = pd.Timestamp("2024-10-01 00:00:00", tz="UTC")
    timestamps = pd.date_range(start=start_time, periods=48, freq="h")
    
    forecast_template = pd.DataFrame({"measured_at": timestamps})
    
    print(f"✓ Created forecast template for {len(forecast_template)} hours")
    return forecast_template


def estimate_future_prices(prices: pd.DataFrame, forecast_timestamps: pd.Series) -> pd.Series:
    """Estimate prices for future timestamps based on historical patterns."""
    prices = prices.copy()
    prices["measured_at"] = pd.to_datetime(prices["measured_at"], utc=True)
    
    # Get last 4 weeks for price estimation
    max_date = prices["measured_at"].max()
    recent_prices = prices[prices["measured_at"] >= max_date - pd.Timedelta(days=28)]
    
    # Calculate average price per hour of day and day of week
    recent_prices["hour"] = recent_prices["measured_at"].dt.hour
    recent_prices["dayofweek"] = recent_prices["measured_at"].dt.dayofweek
    
    hourly_avg = recent_prices.groupby(["dayofweek", "hour"])["eur_per_mwh"].mean()
    
    # Map to forecast timestamps
    forecast_hours = forecast_timestamps.dt.hour
    forecast_days = forecast_timestamps.dt.dayofweek
    
    estimated_prices = []
    for day, hour in zip(forecast_days, forecast_hours):
        if (day, hour) in hourly_avg.index:
            estimated_prices.append(hourly_avg.loc[(day, hour)])
        else:
            # Fallback to overall average for that hour
            hour_avg = recent_prices[recent_prices["hour"] == hour]["eur_per_mwh"].mean()
            estimated_prices.append(hour_avg if not np.isnan(hour_avg) else recent_prices["eur_per_mwh"].mean())
    
    return pd.Series(estimated_prices, index=forecast_timestamps.index)


def generate_forecasts(
    models: Dict[int, any],
    feature_cols: list,
    historical_data: pd.DataFrame,
    forecast_template: pd.DataFrame,
    prices: pd.DataFrame
) -> pd.DataFrame:
    """Generate forecasts for all groups."""
    print("\nGenerating forecasts...")
    
    # Estimate future prices
    forecast_template["eur_per_mwh"] = estimate_future_prices(
        prices, forecast_template["measured_at"]
    )
    
    all_forecasts = []
    
    for group_id in sorted(models.keys()):
        # Get historical data for this group
        group_hist = historical_data[historical_data["group_id"] == group_id].copy()
        
        # Create forecast rows for this group
        group_forecast = forecast_template.copy()
        group_forecast["group_id"] = group_id
        group_forecast["load_mwh"] = np.nan  # Will be predicted
        
        # Combine historical and forecast data
        combined = pd.concat([group_hist, group_forecast], ignore_index=True)
        combined = combined.sort_values("measured_at").reset_index(drop=True)
        
        # Engineer features
        combined = engineer_features(combined)
        
        # Get forecast rows
        forecast_mask = combined["measured_at"] >= forecast_template["measured_at"].min()
        forecast_data = combined[forecast_mask].copy()
        
        # Prepare features
        available_features = [col for col in feature_cols if col in forecast_data.columns]
        X_forecast = forecast_data[available_features]
        
        # Handle missing features
        for col in feature_cols:
            if col not in X_forecast.columns:
                X_forecast[col] = 0  # Default value for missing features
        
        # Reorder to match training
        X_forecast = X_forecast[feature_cols]
        
        # Make predictions
        predictions = models[group_id].predict(X_forecast)
        
        # Store results
        forecast_result = pd.DataFrame({
            "measured_at": forecast_data["measured_at"].values,
            "group_id": group_id,
            "predicted_load": predictions
        })
        
        all_forecasts.append(forecast_result)
        
        if (group_id - min(models.keys())) % 20 == 0:
            print(f"  Generated forecast for group {group_id}")
    
    # Combine all forecasts
    forecasts_df = pd.concat(all_forecasts, ignore_index=True)
    
    print(f"✓ Generated forecasts for {len(models)} groups")
    return forecasts_df


def format_output_wide(forecasts_df: pd.DataFrame) -> pd.DataFrame:
    """Format forecasts to wide format matching the example file."""
    # Pivot to wide format
    wide_df = forecasts_df.pivot(
        index="measured_at",
        columns="group_id",
        values="predicted_load"
    )
    
    # Sort columns by group ID
    wide_df = wide_df[sorted(wide_df.columns)]
    
    # Reset index to make measured_at a column
    wide_df = wide_df.reset_index()
    
    # Format timestamp to match example (ISO format with milliseconds)
    wide_df["measured_at"] = wide_df["measured_at"].dt.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
    
    return wide_df


def save_forecast(wide_df: pd.DataFrame, filename: str = "forecast_48h_october_2024.csv"):
    """Save forecast in wide format."""
    output_path = OUTPUT_DIR / filename
    
    # Save with semicolon separator to match example format
    wide_df.to_csv(output_path, index=False, sep=";", float_format="%.9f")
    
    print(f"\n✓ Forecast saved to: {output_path}")
    print(f"  Format: Wide format with {len(wide_df)} hours × {len(wide_df.columns)-1} groups")
    print(f"  Separator: semicolon (;)")


def main():
    """Generate 48-hour forecast for October 1-2, 2024."""
    print("="*60)
    print("GENERATING 48-HOUR FORECAST: October 1-2, 2024")
    print("="*60)
    
    # 1. Load trained models
    models, feature_cols = load_models()
    
    # 2. Prepare historical data
    historical_data, groups, prices = prepare_forecast_data()
    
    # 3. Create forecast template
    forecast_template = create_forecast_template()
    
    # 4. Generate forecasts
    forecasts_df = generate_forecasts(
        models, feature_cols, historical_data, forecast_template, prices
    )
    
    # 5. Format to wide format
    wide_df = format_output_wide(forecasts_df)
    
    # 6. Save forecast
    save_forecast(wide_df)
    
    # 7. Display summary
    print("\n" + "="*60)
    print("FORECAST SUMMARY")
    print("="*60)
    print(f"Period: October 1-2, 2024 (48 hours)")
    print(f"Groups forecasted: {len(models)}")
    print(f"Total predictions: {len(forecasts_df)}")
    print(f"\nSample predictions (first 5 hours):")
    print(wide_df.head(5).to_string(index=False, max_cols=10))
    
    print("\n" + "="*60)
    print("FORECAST GENERATION COMPLETE!")
    print("="*60)


if __name__ == "__main__":
    from typing import Tuple
    main()
