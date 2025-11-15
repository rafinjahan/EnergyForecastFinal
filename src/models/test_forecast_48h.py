"""
Forecast the first 48 hours after September 30, 2024 (October 1-2, 2024).
This is a true forecasting task - predicting future values not in training data.
According to Fortum hackathon requirements.
"""
from __future__ import annotations
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import joblib

# Add src to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.fortum_loader import load_fortum_training

MODEL_DIR = PROJECT_ROOT / "models"


def get_all_group_columns():
    """Get all group columns from the data."""
    data = load_fortum_training()
    cons = data["consumption"]
    
    # Get group columns
    exclude_cols = ["measured_at"]
    group_cols = [col for col in cons.columns if col not in exclude_cols]
    
    return group_cols


def load_trained_model():
    """Load the trained LightGBM model."""
    model_path = MODEL_DIR / "lgb_simple_model.pkl"
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}. Train it first with: python src/models/train_lightgbm.py")
    
    model = joblib.load(model_path)
    print(f"✓ Loaded model from {model_path}")
    return model


def get_future_prices():
    """
    Get or estimate future prices for October 1-2, 2024.
    For now, we'll use the average price from the last week of September.
    In a real scenario, you'd have price forecasts or use a separate price model.
    """
    print("\nLoading historical data for price estimation...")
    data = load_fortum_training()
    prices = data["prices"]
    
    # Parse timestamps
    prices["measured_at"] = pd.to_datetime(prices["measured_at"])
    
    # Get last week of data for price estimation
    max_date = prices["measured_at"].max()
    last_week = prices[prices["measured_at"] >= max_date - pd.Timedelta(days=7)]
    
    # Calculate average price per hour of day
    last_week["hour"] = last_week["measured_at"].dt.hour
    hourly_avg_price = last_week.groupby("hour")["eur_per_mwh"].mean()
    
    print(f"✓ Estimated prices based on last week average (simple baseline)")
    return hourly_avg_price


def create_forecast_features():
    """Create features for October 1-2, 2024 (first 48 hours after Sep 30)."""
    # Generate hourly timestamps for October 1-2, 2024
    start_time = pd.Timestamp("2024-10-01 00:00:00", tz="UTC")
    timestamps = pd.date_range(start=start_time, periods=48, freq="h")
    
    # Create DataFrame with time features
    forecast_df = pd.DataFrame({
        "measured_at": timestamps,
        "hour": timestamps.hour,
        "dayofweek": timestamps.dayofweek,
        "month": timestamps.month,
        "day": timestamps.day
    })
    
    # Get estimated prices
    hourly_prices = get_future_prices()
    
    # Map prices to forecast hours
    forecast_df["eur_per_mwh"] = forecast_df["hour"].map(hourly_prices)
    
    # Fill any missing prices with overall average
    if forecast_df["eur_per_mwh"].isna().any():
        avg_price = hourly_prices.mean()
        forecast_df["eur_per_mwh"].fillna(avg_price, inplace=True)
    
    print(f"✓ Created features for {len(forecast_df)} hours (Oct 1-2, 2024)")
    return forecast_df


def generate_forecast(model, forecast_df):
    """Generate 48-hour forecast for October 1-2, 2024 for all groups."""
    # Prepare features
    feature_cols = ["hour", "dayofweek", "month", "day", "eur_per_mwh"]
    X_forecast = forecast_df[feature_cols]
    
    # Get all group columns
    group_cols = get_all_group_columns()
    
    # Make predictions (currently model is trained on one group, so all groups get same prediction)
    # Note: In future, you should train separate models for each group
    predictions = model.predict(X_forecast)
    
    # Create base results DataFrame
    results = pd.DataFrame({
        "measured_at": forecast_df["measured_at"],
        "hour": forecast_df["hour"],
        "estimated_price_eur_mwh": forecast_df["eur_per_mwh"]
    })
    
    # Add predictions for each group
    # For now, using same model prediction for all groups (this is a limitation)
    for group in group_cols:
        results[f"group_{group}_predicted"] = predictions
    
    print("\n" + "="*60)
    print("48-HOUR FORECAST: October 1-2, 2024")
    print("="*60)
    print(f"Number of groups forecasted: {len(group_cols)}")
    print(f"Groups: {', '.join(map(str, group_cols))}")
    print(f"\nPer-group predictions (using single model):")
    print(f"Total predicted consumption: {predictions.sum():.2f}")
    print(f"Average hourly consumption: {predictions.mean():.2f}")
    print(f"Min hourly consumption: {predictions.min():.2f}")
    print(f"Max hourly consumption: {predictions.max():.2f}")
    print("="*60)
    
    return results, group_cols


def save_forecast(results, group_cols, filename="forecast_48h_october_2024.csv"):
    """Save forecast results in clean long format."""
    # Convert wide format to long format for cleaner CSV
    id_cols = ["measured_at", "hour", "estimated_price_eur_mwh"]
    
    # Melt the dataframe to long format
    long_df = results.melt(
        id_vars=id_cols,
        var_name="group",
        value_name="predicted_consumption"
    )
    
    # Clean up group names (remove "group_" prefix and "_predicted" suffix)
    long_df["group"] = long_df["group"].str.replace("group_", "").str.replace("_predicted", "")
    
    # Reorder columns
    long_df = long_df[["measured_at", "hour", "group", "predicted_consumption", "estimated_price_eur_mwh"]]
    
    # Sort by timestamp and group
    long_df = long_df.sort_values(["measured_at", "group"]).reset_index(drop=True)
    
    output_path = PROJECT_ROOT / "models" / filename
    long_df.to_csv(output_path, index=False)
    print(f"\n✓ Forecast saved to: {output_path}")
    print(f"  Format: Long format with {len(long_df)} rows (48 hours × {len(group_cols)} groups)")
    print(f"  Columns: measured_at, hour, group, predicted_consumption, estimated_price_eur_mwh")


def main():
    """Run 48-hour forecast for October 1-2, 2024."""
    print("="*60)
    print("GENERATING FORECAST: October 1-2, 2024 (48 hours)")
    print("="*60)
    
    # Load model
    model = load_trained_model()
    
    # Create forecast features
    forecast_df = create_forecast_features()
    
    # Generate forecast
    results, group_cols = generate_forecast(model, forecast_df)
    
    # Save results
    save_forecast(results, group_cols)
    
    # Show predictions for each group
    print("\n" + "="*60)
    print("FORECAST BY GROUP (First 12 hours)")
    print("="*60)
    for group in group_cols:
        group_col = f"group_{group}_predicted"
        total = results[group_col].sum()
        avg = results[group_col].mean()
        print(f"\nGroup {group}:")
        print(f"  48-hour total: {total:.2f}")
        print(f"  Hourly average: {avg:.2f}")
        print(f"  First 12 hours: {results[group_col].head(12).values}")
    
    print("\n" + "="*60)
    print("DETAILED HOURLY FORECAST (First 24 hours)")
    print("="*60)
    print(results.head(24).to_string(index=False))
    
    print("\n" + "="*60)
    print("DETAILED HOURLY FORECAST (Last 24 hours)")
    print("="*60)
    print(results.tail(24).to_string(index=False))


if __name__ == "__main__":
    main()
