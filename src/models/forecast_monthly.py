"""
Generate 12-month forecasts using trained LightGBM models.
Outputs predictions matching the example monthly file structure.
"""
from __future__ import annotations
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
from typing import Dict, Tuple
import warnings
warnings.filterwarnings('ignore')

# Add src to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.fortum_loader import load_fortum_training
from src.data.weather_enrichment import enrich_consumption_with_all_weather

# Paths
MODEL_DIR = PROJECT_ROOT / "models" / "lgb_monthly"
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


def prepare_forecast_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Prepare historical data for forecasting."""
    print("\nPreparing historical data...")
    
    # Load data
    data = load_fortum_training()
    cons = data["consumption"]
    groups = data["groups"]
    prices = data["prices"]
    
    # Parse timestamps
    cons["measured_at"] = pd.to_datetime(cons["measured_at"], utc=True)
    prices["measured_at"] = pd.to_datetime(prices["measured_at"], utc=True)
    
    # Try to enrich with weather
    try:
        enriched = enrich_consumption_with_all_weather(cons, groups)
        enriched = enriched.merge(
            prices[["measured_at", "eur_per_mwh"]],
            on="measured_at",
            how="left"
        )
    except Exception as e:
        print(f"⚠ Weather enrichment failed: {e}")
        
        # Convert to long format
        group_cols = [col for col in cons.columns if col != "measured_at"]
        enriched = cons.melt(
            id_vars="measured_at",
            value_vars=group_cols,
            var_name="group_id",
            value_name="load_mwh"
        )
        enriched["group_id"] = enriched["group_id"].astype(int)
        enriched = enriched.merge(
            prices[["measured_at", "eur_per_mwh"]],
            on="measured_at",
            how="left"
        )
    
    print(f"✓ Prepared historical data: {len(enriched)} records")
    return enriched, groups, prices


def aggregate_to_monthly(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate hourly data to monthly totals."""
    df = df.copy()
    
    # Create period for grouping
    df["month_period"] = df["measured_at"].dt.to_period("M")
    df["year"] = df["measured_at"].dt.year
    df["month"] = df["measured_at"].dt.month
    
    # Aggregation dict
    agg_dict = {
        "load_mwh": "sum",
        "measured_at": "min",
    }
    
    # Add weather aggregations if available
    weather_cols = ["temperature_c", "wind_speed_ms", "precip_mm", "humidity_pct"]
    for col in weather_cols:
        if col in df.columns:
            agg_dict[col] = "mean"
    
    # Add price aggregation
    if "eur_per_mwh" in df.columns:
        agg_dict["eur_per_mwh"] = "mean"
    
    # Group by group_id and month
    monthly = df.groupby(["group_id", "month_period"], as_index=False).agg(agg_dict)
    
    # Add back year and month
    monthly["year"] = monthly["month_period"].dt.year
    monthly["month"] = monthly["month_period"].dt.month
    
    # Sort
    monthly = monthly.sort_values(["group_id", "month_period"]).reset_index(drop=True)
    
    return monthly


def engineer_monthly_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create comprehensive features for monthly forecasting."""
    df = df.copy()
    
    # Cyclical encoding for month
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    
    # Quarter
    df["quarter"] = df["month_period"].dt.quarter
    
    # Lag features (per group)
    for lag in [1, 2, 3, 6, 12]:
        df[f"load_lag_{lag}m"] = df.groupby("group_id")["load_mwh"].shift(lag)
    
    # Rolling statistics (per group)
    for window in [3, 6, 12]:
        df[f"load_rolling_mean_{window}m"] = (
            df.groupby("group_id")["load_mwh"]
            .transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
        )
        df[f"load_rolling_std_{window}m"] = (
            df.groupby("group_id")["load_mwh"]
            .transform(lambda x: x.shift(1).rolling(window, min_periods=1).std())
        )
    
    # Year-over-year change
    df["load_yoy_change"] = (
        df.groupby("group_id")["load_mwh"].pct_change(12) * 100
    )
    
    return df


def create_forecast_template(historical_monthly: pd.DataFrame, n_months: int = 12) -> pd.DataFrame:
    """Create template for next N months after the historical data."""
    # Get the last month in historical data
    max_period = historical_monthly["month_period"].max()
    
    # Create future months
    future_periods = [max_period + i for i in range(1, n_months + 1)]
    
    # Get all unique group IDs
    group_ids = sorted(historical_monthly["group_id"].unique())
    
    # Create template
    template_rows = []
    for period in future_periods:
        for group_id in group_ids:
            template_rows.append({
                "group_id": group_id,
                "month_period": period,
                "year": period.year,
                "month": period.month,
                "measured_at": period.to_timestamp(),
                "load_mwh": np.nan  # To be predicted
            })
    
    template_df = pd.DataFrame(template_rows)
    
    print(f"✓ Created forecast template for {n_months} months, {len(group_ids)} groups")
    return template_df


def estimate_future_monthly_prices(prices: pd.DataFrame, forecast_periods: pd.Series) -> pd.Series:
    """Estimate average monthly prices for future periods."""
    prices = prices.copy()
    prices["measured_at"] = pd.to_datetime(prices["measured_at"], utc=True)
    prices["month_period"] = prices["measured_at"].dt.to_period("M")
    
    # Calculate historical monthly average prices
    monthly_prices = prices.groupby("month_period")["eur_per_mwh"].mean()
    
    # Calculate average price per month of year
    prices["month"] = prices["measured_at"].dt.month
    month_avg_prices = prices.groupby("month")["eur_per_mwh"].mean()
    
    # Estimate future prices based on month of year pattern
    estimated_prices = []
    for period in forecast_periods:
        month = period.month
        if month in month_avg_prices.index:
            estimated_prices.append(month_avg_prices[month])
        else:
            estimated_prices.append(monthly_prices.mean())
    
    return pd.Series(estimated_prices, index=forecast_periods.index)


def generate_forecasts(
    models: Dict[int, any],
    feature_cols: list,
    historical_monthly: pd.DataFrame,
    forecast_template: pd.DataFrame,
    prices: pd.DataFrame
) -> pd.DataFrame:
    """Generate monthly forecasts for all groups."""
    print("\nGenerating monthly forecasts...")
    
    # Estimate future prices
    forecast_template["eur_per_mwh"] = estimate_future_monthly_prices(
        prices, forecast_template["month_period"]
    )
    
    all_forecasts = []
    
    for group_id in sorted(models.keys()):
        # Get historical data for this group
        group_hist = historical_monthly[historical_monthly["group_id"] == group_id].copy()
        
        # Get forecast template for this group
        group_forecast = forecast_template[forecast_template["group_id"] == group_id].copy()
        
        # Combine historical and forecast data
        combined = pd.concat([group_hist, group_forecast], ignore_index=True)
        combined = combined.sort_values("month_period").reset_index(drop=True)
        
        # Engineer features
        combined = engineer_monthly_features(combined)
        
        # Get forecast rows
        forecast_mask = combined["month_period"] >= forecast_template["month_period"].min()
        forecast_data = combined[forecast_mask].copy()
        
        # Prepare features
        available_features = [col for col in feature_cols if col in forecast_data.columns]
        X_forecast = forecast_data[available_features]
        
        # Handle missing features
        for col in feature_cols:
            if col not in X_forecast.columns:
                X_forecast[col] = 0
        
        # Reorder to match training
        X_forecast = X_forecast[feature_cols]
        
        # Make predictions
        predictions = models[group_id].predict(X_forecast)
        
        # Store results
        forecast_result = pd.DataFrame({
            "group_id": group_id,
            "year": forecast_data["year"].values,
            "month": forecast_data["month"].values,
            "month_period": forecast_data["month_period"].values,
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
    """Format forecasts to wide format matching example monthly file."""
    # Create a period column for pivoting
    forecasts_df["period_str"] = forecasts_df["year"].astype(str) + "-" + forecasts_df["month"].astype(str).str.zfill(2)
    
    # Pivot to wide format
    wide_df = forecasts_df.pivot(
        index="period_str",
        columns="group_id",
        values="predicted_load"
    )
    
    # Sort columns by group ID
    wide_df = wide_df[sorted(wide_df.columns)]
    
    # Reset index
    wide_df = wide_df.reset_index()
    wide_df = wide_df.rename(columns={"period_str": "year_month"})
    
    return wide_df


def save_forecast(wide_df: pd.DataFrame, filename: str = "forecast_12m_monthly.csv"):
    """Save monthly forecast."""
    output_path = OUTPUT_DIR / filename
    
    # Save with semicolon separator
    wide_df.to_csv(output_path, index=False, sep=";", float_format="%.9f")
    
    print(f"\n✓ Forecast saved to: {output_path}")
    print(f"  Format: Wide format with {len(wide_df)} months × {len(wide_df.columns)-1} groups")
    print(f"  Separator: semicolon (;)")


def main():
    """Generate 12-month forecast."""
    print("="*60)
    print("GENERATING 12-MONTH FORECAST")
    print("="*60)
    
    # 1. Load trained models
    models, feature_cols = load_models()
    
    # 2. Prepare historical data
    historical_data, groups, prices = prepare_forecast_data()
    
    # 3. Aggregate to monthly
    print("\nAggregating to monthly level...")
    historical_monthly = aggregate_to_monthly(historical_data)
    print(f"✓ Aggregated to {len(historical_monthly)} monthly records")
    
    # 4. Engineer features on historical data
    print("\nEngineering features on historical data...")
    historical_monthly = engineer_monthly_features(historical_monthly)
    
    # 5. Create forecast template
    forecast_template = create_forecast_template(historical_monthly, n_months=12)
    
    # 6. Generate forecasts
    forecasts_df = generate_forecasts(
        models, feature_cols, historical_monthly, forecast_template, prices
    )
    
    # 7. Format to wide format
    wide_df = format_output_wide(forecasts_df)
    
    # 8. Save forecast
    save_forecast(wide_df)
    
    # 9. Display summary
    print("\n" + "="*60)
    print("FORECAST SUMMARY")
    print("="*60)
    print(f"Forecast period: Next 12 months after training data")
    print(f"Groups forecasted: {len(models)}")
    print(f"Total predictions: {len(forecasts_df)}")
    print(f"\nSample predictions:")
    print(wide_df.head(12).to_string(index=False, max_cols=10))
    
    print("\n" + "="*60)
    print("FORECAST GENERATION COMPLETE!")
    print("="*60)


if __name__ == "__main__":
    main()
