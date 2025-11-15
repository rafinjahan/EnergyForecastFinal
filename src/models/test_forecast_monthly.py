"""
Test monthly forecasting - first 12 months (monthly challenge).
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
from sklearn.metrics import mean_absolute_error, mean_squared_error

MODEL_DIR = PROJECT_ROOT / "models"


def load_trained_model():
    """Load the trained LightGBM model."""
    model_path = MODEL_DIR / "lgb_simple_model.pkl"
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}. Train it first with: python src/models/train_lightgbm.py")
    
    model = joblib.load(model_path)
    print(f"✓ Loaded model from {model_path}")
    return model


def prepare_monthly_test_data():
    """Prepare test data aggregated to monthly level."""
    print("\nLoading data...")
    data = load_fortum_training()
    cons = data["consumption"]
    prices = data["prices"]
    
    # Parse timestamps
    cons["measured_at"] = pd.to_datetime(cons["measured_at"])
    prices["measured_at"] = pd.to_datetime(prices["measured_at"])
    
    # Merge prices
    cons = cons.merge(prices[["measured_at", "eur_per_mwh"]], on="measured_at", how="left")
    
    # Add time columns
    cons["year"] = cons["measured_at"].dt.year
    cons["month"] = cons["measured_at"].dt.month
    
    # Get group columns
    exclude_cols = ["measured_at", "year", "month", "eur_per_mwh"]
    group_cols = [col for col in cons.columns if col not in exclude_cols]
    
    if not group_cols:
        raise ValueError("No group columns found")
    
    target_col = group_cols[0]
    print(f"Forecasting for group: {target_col}")
    
    # Aggregate to monthly level
    monthly_data = cons.groupby(["year", "month"]).agg({
        target_col: "sum",  # Total monthly consumption
        "eur_per_mwh": "mean",  # Average monthly price
        "measured_at": "min"  # First timestamp of month
    }).reset_index()
    
    # Sort by time
    monthly_data = monthly_data.sort_values("measured_at").reset_index(drop=True)
    
    # Take last 12 months as test set
    test_data = monthly_data.tail(12).copy()
    
    # Add features
    test_data["hour"] = 12  # Representative hour (noon)
    test_data["dayofweek"] = 2  # Representative day (Wednesday)
    test_data["day"] = 15  # Representative day of month (15th)
    
    print(f"✓ Aggregated to monthly level: {len(monthly_data)} months total")
    print(f"✓ Test period: {test_data['measured_at'].min()} to {test_data['measured_at'].max()}")
    
    return test_data, target_col


def forecast_12_months(model, test_data, target_col):
    """Generate 12-month forecast and evaluate."""
    # Prepare features
    feature_cols = ["hour", "dayofweek", "month", "day", "eur_per_mwh"]
    
    X_test = test_data[feature_cols]
    y_true = test_data[target_col]
    
    # Drop NaN rows
    mask = ~(y_true.isna() | X_test.isna().any(axis=1))
    X_test = X_test[mask]
    y_true = y_true[mask]
    timestamps = test_data.loc[mask, "measured_at"]
    years = test_data.loc[mask, "year"]
    months = test_data.loc[mask, "month"]
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
    
    print("\n" + "="*60)
    print("12-MONTH FORECAST EVALUATION (Last 12 months in data)")
    print("="*60)
    print(f"MAE:  {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAPE: {mape:.2f}%")
    print("="*60)
    
    # Create results DataFrame
    results = pd.DataFrame({
        "year": years.values,
        "month": months.values,
        "measured_at": timestamps.values,
        "actual": y_true.values,
        "predicted": y_pred,
        "error": y_true.values - y_pred
    })
    
    return results


def save_forecast(results, filename="forecast_12m_monthly.csv"):
    """Save forecast results."""
    output_path = PROJECT_ROOT / "models" / filename
    results.to_csv(output_path, index=False)
    print(f"\n✓ Forecast saved to: {output_path}")


def main():
    """Run 12-month forecast test."""
    print("="*60)
    print("MONTHLY FORECAST TEST: Last 12 months in dataset")
    print("="*60)
    
    # Load model
    model = load_trained_model()
    
    # Prepare test data
    test_data, target_col = prepare_monthly_test_data()
    
    # Generate forecast
    results = forecast_12_months(model, test_data, target_col)
    
    # Save results
    save_forecast(results)
    
    # Show all predictions
    print("\nMonthly predictions:")
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()
