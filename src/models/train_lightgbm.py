"""
Simple LightGBM model for energy consumption forecasting.
Uses fortum_loader to load data.
"""
from __future__ import annotations
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib

# Add src to path for imports
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.fortum_loader import load_fortum_training

# Paths
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)


def load_and_prepare_data():
    """Load Fortum data using fortum_loader."""
    print("Loading data using fortum_loader...")
    
    data = load_fortum_training()
    cons = data["consumption"]
    groups = data["groups"]
    prices = data["prices"]
    
    # Parse measured_at timestamp for consumption
    if not pd.api.types.is_datetime64_any_dtype(cons["measured_at"]):
        cons["measured_at"] = pd.to_datetime(cons["measured_at"])
    
    # Parse measured_at timestamp for prices
    if not pd.api.types.is_datetime64_any_dtype(prices["measured_at"]):
        prices["measured_at"] = pd.to_datetime(prices["measured_at"])
    
    # Merge prices into consumption data
    cons = cons.merge(prices[["measured_at", "eur_per_mwh"]], on="measured_at", how="left")
    
    # Add simple time features
    cons["hour"] = cons["measured_at"].dt.hour
    cons["dayofweek"] = cons["measured_at"].dt.dayofweek
    cons["month"] = cons["measured_at"].dt.month
    cons["day"] = cons["measured_at"].dt.day
    
    print(f"Loaded {len(cons)} consumption records")
    print(f"Columns: {list(cons.columns)}")
    return cons, groups


def time_based_split(df: pd.DataFrame, train_ratio: float = 0.8):
    """Split data by time - last 20% for validation."""
    df = df.sort_values("measured_at").reset_index(drop=True)
    split_idx = int(len(df) * train_ratio)
    
    train_df = df.iloc[:split_idx]
    val_df = df.iloc[split_idx:]
    
    print(f"\nTrain: {train_df['measured_at'].min()} to {train_df['measured_at'].max()}")
    print(f"Val:   {val_df['measured_at'].min()} to {val_df['measured_at'].max()}")
    print(f"Train size: {len(train_df)}, Val size: {len(val_df)}")
    
    return train_df, val_df


def prepare_features(df: pd.DataFrame):
    """Prepare X and y for training.
    
    The consumption data is in wide format: measured_at + group columns (28, 29, 30, etc.)
    We'll melt it to long format and use one group for now.
    """
    # Get all consumption columns (everything except measured_at, time features, and price)
    exclude_cols = ["measured_at", "hour", "dayofweek", "month", "day", "eur_per_mwh"]
    group_cols = [col for col in df.columns if col not in exclude_cols]
    
    # For simplicity, pick the first group column
    if not group_cols:
        raise ValueError("No group columns found in data")
    
    target_col = group_cols[0]  # Use first group
    print(f"\nUsing group column '{target_col}' as target")
    
    # Features: time-based + price
    feature_cols = ["hour", "dayofweek", "month", "day", "eur_per_mwh"]
    available_features = [col for col in feature_cols if col in df.columns]
    
    X = df[available_features]
    y = df[target_col]
    
    # Drop rows with missing target or feature values
    mask = ~(y.isna() | X.isna().any(axis=1))
    X = X[mask]
    y = y[mask]
    
    print(f"Features: {list(X.columns)}")
    print(f"After dropping NaN: {len(X)} samples")
    
    return X, y


def train_simple_model(X_train, y_train, X_val, y_val):
    """Train a simple LightGBM model."""
    print("\nTraining LightGBM model...")
    
    model = lgb.LGBMRegressor(
        objective="regression",
        n_estimators=500,
        learning_rate=0.1,
        max_depth=5,
        num_leaves=31,
        n_jobs=-1,
        random_state=42,
        verbose=-1,
    )
    
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="mae",
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=True)],
    )
    
    print("✓ Training complete!")
    return model


def evaluate_model(model, X_val, y_val):
    """Evaluate model performance."""
    y_pred = model.predict(X_val)
    
    mae = mean_absolute_error(y_val, y_pred)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    
    # Avoid division by zero in MAPE
    mask = y_val > 0
    mape = np.mean(np.abs((y_val[mask] - y_pred[mask]) / y_val[mask])) * 100
    
    print("\n" + "="*60)
    print("VALIDATION METRICS")
    print("="*60)
    print(f"MAE:  {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAPE: {mape:.2f}%")
    print("="*60)
    
    return {"mae": mae, "rmse": rmse, "mape": mape}


def save_model(model, metrics, model_name="lgb_simple_model"):
    """Save model and metrics."""
    model_path = MODEL_DIR / f"{model_name}.pkl"
    
    joblib.dump(model, model_path)
    print(f"\n✓ Model saved: {model_path}")
    
    # Save metrics
    metrics_path = MODEL_DIR / f"{model_name}_metrics.txt"
    with open(metrics_path, "w") as f:
        f.write("Validation Metrics\n")
        f.write("="*40 + "\n")
        for k, v in metrics.items():
            f.write(f"{k.upper()}: {v:.4f}\n")
    print(f"✓ Metrics saved: {metrics_path}")


def main():
    """Simple training pipeline."""
    # 1. Load data
    cons, groups = load_and_prepare_data()
    
    # 2. Time-based split
    train_df, val_df = time_based_split(cons, train_ratio=0.8)
    
    # 3. Prepare features
    X_train, y_train = prepare_features(train_df)
    X_val, y_val = prepare_features(val_df)
    
    print(f"\nFeatures: {list(X_train.columns)}")
    print(f"Train X: {X_train.shape}, y: {y_train.shape}")
    print(f"Val X:   {X_val.shape}, y: {y_val.shape}")
    
    # 4. Train
    model = train_simple_model(X_train, y_train, X_val, y_val)
    
    # 5. Evaluate
    metrics = evaluate_model(model, X_val, y_val)
    
    # 6. Save
    save_model(model, metrics)
    
    # 7. Feature importance
    feature_imp = pd.DataFrame({
        "feature": X_train.columns,
        "importance": model.feature_importances_
    }).sort_values("importance", ascending=False)
    
    print("\nFeature Importance:")
    print(feature_imp.to_string(index=False))


if __name__ == "__main__":
    main()
