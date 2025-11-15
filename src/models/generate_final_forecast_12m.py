"""
Generate 12-month forecast in Fortum submission format.
Format: Wide CSV with semicolon delimiter, comma decimal separator.
October 2024 - September 2025 (12 months).
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
from src.data.finnish_holidays import add_holiday_features

MODEL_DIR = PROJECT_ROOT / "models" / "lgb_monthly"
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


def prepare_historical_monthly_data():
    """Prepare historical monthly aggregated data."""
    print("\nPreparing historical monthly data...")
    
    # Load historical data
    data = load_fortum_training()
    cons = data["consumption"]
    prices = data["prices"]
    groups = data["groups"]
    
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
    
    # Aggregate to monthly
    print("Aggregating to monthly...")
    monthly = _aggregate_to_monthly(cons_long)
    
    # Engineer features
    print("Engineering features...")
    monthly = _engineer_monthly_features(monthly, groups)
    
    print(f"✓ Prepared historical data: {len(monthly)} monthly records")
    
    return monthly, groups


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


def _aggregate_to_monthly(df):
    """Aggregate hourly data to monthly."""
    df = df.copy()
    df["month_period"] = df["measured_at"].dt.to_period("M")
    df["year"] = df["measured_at"].dt.year
    df["month"] = df["measured_at"].dt.month
    
    agg_dict = {
        "load_mwh": "sum",
        "measured_at": "min",
    }
    
    weather_cols = ["temperature_c", "wind_speed_ms", "precip_mm", "humidity_pct"]
    for col in weather_cols:
        if col in df.columns:
            agg_dict[col] = "mean"
    
    if "eur_per_mwh" in df.columns:
        agg_dict["eur_per_mwh"] = "mean"
    
    holiday_cols = ["is_holiday", "is_holiday_eve", "is_holiday_after", "is_major_holiday_period"]
    for col in holiday_cols:
        if col in df.columns:
            agg_dict[col] = "sum"
    
    if "days_to_next_holiday" in df.columns:
        agg_dict["days_to_next_holiday"] = "mean"
    if "days_since_last_holiday" in df.columns:
        agg_dict["days_since_last_holiday"] = "mean"
    
    monthly = df.groupby(["group_id", "month_period"], as_index=False).agg(agg_dict)
    monthly["year"] = monthly["month_period"].dt.year
    monthly["month"] = monthly["month_period"].dt.month
    monthly = monthly.sort_values(["group_id", "month_period"]).reset_index(drop=True)
    
    return monthly


def _engineer_monthly_features(df, groups):
    """Create monthly features."""
    from sklearn.preprocessing import LabelEncoder
    
    df = df.copy()
    
    # Time features
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    
    # Sort for lags
    df = df.sort_values(["group_id", "month_period"]).reset_index(drop=True)
    
    # Lag features
    df["load_lag_1m"] = df.groupby("group_id")["load_mwh"].shift(1)
    df["load_lag_12m"] = df.groupby("group_id")["load_mwh"].shift(12)
    
    # Price features
    if "eur_per_mwh" in df.columns:
        df["price_ma_3m"] = (
            df.groupby("group_id")["eur_per_mwh"]
            .transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
        )
        df["price_ma_6m"] = (
            df.groupby("group_id")["eur_per_mwh"]
            .transform(lambda x: x.shift(1).rolling(6, min_periods=1).mean())
        )
        df["price_volatility_3m"] = (
            df.groupby("group_id")["eur_per_mwh"]
            .transform(lambda x: x.shift(1).rolling(3, min_periods=1).std())
        )
        df["price_volatility_6m"] = (
            df.groupby("group_id")["eur_per_mwh"]
            .transform(lambda x: x.shift(1).rolling(6, min_periods=1).std())
        )
    
    # Group features
    df = df.merge(groups[["group_id", "group_label"]], on="group_id", how="left")
    
    df["region"] = df["group_label"].str.split("|").str[0]
    df["province"] = df["group_label"].str.split("|").str[1]
    df["contract_type"] = df["group_label"].str.split("|").str[2]
    df["consumption_level"] = df["group_label"].str.split("|").str[3]
    
    for col in ["region", "province", "contract_type", "consumption_level"]:
        if col in df.columns and df[col].notna().any():
            le = LabelEncoder()
            df[col] = df[col].fillna("Unknown")
            df[f"{col}_encoded"] = le.fit_transform(df[col].astype(str))
    
    return df


def create_forecast_months():
    """Create month periods for October 2024 - September 2025."""
    start = pd.Period("2024-10", freq="M")
    months = [start + i for i in range(12)]
    return months


def generate_forecast(models, feature_cols, historical_data, groups):
    """Generate 12-month forecast for all groups."""
    print("\nGenerating 12-month forecasts...")
    
    forecast_months = create_forecast_months()
    
    # Create timestamps (first day of each month at 00:00 UTC)
    timestamps = [pd.Timestamp(f"{m.year}-{m.month:02d}-01 00:00:00", tz="UTC") for m in forecast_months]
    
    # Initialize results
    results = pd.DataFrame({"measured_at": timestamps})
    
    # For each group, generate forecast
    for group_id in sorted(models.keys()):
        print(f"  Forecasting group {group_id}...")
        
        # Get historical data for this group
        group_data = historical_data[historical_data["group_id"] == group_id].copy()
        group_data = group_data.sort_values("month_period").reset_index(drop=True)
        
        predictions = []
        
        for i, (month_period, timestamp) in enumerate(zip(forecast_months, timestamps)):
            # Create features for this month
            forecast_row = _create_forecast_month_row(
                month_period, timestamp, group_data, group_id, groups
            )
            
            # Ensure all required features are present
            for col in feature_cols:
                if col not in forecast_row:
                    forecast_row[col] = 0
            
            # Make prediction
            X = pd.DataFrame([forecast_row])[feature_cols]
            
            # Fill NaN
            for col in X.columns:
                if X[col].isna().any():
                    median_val = group_data[col].median() if col in group_data.columns else 0
                    X[col].fillna(median_val if not pd.isna(median_val) else 0, inplace=True)
            
            pred = models[group_id].predict(X)[0]
            predictions.append(max(0, pred))
            
            # Update group_data with prediction
            new_row = forecast_row.copy()
            new_row["month_period"] = month_period
            new_row["group_id"] = group_id
            new_row["load_mwh"] = pred
            new_row["measured_at"] = timestamp
            group_data = pd.concat([group_data, pd.DataFrame([new_row])], ignore_index=True)
        
        results[str(group_id)] = predictions
    
    print(f"✓ Generated forecasts for {len(models)} groups")
    return results


def _create_forecast_month_row(month_period, timestamp, historical_data, group_id, groups):
    """Create feature row for a forecast month."""
    row = {}
    
    # Time features
    row["month"] = month_period.month
    row["year"] = month_period.year
    row["month_sin"] = np.sin(2 * np.pi * month_period.month / 12)
    row["month_cos"] = np.cos(2 * np.pi * month_period.month / 12)
    
    # Get latest historical values
    if len(historical_data) > 0:
        latest = historical_data.iloc[-1]
        
        # Weather features (use same month from previous year)
        same_month_hist = historical_data[historical_data["month"] == month_period.month]
        for col in ["temperature_c", "wind_speed_ms", "precip_mm", "humidity_pct"]:
            if col in historical_data.columns:
                if len(same_month_hist) > 0 and col in same_month_hist.columns:
                    row[col] = same_month_hist[col].mean()
                else:
                    row[col] = latest.get(col, 0)
        
        # Price (use average of same month from history)
        if "eur_per_mwh" in historical_data.columns:
            if len(same_month_hist) > 0:
                row["eur_per_mwh"] = same_month_hist["eur_per_mwh"].mean()
            else:
                row["eur_per_mwh"] = latest.get("eur_per_mwh", 50)
        
        # Lag features
        if len(historical_data) >= 1:
            row["load_lag_1m"] = historical_data.iloc[-1]["load_mwh"]
        if len(historical_data) >= 12:
            row["load_lag_12m"] = historical_data.iloc[-12]["load_mwh"]
        
        # Price moving averages and volatility
        if "eur_per_mwh" in historical_data.columns:
            recent_prices = historical_data["eur_per_mwh"].tail(6)
            if len(recent_prices) >= 3:
                row["price_ma_3m"] = recent_prices.tail(3).mean()
                row["price_volatility_3m"] = recent_prices.tail(3).std()
            if len(recent_prices) >= 6:
                row["price_ma_6m"] = recent_prices.mean()
                row["price_volatility_6m"] = recent_prices.std()
    
    # Group features
    group_info = groups[groups["group_id"] == group_id]
    if len(group_info) > 0:
        group_label = group_info.iloc[0]["group_label"]
        parts = group_label.split("|")
        
        # Encode categorical features (simple approach)
        from sklearn.preprocessing import LabelEncoder
        
        all_groups = groups["group_label"].str.split("|", expand=True)
        
        for i, col_name in enumerate(["region", "province", "contract_type", "consumption_level"]):
            if i < len(parts):
                le = LabelEncoder()
                le.fit(all_groups[i].fillna("Unknown"))
                row[f"{col_name}_encoded"] = le.transform([parts[i]])[0]
    
    # Holiday features (set to 0 for future)
    for col in ["is_holiday", "is_holiday_eve", "is_holiday_after", "is_major_holiday_period"]:
        row[col] = 0
    row["days_to_next_holiday"] = 15
    row["days_since_last_holiday"] = 15
    
    return row


def save_forecast_fortum_format(results, filename="forecast_12m.csv"):
    """Save forecast in Fortum submission format."""
    output_path = OUTPUT_DIR / filename
    
    # Format timestamps
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
    """Generate final 12-month forecast."""
    print("="*60)
    print("GENERATING 12-MONTH FORECAST - FORTUM SUBMISSION FORMAT")
    print("="*60)
    
    # Load models
    models, feature_cols = load_models_and_features()
    
    # Prepare historical data
    historical_data, groups = prepare_historical_monthly_data()
    
    # Generate forecast
    results = generate_forecast(models, feature_cols, historical_data, groups)
    
    # Save in Fortum format
    output_path = save_forecast_fortum_format(results)
    
    # Display summary
    print("\n" + "="*60)
    print("FORECAST SUMMARY")
    print("="*60)
    print(f"Period: October 2024 - September 2025 (12 months)")
    print(f"Groups forecasted: {len(models)}")
    print(f"\nSample predictions (all months, first 5 groups):")
    print(results.iloc[:, :6].to_string(index=False))
    
    print("\n" + "="*60)
    print("✓ FORECAST GENERATION COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()
