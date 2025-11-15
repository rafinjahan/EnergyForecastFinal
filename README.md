# Energy Forecast Final – Quick Guide

This repository contains OwnEnergy's submission for the Fortum Energy Forecasting Challenge (Junction 2025). It bundles data loaders, feature engineering, and LightGBM models for two horizons:

- **48-hour** shared LightGBM model (`src/models/train_lightgbm.py`, forecasts via `src/models/test_forecast_48h.py`).
- **12-month** per-group LightGBM models (`src/models/train_lightgbm_monthly.py`, forecasts via `src/models/forecast_monthly.py`).

## Project Structure

```
Data/                  # Local weather spreadsheets (not tracked)
forecasts/             # Generated submissions / artifacts
src/data/              # Loaders, enrichment utilities, holiday helpers
src/models/            # Training + inference scripts for each horizon
METHODOLOGY.md         # Detailed documentation of approach
SUBMISSION_SUMMARY.md  # Competition hand-in summary
requirements.txt       # Python dependencies (LightGBM, pandas, etc.)
```

## Prerequisites

1. Python 3.10+ and a virtual environment (`python -m venv .venv && source .venv/bin/activate`).
2. Install dependencies: `pip install -r requirements.txt`.
3. Place the Fortum training Excel files (and any FMI weather files) in `Data/` using the original structure.

## Running the Pipelines

### 48-Hour Model

```
python src/models/train_lightgbm.py        # trains shared LightGBM + saves metrics
python src/models/test_forecast_48h.py     # produces 48h forecasts for Oct 1–2, 2024
```

Artifacts land under `models/` (e.g., `models/lgb_weather_model.pkl` and validation CSVs).

### 12-Month Model

```
python src/models/train_lightgbm_monthly.py   # trains 112 per-group monthly models
python src/models/forecast_monthly.py         # generates 12-month forecast submission
```

Models, feature lists, and summaries are written to `models/lgb_monthly/`.

## Verification

- All scripts log their train/validation windows and per-group MAPE statistics.
- Check `models/*_metrics.txt` or `models/lgb_monthly/training_metrics.csv` for validation scores.

## Support

See `METHODOLOGY.md` for the methodological deep dive and `SUBMISSION_SUMMARY.md` for the official submission notes.
