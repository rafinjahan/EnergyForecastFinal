# Fortum Energy Forecasting Challenge - Junction 2025

**Team:** OwnEnergy\
**Challenge:** 48-Hour & 12-Month Energy Consumption Forecasting\
**Models:** LightGBM with Advanced Feature Engineering

---

## 📊 Project Overview

This repository contains our solution for the Fortum energy forecasting
challenge at Junction 2025 hackathon. We developed machine learning models to
predict electricity consumption for 112 customer groups across two time
horizons:

- **48-Hour Forecast**: Hourly predictions for October 1-2, 2024 (short-term
  operational planning)
- **12-Month Forecast**: Monthly predictions for October 2024 - September 2025
  (long-term hedging strategy)

### Key Results

| Model                  | Average MAPE | Median MAPE | Best Group MAPE | Groups Trained |
| ---------------------- | ------------ | ----------- | --------------- | -------------- |
| **48-Hour (Hourly)**   | **4.24%**    | 4.07%       | 2.30%           | 112/112        |
| **12-Month (Monthly)** | **10.42%**   | 9.07%       | 3.29%           | 112/112        |

Both models significantly outperform naive baselines, delivering actionable
forecasts for energy trading and grid management.

---

## 🗂️ Repository Structure

```
EnergyForecasting/
├── forecasts/                      # Final submission forecasts
│   ├── forecast_48h.csv           # 48-hour forecast (Oct 1-2, 2024)
│   └── forecast_12m.csv           # 12-month forecast (Oct 2024-Sep 2025)
├── src/
│   ├── data/                      # Data loading and processing
│   │   ├── fortum_loader.py       # Load Fortum training data
│   │   ├── weather_enrichment.py  # Weather data integration
│   │   └── finnish_holidays.py    # Finnish holiday API integration
│   ├── models/                    # Model training and forecasting
│   │   ├── train_lightgbm_48h.py  # Train 48-hour models
│   │   ├── train_lightgbm_monthly.py  # Train 12-month models
│   │   ├── generate_final_forecast_48h.py  # Generate 48h forecast
│   │   └── generate_final_forecast_12m.py  # Generate 12m forecast
│   └── visualization/             # Analysis and plotting
├── models/                        # Trained model artifacts
│   ├── lgb_48h/                   # 48-hour models (112 models)
│   └── lgb_monthly/               # Monthly models (112 models)
├── reports/                       # Training metrics and summaries
├── Data/                          # Input data (Fortum + Weather)
├── METHODOLOGY.md                 # Detailed methodology document
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- Virtual environment (recommended)

### Installation

```bash
# Clone the repository
git clone https://github.com/anton-saari/EnergyForecasting.git
cd EnergyForecasting

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Training Models

```bash
# Train 48-hour forecast models
python src/models/train_lightgbm_48h.py

# Train 12-month forecast models
python src/models/train_lightgbm_monthly.py
```

### Generating Forecasts

```bash
# Generate 48-hour forecast (Oct 1-2, 2024)
python src/models/generate_final_forecast_48h.py

# Generate 12-month forecast (Oct 2024-Sep 2025)
python src/models/generate_final_forecast_12m.py
```

Forecasts will be saved to `forecasts/` directory in Fortum submission format
(semicolon delimiter, comma decimal separator).

---

## 🔬 Methodology Highlights

### Approach

1. **Individual Group Models**: Trained separate LightGBM models for each of 112
   customer groups to capture unique consumption patterns
2. **Rich Feature Engineering**: 64 features for hourly models, 23 features for
   monthly models
3. **External Data Integration**: Finnish weather data (FMI) and public holidays
   (holiday-calendar.fi API)
4. **80/20 Time-Based Split**: Proper validation without data leakage

### Feature Categories

**48-Hour Model (64 features):**

- Time features (hour, day, week with cyclical encoding)
- Weather features (temperature, humidity, precipitation, wind)
- Price features (spot prices with lags and volatility)
- Lag features (1h, 24h, 168h, 336h consumption lags)
- Rolling statistics (24h, 168h means, std, max, min)
- Holiday features (Finnish public holidays and proximity)
- Advanced patterns (EMAs, interactions, seasonal indicators)

**12-Month Model (23 features - simplified for better generalization):**

- Time features (month with cyclical encoding)
- Weather features (monthly averages)
- Price features (monthly averages with 3m/6m MAs and volatility)
- Lag features (1-month, 12-month for YoY patterns)
- Group features (region, province, contract type, consumption level)
- Holiday features (monthly aggregations)

### Key Innovations

1. **Finnish Holiday Integration**: REST API integration with
   holiday-calendar.fi for accurate holiday effects
2. **Cyclical Time Encoding**: Sine/cosine transformations ensure December and
   January are "close" in feature space
3. **Group-Specific Feature Extraction**: Parsed customer metadata from group
   labels (region, contract type, consumption tier)
4. **Feature Simplification**: Reduced monthly model from 56 to 23 features,
   improving MAPE by 12.5%

---

## 📈 Performance Analysis

### 48-Hour Model Performance

- **Average MAPE**: 4.24% (excellent for hourly energy forecasting)
- **Progression**: 4.83% (baseline) → 4.28% (enhanced) → 4.24% (with holidays)
- **Best Groups**: Groups with stable consumption patterns (2.30% - 3.50% MAPE)
- **Challenging Groups**: Small customers with high variability (7% - 9% MAPE)

### 12-Month Model Performance

- **Average MAPE**: 10.42% (above industry average for monthly forecasting)
- **Improvement**: 11.91% (complex) → 10.42% (simplified) = 12.5% reduction
- **Best Groups**: Large stable customers (3.29% - 4% MAPE)
- **Industry Benchmark**: 8-15% MAPE typical; our model at 10.42% is competitive

### Forecast Value Added (FVA)

Compared to naive baselines:

- **48-hour model**: ~15-20% improvement over weekly seasonal naive
- **12-month model**: ~30-35% improvement over yearly seasonal naive

---

## 🛠️ Technology Stack

| Component           | Technology                           | Version |
| ------------------- | ------------------------------------ | ------- |
| **ML Framework**    | LightGBM                             | 4.6.0   |
| **Programming**     | Python                               | 3.10.12 |
| **Data Processing** | pandas, numpy                        | Latest  |
| **External Data**   | FMI Weather, holiday-calendar.fi API | -       |
| **Environment**     | Virtual Environment (.venv)          | -       |

---

## 📊 Data Sources

### Provided Data

- **Fortum Training Consumption**: 32,856 hourly records (2021-2024) for 112
  customer groups
- **Fortum Prices**: Day-ahead electricity prices (EUR/MWh)
- **Customer Groups**: Metadata with region, contract type, consumption level

### External Data

- **Finnish Meteorological Institute (FMI)**: Historical weather data
  - Temperature, wind speed, precipitation, humidity
  - Multiple weather stations averaged for robustness
- **holiday-calendar.fi API**: Finnish public holidays (2020-2024)
  - 46 holidays fetched
  - Excludes weekends (only actual non-working days)

---

## 📝 Submission Files

### Forecast Files

Both files follow Fortum submission format:

- **Encoding**: UTF-8
- **Delimiter**: Semicolon (;)
- **Decimal Separator**: Comma (,)
- **Timestamp Format**: ISO 8601 with .000Z suffix (e.g.,
  `2024-10-01T00:00:00.000Z`)
- **Structure**: Wide format (one row per timestamp, one column per group)

**1. forecast_48h.csv**

- 48 rows (hourly timestamps from Oct 1 00:00 to Oct 2 23:00)
- 113 columns (measured_at + 112 group forecasts)

**2. forecast_12m.csv**

- 12 rows (monthly timestamps from Oct 2024 to Sep 2025)
- 113 columns (measured_at + 112 group forecasts)

---

## 🏆 Key Achievements

✅ **All 112 groups trained successfully** - No failures or missing predictions\
✅ **4.24% MAPE on 48-hour forecast** - Excellent short-term accuracy\
✅ **10.42% MAPE on 12-month forecast** - Strong long-term performance\
✅ **Production-ready code** - Modular, documented, reproducible\
✅ **External data integration** - Weather and holidays improve accuracy\
✅ **Proper validation** - Time-based 80/20 split prevents data leakage

---

## 📖 Documentation

- **METHODOLOGY.md**: Comprehensive methodology document (see separate file)
- **HOLIDAY_API_INTEGRATION.md**: Details on Finnish holiday feature integration
- **MONTHLY_MODEL_SIMPLIFICATION_RESULTS.md**: Analysis of feature
  simplification impact

---

## 👥 Team

**Team Name**: OwnEnergy\
**Hackathon**: Junction 2025\
**Challenge**: Fortum Energy Forecasting

---

## 📄 License

This project was developed for the Junction 2025 hackathon. All rights reserved.

---

## 🙏 Acknowledgments

- **Fortum** for providing the challenge and dataset
- **Junction 2025** for organizing the hackathon
- **Finnish Meteorological Institute (FMI)** for weather data
- **holiday-calendar.fi** for Finnish public holiday API

---

## 📧 Contact

For questions about this project, please contact through the Junction 2025
platform.

---

**Last Updated**: November 15, 2025\
**Status**: ✅ Submission Ready
