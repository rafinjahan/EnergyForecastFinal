# QUICK REFERENCE GUIDE

## Fortum Energy Forecasting - Junction 2025

---

## 🚀 QUICK COMMANDS

### Activate Environment

```bash
cd /home/devcontainers/junction/OwnEnergy/EnergyForecasting
source ../.venv/bin/activate
```

### Train Models

```bash
# 48-hour model (takes ~5 minutes)
python src/models/train_lightgbm_48h.py

# 12-month model (takes ~2 minutes)
python src/models/train_lightgbm_monthly.py
```

### Generate Forecasts

```bash
# 48-hour forecast → forecasts/forecast_48h.csv
python src/models/generate_final_forecast_48h.py

# 12-month forecast → forecasts/forecast_12m.csv
python src/models/generate_final_forecast_12m.py
```

---

## 📊 KEY METRICS

| Model        | MAPE       | Median MAPE | Best  | Worst  | Groups  |
| ------------ | ---------- | ----------- | ----- | ------ | ------- |
| **48-Hour**  | **4.24%**  | 4.07%       | 2.30% | 9.49%  | 112/112 |
| **12-Month** | **10.42%** | 9.07%       | 3.29% | 23.47% | 112/112 |

---

## 📁 KEY FILES

### Submission Files

- `forecasts/forecast_48h.csv` - 48-hour forecast
- `forecasts/forecast_12m.csv` - 12-month forecast

### Documentation

- `README.md` - Project overview
- `METHODOLOGY.md` - Detailed methodology (27KB)
- `SUBMISSION_SUMMARY.md` - Submission checklist

### Training Scripts

- `src/models/train_lightgbm_48h.py` - 48h training
- `src/models/train_lightgbm_monthly.py` - 12m training

### Forecast Scripts

- `src/models/generate_final_forecast_48h.py` - 48h generation
- `src/models/generate_final_forecast_12m.py` - 12m generation

---

## 🔧 FEATURE COUNTS

**48-Hour Model**: 64 features

- 14 time features (hour, day, cyclical encoding)
- 12 weather features (temp, wind, humidity, interactions)
- 9 price features (current, lags, volatility)
- 13 lag features (1h, 24h, 168h, rolling stats)
- 7 holiday features (Finnish holidays)
- 9 pattern features (business hours, peaks)

**12-Month Model**: 23 features

- 2 time features (month cyclical)
- 4 weather features (monthly averages)
- 5 price features (monthly MAs, volatility)
- 2 lag features (1m, 12m)
- 4 group features (region, contract, consumption)
- 6 holiday features (monthly aggregations)

---

## 📈 PERFORMANCE BREAKDOWN

### 48-Hour Model

- **Excellent (< 3.5%)**: 25 groups (22%)
- **Good (3.5-5%)**: 58 groups (52%)
- **Acceptable (5-7%)**: 23 groups (21%)
- **Challenging (> 7%)**: 6 groups (5%)

### 12-Month Model

- **Excellent (< 7%)**: 32 groups (29%)
- **Good (7-10%)**: 31 groups (28%)
- **Acceptable (10-15%)**: 38 groups (34%)
- **Challenging (> 15%)**: 11 groups (10%)

---

## 🎯 FORECAST VALUE ADDED

**48-Hour FVA**: ~23% improvement over weekly naive baseline\
**12-Month FVA**: ~31% improvement over yearly naive baseline\
**Average FVA**: ~27% improvement

---

## 📝 SUBMISSION FORMAT

### CSV Format Requirements

- **Delimiter**: Semicolon (;)
- **Decimal**: Comma (,)
- **Encoding**: UTF-8
- **Timestamp**: ISO 8601 with .000Z suffix
- **Structure**: Wide format (rows=time, columns=groups)

### Example Timestamp

```
2024-10-01T00:00:00.000Z
```

### Example Value

```
3,8633255124361874
```

---

## 🔬 TECHNOLOGY STACK

```
Python:    3.10.12
LightGBM:  4.6.0
pandas:    Latest
numpy:     Latest
sklearn:   Latest
```

---

## 📊 DATA SOURCES

**Provided**:

- Fortum training consumption (32,856 hourly records)
- Fortum prices (day-ahead EUR/MWh)
- Customer groups (112 groups with metadata)

**External**:

- Finnish Meteorological Institute (FMI) weather
- holiday-calendar.fi API (46 Finnish holidays)

---

## 🏆 KEY INNOVATIONS

1. **Finnish Holiday Integration** ⭐
   - REST API from holiday-calendar.fi
   - 46 holidays (2020-2024)
   - 7 holiday-related features

2. **Feature Simplification** ⭐
   - Reduced monthly model: 56 → 23 features
   - Improved MAPE: 11.91% → 10.42%

3. **Group-Specific Models** ⭐
   - 112 individual models
   - Captures unique patterns
   - Better than global model

4. **Cyclical Encoding** ⭐
   - Sin/cos for time features
   - Smooth seasonal transitions

---

## ⚡ TROUBLESHOOTING

### Model Training Fails

```bash
# Check data location
ls Data/*.xlsx

# Check environment
which python
python --version

# Check dependencies
pip list | grep lightgbm
```

### Forecast Generation Fails

```bash
# Verify models exist
ls models/lgb_48h/*.pkl | wc -l  # Should be 112
ls models/lgb_monthly/*.pkl | wc -l  # Should be 112
```

### CSV Format Issues

```bash
# Check delimiter and decimal
head -3 forecasts/forecast_48h.csv

# Should see: semicolon (;) delimiter, comma (,) decimal
```

---

## 📧 SUPPORT

**GitHub**: https://github.com/anton-saari/EnergyForecasting\
**Team**: OwnEnergy\
**Challenge**: Fortum Energy Forecasting - Junction 2025

---

## ✅ PRE-SUBMISSION CHECKLIST

- [x] Both CSV files generated
- [x] Correct format (semicolon, comma decimal)
- [x] All 112 groups present
- [x] No missing values
- [x] README.md complete
- [x] METHODOLOGY.md complete
- [x] GitHub repository public
- [x] Code documented
- [ ] Demo video created
- [ ] Submitted via Junction platform

---

**Last Updated**: November 15, 2025\
**Status**: ✅ **READY FOR SUBMISSION**

---

## 🎬 DEMO VIDEO OUTLINE

**Duration**: 3-5 minutes

**Structure**:

1. **Introduction (30s)**
   - Team name and challenge
   - Problem overview

2. **Solution Approach (1.5min)**
   - LightGBM with 112 individual models
   - 64/23 features overview
   - External data integration

3. **Technical Highlights (1.5min)**
   - Finnish holiday API integration
   - Feature simplification success
   - Cyclical time encoding

4. **Results (1min)**
   - 4.24% MAPE for 48h
   - 10.42% MAPE for 12m
   - 27% FVA over baseline

5. **Business Impact (30s)**
   - Cost savings potential
   - Risk reduction
   - Production-ready solution

---

**🚀 GOOD LUCK! 🚀**
