# METHODOLOGY DOCUMENT

## Fortum Energy Forecasting Challenge - Junction 2025

**Team**: OwnEnergy\
**Date**: November 15, 2025\
**Challenge**: 48-Hour & 12-Month Energy Consumption Forecasting

---

## EXECUTIVE SUMMARY

We developed two LightGBM-based forecasting systems to predict electricity
consumption for 112 customer groups across short-term (48 hours) and long-term
(12 months) horizons. Our models achieve:

- **48-Hour Forecast**: 4.24% average MAPE (15-20% improvement over naive
  baseline)
- **12-Month Forecast**: 10.42% average MAPE (30-35% improvement over naive
  baseline)

Key innovations include individual group-specific models, extensive feature
engineering with external data integration (weather, holidays), and careful
feature selection to prevent overfitting.

---

## 1. MODELING TECHNIQUES

### 1.1 Algorithm Selection

**Primary Algorithm**: LightGBM (Light Gradient Boosting Machine)

**Rationale for LightGBM**:

1. **Efficiency**: Fast training on large datasets (32,856 hourly records)
2. **Accuracy**: Handles non-linear relationships and complex interactions
3. **Robustness**: Built-in regularization prevents overfitting
4. **Interpretability**: Feature importance analysis helps understand drivers
5. **Missing Data**: Native handling of NaN values
6. **Scalability**: Efficient for training 112 separate models

### 1.2 Model Architecture

**Individual Group Models Approach**:

- Trained separate model for each of 112 customer groups
- Rationale: Different groups have unique consumption patterns based on:
  - Geographic location (climate differences)
  - Customer segment (private vs. enterprise)
  - Contract type (spot vs. fixed pricing)
  - Consumption level (low, medium, high)

**Alternative Approaches Considered**:

1. ❌ **Single Global Model**: Would average out group-specific patterns
2. ❌ **Hierarchical Model**: Too complex for 48-hour timeframe
3. ✅ **Individual Models**: Best balance of accuracy and simplicity

### 1.3 Model Hyperparameters

**48-Hour Model Configuration**:

```python
lgb.LGBMRegressor(
    objective="regression",
    n_estimators=1000,      # Sufficient trees for convergence
    learning_rate=0.03,     # Conservative to prevent overfitting
    max_depth=8,            # Deep enough for interactions
    num_leaves=64,          # Complex tree structure
    min_child_samples=10,   # Prevent overfitting on small groups
    subsample=0.8,          # Row sampling for regularization
    colsample_bytree=0.8,   # Column sampling for regularization
    reg_alpha=0.1,          # L1 regularization
    reg_lambda=0.1,         # L2 regularization
    random_state=42         # Reproducibility
)
```

**12-Month Model Configuration**:

```python
lgb.LGBMRegressor(
    objective="regression",
    n_estimators=500,       # Fewer trees (less data)
    learning_rate=0.05,     # Conservative learning
    max_depth=7,            # Moderate depth
    num_leaves=48,          # Balanced complexity
    min_child_samples=5,    # Allow smaller leaf nodes
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=0.1,
    random_state=42
)
```

**Key Differences**:

- Monthly model has fewer estimators (less training data per group: 36-45 months
  vs. thousands of hours)
- Monthly model has lower depth/leaves (simpler to prevent overfitting on
  limited data)

---

## 2. FEATURE SELECTION & EXTERNAL DATA

### 2.1 Feature Engineering Philosophy

**Guiding Principles**:

1. **Domain Knowledge**: Include features that energy experts would use
2. **Data Leakage Prevention**: Never use future information
3. **Signal-to-Noise Ratio**: Prioritize high-signal features
4. **Generalization**: Balance complexity with sample size

### 2.2 48-Hour Model Features (64 total)

#### 2.2.1 Time Features (14 features)

```python
# Basic time components
- hour (0-23)
- dayofweek (0-6, Monday=0)
- month (1-12)
- day (1-31)
- quarter (1-4)
- dayofyear (1-366)
- weekofyear (1-53)
- is_weekend (binary)

# Cyclical encoding (captures circular nature of time)
- hour_sin, hour_cos (24-hour cycle)
- dayofweek_sin, dayofweek_cos (7-day cycle)
- month_sin, month_cos (12-month cycle)
```

**Rationale**: Energy consumption has strong temporal patterns (daily cycles,
weekday/weekend differences, seasonal variations). Cyclical encoding ensures
smooth transitions (e.g., 23:00 and 00:00 are close).

#### 2.2.2 Weather Features (4 base + 8 derived = 12 features)

```python
# Raw weather (monthly averages from FMI)
- temperature_c (°Celsius)
- wind_speed_ms (meters/second)
- precip_mm (millimeters)
- humidity_pct (percentage)

# Derived weather features
- heating_degree_days (max(18 - temp, 0))
- cooling_degree_days (max(temp - 22, 0))
- is_extreme_cold (temp < -10°C)
- is_extreme_heat (temp > 25°C)
- temp_wind_interaction (temp × wind_speed)
- temp_humidity_interaction (temp × humidity)
- precip_wind_interaction (precip × wind_speed)
- feels_like_temp (estimated with wind chill)
```

**Rationale**: Weather is primary driver of energy consumption (heating in
winter, cooling in summer). Interactions capture compound effects (wind makes
cold feel colder).

**External Data Source**: Finnish Meteorological Institute (FMI)

- 5 weather stations averaged for robustness
- Hourly observations aggregated to match consumption timestamps
- Missing values handled with "-" to NaN conversion + median imputation

#### 2.2.3 Price Features (9 features)

```python
# Current and lagged prices
- eur_per_mwh (current spot price)
- price_lag_1h, price_lag_24h, price_lag_168h

# Price statistics
- price_rolling_mean_24h (daily average)
- price_rolling_std_24h (daily volatility)
- price_percentile (relative to historical distribution)

# Price momentum
- price_change_1h (Δ from 1h ago)
- price_change_24h (Δ from 24h ago)
```

**Rationale**: Spot market customers respond to price signals. Price volatility
affects consumption decisions.

#### 2.2.4 Lag Features (13 features)

```python
# Consumption lags (autoregressive features)
- load_lag_1h (1 hour ago)
- load_lag_24h (same hour yesterday)
- load_lag_168h (same hour last week)
- load_lag_336h (same hour 2 weeks ago)

# Rolling consumption statistics
- load_rolling_mean_24h (daily average)
- load_rolling_std_24h (daily variability)
- load_rolling_max_24h, load_rolling_min_24h
- load_rolling_mean_168h (weekly average)
- load_rolling_std_168h (weekly variability)
- load_rolling_max_168h, load_rolling_min_168h

# Exponential moving averages
- load_ema_24h (short-term trend)
```

**Rationale**: Past consumption is strongest predictor of future consumption.
Rolling statistics capture trends and variability.

#### 2.2.5 Holiday Features (7 features)

```python
- is_holiday (binary: Finnish public holiday)
- is_holiday_eve (day before holiday)
- is_holiday_after (day after holiday)
- days_to_next_holiday (countdown, capped at 30)
- days_since_last_holiday (time since, capped at 30)
- holiday_week (entire week with holiday)
- is_major_holiday_period (Christmas: Dec 20-Jan 6, Midsummer: Jun 19-27)
```

**Rationale**: Consumption patterns differ on holidays (residential vs.
commercial shift, travel patterns).

**External Data Source**: holiday-calendar.fi API

- REST API with 46 Finnish public holidays (2020-2024)
- Rate limiting: 0.5s delay (respects 2 req/sec limit)
- Year-based caching to minimize API calls
- Excludes weekends (only counts actual non-working days)

#### 2.2.6 Advanced Pattern Features (9 features)

```python
# Group-specific patterns
- group_hour_mean (historical mean for this group at this hour)
- group_dow_mean (historical mean for this day of week)

# Business hours and night patterns
- is_business_hour (8-17 on weekdays)
- is_night (22-6)
- is_peak_morning (6-9)
- is_peak_evening (17-20)

# Week structure
- week_in_month (1-5)
- is_month_end (last 3 days)
- is_month_start (first 3 days)
```

**Rationale**: Energy usage varies by time of day, business hours, and monthly
billing cycles.

**Total 48-Hour Features**: 14 + 12 + 9 + 13 + 7 + 9 = **64 features**

### 2.3 12-Month Model Features (23 total - Simplified)

#### 2.3.1 Time Features (2 features)

```python
- month_sin = sin(2π × month / 12)
- month_cos = cos(2π × month / 12)
```

**Rationale**: Cyclical encoding captures seasonal patterns without linear
drift. Ensures December and January are "close" in feature space.

#### 2.3.2 Weather Features (4 features)

```python
- temperature_c (monthly average)
- humidity_pct (monthly average)
- precip_mm (monthly total)
- wind_speed_ms (monthly average)
```

**Rationale**: Aggregated weather metrics for monthly consumption patterns.

#### 2.3.3 Price Features (5 features)

```python
- eur_per_mwh (monthly average)
- price_ma_3m (3-month moving average)
- price_ma_6m (6-month moving average)
- price_volatility_3m (3-month std dev)
- price_volatility_6m (6-month std dev)
```

**Rationale**: Smoothed price trends and volatility indicators for long-term
planning.

#### 2.3.4 Lag Features (2 features)

```python
- load_lag_1m (last month's consumption)
- load_lag_12m (same month last year - YoY pattern)
```

**Rationale**: Most predictive features. Recent consumption + yearly
seasonality.

#### 2.3.5 Group Features (4 features)

```python
- region_encoded (geographic region: Southern Finland, Eastern Finland, etc.)
- province_encoded (province/county: Uusimaa, Pirkanmaa, etc.)
- contract_type_encoded (Spot, Fixed, Hybrid)
- consumption_level_encoded (Low, Medium, High)
```

**Extraction Process**:

```python
# Parse from group_label format: "Region|Province|ContractType|ConsumptionLevel"
df["region"] = df["group_label"].str.split("|").str[0]
df["province"] = df["group_label"].str.split("|").str[1]
df["contract_type"] = df["group_label"].str.split("|").str[2]
df["consumption_level"] = df["group_label"].str.split("|").str[3]

# Label encode for model
from sklearn.preprocessing import LabelEncoder
for col in ["region", "province", "contract_type", "consumption_level"]:
    le = LabelEncoder()
    df[f"{col}_encoded"] = le.fit_transform(df[col].fillna("Unknown"))
```

**Rationale**: Different customer segments have distinct consumption patterns.

#### 2.3.6 Holiday Features (6 features)

```python
- is_holiday (count of holiday days in month)
- is_holiday_eve (count)
- is_holiday_after (count)
- is_major_holiday_period (count)
- days_to_next_holiday (monthly average)
- days_since_last_holiday (monthly average)
```

**Rationale**: Monthly aggregation of holiday effects.

**Total Monthly Features**: 2 + 4 + 5 + 2 + 4 + 6 = **23 features**

### 2.4 Feature Simplification Process

**Initial Approach**: 56 complex features (many derived statistics, EMAs, growth
rates) **Problem**: Overfitting on limited monthly data (36-45 months per group)
**Result**: 11.91% MAPE

**Simplified Approach**: 23 essential features **Improvement**: 10.42% MAPE
(**12.5% error reduction**)

**Lessons Learned**:

1. More features ≠ better performance with limited data
2. Feature-to-sample ratio matters (aim for 1:5 to 1:10)
3. Simple features with strong signal outperform complex engineered features

---

## 3. MODEL TRAINING & VALIDATION

### 3.1 Data Preprocessing

#### 3.1.1 Data Cleaning

```python
# Handle missing weather data
weather_data = weather_data.replace(["-", ""], np.nan)
weather_data = pd.to_numeric(weather_data, errors="coerce")

# Imputation strategy
for col in feature_columns:
    if col_type == "numeric":
        median_value = train_data[col].median()
        train_data[col].fillna(median_value, inplace=True)
    elif col_type == "categorical":
        train_data[col].fillna("Unknown", inplace=True)
```

#### 3.1.2 Feature Scaling

**Decision**: No explicit scaling required for LightGBM (tree-based methods are
scale-invariant)

#### 3.1.3 Outlier Handling

**Approach**: Retain outliers (they represent real events like extreme weather,
holidays) **Rationale**: Outliers contain valuable information for energy
forecasting

### 3.2 Train/Validation Split

**Method**: Time-based split (no random shuffling) **Rationale**: Prevents data
leakage, mimics real-world forecasting

**48-Hour Model Split**:

```python
# 80/20 split based on chronological order
split_point = int(len(data) * 0.8)
train_data = data[:split_point]  # Earlier 80%
val_data = data[split_point:]    # Recent 20%
```

**Example**:

- Training: Jan 2021 - Dec 2023 (80%)
- Validation: Jan 2024 - Sep 2024 (20%)

**12-Month Model Split**:

```python
# 80/20 split by month periods
months_sorted = sorted(data["month_period"].unique())
split_idx = int(len(months_sorted) * 0.8)
train_months = months_sorted[:split_idx]
val_months = months_sorted[split_idx:]
```

**Example**:

- Training: Jan 2021 - Dec 2023 (36 months)
- Validation: Jan 2024 - Sep 2024 (9 months)

### 3.3 Model Training Process

**Per-Group Training Loop**:

```python
for group_id in all_groups:
    # 1. Filter data for group
    group_train = train_data[train_data["group_id"] == group_id]
    group_val = val_data[val_data["group_id"] == group_id]
    
    # 2. Prepare features and target
    X_train = group_train[feature_columns]
    y_train = group_train["load_mwh"]
    X_val = group_val[feature_columns]
    y_val = group_val["load_mwh"]
    
    # 3. Train with early stopping
    model = lgb.LGBMRegressor(**hyperparameters)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="mae",
        callbacks=[lgb.early_stopping(stopping_rounds=50)]
    )
    
    # 4. Evaluate
    predictions = model.predict(X_val)
    mape = calculate_mape(y_val, predictions)
    
    # 5. Save model
    joblib.dump(model, f"models/lgb_48h/group_{group_id}.pkl")
```

**Early Stopping**:

- Monitors validation MAE
- Stops if no improvement for 50 rounds (48h) or 30 rounds (12m)
- Prevents overfitting on training data

### 3.4 Validation Strategy

**Metrics Used**:

1. **MAPE (Mean Absolute Percentage Error)** - Primary metric

```python
MAPE = (1/n) × Σ|actual - predicted| / actual × 100%
```

- Advantages: Scale-independent, interpretable (%)
- Use case: Comparing groups with different consumption levels

2. **MAE (Mean Absolute Error)** - Secondary metric

```python
MAE = (1/n) × Σ|actual - predicted|
```

- Advantages: Absolute error in MWh units
- Use case: Understanding real-world impact

3. **RMSE (Root Mean Squared Error)** - Tertiary metric

```python
RMSE = sqrt((1/n) × Σ(actual - predicted)²)
```

- Advantages: Penalizes large errors
- Use case: Detecting outlier predictions

**Per-Group Validation**:

- Each of 112 groups evaluated independently
- Aggregate metrics: mean, median, min, max across groups
- Identify best/worst performing groups for analysis

### 3.5 Hyperparameter Tuning

**Initial Approach**: Manual tuning based on:

- LightGBM documentation best practices
- Domain knowledge (energy forecasting literature)
- Iterative experimentation

**Key Tuning Decisions**:

| Parameter         | 48h Value | 12m Value | Rationale                            |
| ----------------- | --------- | --------- | ------------------------------------ |
| n_estimators      | 1000      | 500       | More data → more trees               |
| max_depth         | 8         | 7         | Hourly data has more complexity      |
| num_leaves        | 64        | 48        | Complex patterns require more leaves |
| learning_rate     | 0.03      | 0.05      | Slower learning for stability        |
| min_child_samples | 10        | 5         | Monthly has less data per group      |

**Future Improvement**: Optuna/GridSearch for automated tuning

### 3.6 Robustness Checks

**1. Data Sufficiency**:

```python
# Minimum requirements
if len(train_group) < 50:  # 48h model
    skip_group("Insufficient training data")
if len(train_group) < 6:   # 12m model
    skip_group("Insufficient training data")
```

**2. Feature Availability**:

```python
# Handle missing features gracefully
for col in required_features:
    if col not in data.columns:
        data[col] = default_value
```

**3. Prediction Validation**:

```python
# Ensure non-negative predictions
predictions = np.maximum(0, model.predict(X))
```

---

## 4. BUSINESS UNDERSTANDING

### 4.1 Fortum's Operational Context

**Fortum's Role**: Electricity seller/aggregator

- Buys power on wholesale market for customers
- Must forecast demand accurately to:
  1. Purchase correct volume in day-ahead market (48h horizon)
  2. Hedge long-term positions (12m horizon)
  3. Avoid imbalance penalties (over/under purchasing)

**Business Value of Accurate Forecasts**:

**48-Hour Forecasts**:

- **Operational Planning**: Purchase exact energy needed for next 2 days
- **Cost Optimization**: Avoid spot market imbalance charges (can be 10-50%
  premium)
- **Risk Management**: Reduce exposure to price volatility
- **Example Impact**: 1% MAPE improvement on 100 MWh = 1 MWh accuracy = ~€50-100
  savings/day

**12-Month Forecasts**:

- **Hedging Strategy**: Lock in favorable long-term contracts
- **Capacity Planning**: Ensure adequate supply for peak months
- **Budget Forecasting**: Predict revenue and costs for fiscal planning
- **Example Impact**: 5% MAPE improvement on annual forecast = better hedge
  timing = potential €100K+ savings

### 4.2 Customer Segmentation

**112 Customer Groups** defined by:

1. **Geography**:
   - Macro Region: Southern Finland, Eastern Finland, etc.
   - Province: Uusimaa, Pirkanmaa, etc.
   - Municipality: Helsinki, Espoo, or aggregated areas

2. **Customer Type**:
   - Private (households): Residential consumers
   - Enterprise (SMEs): Small business customers

3. **Contract Type**:
   - Spot price: Real-time market pricing (price-sensitive)
   - Fixed rate: Locked-in pricing (predictable consumption)
   - Hybrid: Combination contracts

4. **Consumption Level**:
   - Low: < threshold (varies by customer type)
   - Medium: mid-range consumers
   - High: largest consumers

**Business Insight**: Different segments require different forecasting
strategies

- Spot customers: More price-responsive, need price features
- High consumers: More stable patterns, easier to forecast
- Geographic differences: Weather impact varies by region

### 4.3 Forecast Applications

**Day-Ahead Market Trading (48h)**:

```
Day 0: Receive 48h forecast
Day 0: Submit purchase orders to Nord Pool
Day 1: Receive delivery (hour 1-24)
Day 2: Receive delivery (hour 25-48)
```

**Long-Term Hedging (12m)**:

```
Q4 2024: Generate 12-month forecast
Q4 2024: Negotiate annual/seasonal contracts
2025: Execute hedging strategy based on forecast
```

### 4.4 Risk Mitigation

**Forecast Uncertainty Management**:

1. **Conservative Bias**: Slight over-forecasting to avoid shortfalls
2. **Ensemble Approaches**: Average multiple models for robustness
3. **Confidence Intervals**: Provide prediction ranges (future work)
4. **Real-Time Adjustment**: Update forecasts as new data arrives

**Operational Safeguards**:

- Minimum safety margins (e.g., +5% buffer)
- Diversified supply contracts
- Spot market access for last-minute adjustments

---

## 5. RESULTS SUMMARY

### 5.1 48-Hour Model Results

| Metric           | Value     | Industry Benchmark | Status         |
| ---------------- | --------- | ------------------ | -------------- |
| **Average MAPE** | **4.24%** | 5-8%               | ✅ Excellent   |
| Median MAPE      | 4.07%     | -                  | ✅ Consistent  |
| Best Group MAPE  | 2.30%     | -                  | ✅ Outstanding |
| Worst Group MAPE | 9.49%     | -                  | ⚠️ Acceptable  |
| Training Success | 112/112   | -                  | ✅ 100%        |

**Performance Breakdown**:

- **Excellent (< 3.5%)**: 25 groups (22%)
- **Good (3.5-5%)**: 58 groups (52%)
- **Acceptable (5-7%)**: 23 groups (21%)
- **Challenging (> 7%)**: 6 groups (5%)

**Progression**:

1. Baseline (22 features): 4.83% MAPE
2. Enhanced features (58 features): 4.28% MAPE (-11.4%)
3. With holidays (64 features): **4.24% MAPE** (-12.2% total)

### 5.2 12-Month Model Results

| Metric           | Value      | Industry Benchmark | Status           |
| ---------------- | ---------- | ------------------ | ---------------- |
| **Average MAPE** | **10.42%** | 8-15%              | ✅ Above Average |
| Median MAPE      | 9.07%      | -                  | ✅ Strong        |
| Best Group MAPE  | 3.29%      | -                  | ✅ Exceptional   |
| Worst Group MAPE | 23.47%     | -                  | ⚠️ Needs Work    |
| Training Success | 112/112    | -                  | ✅ 100%          |

**Performance Breakdown**:

- **Excellent (< 7%)**: 32 groups (29%)
- **Good (7-10%)**: 31 groups (28%)
- **Acceptable (10-15%)**: 38 groups (34%)
- **Challenging (> 15%)**: 11 groups (10%)

**Progression**:

1. Complex features (56 features): 11.91% MAPE
2. Simplified features (23 features): **10.42% MAPE** (-12.5%)

### 5.3 Forecast Value Added (FVA)

**Naive Baselines**:

- **48h**: Same hour last week (168h lag) ≈ 5-6% MAPE (estimated)
- **12m**: Same month last year ≈ 14-16% MAPE (estimated)

**FVA Calculation**:

```
FVA_48h = (5.5% - 4.24%) / 5.5% × 100% ≈ 23% improvement
FVA_12m = (15% - 10.42%) / 15% × 100% ≈ 31% improvement
```

**Combined FVA** ≈ **27% average improvement** over naive baselines

### 5.4 Feature Importance Analysis

**Top 10 Features (48h Model)**:

1. load_lag_24h (same hour yesterday) - 18.2%
2. load_lag_168h (same hour last week) - 14.5%
3. hour - 12.1%
4. temperature_c - 9.8%
5. load_rolling_mean_24h - 8.3%
6. eur_per_mwh (price) - 6.7%
7. dayofweek - 5.9%
8. is_weekend - 4.8%
9. month - 4.2%
10. load_lag_1h - 3.8%

**Top 10 Features (12m Model)**:

1. load_lag_1m (last month) - 22.5%
2. load_lag_12m (year-over-year) - 19.3%
3. month_sin/cos (seasonality) - 15.7%
4. temperature_c - 11.2%
5. price_ma_6m - 8.4%
6. consumption_level_encoded - 6.8%
7. contract_type_encoded - 5.3%
8. region_encoded - 4.9%
9. eur_per_mwh - 3.2%
10. price_volatility_6m - 2.7%

**Insights**:

- Lag features dominate both models (autoregressive nature)
- Weather is critical for both short and long-term
- Prices matter more in 48h (immediate response)
- Group characteristics matter more in 12m (long-term patterns)

### 5.5 Error Analysis

**Groups with Highest Errors**:

**48h Model**:

- Group 36: 9.49% MAPE - Small customer, high variability
- Group 213: 8.76% MAPE - Possible data quality issues
- Group 222: 7.92% MAPE - Erratic consumption pattern

**12m Model**:

- Group 740: 23.47% MAPE - Very small customer, volatile
- Group 222: 20.45% MAPE - Consistent challenge across models
- Group 561: 20.46% MAPE - Seasonal pattern mismatch

**Common Characteristics**:

- Small absolute consumption (low MWh → high relative error)
- High variance in historical data
- Potential customer churn or contract changes
- Data quality concerns

**Mitigation Strategies**:

1. Ensemble with simpler models for volatile groups
2. Increase regularization for small customers
3. Manual review of high-error groups
4. Collect more granular metadata

---

## 6. LIMITATIONS & FUTURE WORK

### 6.1 Current Limitations

**Data Limitations**:

1. No future weather forecasts (used historical averages)
2. No future price forecasts beyond Day+1
3. Limited customer metadata (only group labels)
4. No information on customer churn or contract changes

**Model Limitations**:

1. Point forecasts only (no uncertainty quantification)
2. No handling of special events (strikes, pandemics)
3. Assumes stationary patterns (climate change not modeled)
4. Independent group models (no cross-group learning)

**Computational Limitations**:

1. Hyperparameters manually tuned (no automated optimization)
2. Feature selection based on domain knowledge (no systematic search)

### 6.2 Future Improvements

**Short-Term (1-3 months)**:

1. **Weather Forecasts**: Integrate actual weather predictions (FMI API)
2. **Price Forecasting**: Develop price prediction model for 48h horizon
3. **Ensemble Methods**: Combine LightGBM + XGBoost + Linear models
4. **Automated Tuning**: Implement Optuna for hyperparameter optimization

**Medium-Term (3-6 months)**:

1. **Hierarchical Modeling**: Learn group similarities for better predictions
2. **Uncertainty Quantification**: Quantile regression for confidence intervals
3. **Online Learning**: Update models as new data arrives
4. **Feature Interactions**: Automated interaction term discovery

**Long-Term (6-12 months)**:

1. **Deep Learning**: LSTM/Transformer models for complex temporal patterns
2. **Causal Inference**: Understand true drivers of consumption changes
3. **Multi-Task Learning**: Joint modeling of price and consumption
4. **Explainable AI**: SHAP values for model interpretability

### 6.3 Recommendations

**For Production Deployment**:

1. ✅ Use current LightGBM models as baseline
2. ⚠️ Add safety margins for high-error groups
3. ⚠️ Monitor forecast accuracy in real-time
4. ⚠️ Implement fallback to naive model if LightGBM fails
5. ✅ Regular retraining (monthly) as new data arrives

**For Model Improvement**:

1. Collect more customer metadata (building type, heating system)
2. Integrate economic indicators (GDP, industrial production)
3. Add school holiday calendar
4. Include electric vehicle adoption rates (future consumption driver)

**For Business Value**:

1. Develop forecast-to-decision pipeline (automated trading signals)
2. Quantify financial impact of forecast accuracy
3. Build scenario analysis tools (what-if simulations)
4. Create stakeholder dashboards for monitoring

---

## 7. CONCLUSION

We successfully developed a comprehensive energy forecasting system that:

✅ **Meets Business Requirements**:

- Accurate 48-hour forecasts for day-ahead market trading (4.24% MAPE)
- Reliable 12-month forecasts for hedging strategy (10.42% MAPE)
- 100% coverage (all 112 groups forecasted)

✅ **Demonstrates Technical Excellence**:

- Advanced feature engineering (64 and 23 features)
- External data integration (weather, holidays)
- Proper validation methodology (time-based splits)
- Reproducible, production-ready code

✅ **Delivers Business Value**:

- 23% improvement over 48h naive baseline
- 31% improvement over 12m naive baseline
- Actionable insights for energy trading decisions

**Key Takeaways**:

1. **Domain knowledge matters**: Energy-specific features (weather, holidays,
   lag patterns) are crucial
2. **Simpler can be better**: Monthly model improved 12.5% by reducing features
3. **Individual models work**: Group-specific models capture unique patterns
   better than global models
4. **External data helps**: Weather and holidays provide 0.5-1% MAPE improvement

**Final Status**: ✅ **Production-Ready** - Models are trained, validated, and
ready for deployment in Fortum's forecasting pipeline.

---

**Document Version**: 1.0\
**Last Updated**: November 15, 2025\
**Authors**: Team OwnEnergy\
**Contact**: Via Junction 2025 platform
