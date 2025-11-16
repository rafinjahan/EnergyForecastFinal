# METHODOLOGY DOCUMENT

## Fortum Energy Forecasting Challenge — Junction 2025

**Team**: OwnEnergy    **Date**: 15 Nov 2025    **Scope**: 48-hour and 12-month demand forecasts for 112 Fortum customer groups.

---

## Executive Summary

- LightGBM underpins both horizons: a single weather-aware model for 48 hours and 112 lightweight models for 12 months.
- Forecast accuracy: **3.68% MAPE** (48h) and **10.25% MAPE** (12m), improving on naive lag baselines by roughly 23% and 31% respectively.
- Differentiators: FMI weather enrichment, Finnish holiday awareness, categorical customer descriptors, disciplined time-based validation, and automated artifact export.

---

## 1. Models & Rationale

### 48-Hour Horizon
- **Architecture**: Shared LightGBM trained on the full long-form hourly table with `group_id`, `customer_type`, `contract_type`, and `consumption_level` treated as categorical features; captures global weather effects and per-group offsets simultaneously.
- **Tuning choices**: 2 000 trees, learning rate 0.05, 128 leaves, strong row/column subsampling, early stopping after 100 no-improvement rounds.
- **Why shared?** ~185k hourly samples benefit from pooled learning, while categorical splits prevent leakage between dissimilar segments.

### 12-Month Horizon
- **Architecture**: Independent LightGBM per group using monthly aggregates (36‑45 observations each) with rich lags and seasonal encodings.
- **Tuning choices**: 500 trees, depth ≤7, 48 leaves, stronger regularization, early stopping after 30 stagnant rounds.
- **Why per-group?** Limited monthly history makes pooled approaches unstable; per-group boosters keep the feature-to-sample ratio manageable.

### Alternatives Considered
- Per-group hourly models lacked data for small customers.
- Hierarchical deep nets were heavier to train and provided no lift during rapid prototyping.
- The final hybrid balances accuracy, interpretability, and runtime.

---

## 2. Features & External Data

### 48-Hour Feature Set (19 numeric + 4 categorical)
- **Time & calendar**: hour, weekday, weekend flag, month, ISO week, year.
- **Holiday**: Finnish statutory holiday indicator from the `holidays` package.
- **Customer metadata**: categorical splits for customer type, contract type, consumption tier, and the `group_id` identifier.
- **Weather**: temperature, wind speed, precipitation, humidity from all Finnish Meteorological Institute stations mapped to each group; multiple stations are averaged per hour and missing values are imputed with group-specific medians.
- **Price**: day-ahead spot price plus a 24-hour lag to capture price momentum.
- **Autoregressive load**: lags at 1, 24, and 168 hours with per-group median backfilling.

### 12-Month Feature Set (23 total)
- **Seasonality**: month sine/cosine encodings.
- **Weather**: monthly means/totals of the FMI variables above.
- **Price**: monthly mean, 3- and 6-month moving averages, and rolling volatility indicators.
- **Autoregressive signal**: previous month and same month last year.
- **Group descriptors**: encoded region, province, contract type, and consumption level parsed from Fortum’s group labels.
- **Holiday burden**: counts of holiday days/periods plus average distance to the next/previous holiday.

### External Sources
- Fortum training bundle (consumption, groups, prices).
- FMI weather spreadsheets (multiple stations per region).
- Finnish holiday calendar via the `holidays` Python package.

---

## 3. Training & Validation Approach

1. **Preparation**
   - Convert consumption wide tables to long format; align timestamps in UTC for both consumption and price series.
   - Enrich with FMI weather (`enrich_consumption_with_all_weather`), aggregate multi-station observations, and impute gaps with station medians.
   - Add Finnish holiday indicators plus lagged/rolling price features; aggregate to monthly totals for the 12-month workflow.
2. **Splits**
   - Purely time-based.
   - 48h model trained on data before 1 Jan 2024 and validated on Jan–Sep 2024 (aligned with the Oct 1–2, 2024 forecast window).
   - 12m models use the first 80% of months (Jan 2021–Dec 2023) for training and the remaining months (Jan–Sep 2024) for validation.
3. **Modeling workflow**
   - Shared 48h model fitted once with categorical handling, early stopping, and persistence via joblib; category levels are frozen for inference stability.
   - Monthly models loop over group IDs, skip groups with <6 monthly entries, train with the simplified feature list, and save per-group pickles plus metrics CSVs.
4. **Metrics & monitoring**
   - MAPE (primary), MAE, and RMSE reported per group and in aggregate; highest-error segments flagged for manual review.
   - Automated checks ensure non-negative predictions, feature availability, and sufficient training rows per group.

---

## 4. Business Alignment

- **Day-ahead trading**: 48-hour forecasts feed Nord Pool purchase planning, reduce imbalance fees (10‑50% premiums), and highlight price-sensitive spot contracts.
- **Hedging & budgeting**: 12-month forecasts guide long-term hedges, peak capacity planning, and financial budgeting for both residential and SME portfolios.
- **Customer segmentation**: metadata-aware modeling respects geographic weather exposure, contract incentives, and consumption tiers.
- **Risk mitigation**: operations teams can hold safety buffers for volatile groups, monitor live accuracy, and fall back to naive lags during data incidents.


---

## 5. Limitations & Next Steps

- **Data gaps**: only historical weather/price observations; no probabilistic or forecast inputs yet.
- **Model scope**: monthly models remain independent (no cross-learning) and provide point estimates only.
- **Tuning**: manual hyperparameters; automated search (e.g., Optuna) is planned.
- **Engineering polish**: logging is fairly verbose/duplicated between scripts and some modules keep unused imports; tightening those would make the toolkit leaner.

**Recommended roadmap**
1. Integrate short-term weather and price forecasts into the 48h pipeline.
2. Capture richer customer metadata (building type, heating systems) and local events (school breaks, industrial shutdowns).
3. Automate retraining, monitoring, and alerting for high-MAPE groups.
4. Explore probabilistic outputs (quantile boosting) for risk-aware decision making.

---

## 6. Conclusion

- The hybrid LightGBM strategy is production-ready, satisfies Fortum’s operational requirements, and delivers double-digit improvements over naive baselines.
- Weather enrichment, holiday context, and categorical customer descriptors are the primary accuracy levers.
- Future work should prioritize richer external forecasts, automated tuning, and business-facing monitoring dashboards.

**Document Version**: 2.0    **Last Updated**: 15 Nov 2025
