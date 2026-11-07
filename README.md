# ✈️ From Box-Jenkins to Prophet: Forecasting Air Travel With 60 Years of Methods

> **Can we forecast monthly airline passenger demand using 12 years of historical data?**  
> A full SARIMA forecasting workflow on the classic Box-Jenkins AirPassengers dataset — in both R and Python, with a Prophet comparison.

❗[Forecast](figures/py_06_forecast.png)

---

## 📋 Table of Contents
- [Problem](#problem)
- [Data](#data)
- [Methodology](#methodology)
- [Key Results](#key-results)
- [How to Reproduce](#how-to-reproduce)
- [What I'd Do Next](#what-id-do-next)

---

## Problem

This project builds a complete **time series forecasting pipeline** for monthly airline passenger counts (1949–1960). The workflow follows the classic **Box-Jenkins methodology**:

1. **Visualise** — identify trend, seasonality, and variance behaviour
2. **Decompose** — separate trend, seasonal, and residual components
3. **Transform** — stabilise variance (log) and achieve stationarity (differencing)
4. **Identify** — use ACF/PACF to suggest ARIMA order; confirm with `auto.arima`/`pmdarima`
5. **Diagnose** — check residuals with Ljung-Box test
6. **Forecast** — 36 months ahead with prediction intervals
7. **Validate** — hold-out test on 1959–1960 (MAPE)
8. **Compare** — benchmark SARIMA against Facebook Prophet

---

## Data

| Property | Detail |
|---|---|
| **Source** | Built into base R (`data("AirPassengers")`) / statsmodels / Kaggle |
| **Original** | Box, G.E.P & Jenkins, G.M. (1976). *Time Series Analysis* |
| **Size** | 144 monthly observations (Jan 1949 – Dec 1960) |
| **Unit** | Thousands of international airline passengers |
| **License** | Public domain |

### Why this dataset?
- Perfect for demonstrating multiplicative seasonality (amplitude grows with trend)
- Small enough to understand completely
- Canonical — every time-series textbook uses it, so interviewers know it

---

## Methodology

### Why multiplicative decomposition?
The seasonal swings **grow proportionally with the trend** (summers get bigger as the series grows). Additive decomposition assumes constant seasonal amplitude — wrong here. Multiplicative handles this correctly.

### Transformation steps
```
Raw series
  → Log transform         (stabilises growing variance)
  → d=1 first difference  (removes trend → mean-stationary)
  → D=1 seasonal diff     (removes seasonality → fully stationary)
```

ADF test p-value after all three steps: **p < 0.05** → stationary ✓

### Model selection
`auto.arima` (R) and `pmdarima.auto_arima` (Python) both perform exhaustive AIC-based search.  
Typical result: **ARIMA(2,1,1)(0,1,0)[12]** or **ARIMA(0,1,1)(0,1,1)[12]**

### R (`R/timeseries_analysis.R`)
| Step | Function | Package |
|---|---|---|
| Time series plot | `autoplot()`, `ggplot2` | forecast, ggplot2 |
| Seasonal plot | `ggseasonplot()` | forecast |
| Decomposition | `decompose(type="multiplicative")` | stats |
| ADF test | `adf.test()` | tseries |
| KPSS test | `kpss.test()` | tseries |
| ACF / PACF | `acf()`, `pacf()` | stats |
| Model selection | `auto.arima(stepwise=FALSE)` | forecast |
| Residual check | `checkresiduals()` | forecast |
| Ljung-Box | `Box.test(type="Ljung-Box")` | stats |
| Forecast | `forecast(h=36)` | forecast |
| Accuracy | `accuracy()` | forecast |

### Python (`Python/timeseries_analysis.py`)
| Step | Function | Package |
|---|---|---|
| Decomposition | `seasonal_decompose(model="multiplicative")` | statsmodels |
| ADF / KPSS | `adfuller()`, `kpss()` | statsmodels |
| ACF / PACF | `plot_acf()`, `plot_pacf()` | statsmodels |
| Model selection | `pm.auto_arima(seasonal=True, m=12)` | pmdarima |
| Fit SARIMA | `SARIMAX().fit()` | statsmodels |
| Ljung-Box | `acorr_ljungbox()` | statsmodels |
| Forecast | `model.predict(return_conf_int=True)` | pmdarima |
| Prophet | `Prophet(seasonality_mode="multiplicative")` | prophet |

---

## Key Results

### Decomposition
| Component | Finding |
|---|---|
| Trend | Steady linear growth from ~125k to ~450k passengers |
| Seasonality | Strong July–August peak every year (multiplier ~1.35) |
| Residual | Small, random — model captures most variation |

### Seasonal Indices (multiplicative)
| Jan | Feb | Mar | Apr | May | Jun | Jul | Aug | Sep | Oct | Nov | Dec |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.91 | 0.88 | 1.00 | 1.01 | 1.02 | 1.14 | **1.23** | **1.22** | 1.06 | 0.93 | 0.83 | 0.89 |

→ July is the peak month (23% above average), November is the trough (17% below)

### Model Fit
| Metric | Value |
|---|---|
| Selected model | ARIMA(0,1,1)(0,1,1)[12] |
| AIC | -483.39 (log scale, full sample n=144) |
| Ljung-Box p-value (lag 24) | > 0.05 → white noise residuals ✓ |
| Hold-out MAPE, 24 months (1959-60) | **8.52%** |
| Hold-out MAPE, 12 months (1960 only) | 2.90% |

AIC is for SARIMAX(0,1,1)(0,1,1)[12] fitted by statsmodels on log(AirPassengers), all 144 months. The R script (`timeseries_analysis.R`, re-run on R 4.6 with `auto.arima`) gives the same order and AIC = -483.40, and a hold-out MAPE of 8.52% on the passenger scale. AIC on the raw series (1021.00) is not comparable with the log-scale value. The Ljung-Box row was checked with statsmodels residuals: p at lag 24 is 0.32 (all residuals) and 0.48 (first 13 start-up residuals dropped).

### SARIMA vs Prophet (Hold-out 1959–1960)
| Model | MAPE |
|---|---|
| SARIMA (0,1,1)(0,1,1)[12] | 8.52% |
| Prophet | not re-measured |

→ MAPE is on the original passenger scale (forecasts back-transformed with exp), model fitted on log(passengers), order (0,1,1)(0,1,1)[12] refitted on the training window only, statsmodels SARIMAX. The earlier ~3% figure is numerically consistent with the 12-month 1960 hold-out (2.90%) but not with the 24-month 1959-60 hold-out. The Prophet comparison has not been re-run, so no Prophet number is claimed.

Reproduce: `python -c "import numpy as np;from statsmodels.datasets import get_rdataset as g;from statsmodels.tsa.statespace.sarimax import SARIMAX as S;y=np.log(g('AirPassengers','datasets').data.iloc[:,-1].astype(float).values);[print(n,np.mean(abs(1-np.exp(S(y[:144-n],order=(0,1,1),seasonal_order=(0,1,1,12)).fit(disp=False).forecast(n)-y[144-n:])))*100) for n in (24,12)]"`

---

## How to Reproduce

### R
```r
# No data download needed — AirPassengers is built into R
install.packages(c("tidyverse","forecast","tseries"))
source("R/timeseries_analysis.R")
```

### Python
```bash
pip install pandas numpy matplotlib statsmodels pmdarima prophet
python Python/timeseries_analysis.py
```

> **Note:** `prophet` can be slow to install. If you skip it, comment out Section 10 in the Python script — the rest runs fine without it.

---
## What I'd Do Next
- Apply the same pipeline to a **modern dataset** (e.g., monthly Winnipeg transit ridership or electricity consumption)
- Try **ETS (Exponential Smoothing)** models via `ets()` in R and compare AIC with SARIMA
- Use **cross-validation with rolling windows** (`tsCV()` in R) for more robust MAPE estimation
- Build a **Streamlit dashboard** that lets a user upload any monthly CSV and get an auto-SARIMA forecast

---

*Data: Box, G.E.P & Jenkins, G.M. (1976). Time Series Analysis: Forecasting and Control. San Francisco: Holden-Day. Available in base R via `data("AirPassengers")`.*
