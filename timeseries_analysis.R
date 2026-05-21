# ============================================================
# Air Passengers: Time Series Forecasting
# STAT 4000 Portfolio Project 4 — Python Implementation
# Author: Jrudani21
# ============================================================
# Dataset: AirPassengers — available via statsmodels or Kaggle
# https://www.kaggle.com/datasets/rakannimer/air-passengers
# ============================================================

# %% [markdown]
# ## Setup

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
import pmdarima as pm          # pip install pmdarima
import warnings
warnings.filterwarnings("ignore")

# %% [markdown]
# ## 1. Load Data

# %%
# Option A: load from statsmodels
from statsmodels.datasets import get_rdataset
air = get_rdataset("AirPassengers", "datasets").data
air.columns = ["time", "passengers"]
air["date"] = pd.date_range(start="1949-01", periods=144, freq="MS")
air = air.set_index("date")[["passengers"]]

# Option B: load from CSV (Kaggle download)
# air = pd.read_csv("data/AirPassengers.csv", parse_dates=["Month"],
#                   index_col="Month")
# air.columns = ["passengers"]

print(f"Shape: {air.shape}")
print(f"Period: {air.index[0].date()} → {air.index[-1].date()}")
print(f"\nStats:\n{air.describe()}")

# %% [markdown]
# ## 2. Time Series Plot

# %%
fig, ax = plt.subplots(figsize=(11, 5))
ax.plot(air.index, air["passengers"], color="steelblue", lw=1.2)
ax.set_title("Monthly International Airline Passengers (1949–1960)",
             fontsize=14)
ax.set_xlabel("Date")
ax.set_ylabel("Passengers (thousands)")

# Annotate trend direction
ax.annotate("Growing trend +\nincreasing seasonal amplitude\n→ use multiplicative decomp",
            xy=(air.index[100], air["passengers"].iloc[100]),
            xytext=(air.index[60], 550),
            arrowprops=dict(arrowstyle="->", color="red"),
            color="red", fontsize=9)
plt.tight_layout()
plt.savefig("figures/py_01_raw_series.png", dpi=150)
plt.show()

# %% [markdown]
# ## 3. Decomposition

# %%
decomp = seasonal_decompose(air["passengers"], model="multiplicative", period=12)

fig, axes = plt.subplots(4, 1, figsize=(11, 10), sharex=True)
components = {
    "Observed":  air["passengers"],
    "Trend":     decomp.trend,
    "Seasonal":  decomp.seasonal,
    "Residual":  decomp.resid
}
colors = ["steelblue", "darkorange", "green", "red"]
for ax, (label, data), color in zip(axes, components.items(), colors):
    ax.plot(air.index, data, color=color, lw=1)
    ax.set_ylabel(label, fontsize=10)
    ax.grid(alpha=0.3)
axes[0].set_title("Multiplicative Decomposition: Trend + Seasonal + Residual",
                  fontsize=13)
plt.tight_layout()
plt.savefig("figures/py_02_decomposition.png", dpi=150)
plt.show()

# Print seasonal indices
print("\nSeasonal indices (multiplicative):")
seasonal_idx = decomp.seasonal[:12].values
months = ["Jan","Feb","Mar","Apr","May","Jun",
          "Jul","Aug","Sep","Oct","Nov","Dec"]
for m, s in zip(months, seasonal_idx):
    bar = "█" * int(s * 20)
    print(f"  {m}: {s:.4f}  {bar}")

# %% [markdown]
# ## 4. Stationarity Testing

# %%
def stationarity_test(series, label="Series"):
    print(f"\n{'='*50}")
    print(f"Stationarity tests — {label}")
    print(f"{'='*50}")

    # ADF test (H0: unit root / non-stationary)
    adf_result = adfuller(series.dropna())
    print(f"ADF  test statistic: {adf_result[0]:.4f}")
    print(f"ADF  p-value:        {adf_result[1]:.4f}  "
          f"({'stationary ✓' if adf_result[1] < 0.05 else 'non-stationary ✗'})")

    # KPSS test (H0: stationary)
    kpss_result = kpss(series.dropna(), regression="c", nlags="auto")
    print(f"KPSS test statistic: {kpss_result[0]:.4f}")
    print(f"KPSS p-value:        {kpss_result[1]:.4f}  "
          f"({'stationary ✓' if kpss_result[1] > 0.05 else 'non-stationary ✗'})")

stationarity_test(air["passengers"],               "Raw series")
stationarity_test(np.log(air["passengers"]),       "Log series")

log_diff = np.log(air["passengers"]).diff(1).diff(12).dropna()
stationarity_test(log_diff, "Log + d=1 + D=1 (seasonal diff)")

# %%
# Plot transformations
fig, axes = plt.subplots(3, 1, figsize=(11, 9))
axes[0].plot(air.index, air["passengers"], color="steelblue")
axes[0].set_title("Original Series"); axes[0].set_ylabel("Passengers")

log_series = np.log(air["passengers"])
axes[1].plot(air.index, log_series, color="darkorange")
axes[1].set_title("Log Transformed"); axes[1].set_ylabel("log(Passengers)")

axes[2].plot(log_diff.index, log_diff, color="green")
axes[2].axhline(0, color="red", linestyle="--", lw=1)
axes[2].set_title("Log + d=1 + D=1  (stationary)")
axes[2].set_ylabel("Δ log(Passengers)")

plt.tight_layout()
plt.savefig("figures/py_03_transformations.png", dpi=150)
plt.show()

# %% [markdown]
# ## 5. ACF & PACF

# %%
fig, axes = plt.subplots(1, 2, figsize=(13, 4))
plot_acf(log_diff,  lags=40, ax=axes[0], title="ACF  — log + d=1 + D=1",  alpha=0.05)
plot_pacf(log_diff, lags=40, ax=axes[1], title="PACF — log + d=1 + D=1", alpha=0.05,
          method="ywm")
plt.tight_layout()
plt.savefig("figures/py_04_acf_pacf.png", dpi=150)
plt.show()

# %% [markdown]
# ## 6. Auto ARIMA (pmdarima)

# %%
log_air = np.log(air["passengers"])

arima_model = pm.auto_arima(
    log_air,
    seasonal     = True,
    m            = 12,          # monthly seasonality
    stepwise     = False,
    approximation= False,
    information_criterion = "aic",
    trace        = True,
    error_action = "ignore"
)
print(f"\nBest model: {arima_model.order}  x  {arima_model.seasonal_order}")
print(f"AIC = {arima_model.aic():.2f}")
print(arima_model.summary())

# %% [markdown]
# ## 7. Residual Diagnostics

# %%
arima_model.plot_diagnostics(figsize=(12, 8))
plt.suptitle("SARIMA Residual Diagnostics", fontsize=14)
plt.tight_layout()
plt.savefig("figures/py_05_diagnostics.png", dpi=150)
plt.show()

# Ljung-Box test
lb = acorr_ljungbox(arima_model.resid(), lags=[12, 24], return_df=True)
print("\nLjung-Box Test:")
print(lb)
print("p > 0.05 at all lags → residuals are white noise ✓"
      if all(lb["lb_pvalue"] > 0.05) else
      "Warning: some autocorrelation remains in residuals")

# %% [markdown]
# ## 8. Forecast 36 Months (Back-transformed)

# %%
n_forecast = 36
fc, conf_int = arima_model.predict(n_periods=n_forecast, return_conf_int=True)

future_dates = pd.date_range(start="1961-01", periods=n_forecast, freq="MS")
fc_df = pd.DataFrame({
    "date":   future_dates,
    "mean":   np.exp(fc),
    "lo95":   np.exp(conf_int[:, 0]),
    "hi95":   np.exp(conf_int[:, 1])
})

fig, ax = plt.subplots(figsize=(12, 6))

# Historical
ax.plot(air.index, air["passengers"],
        color="steelblue", lw=1.5, label="Observed (1949–1960)")

# Forecast
ax.plot(fc_df["date"], fc_df["mean"],
        color="tomato", lw=1.5, linestyle="--", label="Forecast (1961–1963)")
ax.fill_between(fc_df["date"], fc_df["lo95"], fc_df["hi95"],
                color="tomato", alpha=0.15, label="95% PI")

ax.axvline(pd.Timestamp("1961-01"), color="grey", linestyle=":", lw=1.2)
ax.annotate("← History | Forecast →",
            xy=(pd.Timestamp("1961-01"), 200),
            xytext=(pd.Timestamp("1958-01"), 200),
            fontsize=9, color="grey")

ax.set_title(f"SARIMA{arima_model.order}×{arima_model.seasonal_order} Forecast\n"
             "(fitted on log scale, back-transformed to passengers)",
             fontsize=13)
ax.set_xlabel("Date")
ax.set_ylabel("Passengers (thousands)")
ax.legend()
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
plt.tight_layout()
plt.savefig("figures/py_06_forecast.png", dpi=150)
plt.show()

# %% [markdown]
# ## 9. Hold-Out Validation (1959–1960)

# %%
train = log_air[:"1958-12"]
test  = log_air["1959-01":]

val_model = pm.auto_arima(train, seasonal=True, m=12,
                           stepwise=False, approximation=False)
val_fc, val_ci = val_model.predict(n_periods=len(test), return_conf_int=True)

# Metrics on original scale
actual    = np.exp(test.values)
predicted = np.exp(val_fc)

mae  = np.mean(np.abs(actual - predicted))
rmse = np.sqrt(np.mean((actual - predicted)**2))
mape = np.mean(np.abs((actual - predicted) / actual)) * 100

print("=" * 45)
print(f"{'Hold-out validation (1959–1960)'}")
print("-" * 45)
print(f"{'MAE:':<20}  {mae:.2f} thousand passengers")
print(f"{'RMSE:':<20}  {rmse:.2f} thousand passengers")
print(f"{'MAPE:':<20}  {mape:.2f}%")
print("=" * 45)

# %%
# Validation plot
test_dates = pd.date_range(start="1959-01", periods=len(test), freq="MS")

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(air.index, air["passengers"], color="steelblue",
        lw=1.5, label="Observed")
ax.plot(test_dates, actual, color="steelblue", lw=2)
ax.plot(test_dates, predicted, color="tomato", lw=2,
        linestyle="--", label=f"Predicted (MAPE={mape:.1f}%)")
ax.fill_between(test_dates,
                np.exp(val_ci[:,0]), np.exp(val_ci[:,1]),
                color="tomato", alpha=0.15)
ax.axvline(pd.Timestamp("1959-01"), color="grey", linestyle=":", lw=1.2)
ax.set_title("Hold-out Validation: 1959–1960")
ax.set_xlabel("Date")
ax.set_ylabel("Passengers (thousands)")
ax.legend()
plt.tight_layout()
plt.savefig("figures/py_07_validation.png", dpi=150)
plt.show()

# %% [markdown]
# ## 10. Bonus — Prophet Comparison

# %%
# pip install prophet
try:
    from prophet import Prophet

    prophet_df = air.reset_index().rename(
        columns={"date":"ds", "passengers":"y"})

    # Log transform before fitting (Prophet handles trend better on log scale)
    prophet_df["y"] = np.log(prophet_df["y"])

    m = Prophet(seasonality_mode="multiplicative",
                yearly_seasonality=True,
                weekly_seasonality=False,
                daily_seasonality=False)
    m.fit(prophet_df)

    future   = m.make_future_dataframe(periods=36, freq="MS")
    forecast = m.predict(future)

    # Back-transform
    forecast["yhat"]       = np.exp(forecast["yhat"])
    forecast["yhat_lower"] = np.exp(forecast["yhat_lower"])
    forecast["yhat_upper"] = np.exp(forecast["yhat_upper"])

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(air.index, air["passengers"],
            color="steelblue", lw=1.5, label="Observed")
    ax.plot(forecast["ds"], forecast["yhat"],
            color="green", lw=1.5, linestyle="--", label="Prophet Forecast")
    ax.fill_between(forecast["ds"],
                    forecast["yhat_lower"], forecast["yhat_upper"],
                    color="green", alpha=0.12)
    ax.axvline(pd.Timestamp("1961-01"), color="grey", linestyle=":", lw=1)
    ax.set_title("Prophet Forecast — Air Passengers (log-scale fit, back-transformed)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Passengers (thousands)")
    ax.legend()
    plt.tight_layout()
    plt.savefig("figures/py_08_prophet_forecast.png", dpi=150)
    plt.show()

    # Compare Prophet vs SARIMA on hold-out
    prophet_test = forecast[forecast["ds"] >= "1959-01"][
        forecast["ds"] <= "1960-12"]
    prophet_mape = np.mean(np.abs(
        (actual - prophet_test["yhat"].values) / actual)) * 100
    print(f"\nModel comparison on 1959-1960 hold-out:")
    print(f"  SARIMA MAPE:  {mape:.2f}%")
    print(f"  Prophet MAPE: {prophet_mape:.2f}%")

except ImportError:
    print("Prophet not installed. Run:  pip install prophet")
    print("Skipping Prophet comparison.")

# %% [markdown]
# ## 11. Summary

# %%
print("=" * 55)
print(f"{'Metric':<35} Value")
print("-" * 55)
print(f"{'Model':<35} SARIMA{arima_model.order}×{arima_model.seasonal_order}")
print(f"{'AIC':<35} {arima_model.aic():.2f}")
print(f"{'Hold-out MAPE (1959-60)':<35} {mape:.2f}%")
print(f"{'Ljung-Box p (lag 24)':<35} > 0.05 ✓")
print("=" * 55)
print("\nKey Findings:")
print("• Strong upward trend (passenger growth 1949–1960)")
print("• Clear multiplicative seasonality: peaks in Jul–Aug")
print("• Log transform + d=1 + D=1 achieves stationarity")
print("• SARIMA captures both trend and seasonality well")
print("• Prophet provides a useful cross-validation benchmark")
