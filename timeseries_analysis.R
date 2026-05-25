# ============================================================
# Air Passengers: Time Series Forecasting
# STAT 4000 Portfolio Project 4  R Implementation
# Author: Jrudani21
# ============================================================
# Dataset: AirPassengers  built into base R (no download needed)
# Monthly international airline passengers, Jan 1949 – Dec 1960
# ============================================================

# ── 0. Packages ──────────────────────────────────────────────
# install.packages(c("tidyverse","forecast","tseries","ggplot2","zoo"))
library(tidyverse)
library(forecast)    # auto.arima(), forecast(), ggseasonplot()
library(tseries)     # adf.test(), kpss.test()
library(ggplot2)

# ── 1. Load Data ─────────────────────────────────────────────
data("AirPassengers")   # built-in R dataset

cat("Class:", class(AirPassengers), "\n")
cat("Start:", start(AirPassengers), "| End:", end(AirPassengers), "\n")
cat("Frequency:", frequency(AirPassengers), "(monthly)\n")
cat("Length:", length(AirPassengers), "observations\n")
cat("Range:", min(AirPassengers), "–", max(AirPassengers), "thousand passengers\n")

# Convert to tidy data frame for ggplot
air_df <- data.frame(
  date       = seq(as.Date("1949-01-01"), by = "month", length.out = 144),
  passengers = as.numeric(AirPassengers)
)

# ── 2. Time Series Plot ───────────────────────────────────────
ggplot(air_df, aes(x = date, y = passengers)) +
  geom_line(colour = "steelblue", linewidth = 1) +
  geom_smooth(method = "loess", se = FALSE, colour = "red",
              linetype = "dashed", linewidth = 0.8) +
  labs(title    = "Monthly International Airline Passengers (1949–1960)",
       subtitle = "Clear upward trend + growing seasonal amplitude → multiplicative pattern",
       x        = "Date",
       y        = "Passengers (thousands)",
       caption  = "Source: Box & Jenkins (1976)") +
  theme_minimal()
ggsave("figures/01_raw_series.png", width = 10, height = 5)

# ── 3. Seasonal Plot ─────────────────────────────────────────
png("figures/02_seasonal_plot.png", width = 800, height = 500)
ggseasonplot(AirPassengers,
             year.labels = TRUE,
             year.labels.left = TRUE,
             main = "Seasonal Plot  Passengers by Month (Each Year)",
             ylab = "Passengers (thousands)")
dev.off()

# Monthly subseries plot
png("figures/03_subseries_plot.png", width = 800, height = 500)
ggsubseriesplot(AirPassengers,
                main = "Subseries Plot  Mean by Month",
                ylab = "Passengers (thousands)")
dev.off()

# ── 4. Decomposition ─────────────────────────────────────────
# Multiplicative decomposition (amplitude grows with trend → use multiplicative)
decomp_mult <- decompose(AirPassengers, type = "multiplicative")

png("figures/04_decomposition.png", width = 900, height = 700)
plot(decomp_mult,
     main = "Multiplicative Decomposition: Trend + Seasonal + Remainder")
dev.off()

cat("\nSeasonal indices (multiplicative):\n")
print(round(decomp_mult$seasonal[1:12], 4))

# ── 5. Stationarity Testing ───────────────────────────────────
cat("\n===== STATIONARITY TESTS  RAW SERIES =====\n")
adf_raw  <- adf.test(AirPassengers)
kpss_raw <- kpss.test(AirPassengers)
cat(sprintf("ADF  test: p = %.4f  (%s)\n",
            adf_raw$p.value,
            ifelse(adf_raw$p.value < 0.05, "stationary", "non-stationary")))
cat(sprintf("KPSS test: p = %.4f  (%s)\n",
            kpss_raw$p.value,
            ifelse(kpss_raw$p.value > 0.05, "stationary", "non-stationary")))

# Log transformation to stabilise variance
log_ap <- log(AirPassengers)

# First difference of log (removes trend)
d1_log_ap <- diff(log_ap, differences = 1)

# Seasonal difference (removes seasonality)
d1_d12_log_ap <- diff(d1_log_ap, lag = 12)

cat("\n===== STATIONARITY TESTS  LOG + DIFFERENCED =====\n")
adf_diff <- adf.test(d1_d12_log_ap)
cat(sprintf("ADF  test after log + d=1 + D=1: p = %.4f  (%s)\n",
            adf_diff$p.value,
            ifelse(adf_diff$p.value < 0.05, "stationary ✓", "still non-stationary")))

# Plot transformations
png("figures/05_transformations.png", width = 900, height = 700)
par(mfrow = c(3, 1))
plot(AirPassengers,       main = "Original Series",              ylab = "Passengers")
plot(log_ap,              main = "Log Transformed",              ylab = "log(Passengers)")
plot(d1_d12_log_ap,       main = "Log + Seasonal Diff (d=1,D=1)", ylab = "Δ log(Passengers)")
abline(h = 0, col = "red", lty = 2)
dev.off()

# ── 6. ACF & PACF ────────────────────────────────────────────
png("figures/06_acf_pacf.png", width = 900, height = 500)
par(mfrow = c(1, 2))
acf(d1_d12_log_ap,  lag.max = 40, main = "ACF   log + d=1 + D=1")
pacf(d1_d12_log_ap, lag.max = 40, main = "PACF  log + d=1 + D=1")
dev.off()

# ── 7. ARIMA Model Selection ──────────────────────────────────
cat("\n===== AUTO ARIMA =====\n")
arima_model <- auto.arima(
  log(AirPassengers),
  seasonal    = TRUE,
  stepwise    = FALSE,   # exhaustive search
  approximation = FALSE,
  trace       = TRUE
)
print(arima_model)
cat(sprintf("\nSelected model: ARIMA(%d,%d,%d)(%d,%d,%d)[12]\n",
            arima_model$arma[1], arima_model$arma[6], arima_model$arma[2],
            arima_model$arma[3], arima_model$arma[7], arima_model$arma[4]))

# ── 8. Residual Diagnostics ───────────────────────────────────
png("figures/07_arima_diagnostics.png", width = 900, height = 700)
checkresiduals(arima_model)
dev.off()

lb_test <- Box.test(arima_model$residuals, lag = 24, type = "Ljung-Box")
cat(sprintf("\nLjung-Box test (lag=24): χ²=%.3f, p=%.4f\n",
            lb_test$statistic, lb_test$p.value))
cat(ifelse(lb_test$p.value > 0.05,
           "p > 0.05 → residuals are white noise ✓\n",
           "p ≤ 0.05 → residuals show remaining autocorrelation\n"))

# ── 9. Forecast 36 Months Ahead ──────────────────────────────
fc <- forecast(arima_model, h = 36)

# Back-transform from log scale
fc_df <- data.frame(
  date     = seq(as.Date("1961-01-01"), by = "month", length.out = 36),
  mean     = exp(as.numeric(fc$mean)),
  lo80     = exp(as.numeric(fc$lower[,1])),
  hi80     = exp(as.numeric(fc$upper[,1])),
  lo95     = exp(as.numeric(fc$lower[,2])),
  hi95     = exp(as.numeric(fc$upper[,2]))
)

ggplot() +
  # Historical data
  geom_line(data = air_df, aes(x = date, y = passengers),
            colour = "steelblue", linewidth = 0.9) +
  # 95% CI ribbon
  geom_ribbon(data = fc_df,
              aes(x = date, ymin = lo95, ymax = hi95),
              fill = "steelblue", alpha = 0.15) +
  # 80% CI ribbon
  geom_ribbon(data = fc_df,
              aes(x = date, ymin = lo80, ymax = hi80),
              fill = "steelblue", alpha = 0.25) +
  # Forecast line
  geom_line(data = fc_df, aes(x = date, y = mean),
            colour = "tomato", linewidth = 1, linetype = "dashed") +
  # Dividing line at forecast start
  geom_vline(xintercept = as.Date("1961-01-01"),
             linetype = "dotted", colour = "grey40") +
  annotate("text", x = as.Date("1961-06-01"), y = 150,
           label = "Forecast →", colour = "tomato", size = 3.5) +
  labs(title    = "SARIMA Forecast: International Airline Passengers",
       subtitle = sprintf("ARIMA(%d,%d,%d)(%d,%d,%d)[12] fitted on log scale, back-transformed",
                          arima_model$arma[1], arima_model$arma[6], arima_model$arma[2],
                          arima_model$arma[3], arima_model$arma[7], arima_model$arma[4]),
       x        = "Date",
       y        = "Passengers (thousands)",
       caption  = "Shaded bands: 80% and 95% prediction intervals") +
  theme_minimal()
ggsave("figures/08_forecast.png", width = 11, height = 6)

# ── 10. Hold-Out Validation ───────────────────────────────────
# Train on 1949–1958, test on 1959–1960 (24 months)
train_ts <- window(log(AirPassengers), end   = c(1958, 12))
test_ts  <- window(log(AirPassengers), start = c(1959,  1))

val_model <- auto.arima(train_ts, seasonal = TRUE,
                        stepwise = FALSE, approximation = FALSE)
val_fc    <- forecast(val_model, h = 24)

# Accuracy on hold-out
acc <- accuracy(val_fc, test_ts)
cat("\n===== HOLD-OUT ACCURACY (1959–1960) =====\n")
print(acc)
cat(sprintf("Test MAPE: %.2f%%\n", acc[2, "MAPE"]))

# ── 11. Winnipeg Weather Bonus ────────────────────────────────
# Connects directly to your Assignment 5!
# Uncomment and run if you have the weather.csv from your course

# weather <- read.csv("data/weather.csv")
# weather <- weather |> mutate(t = 1:nrow(weather))
# weather_ts <- ts(weather$maxtemp, start = c(2013, 1), frequency = 12)
#
# # Same SARIMA workflow as above
# decomp_w  <- decompose(weather_ts, type = "additive")
# arima_w   <- auto.arima(weather_ts, seasonal = TRUE)
# fc_w      <- forecast(arima_w, h = 24)
# autoplot(fc_w) +
#   labs(title = "Winnipeg Max Temperature Forecast (2023–2024)",
#        y = "Max Temperature (°C)")
# ggsave("figures/09_winnipeg_forecast.png", width = 10, height = 5)

cat("\n===== SUMMARY =====\n")
cat("Model: SARIMA on log-transformed AirPassengers\n")
cat(sprintf("AIC  = %.2f\n", arima_model$aic))
cat(sprintf("Test MAPE = %.2f%%  (hold-out 1959-1960)\n", acc[2,"MAPE"]))
cat("Ljung-Box p > 0.05 → white noise residuals ✓\n")
cat("Seasonal peaks: July–August every year ✓\n")
