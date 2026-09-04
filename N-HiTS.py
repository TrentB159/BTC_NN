import numpy as np
import pandas as pd
import torch

from neuralforecast import NeuralForecast
from neuralforecast.models import NHITS
from neuralforecast.auto import AutoNHITS
from neuralforecast.losses.pytorch import HuberLoss, MAE
from ray import tune
import matplotlib.pyplot as plt

horizon = 180

# --- 1. Load and Clean Dataset ---
df = pd.read_csv("BTC_data.csv", index_col=0)
df.columns = ["open", "high", "low", "close", "volume"]
df.index = pd.to_datetime(df.index)
df.index.name = "ds"
df = df.sort_index()

# --- 2. Feature Engineering ---
close = df["close"]

df["log_ret_1d"] = np.log(close / close.shift(1))
df["log_ret_30d"] = np.log(close / close.shift(30))
df["log_ret_90d"] = np.log(close / close.shift(90))

df["sma_50"] = close.rolling(50).mean()
df["sma_200"] = close.rolling(200).mean()

df["price_to_sma200"] = close / df["sma_200"]
df["sma50_to_sma200"] = df["sma_50"] / df["sma_200"]

df["volatility_30d"] = df["log_ret_1d"].rolling(30).std() * np.sqrt(365)

high_180 = df["high"].rolling(180).max()
low_180 = df["low"].rolling(180).min()
denom = (high_180 - low_180).replace(0, np.nan)
df["channel_pos_180d"] = ((close - low_180) / denom).clip(0, 1)

df["day_of_week"] = df.index.dayofweek / 6.0
df["day_of_week_sin"] = np.sin(2 * np.pi * df.index.dayofweek / 7.0)
df["day_of_week_cos"] = np.cos(2 * np.pi * df.index.dayofweek / 7.0)
df["day_of_year_sin"] = np.sin(2 * np.pi * df.index.dayofyear / 365.25)
df["day_of_year_cos"] = np.cos(2 * np.pi * df.index.dayofyear / 365.25)

df["y"] = np.log(close / close.shift(1))
df = df.replace([np.inf, -np.inf], np.nan).dropna().reset_index()

hist_exog_cols = [
    "log_ret_1d",
    "log_ret_30d",
    "log_ret_90d",
    "price_to_sma200",
    "sma50_to_sma200",
    "volatility_30d",
    "channel_pos_180d",
]

futr_exog_cols = [
    "day_of_week",
    "day_of_week_sin",
    "day_of_week_cos",
    "day_of_year_sin",
    "day_of_year_cos",
]

Y_df = df.copy()
Y_df["unique_id"] = "BTC"

Y_df = Y_df[
    ["unique_id", "ds", "y", "close"] + hist_exog_cols + futr_exog_cols
].sort_values("ds")

# --- 3. Hyperparameter Tuning ---
nhits_config = {
    "max_steps": tune.choice([400, 800]),
    "val_check_steps": 5,
    "early_stop_patience_steps": 4,
    "learning_rate": tune.choice([1e-4, 3e-4]),
    "input_size": tune.choice([360, 720]),
    "batch_size": tune.choice([32, 64]),
    "windows_batch_size": 256,
    "n_pool_kernel_size": tune.choice([[2, 2, 1], [4, 2, 1]]),
    "n_freq_downsample": tune.choice([[7, 1, 1], [14, 7, 1]]),
    "activation": "ReLU",
    "n_blocks": [1, 1, 1],
    "mlp_units": tune.choice([
        [[256, 256], [256, 256], [256, 256]],
        [[512, 256], [256, 128], [128, 64]],
    ]),
    "dropout_prob_theta": tune.choice([0.1, 0.2]),
    "optimizer": torch.optim.AdamW,
    "optimizer_kwargs": {
        "weight_decay": tune.choice([1e-3, 1e-2]),
    },
    "scaler_type": "standard",
    "random_seed": 42,
    "hist_exog_list": hist_exog_cols,
    "futr_exog_list": futr_exog_cols,
}

auto_model = AutoNHITS(
    h=horizon,
    loss=HuberLoss(),
    valid_loss=MAE(),
    config=nhits_config,
    num_samples=10,
    backend="ray",
)

nf_auto = NeuralForecast(models=[auto_model], freq="D")
nf_auto.fit(
    df=Y_df[["unique_id", "ds", "y"] + hist_exog_cols + futr_exog_cols],
    val_size=horizon,
)

# 1. Find the last date in your training data (Y_df)
last_date = Y_df["ds"].max()

# 2. Generate the 180 future dates (starting the day after last_date)
future_dates = pd.date_range(
    start=last_date + pd.Timedelta(days=1), periods=180, freq="D"
)

# 3. Create the future dataframe structure
futr_df = pd.DataFrame({"unique_id": "BTC", "ds": future_dates})

# 4. Compute the exact same exogenous features used in training
futr_df["day_of_week"] = futr_df["ds"].dt.dayofweek / 6.0
futr_df["day_of_week_sin"] = np.sin(2 * np.pi * futr_df["ds"].dt.dayofweek / 7.0)
futr_df["day_of_week_cos"] = np.cos(2 * np.pi * futr_df["ds"].dt.dayofweek / 7.0)
futr_df["day_of_year_sin"] = np.sin(2 * np.pi * futr_df["ds"].dt.dayofyear / 365.25)
futr_df["day_of_year_cos"] = np.cos(2 * np.pi * futr_df["ds"].dt.dayofyear / 365.25)

# 5. Generate the 180-day forecast using the fitted best model
forecasts = nf_auto.predict(futr_df=futr_df)

# 6. Reconstruct absolute prices (if training target 'y' was log returns)
last_close_price = Y_df["close"].iloc[-1]
forecasts["cum_log_return"] = forecasts["AutoNHITS"].cumsum()
forecasts["predicted_price"] = last_close_price * np.exp(
    forecasts["cum_log_return"]
)

# 1. Generate historical in-sample predictions
insample_df = nf_auto.predict_insample(step_size=horizon)

# 2. Reconstruct Prices
last_close = Y_df["close"].iloc[-1]
forecasts["predicted_price"] = last_close * np.exp(
    forecasts["AutoNHITS"].cumsum()
)

# 3. Create the Plot
plt.figure(figsize=(14, 7))

# Plot historical close price (last 360 days)
historical_tail = Y_df.iloc[-360:]
plt.plot(
    historical_tail["ds"],
    historical_tail["close"],
    label="Actual History",
    color="black",
    linewidth=1.5,
)

# Plot 180-Day Future Forecast
plt.plot(
    forecasts["ds"],
    forecasts["predicted_price"],
    label="180-Day Forecast",
    color="blue",
    linestyle="--",
    linewidth=2,
)

plt.title("Bitcoin Price: Historical vs 180-Day N-HiTS Forecast")
plt.xlabel("Date")
plt.ylabel("Price ($)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
