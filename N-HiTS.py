import numpy as np
import pandas as pd
import torch

from neuralforecast import NeuralForecast
from neuralforecast.models import NHITS
from neuralforecast.auto import AutoNHITS
from neuralforecast.losses.pytorch import HuberLoss, MAE
from ray import tune

horizon = 180

# 1. Load dataset
df = pd.read_csv("BTC_data.csv", index_col=0)
df.columns = ["open", "high", "low", "close", "volume"]

# Make date column robust
df.index = pd.to_datetime(df.index)
df.index.name = "ds"

# Optional but recommended: enforce daily frequency
# Decide how you want to handle missing days.
# df = df.asfreq("D")
# df[["open", "high", "low", "close", "volume"]] = df[["open", "high", "low", "close", "volume"]].ffill()

df = df.sort_index()

# 2. Feature engineering
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

# Calendar features: these are known in the future too
df["day_of_week"] = df.index.dayofweek / 6.0
df["day_of_week_sin"] = np.sin(2 * np.pi * df.index.dayofweek / 7.0)
df["day_of_week_cos"] = np.cos(2 * np.pi * df.index.dayofweek / 7.0)
df["day_of_year_sin"] = np.sin(2 * np.pi * df.index.dayofyear / 365.25)
df["day_of_year_cos"] = np.cos(2 * np.pi * df.index.dayofyear / 365.25)

# Target: daily log return
df["y"] = np.log(close / close.shift(1))

# Clean
df = df.replace([np.inf, -np.inf], np.nan).dropna().reset_index()

# Historical exogenous: known only up to forecast cutoff
hist_exog_cols = [
    "log_ret_1d",
    "log_ret_30d",
    "log_ret_90d",
    "price_to_sma200",
    "sma50_to_sma200",
    "volatility_30d",
    "channel_pos_180d",
]

# Future exogenous: calendar variables are known for the next 180 days
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
    ["unique_id", "ds", "y", "close"]
    + hist_exog_cols
    + futr_exog_cols
].sort_values("ds")


# ---------------------------------------------------------------------------
# 3. AutoNHITS with a probabilistic (quantile) loss instead of a point loss.
#    A 180-day BTC forecast has enormous uncertainty; a single point number
#    hides that.
#    TODO: Review N-HiTS and LSTM models and configs for them 
# ---------------------------------------------------------------------------
nhits_config = {
    "max_steps": tune.choice([400, 800]),
    "val_check_steps": 20,
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


#----------------------------------------------------------------------------
# 4. Setup Model for cross validation
#    TODO: Look into how to parse the dataframe that results from cross validation 
#----------------------------------------------------------------------------
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

best_result = nf_auto.models[0].results.get_best_result(
    metric="valid_loss",
    mode="min",
)

best_config = best_result.config
print(best_config)

model = NHITS(
    h=horizon,
    loss=HuberLoss(),
    valid_loss=MAE(),
    **best_config,
)

nf = NeuralForecast(models=[model], freq="D")

val_size = horizon
test_size = horizon

Y_hat_df = nf.cross_validation(
    df=Y_df[["unique_id", "ds", "y"] + hist_exog_cols + futr_exog_cols],
    val_size=val_size,
    test_size=test_size,
    n_windows=1,
    step_size=horizon,
)



Y_hat_df = Y_hat_df.sort_values(["unique_id", "cutoff", "ds"])

# Actual close at each cutoff date
cutoff_prices = (
    Y_df[["ds", "close"]]
    .rename(columns={"ds": "cutoff", "close": "cutoff_close"})
)

Y_hat_df = Y_hat_df.merge(cutoff_prices, on="cutoff", how="left")

model_cols = [c for c in Y_hat_df.columns if c not in ["unique_id", "ds", "cutoff", "y", "cutoff_close"]]
model_cols = [c for c in model_cols if not c.endswith("_price")]

for col in model_cols:
    Y_hat_df[f"{col}_cum_log_ret"] = Y_hat_df.groupby("cutoff")[col].cumsum()
    Y_hat_df[f"{col}_price"] = (
        Y_hat_df["cutoff_close"] * np.exp(Y_hat_df[f"{col}_cum_log_ret"])
    )

# Actual reconstructed price from actual daily log returns
Y_hat_df["actual_cum_log_ret"] = Y_hat_df.groupby("cutoff")["y"].cumsum()
Y_hat_df["actual_price"] = (
    Y_hat_df["cutoff_close"] * np.exp(Y_hat_df["actual_cum_log_ret"])
)

print(
    Y_hat_df[
        ["unique_id", "cutoff", "ds", "cutoff_close", "NHITS_price", "actual_price"]
    ].head()
)


terminal = Y_hat_df.groupby("cutoff").tail(1).copy()

terminal["abs_error"] = np.abs(terminal["NHITS_price"] - terminal["actual_price"])
terminal["pct_error"] = (
    np.abs(terminal["NHITS_price"] - terminal["actual_price"]) / terminal["actual_price"]
)

print(terminal[["cutoff", "ds", "actual_price", "NHITS_price", "pct_error"]])

print("MAPE:", terminal["pct_error"].mean())
print("RMSE:", np.sqrt(np.mean((terminal["NHITS_price"] - terminal["actual_price"]) ** 2)))
