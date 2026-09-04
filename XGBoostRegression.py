import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("BTC_data.csv", index_col=0)
df.columns = ["open", "high", "low", "close", "volume"]

#Returns
df["return_1d"] = df["close"].pct_change()
df["return_30d"] = df["close"].pct_change(30)
df["return_90d"] = df["close"].pct_change(90)

# Moving averages
df["sma_50"] = df["close"].rolling(50).mean()
df["sma_200"] = df["close"].rolling(200).mean()

# Safe Ratio Features (avoiding raw high prices or uncalibrated scales)
df["price_to_sma200"] = df["close"] / df["sma_200"]
df["sma50_to_sma200"] = df["sma_50"] / df["sma_200"]

# Volatility (30-day)
df["volatility_30d"] = df["return_1d"].rolling(30).std() * np.sqrt(365)

# Safe Channel Position (clip between 0 and 1 to prevent Inf)
high_180 = df["high"].rolling(180).max()
low_180 = df["low"].rolling(180).min()
denom = (high_180 - low_180).replace(0, np.nan)
df["channel_pos_180d"] = ((df["close"] - low_180) / denom).clip(0, 1)

# Target: 180-day future percentage return
df["target_180d_return"] = df["close"].pct_change(180).shift(-180)

# Clean dataset
df_clean = df.replace([np.inf, -np.inf], np.nan).dropna()

feature_cols = [
    "return_1d",
    "return_30d",
    "return_90d",
    "price_to_sma200",
    "sma50_to_sma200",
    "volatility_30d",
    "channel_pos_180d",
]

X = df_clean[feature_cols]
y = df_clean["target_180d_return"]

#Setup Time series split
tscv = TimeSeriesSplit(n_splits=4)
results_list = []

for fold, (train_index, test_index) in enumerate(tscv.split(X)):
    # Purge 180 days from training end to eliminate target overlap
    train_index_purged = train_index[:-180]
    if len(train_index_purged) < 50:
        continue

    X_tr, X_va = X.iloc[train_index_purged], X.iloc[test_index]
    y_tr, y_va = y.iloc[train_index_purged], y.iloc[test_index]

     # Fit scaler ONLY on training data to prevent leakage
    scaler = StandardScaler()
    X_tr_scaled = scaler.fit_transform(X_tr)
    X_va_scaled = scaler.transform(X_va)

   # 2. Initialize the regressor
    regressor = xgb.XGBRegressor(
        n_estimators=30,       # Keep trees low to prevent overfitting noise
        max_depth=3,           # Shallow trees force macro-level rules
        learning_rate=0.02,    # Small steps
        reg_alpha=5.0,         # L1 Regularization
        reg_lambda=10.0,       # L2 Regularization
        subsample=0.8,
        random_state=42
    )

    # 3. Train the model
    regressor.fit(X_tr, y_tr)

    # Predict continuous returns
    y_pred = regressor.predict(X_va_scaled)

    fold_df = pd.DataFrame(
        {
            "actual_return": y_va,
            "pred_return": y_pred,
            "naive_pred": 0.0,
        },
        index=y_va.index,
    )
    results_list.append(fold_df)
    
# Combine all validation predictions across folds
results_df = pd.concat(results_list)

# Sanity Check
print("--- PREDICTION RANGE CHECK ---")
print(
    f"Min Pred Return: {results_df['pred_return'].min():.2f} | Max Pred Return: {results_df['pred_return'].max():.2f}"
)

# Model Evaluation Metrics
mse = mean_squared_error(results_df["actual_return"], results_df["pred_return"])
rmse = np.sqrt(mse)
mae = mean_absolute_error(
    results_df["actual_return"], results_df["pred_return"]
)
r2 = r2_score(results_df["actual_return"], results_df["pred_return"])
naive_rmse = np.sqrt(
    mean_squared_error(results_df["actual_return"], results_df["naive_pred"])
)

print("\n=== STATISTICAL METRICS ===")
print(f"Model RMSE: {rmse:.4f}  |  Naive Baseline RMSE: {naive_rmse:.4f}")
print(f"Model MAE:  {mae:.4f}")
print(f"Model R^2:   {r2:.4f}")

correct_direction = (
    np.sign(results_df["pred_return"]) == np.sign(results_df["actual_return"])
).mean()
print(f"Directional Hit Rate: {correct_direction * 100:.2f}%")