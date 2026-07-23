import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

df = pd.read_csv("BTC_data.csv", index_col=0)

df.columns = ["open", "high", "low", "close", "volume"]


# Percentage price changes
df["return"] = df["close"].pct_change()
df["high_low_spread"] = (df["high"] - df["low"]) / df["close"]

# Lagged features (e.g., past 1, 2, and 3 steps)
df["close_lag1"] = df["close"].shift(1)
df["close_lag2"] = df["close"].shift(2)
df["volume_lag1"] = df["volume"].shift(1)

# Moving averages
df["sma_5"] = df["close"].rolling(window=5).mean()


# Next Period Close Price (Regression)
df["target_close"] = df["close"].shift(-1)
# Next Period Return
df["target_return"] = df["close"].pct_change().shift(-1)

# Dropping NaN rows created by lags (.shift(1)) and targets (.shift(-1))
df_clean = df.dropna()

# Features (X) exclude target columns
feature_cols = ["open", "high", "low", "close", "volume", "return", "high_low_spread", "close_lag1", "close_lag2", "volume_lag1", "sma_5"]

X = df_clean[feature_cols]
y = df_clean["target_return"] 


#Setup time series split
tscv = TimeSeriesSplit(n_splits=5)
results_list = []

for fold, (train_index, test_index) in enumerate(tscv.split(X)):
    X_tr, X_va = X.iloc[train_index], X.iloc[test_index]
    y_tr, y_va = y.iloc[train_index], y.iloc[test_index]
   
   # Fit scaler ONLY on training data to prevent leakage
    scaler = StandardScaler()
    X_tr_scaled = scaler.fit_transform(X_tr)
    X_va_scaled = scaler.transform(X_va)

    # Fit Linear Regression
    model = LinearRegression()
    model.fit(X_tr_scaled, y_tr)

    # Predict continuous returns
    y_pred = model.predict(X_va_scaled)

    # Save validation index, actuals, and predictions
    fold_df = pd.DataFrame(
        {
            "fold": fold + 1,
            "actual_return": y_va,
            "pred_return": y_pred,
            # Naive Model Baseline: assumes tomorrow's return equals 0
            "naive_pred": 0.0,
        },
        index=y_va.index,
    )

    results_list.append(fold_df)

# Combine all validation predictions across folds
results_df = pd.concat(results_list)

#Evaluate Each Fold
mse = mean_squared_error(results_df["actual_return"], results_df["pred_return"])
rmse = np.sqrt(mse)
mae = mean_absolute_error(results_df["actual_return"], results_df["pred_return"])
r2 = r2_score(results_df["actual_return"], results_df["pred_return"])

# Naive Baseline Metrics (predicting 0 return every day)
naive_rmse = np.sqrt(
    mean_squared_error(
        results_df["actual_return"], results_df["naive_pred"]
    )
)

print("=== STATISTICAL METRICS ===")
print(f"Model RMSE: {rmse}  |  Naive Baseline RMSE: {naive_rmse}")
print(f"Model MAE:  {mae}")
print(f"Model R^2:   {r2}")

# --- 4. Directional Accuracy (Financial Relevance) ---
# Check if predicted sign (+) or (-) matched actual return sign
correct_direction = (
    np.sign(results_df["pred_return"]) == np.sign(results_df["actual_return"])
).mean()

print("\n=== DIRECTIONAL ACCURACY ===")
print(
    f"Directional Hit Rate: {correct_direction * 100:.2f}% (Target: > 50%)"
)