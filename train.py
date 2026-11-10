import pandas as pd
import numpy as np
import os
import time
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ==========================================================
# CONFIG
# ==========================================================

DATA_FILE = "Dataset/Maharashtra/final_model_data.csv"
SPLIT_DATE = "2025-01-01"   # before this = train, on/after = test

NUMERIC_FEATURES = [
    "Year", "Month", "Day", "DayOfWeek", "DayOfYear",
    "Lag_1", "Lag_7", "Lag_14", "Lag_30",
    "Rolling_Mean_7", "Rolling_Mean_30"
]
CATEGORICAL_FEATURES = ["Commodity", "Market"]
TARGET = "Modal_Price"


# ==========================================================
# LOAD DATA + BUILD FEATURES
# ==========================================================

df = pd.read_csv(DATA_FILE)
df["Arrival_Date"] = pd.to_datetime(df["Arrival_Date"])

print("Raw rows:", len(df))

# Sort so lag/rolling features look at the correct past rows
df = df.sort_values(["Commodity", "Market", "Arrival_Date"])

grouped = df.groupby(["Commodity", "Market"])["Modal_Price"]

df["Lag_1"] = grouped.shift(1)
df["Lag_7"] = grouped.shift(7)
df["Lag_14"] = grouped.shift(14)
df["Lag_30"] = grouped.shift(30)

# rolling mean of the *previous* prices only (no leakage of current price)
shifted = grouped.shift(1)
df["Rolling_Mean_7"] = shifted.groupby([df["Commodity"], df["Market"]]).transform(lambda s: s.rolling(7).mean())
df["Rolling_Mean_30"] = shifted.groupby([df["Commodity"], df["Market"]]).transform(lambda s: s.rolling(30).mean())

df["Year"] = df["Arrival_Date"].dt.year
df["Month"] = df["Arrival_Date"].dt.month
df["Day"] = df["Arrival_Date"].dt.day
df["DayOfWeek"] = df["Arrival_Date"].dt.dayofweek
df["DayOfYear"] = df["Arrival_Date"].dt.dayofyear

# drop rows that don't have 30 days of history yet
df = df.dropna(subset=NUMERIC_FEATURES)

print("Rows after feature engineering:", len(df))


# ==========================================================
# TRAIN / TEST SPLIT (by date)
# ==========================================================

split_date = pd.to_datetime(SPLIT_DATE)
train_df = df[df["Arrival_Date"] < split_date]
test_df = df[df["Arrival_Date"] >= split_date]

print("Train rows:", len(train_df))
print("Test rows :", len(test_df))

X_train = train_df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
y_train = train_df[TARGET]
X_test = test_df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
y_test = test_df[TARGET]


# ==========================================================
# PREPROCESSING
# ==========================================================

preprocessor = ColumnTransformer(
    transformers=[
        ("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES)
    ],
    remainder="passthrough"
)


# ==========================================================
# MODELS
# ==========================================================

models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=200, max_depth=20, random_state=42, n_jobs=-1),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, max_depth=5, random_state=42),
}


# ==========================================================
# TRAIN + EVALUATE
# ==========================================================

os.makedirs("Models", exist_ok=True)
results = []

for name, model in models.items():
    print("\nTraining:", name)
    start = time.time()

    pipeline = Pipeline([("preprocessor", preprocessor), ("model", model)])
    pipeline.fit(X_train, y_train)
    preds = pipeline.predict(X_test)

    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)
    elapsed = time.time() - start

    print(f"MAE={mae:.2f}  RMSE={rmse:.2f}  R2={r2:.4f}  time={elapsed:.1f}s")

    results.append({"Model": name, "MAE": mae, "RMSE": rmse, "R2": r2, "Training_Time_Seconds": elapsed})

    filename = f"Models/{name.lower().replace(' ', '_')}.pkl"
    joblib.dump(pipeline, filename)
    print("Saved:", filename)


# ==========================================================
# SAVE COMPARISON
# ==========================================================

results_df = pd.DataFrame(results).sort_values("RMSE")
results_df.to_csv("Models/model_comparison.csv", index=False)

print("\n" + "=" * 60)
print(results_df.to_string(index=False))
print("\nBest model:", results_df.iloc[0]["Model"])