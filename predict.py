import pandas as pd
import joblib


# ==========================================================
# CONFIG
# ==========================================================

DATA_FILE = "Dataset/Maharashtra/final_model_data.csv"
MODELS_DIR = "Models"


# ==========================================================
# LOAD DATA
# ==========================================================

df = pd.read_csv(DATA_FILE)
df["Arrival_Date"] = pd.to_datetime(df["Arrival_Date"])


# ==========================================================
# PICK BEST MODEL AUTOMATICALLY (from model_comparison.csv)
# ==========================================================

comparison = pd.read_csv(f"{MODELS_DIR}/model_comparison.csv")
best_model_name = comparison.sort_values("RMSE").iloc[0]["Model"]
model_file = f"{MODELS_DIR}/{best_model_name.lower().replace(' ', '_')}.pkl"

print("Using model:", best_model_name, "->", model_file)
model = joblib.load(model_file)


# ==========================================================
# PREDICTION FUNCTION
# ==========================================================

def predict_crop_price(commodity, market, forecast_date):

    forecast_date = pd.to_datetime(forecast_date)

    history = df[
        (df["Commodity"] == commodity) &
        (df["Market"] == market) &
        (df["Arrival_Date"] < forecast_date)
    ].sort_values("Arrival_Date")

    if len(history) < 30:
        raise ValueError("Not enough historical data for this crop and market.")

    prices = history["Modal_Price"]

    input_data = pd.DataFrame({
        "Year": [forecast_date.year],
        "Month": [forecast_date.month],
        "Day": [forecast_date.day],
        "DayOfWeek": [forecast_date.dayofweek],
        "DayOfYear": [forecast_date.dayofyear],
        "Lag_1": [prices.iloc[-1]],
        "Lag_7": [prices.iloc[-7]],
        "Lag_14": [prices.iloc[-14]],
        "Lag_30": [prices.iloc[-30]],
        "Rolling_Mean_7": [prices.iloc[-7:].mean()],
        "Rolling_Mean_30": [prices.iloc[-30:].mean()],
        "Commodity": [commodity],
        "Market": [market],
    })

    prediction = model.predict(input_data)[0]
    return max(0, prediction)


# ==========================================================
# TEST PREDICTION
# ==========================================================

if __name__ == "__main__":

    commodity = "Wheat"
    market = "Pune"
    forecast_date = "2025-12-01"

    try:
        prediction = predict_crop_price(commodity, market, forecast_date)

        print("=" * 60)
        print("CROP PRICE FORECAST")
        print("=" * 60)
        print("Crop:", commodity)
        print("Market:", market)
        print("Forecast Date:", forecast_date)
        print("Predicted Modal Price:", round(prediction, 2), "₹/quintal")

    except ValueError as e:
        print("Error:", e)