import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt


# ==========================================================
# PAGE CONFIG
# ==========================================================

st.set_page_config(page_title="Maharashtra Crop Price Forecasting", page_icon="🌾", layout="wide")

DATA_FILE = "Dataset/Maharashtra/final_model_data.csv"
MODELS_DIR = "Models"


# ==========================================================
# LOAD DATA + MODEL (cached so it only runs once)
# ==========================================================

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_FILE)
    df["Arrival_Date"] = pd.to_datetime(df["Arrival_Date"])
    return df


@st.cache_resource
def load_best_model():
    comparison = pd.read_csv(f"{MODELS_DIR}/model_comparison.csv")
    best = comparison.sort_values("RMSE").iloc[0]
    model_file = f"{MODELS_DIR}/{best['Model'].lower().replace(' ', '_')}.pkl"
    return joblib.load(model_file), best


df = load_data()
model, best_info = load_best_model()


# ==========================================================
# TITLE
# ==========================================================

st.title("🌾 Maharashtra Crop Price Forecasting")
st.write("Machine Learning based prediction of crop Modal Price in Maharashtra.")
st.divider()


# ==========================================================
# SIDEBAR
# ==========================================================

st.sidebar.header("🔧 Forecast Settings")

commodities = sorted(df["Commodity"].dropna().unique())
commodity = st.sidebar.selectbox("Select Crop", commodities)

available_markets = sorted(df[df["Commodity"] == commodity]["Market"].dropna().unique())
market = st.sidebar.selectbox("Select Market", available_markets)

min_date = df["Arrival_Date"].max() + pd.Timedelta(days=1)
max_date = min_date + pd.Timedelta(days=365)

forecast_date = st.sidebar.date_input("Select Forecast Date", value=min_date.date(), min_value=min_date.date(), max_value=max_date.date())
forecast_date = pd.to_datetime(forecast_date)


# ==========================================================
# PREDICTION FUNCTION
# ==========================================================

def predict_price(commodity, market, forecast_date):

    history = df[
        (df["Commodity"] == commodity) &
        (df["Market"] == market) &
        (df["Arrival_Date"] < forecast_date)
    ].sort_values("Arrival_Date")

    if len(history) < 30:
        return None, "Not enough historical data for this crop and market."

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
    return max(0, prediction), None


# ==========================================================
# FORECAST BUTTON
# ==========================================================

if st.sidebar.button("🔮 Predict Price", use_container_width=True):

    prediction, error = predict_price(commodity, market, forecast_date)

    if error:
        st.error(error)
    else:
        st.success("Prediction generated successfully!")
        st.subheader("💰 Predicted Modal Price")

        col1, col2, col3 = st.columns(3)
        col1.metric("Crop", commodity)
        col2.metric("Market", market)
        col3.metric("Forecast Date", forecast_date.strftime("%d-%m-%Y"))

        st.markdown("###")
        st.metric("Predicted Price", f"₹ {prediction:,.2f} / quintal")

        history = df[(df["Commodity"] == commodity) & (df["Market"] == market)].sort_values("Arrival_Date")
        recent_history = history.tail(30)

        st.subheader("📈 Recent Historical Price")
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(recent_history["Arrival_Date"], recent_history["Modal_Price"], marker="o")
        ax.set_xlabel("Date")
        ax.set_ylabel("Modal Price (₹/quintal)")
        ax.set_title(f"{commodity} - {market}")
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig)

        st.subheader("📋 Recent Price Records")
        display_data = recent_history[["Arrival_Date", "Modal_Price"]].copy()
        display_data.columns = ["Date", "Modal Price (₹/quintal)"]
        st.dataframe(display_data, use_container_width=True, hide_index=True)


# ==========================================================
# MODEL INFO
# ==========================================================

st.divider()
st.subheader("🤖 Machine Learning Model")

info_col1, info_col2, info_col3 = st.columns(3)
info_col1.info(f"**Model:** {best_info['Model']}")
info_col2.info(f"**R²:** {best_info['R2']:.4f}")
info_col3.info(f"**MAE:** ₹{best_info['MAE']:.2f}/quintal")


# ==========================================================
# ABOUT
# ==========================================================

with st.expander("ℹ️ About This Project"):
    st.write("""
    This project predicts the Modal Price of selected agricultural commodities
    in Maharashtra using Machine Learning.

    Features: historical price, lag features, rolling mean features, date features,
    crop, market. Target: Modal_Price (₹/quintal).
    """)