import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from openai import AzureOpenAI
import plotly.graph_objects as go
import os
from dotenv import load_dotenv
import yfinance as yf
import requests

# -----------------------------
# Load .env file
# -----------------------------
load_dotenv()

AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_VERSION = "2024-12-01-preview"
AZURE_OPENAI_DEPLOYMENT_NAME = "gpt-4o-raj"

# -----------------------------
# Azure OpenAI Client
# -----------------------------
client = AzureOpenAI(
    api_key=AZURE_OPENAI_API_KEY,
    api_version=AZURE_OPENAI_API_VERSION,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
)

# -----------------------------
# Streamlit Config
# -----------------------------
st.set_page_config(page_title="UK Oil & Gas Forecasting", layout="wide")
st.cache_data.clear()

# -----------------------------
# Forecast logic
# -----------------------------
def forecast_series(series, volatility_factor):
    future_dates = pd.date_range(datetime.today() + timedelta(days=1), periods=30)
    last_value = series.iloc[-1] if not series.empty else 0
    forecast = np.random.normal(loc=last_value, scale=volatility_factor, size=30)
    return pd.DataFrame({"date": future_dates, "forecast (£)": forecast})

# -----------------------------
# Scenario logic
# -----------------------------
def scenario_forecast(series, scenario):
    adjustment = {
        "Base Case": 1.0,
        "Recession": 0.9,
        "Severe Cold Weather": 1.2,
        "Supply Shock": 1.3,
        "Geo-Political Tensions": 1.15
    }.get(scenario, 1.0)

    future_dates = pd.date_range(datetime.today() + timedelta(days=1), periods=30)
    last_price = series.iloc[-1] if not series.empty else 0
    simulated = np.random.normal(loc=last_price * adjustment, scale=2, size=30)
    return pd.DataFrame({"date": future_dates, "forecast": simulated})

# -----------------------------
# Load Yahoo data
# -----------------------------
def load_data():
    today = datetime.today()
    try:
        brent_raw = yf.download("BZ=F", period="90d", interval="1d", progress=False)
        gas_raw = yf.download("NG=F", period="90d", interval="1d", progress=False)

        if brent_raw.empty or gas_raw.empty:
            raise ValueError("Yahoo API returned empty data")

        if isinstance(brent_raw.columns, pd.MultiIndex):
            brent_raw.columns = brent_raw.columns.get_level_values(0)
        brent_raw = brent_raw.reset_index()

        if isinstance(gas_raw.columns, pd.MultiIndex):
            gas_raw.columns = gas_raw.columns.get_level_values(0)
        gas_raw = gas_raw.reset_index()

        brent_df = brent_raw.rename(columns={"Date": "date", "Close": "Brent_Price"})
        brent_df = brent_df[["date", "Brent_Price"]].dropna()
        brent_df["date"] = pd.to_datetime(brent_df["date"])

        gas_df = gas_raw.rename(columns={"Date": "date", "Close": "UK_NatGas_Price"})
        gas_df = gas_df[["date", "UK_NatGas_Price"]].dropna()
        gas_df["date"] = pd.to_datetime(gas_df["date"])

        return brent_df, gas_df

    except Exception as e:
        st.warning(f"⚠️ Yahoo Finance API not responding. Simulating sample data. [Reason: {e}]")
        dates = pd.date_range(today - timedelta(days=90), periods=90)
        brent_df = pd.DataFrame({
            "date": dates,
            "Brent_Price": np.random.normal(80, 5, size=90)
        })
        gas_df = pd.DataFrame({
            "date": dates,
            "UK_NatGas_Price": np.random.normal(2.5, 0.3, size=90)
        })
        return brent_df, gas_df

# -----------------------------
# Utility to fetch actual price (dummy fallback)
# -----------------------------
def fetch_actual_market_price(ticker):
    try:
        df = yf.download(ticker, period="1d", interval="1d", progress=False)
        return float(df['Close'].iloc[-1]) if not df.empty else None
    except:
        return None

# -----------------------------
# Sidebar Navigation
# -----------------------------
page = st.sidebar.radio("📂 Navigation", ["📘 Overview", "🛠️ Main App"])

# -----------------------------
# Overview Page
# -----------------------------
if page == "📘 Overview":
    st.title("📂 UK Oil & Gas Price Forecasting Assistant")
    st.markdown("""
### 📂 About This App

This GenAI-powered assistant forecasts UK oil and natural gas prices and provides AI-driven market insights for short-term decision-making.

#### 🔍 Key Features:
- Real-time historical price retrieval for Brent Crude & UK Natural Gas (via Yahoo Finance)
- 30-day future price forecasts using statistical simulation
- AI-generated market commentary and investment suggestions
- Interactive scenario-based forecasting for various market conditions
- Natural language Q&A assistant for business queries

#### 🧠 How GenAI Helps:
- Converts forecasts into easy-to-understand summaries
- Recommends hedging or buying/selling strategies
- Answers custom questions about forecast deviations or risks

#### 📊 What Is Real vs. Simulated:
- ✅ Real Data: Past 90 days of market prices fetched from Yahoo Finance
- ⚙️ Simulated Data:
  - Forecast prices are generated using statistical assumptions
  - Scenario forecasts are mock-adjusted based on user-selected situations
  - Confidence levels are mock-evaluated by comparing simulated forecasts with current prices

#### 🛠️ Making This Production-Ready (Layman Terms):
- Replace mock forecasts with ML models trained on historical trends
- Use secure financial data APIs for high-accuracy pricing
- Add logging to track forecast accuracy over time
- Secure API keys and handle errors automatically
- Host on a cloud platform with daily refresh, versioning, and user access control

#### 📱 Live Data Sources:
- Brent Crude: `BZ=F`
- Natural Gas: `NG=F`
""")

# -----------------------------
# Main App
# -----------------------------
elif page == "🛠️ Main App":
    st.title("🛠️ UK Oil & Gas Forecasting Dashboard")

    brent_df, gas_df = load_data()
    brent_forecast = forecast_series(brent_df["Brent_Price"], 1.5) if not brent_df.empty else pd.DataFrame()
    gas_forecast = forecast_series(gas_df["UK_NatGas_Price"], 0.3) if not gas_df.empty else pd.DataFrame()

    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
        "📈 Historical Prices", "🔮 Forecast", "📉 Forecast Confidence",
        "🧠 AI Advisory", "❓ Ask a Question", "📊 Scenario-Based Forecasting",
        "📦 Production vs Forecast", "📊 Forecast Accuracy"
    ])

    # -----------------------------
    # Tab 1: Historical Prices
    # -----------------------------
    with tab1:
        subtab1, subtab2 = st.tabs(["Brent Crude", "UK Natural Gas"])
        with subtab1:
            st.markdown("### ✅ Brent Crude Price History (Last 90 Days)")
            if not brent_df.empty:
                st.dataframe(brent_df.tail(15))
                st.line_chart(data=brent_df.set_index("date"), y="Brent_Price")
        with subtab2:
            st.markdown("### ✅ UK Natural Gas Price History (Last 90 Days)")
            if not gas_df.empty:
                st.dataframe(gas_df.tail(15))
                st.line_chart(data=gas_df.set_index("date"), y="UK_NatGas_Price")

    # -----------------------------
    # Tab 2: Forecast
    # -----------------------------
    with tab2:
        st.subheader("🔮 Forecast Prices (Next 30 Days)")
        st.markdown("#### 📋 Brent Forecast Table")
        st.dataframe(brent_forecast if not brent_forecast.empty else "Brent forecast unavailable")

        st.markdown("#### 📋 UK Natural Gas Forecast Table")
        st.dataframe(gas_forecast if not gas_forecast.empty else "Gas forecast unavailable")

    # -----------------------------
    # Tab 3: Forecast Confidence
    # -----------------------------
    with tab3:
        st.subheader("📈 Forecast Confidence Level (vs Real Market Data)")
        today = datetime.today().date()
        brent_today_price = fetch_actual_market_price("BZ=F")
        gas_today_price = fetch_actual_market_price("NG=F")

        def evaluate_today_confidence(forecast_df, actual_price):
            future_rows = forecast_df[forecast_df["date"].dt.date >= today]
            if not future_rows.empty and actual_price:
                forecast_price = float(future_rows.iloc[0]["forecast (£)"])
                abs_error = abs(forecast_price - actual_price)
                pct_error = (abs_error / actual_price) * 100
                confidence = max(0, 100 - pct_error)
                return forecast_price, actual_price, confidence
            else:
                return None, None, None

        st.markdown("### 🔎 Brent Forecast Confidence vs Market")
        if not brent_forecast.empty and brent_today_price:
            f_price, a_price, conf = evaluate_today_confidence(brent_forecast, brent_today_price)
            if f_price is not None:
                st.info(f"📈 Forecast: £{f_price}, 🏷️ Market: £{a_price}, 🔒 Confidence: {conf:.2f}%")
        else:
            st.warning("⚠️ Could not evaluate Brent confidence with live data.")

        st.markdown("### 🔎 UK NatGas Forecast Confidence vs Market")
        if not gas_forecast.empty and gas_today_price:
            f_price, a_price, conf = evaluate_today_confidence(gas_forecast, gas_today_price)
            if f_price is not None:
                st.info(f"📈 Forecast: £{f_price}, 🏷️ Market: £{a_price}, 🔒 Confidence: {conf:.2f}%")
        else:
            st.warning("⚠️ Could not evaluate Gas confidence with live data.")

    # -----------------------------
    # Tab 4: AI Advisory (Azure OpenAI)
    # -----------------------------
    with tab4:
        st.subheader("🧠 GenAI Market Advisory")
        try:
            clean_brent = brent_forecast.dropna().head(5) if not brent_forecast.empty else pd.DataFrame()
            clean_gas = gas_forecast.dropna().head(5) if not gas_forecast.empty else pd.DataFrame()

            if clean_brent.empty or clean_gas.empty:
                st.warning("⚠️ Forecast data is incomplete. GenAI advisory might be limited.")
            else:
                prompt = f"""
Analyze the following UK Oil & Gas forecast:

Brent (next 5 days):
{clean_brent.to_string(index=False)}

UK Natural Gas (next 5 days):
{clean_gas.to_string(index=False)}

1. Highlight short-term trends.
2. Suggest any investment or hedging actions.
3. Comment on potential risks or volatility.
"""
                completion = client.chat.completions.create(
                    model=AZURE_OPENAI_DEPLOYMENT_NAME,
                    messages=[
                        {"role": "system", "content": "You are a UK oil and gas market analyst."},
                        {"role": "user", "content": prompt}
                    ]
                )
                st.success("✅ Advisory Generated")
                st.markdown(completion.choices[0].message.content)
        except Exception as e:
            st.error(f"❌ Error generating GenAI advisory: {e}")

    # -----------------------------
    # Tab 5: Q&A with Azure OpenAI
    # -----------------------------
    with tab5:
        st.subheader("❓ Ask About the Forecast")
        user_query = st.text_input("Your question:")
        if st.button("Ask GenAI") and user_query:
            context = f"""
Brent Forecast:
{brent_forecast.head(5).to_string(index=False) if not brent_forecast.empty else 'N/A'}

UK NatGas Forecast:
{gas_forecast.head(5).to_string(index=False) if not gas_forecast.empty else 'N/A'}
"""
            prompt = f"""
Context:
{context}

Answer this question:
{user_query}

Note: This assistant uses market **price data only** (Brent and UK NatGas from Yahoo Finance).
"""
            try:
                completion = client.chat.completions.create(
                    model=AZURE_OPENAI_DEPLOYMENT_NAME,
                    messages=[
                        {"role": "system", "content": "You are a UK energy forecast assistant."},
                        {"role": "user", "content": prompt}
                    ]
                )
                st.success("✅ GenAI Response")
                st.markdown(completion.choices[0].message.content)
            except Exception as e:
                st.error(f"❌ Error generating response: {e}")

    # -----------------------------
    # Tab 6: Scenario-Based Forecasting
    # -----------------------------
    with tab6:
        st.subheader("📊 Scenario-Based Forecasting")
        scenario = st.selectbox("Choose a Scenario:", ["Base Case", "Recession", "Severe Cold Weather", "Supply Shock", "Geo-Political Tensions"])
        st.markdown(f"### Scenario: {scenario}")

        if not brent_df.empty:
            scenario_brent = scenario_forecast(brent_df["Brent_Price"], scenario)
            st.markdown("#### 📍 Brent Forecast (Scenario-Based)")
            st.dataframe(scenario_brent)
            st.line_chart(scenario_brent.set_index("date"))

        if not gas_df.empty:
            scenario_gas = scenario_forecast(gas_df["UK_NatGas_Price"], scenario)
            st.markdown("#### 📍 UK Natural Gas Forecast (Scenario-Based)")
            st.dataframe(scenario_gas)
            st.line_chart(scenario_gas.set_index("date"))

    # -----------------------------
    # Tab 7: Production vs Forecast
    # -----------------------------
    with tab7:
        st.subheader("📦 Production vs Forecast")
        prod_dates = pd.date_range(datetime.today() - timedelta(days=30), periods=30)
        simulated_prod = np.random.normal(loc=950, scale=100, size=30)
        predicted_prod = simulated_prod * np.random.normal(1.02, 0.04, size=30)

        df_prod = pd.DataFrame({
            "date": prod_dates,
            "Simulated_Production": simulated_prod,
            "Forecast_Production": predicted_prod
        })

        st.line_chart(df_prod.set_index("date"))
        st.dataframe(df_prod)

    # -----------------------------
    # Tab 8: Forecast Accuracy
    # -----------------------------
    with tab8:
        st.subheader("📊 Forecast Accuracy")
        try:
            actual_price = np.random.normal(2.5, 0.2)  # simulated UK NatGas price

            if not gas_forecast.empty:
                predicted = gas_forecast.iloc[0]["forecast (£)"]
                abs_error = abs(predicted - actual_price)
                pct_error = (abs_error / actual_price) * 100
                confidence = max(0, 100 - pct_error)
                st.success(f"📈 Predicted: £{predicted:.2f}, 🏷️ Actual (simulated market): £{actual_price:.2f}, ✅ Accuracy: {confidence:.2f}%")
            else:
                st.warning("⚠️ No forecast data available to compute accuracy.")
        except Exception as e:
            st.error(f"❌ Error during accuracy computation: {e}")
