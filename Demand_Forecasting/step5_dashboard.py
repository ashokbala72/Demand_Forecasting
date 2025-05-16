# step5_dashboard.py

import streamlit as st
import pandas as pd
import plotly.express as px

def run():
    st.subheader("📊 Forecast Visualization Dashboard")

    # ✅ Check if forecast data is available
    try:
        df = pd.read_csv("forecast_with_genai_summary.csv")
    except FileNotFoundError:
        st.warning("⚠️ Forecast not found. Please run Step 5: 'Generate Forecast with GenAI Summary'.")
        return

    df.columns = df.columns.str.strip().str.title()

    # ✅ Show data
    st.markdown("### 📄 Forecast Data Preview")
    st.dataframe(df.head())

    # ✅ Line chart for demand
    st.markdown("### 🔺 Forecasted Demand Over Time")
    fig = px.line(df, x="Date", y="Demand(Mw)", title="Forecasted Demand")
    st.plotly_chart(fig, use_container_width=True)

    # ✅ Show GenAI summaries if available
    if "Genai Summary" in df.columns:
        st.markdown("### 🤖 GenAI Summary Narratives")
        for i, row in df.iterrows():
            st.markdown(f"**{row['Date']}** — {row['Genai Summary']}")
    else:
        st.info("No GenAI summaries found in the data.")
