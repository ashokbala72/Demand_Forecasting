import streamlit as st
import pandas as pd
import plotly.express as px

def run():
    st.subheader("📊 Aggregated Demand Trends")

    if "uploaded_data" not in st.session_state:
        st.warning("⚠️ Please upload historical data first in Step 1.")
        return

    df = st.session_state["uploaded_data"]

    # Show column names to debug
    st.markdown("### 🧾 Uploaded Data Columns")
    st.write(df.columns.tolist())

    # Attempt to normalize the column name
    if "Date" not in df.columns:
        st.error("❌ 'Date' column not found. Please check your CSV format.")
        return

    df["Date"] = pd.to_datetime(df["Date"])

    st.markdown("### 📈 Daily Demand Trend")
    daily_avg = df.groupby("Date")["Demand(MW)"].mean().reset_index()

    fig = px.line(daily_avg, x="Date", y="Demand(MW)", title="Average Daily Demand Over Time")
    st.plotly_chart(fig, use_container_width=True)
