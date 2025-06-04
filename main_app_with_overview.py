from dotenv import load_dotenv
load_dotenv()

import streamlit as st
import step2_aggregate
import step3_prompt
import step4_narrative
import step5_dashboard
from datetime import datetime
from openai import OpenAI

def generate_summary_from_forecast(forecast_df):
    return ""
import pandas as pd
import os
import numpy as np


# Inline forecast generation function
def generate_forecast(reference_date, duration_days=30):
    future_dates = pd.date_range(start=reference_date, periods=duration_days, freq='D')
    base_forecast = 1300 + 100 * np.sin(2 * np.pi * future_dates.dayofyear / 365.25)
    noise = np.random.normal(0, 30, len(future_dates))
    forecast_values = base_forecast + noise
    return pd.DataFrame({"date": future_dates, "forecast": forecast_values.astype(int)})

# Streamlit page setup
st.set_page_config(page_title="GenAI Demand Forecast Narrator", layout="wide")
st.title("📊 Historical Demand Analysis & Forecasting with Scenario Narratives")

# Sidebar step selection
step = st.sidebar.radio("Select Step", [
    "0. Overview",
    "1. Historical Data",
    "2. View Aggregated Trends",
    "3. Generate Prompt",
    "4. Generate Scenario Narrative",
    "5. Generate Forecast",
    "6. Visualize Output"
])

# Step-wise logic
if step == "0. Overview":
    st.subheader("🔍 Application Overview")
    st.markdown("""
    ### 📌 Purpose of the Application
    This tool assists energy utilities in **analyzing historical demand**, generating **AI-driven scenario narratives**, and producing **forecasts enhanced by GenAI insights**. It provides a comprehensive decision-support interface for demand planners and grid analysts.

    ### 🧩 Key Features
    - **Step-wise Guided Workflow:** From default **Historical Data** display with pagination to GenAI-enhanced forecast generation.
    - **GenAI Scenario Narrative:** Automatically generates scenario-based demand insights using custom prompts.
    - **Integrated Forecast Generator:** Uses statistical methods to forecast demand and applies AI summarization.
    - **Dynamic Dashboard:** Visualizes aggregated trends, narrative summaries, and forecast outputs.
    - **Downloadable Forecast Output:** Save forecast with GenAI annotations as CSV.

    ### 📥 Inputs Used
    - **Historical Data (CSV or default sample):** Must contain date/time and load demand in MW.
    - **Optional Narrative Prompt:** To guide GenAI-based scenario explanations.
    - **System Date (auto-injected):** Used for generating forecasts.

    ### 🛠️ Technologies Used
    - **Python & Streamlit:** Front-end interaction and visualization.
    - **Pandas & Numpy:** Data processing and aggregation.
    - **OpenAI (GenAI):** Prompt-based scenario narratives and summaries.
    - **Matplotlib / Altair / Plotly:** Visualization in dashboard.

    ### 🚀 What to Look For
    - Step 1 displays default demand data with pagination. Load is shown in MW.
    - In each step, ensure your data is successfully loaded and intermediate outputs are meaningful.
    - Focus on how GenAI interprets different trends and generates narratives.
    - Final visualization helps confirm forecasting quality and relevance.

    ### ✅ Production Readiness Notes
    - **✅ Authentication:** Add user login for secure access.
    - **✅ API Abstraction:** Externalize OpenAI API usage into a service layer.
    - **✅ Logging & Monitoring:** Add log tracking, exception capture, and performance metrics.
    - **✅ Real-Time Inputs:** Integrate live demand feed instead of static CSVs.
    - **✅ Deployment:** Containerize with Docker and deploy on a cloud platform (e.g., Azure, AWS, GCP).
    """)

elif step == "1. Historical Data":
    st.subheader("📊 Historical Demand Data (Load in MW)")
    st.markdown("**ℹ️ This demo uses demand data from 2020 to 2024 with seasonal patterns.**")

    date_rng = pd.date_range(start='2020-01-01', end='2024-12-31', freq='D')
    seasonal_trend = 200 * np.sin(2 * np.pi * date_rng.dayofyear / 365.25)
    random_noise = np.random.normal(0, 50, len(date_rng))
    load_values = 1200 + seasonal_trend + random_noise
    sample_df = pd.DataFrame({'date': date_rng, 'load': load_values.astype(int)})
    sample_df.to_csv("sample_demand.csv", index=False)
    st.success("✅ Default sample_demand.csv generated with realistic seasonal demand from 2020 to 2024")

        # Show paginated sample data
    page_size = 20
    total_rows = sample_df.shape[0]
    total_pages = (total_rows - 1) // page_size + 1
    page = st.number_input("Page", min_value=1, max_value=total_pages, value=1, step=1)
    start_idx = (page - 1) * page_size
    end_idx = start_idx + page_size
    st.dataframe(sample_df.iloc[start_idx:end_idx])

    # Save aggregated file
    agg_df = sample_df.groupby('date').agg({'load': 'sum'}).reset_index()
    agg_df.to_csv("aggregated_output.csv", index=False)
    st.success("✅ Aggregated file 'aggregated_output.csv' generated successfully from default sample.")
    

    

elif step == "2. View Aggregated Trends":
    if os.path.exists("aggregated_output.csv"):
        df = pd.read_csv("aggregated_output.csv")
        st.line_chart(df.set_index("date")["load"])
    else:
        st.warning("⚠️ Please run Step 1 first to generate aggregated demand data.")

    # Add GenAI insight block
    st.markdown("""
    ---
    ### 🤖 GenAI Insight on Aggregated Trends
    """)
    try:
        aggregated_csv = "aggregated_output.csv"
        if os.path.exists(aggregated_csv):
            df = pd.read_csv(aggregated_csv)
            if not df.empty:
                prompt = f"""
                Analyze the following historical energy demand trends:

                {df.head(10).to_string(index=False)}

                Summarize key patterns and potential implications.
                """
                client = OpenAI()
                completion = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "You are an energy demand analyst."},
                        {"role": "user", "content": prompt}
                    ]
                )
                insight = completion.choices[0].message.content
                st.success("✅ GenAI Insight Generated")
                st.markdown(f"**Insight:** {insight}")
            else:
                st.warning("⚠️ Aggregated data file is empty.")
        else:
            st.info("ℹ️ 'aggregated_output.csv' not found. Please run Step 1 to upload data.")
    except Exception as e:
        st.error(f"❌ Error generating GenAI insight: {e}")

elif step == "3. Generate Prompt":
    st.subheader("🧠 AI-Generated Scenario Prompt")

    if os.path.exists("sample_demand.csv"):
        df = pd.read_csv("sample_demand.csv")
        st.dataframe(df.head())

        prompt_text = "Due to an extended heatwave in July and August across the United Kingdom, residential electricity demand is expected to rise by 20%, especially during evening peak hours. Increased air conditioning and fan usage will drive the surge. Assume historical baseline conditions from 2020–2024."

        st.markdown("**Generated Prompt:**")
        st.markdown(f"""
<div style='text-align: left; padding: 1em; background-color: #f8f9fa; border-radius: 8px;'>
<pre style='white-space: pre-wrap; word-wrap: break-word;'>{prompt_text}</pre>
</div>
""", unsafe_allow_html=True)
    else:
        st.warning("⚠️ Please generate the default sample data first in Step 1.")

elif step == "4. Generate Scenario Narrative":
    st.subheader("📝 Scenario Narrative")

    if os.path.exists("sample_demand.csv"):
        df = pd.read_csv("sample_demand.csv")
        st.dataframe(df.head())

        prompt_text = "Due to an extended heatwave in July and August across the United Kingdom, residential electricity demand is expected to rise by 20%, especially during evening peak hours. Increased air conditioning and fan usage will drive the surge. Assume historical baseline conditions from 2020–2024."
        st.markdown("**Using Generated Prompt:**")
        st.markdown(f"""
<div style='text-align: left; padding: 1em; background-color: #f8f9fa; border-radius: 8px;'>
<pre style='white-space: pre-wrap; word-wrap: break-word;'>{prompt_text}</pre>
</div>
""", unsafe_allow_html=True)

        try:
            client = OpenAI()
            completion = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a scenario modeling expert."},
                    {"role": "user", "content": f"Given the demand data and the scenario: '{prompt_text}', describe how demand may evolve."}
                ]
            )
            narrative = completion.choices[0].message.content
            st.success("✅ Scenario Narrative Generated")
            st.markdown(narrative)
        except Exception as e:
            st.error(f"❌ Error generating scenario narrative: {e}")
    else:
        st.warning("⚠️ Please run Step 1 to generate the default demand data.")

elif step == "5. Generate Forecast":
    st.subheader("📈 Generate Forecast")

    duration_days = st.slider("Select forecast duration (days)", min_value=7, max_value=90, value=30, step=1)
    if st.button("Run Forecast"):
        with st.spinner("Generating forecast..."):
            forecast_df = generate_forecast(pd.to_datetime("2025-01-01"), duration_days)
            if 'forecast' not in forecast_df.columns:
                raise KeyError("The generated forecast does not contain a 'forecast' column.")
            forecast_df.to_csv("forecast_with_genai_summary.csv", index=False)
            st.success("✅ Forecast saved to forecast_with_genai_summary.csv")
            st.dataframe(forecast_df.tail(duration_days))  # Display first 30 days forecast

elif step == "6. Visualize Output":
    import plotly.express as px

    if os.path.exists("forecast_with_genai_summary.csv") and os.path.exists("sample_demand.csv"):
        forecast_df = pd.read_csv("forecast_with_genai_summary.csv")
        forecast_df.columns = [col.strip().lower() for col in forecast_df.columns]
        forecast_df.rename(columns={"date": "Date", "forecast": "Forecast"}, inplace=True)
        forecast_df = forecast_df.tail(30)

        historical_df = pd.read_csv("sample_demand.csv")
        historical_df.columns = [col.strip().lower() for col in historical_df.columns]
        historical_df = historical_df.rename(columns={"date": "Date", "load": "Historical"})
        historical_df = historical_df.tail(30)

        fig = px.line()
        fig.add_scatter(x=forecast_df['Date'], y=forecast_df['Forecast'], mode='lines+markers', name='Forecast')
        fig.add_scatter(x=historical_df['Date'], y=historical_df['Historical'], mode='lines+markers', name='Historical')
        st.plotly_chart(fig)

        # GenAI Insight Block
        st.markdown("### 🤖 GenAI View")
        try:
            preview = forecast_df.tail(10).to_string(index=False)
            prompt = f"""
            Analyze the following 10-day forecasted electricity demand values for the UK:

            {preview}

            1. Highlight any anomalies (unexpected deviations).
            2. Provide operational strategy suggestions (load balancing, reserve planning, demand-side response, etc).
            3. Suggest recommendations for grid planners.
            """
            client = OpenAI()
            completion = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a UK energy grid analyst."},
                    {"role": "user", "content": prompt}
                ]
            )
            insight = completion.choices[0].message.content
            st.success("✅ GenAI Insight Generated")
            st.markdown(insight)
        except Exception as e:
            st.error(f"❌ Error generating GenAI insight: {e}")
    else:
        st.warning("⚠️ Please generate forecast in Step 5 and historical data in Step 1 before viewing output.")
