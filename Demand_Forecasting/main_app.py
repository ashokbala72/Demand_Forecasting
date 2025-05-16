from dotenv import load_dotenv
load_dotenv()

import streamlit as st
import step1_upload
import step2_aggregate
import step3_prompt
import step4_narrative
import step5_dashboard
from forecast_with_genai import generate_forecast, apply_genai_summary
from datetime import datetime

# Streamlit page setup
st.set_page_config(page_title="GenAI Demand Forecast Narrator", layout="wide")
st.title("📊 Historical Demand Analysis & Forecasting with Scenario Narratives")

# Sidebar step selection
step = st.sidebar.radio("Select Step", [
    "1. Upload Historical Data",
    "2. View Aggregated Trends",
    "3. Generate Prompt",
    "4. Generate Scenario Narrative",
    "5. Generate Forecast with GenAI Summary",
    "6. Visualize Output"
])

# Step-wise logic
if step == "1. Upload Historical Data":
    step1_upload.run()

elif step == "2. View Aggregated Trends":
    step2_aggregate.run()

elif step == "3. Generate Prompt":
    step3_prompt.run()

elif step == "4. Generate Scenario Narrative":
    step4_narrative.run()

elif step == "5. Generate Forecast with GenAI Summary":
    st.subheader("📈 Generating Forecast and Applying GenAI")

    if st.button("Run Forecast"):
        with st.spinner("Generating forecast and applying GenAI..."):
            forecast_df = generate_forecast(datetime.today())
            forecast_df['GenAI Summary'] = forecast_df.apply(apply_genai_summary, axis=1)
            forecast_df.to_csv("forecast_with_genai_summary.csv", index=False)
            st.success("✅ Forecast with GenAI summaries saved to forecast_with_genai_summary.csv")
            st.dataframe(forecast_df)

elif step == "6. Visualize Output":
    step5_dashboard.run()
