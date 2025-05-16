# step3_prompt.py
import streamlit as st
import pandas as pd

def run():
    st.subheader("📝 Generate Prompt for Scenario")

    if "uploaded_data" not in st.session_state:
        st.warning("⚠️ Please upload historical data first in Step 1.")
        return

    df = st.session_state["uploaded_data"]
    df.columns = df.columns.str.strip().str.title()

    st.markdown("### 📄 Sample of Uploaded Data")
    st.dataframe(df.head())

    if st.button("Generate Prompt"):
        sample = df.iloc[0]
        prompt = (
            f"You are an energy analyst. Based on the demand data for {sample['Date']}, "
            f"in region {sample['Region']}:\n"
            f"- Temperature: {sample['Temperature(C)']}°C\n"
            f"- Demand: {sample['Demand(Mw)']} MW\n"
            f"- Holiday: {'Yes' if sample['Is_Holiday'] else 'No'}\n\n"
            f"Generate a short summary describing why demand is high or low based on temperature and holiday status."
        )

        # ✅ Save to session
        st.session_state["scenario_prompt"] = prompt

        st.markdown("### ✨ Prompt Saved to Use in Step 4")
        st.text_area("Prompt", value=prompt, height=200)
