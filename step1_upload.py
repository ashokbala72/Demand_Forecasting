# step1_upload.py

import streamlit as st
import pandas as pd

def run():
    st.subheader("📁 Upload Historical Demand Data")

    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"], key="upload_file")

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.session_state["uploaded_data"] = df  # Store in session
            st.success("✅ File uploaded successfully.")
            st.dataframe(df.head())
        except Exception as e:
            st.error(f"❌ Error reading file: {e}")
    else:
        st.info("Please upload a CSV file to continue.")
