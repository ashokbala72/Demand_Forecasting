import streamlit as st
import pandas as pd

st.set_page_config(page_title="Test Upload", layout="centered")
st.title("📁 File Upload Test")

uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("✅ File uploaded successfully.")
    st.dataframe(df.head())
else:
    st.info("👆 Please upload a CSV file to proceed.")
