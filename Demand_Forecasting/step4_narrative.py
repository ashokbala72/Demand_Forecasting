# step4_narrative.py
import streamlit as st
import os
from openai import OpenAI

def run():
    st.subheader("📖 Scenario Narrative")

    if "scenario_prompt" not in st.session_state:
        st.warning("⚠️ Please go to Step 3 and click 'Generate Prompt' first.")
        return

    prompt = st.session_state["scenario_prompt"]
    st.text_area("🧾 Prompt Being Used", value=prompt, height=200)

    if st.button("Generate Narrative"):
        with st.spinner("Calling GenAI..."):
            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}]
            )

            result = response.choices[0].message.content
            st.success("✅ Scenario Narrative")
            st.text_area("Narrative", value=result, height=300)
