from openai import OpenAI
import os
import pandas as pd
import numpy as np

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def generate_forecast(start_date):
    dates = pd.date_range(start=start_date, periods=7, freq='D')
    data = {
        "Date": dates,
        "Temperature(C)": np.random.normal(35, 3, 7).round(1),
        "Demand(Mw)": np.random.normal(420, 60, 7).round(2),
        "Is_Holiday": [0, 0, 1, 0, 0, 1, 0]
    }
    return pd.DataFrame(data)

def apply_genai_summary(row):
    prompt = (
        f"As an energy analyst, summarize the forecast for {row['Date']}:\n"
        f"- Avg Temperature: {row['Temperature(C)']} °C\n"
        f"- Avg Demand: {row['Demand(Mw)']} MW\n"
        f"- Is Holiday: {'Yes' if row['Is_Holiday'] else 'No'}\n\n"
        f"Write a short narrative explaining why demand may be high or low."
    )
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"GenAI Error: {str(e)}"
