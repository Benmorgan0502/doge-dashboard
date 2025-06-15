import os
from openai import OpenAI

def format_billions(val):
    """Format value as billions"""
    return f"${val / 1_000_000_000:.1f}B"

def format_millions(val):
    """Format value as millions"""
    return f"${val / 1_000_000:.1f}M"

def generate_chart_commentary(prompt_text):
    """Generate AI commentary for charts using OpenAI API"""
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        return "⚠️ OpenAI API key not set."

    try:
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a data analyst who explains charts in clear, concise language."},
                {"role": "user", "content": prompt_text}
            ],
            max_tokens=150,
            temperature=0.5
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"⚠️ Error generating commentary: {e}"

def create_download_button(df, dataset_name, key_suffix=""):
    """Create a download button for CSV data"""
    import streamlit as st
    
    if not df.empty:
        csv_data = df.to_csv(index=False).encode("utf-8")
        return st.download_button(
            label=f"Download {dataset_name} Data as CSV",
            data=csv_data,
            file_name=f"{dataset_name.lower()}_data.csv",
            mime="text/csv",
            key=f"download_{dataset_name.lower()}_{key_suffix}"
        )
    return None