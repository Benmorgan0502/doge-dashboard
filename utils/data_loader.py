import pandas as pd
import requests
import os
import time
import streamlit as st
from config.settings import API_CONFIGS, CACHE_DURATION_HOURS

def is_cache_fresh(file_path, max_age_hours):
    """Check if cache file is fresh"""
    if not os.path.exists(file_path):
        return False
    last_modified = os.path.getmtime(file_path)
    age_hours = (time.time() - last_modified) / 3600
    return age_hours < max_age_hours

def fetch_data(api_url, result_key, cache_file):
    """Fetch data from API with pagination"""
    data = []
    page = 1
    try:
        while True:
            res = requests.get(api_url, params={"page": page, "per_page": 500})
            if res.status_code != 200:
                st.warning(f"Failed to fetch data: {res.status_code}")
                break
            json_data = res.json()

            if "result" in json_data and result_key in json_data["result"]:
                page_data = json_data["result"][result_key]
            elif result_key in json_data:
                page_data = json_data[result_key]
            elif isinstance(json_data, list):
                page_data = json_data
            elif isinstance(json_data, dict):
                page_data = [json_data]
            else:
                st.warning(f"Unexpected format in API response for {api_url}")
                break

            if not page_data:
                break
            data.extend(page_data)

            meta = json_data.get("meta", {})
            if not meta or page >= meta.get("pages", 1):
                break
            page += 1

        # Ensure data directory exists
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)
        
        df = pd.json_normalize(data)
        df.to_csv(cache_file, index=False)
        return df
    except Exception as e:
        st.error(f"Error loading data from {api_url}: {e}")
        return pd.DataFrame()

def load_data(config):
    """Load data from cache or API"""
    cache_file = config["cache"]
    if is_cache_fresh(cache_file, CACHE_DURATION_HOURS):
        return pd.read_csv(cache_file)
    else:
        return fetch_data(config["url"], config["result_key"], cache_file)

def load_all_datasets():
    """Load all datasets defined in API_CONFIGS"""
    datasets = {}
    for dataset_name, config in API_CONFIGS.items():
        with st.spinner(f"Loading {dataset_name} data (from cache or API)..."):
            datasets[dataset_name] = load_data(config)
    return datasets