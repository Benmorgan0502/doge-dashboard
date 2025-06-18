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
    """Fetch data from API with pagination and better error handling"""
    data = []
    page = 1
    total_records = 0
    
    try:
        # Create progress bar for data loading
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        while True:
            status_text.text(f"Fetching page {page}... ({total_records} records so far)")
            
            res = requests.get(api_url, params={"page": page, "per_page": 500}, timeout=30)
            if res.status_code != 200:
                st.warning(f"Failed to fetch data from {api_url}: HTTP {res.status_code}")
                break
            
            json_data = res.json()

            # Handle different API response structures
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
            total_records = len(data)

            # Update progress
            meta = json_data.get("meta", {})
            if meta and "pages" in meta:
                progress = min(page / meta["pages"], 1.0)
                progress_bar.progress(progress)
            
            if not meta or page >= meta.get("pages", 1):
                break
            page += 1

        # Clear progress indicators
        progress_bar.empty()
        status_text.empty()
        
        # Ensure data directory exists
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)
        
        # Convert to DataFrame and save
        if data:
            df = pd.json_normalize(data)
            df.to_csv(cache_file, index=False)
            st.success(f"✅ Successfully loaded {len(df):,} records from {api_url}")
            return df
        else:
            st.warning(f"⚠️ No data returned from {api_url}")
            return pd.DataFrame()
            
    except requests.exceptions.Timeout:
        st.error(f"⏰ Timeout while fetching data from {api_url}. Using cached data if available.")
        return load_from_cache_only(cache_file)
    except requests.exceptions.ConnectionError:
        st.error(f"🌐 Connection error while fetching data from {api_url}. Using cached data if available.")
        return load_from_cache_only(cache_file)
    except Exception as e:
        st.error(f"❌ Error loading data from {api_url}: {e}")
        return load_from_cache_only(cache_file)

def load_from_cache_only(cache_file):
    """Load data only from cache file"""
    if os.path.exists(cache_file):
        try:
            df = pd.read_csv(cache_file)
            st.info(f"📁 Loaded {len(df):,} records from cache: {cache_file}")
            return df
        except Exception as e:
            st.error(f"❌ Error reading cache file {cache_file}: {e}")
            return pd.DataFrame()
    else:
        st.warning(f"⚠️ No cache file found: {cache_file}")
        return pd.DataFrame()

def load_data(config):
    """Load data from cache or API with enhanced error handling"""
    cache_file = config["cache"]
    
    # Check if cache is fresh
    if is_cache_fresh(cache_file, CACHE_DURATION_HOURS):
        try:
            df = pd.read_csv(cache_file)
            # Calculate cache age for display
            cache_age = (time.time() - os.path.getmtime(cache_file)) / 3600
            st.info(f"📦 Using cached data ({cache_age:.1f} hours old) - {len(df):,} records")
            return df
        except Exception as e:
            st.warning(f"⚠️ Error reading cache file, fetching fresh data: {e}")
            return fetch_data(config["url"], config["result_key"], cache_file)
    else:
        # Cache is stale or doesn't exist, fetch fresh data
        if os.path.exists(cache_file):
            cache_age = (time.time() - os.path.getmtime(cache_file)) / 3600
            st.info(f"🔄 Cache is {cache_age:.1f} hours old, fetching fresh data...")
        else:
            st.info("🆕 No cache found, fetching data for the first time...")
        
        return fetch_data(config["url"], config["result_key"], cache_file)

def load_all_datasets():
    """Load all datasets defined in API_CONFIGS with enhanced progress tracking"""
    datasets = {}
    total_datasets = len(API_CONFIGS)
    
    # Overall progress tracking
    st.markdown("### 📊 Loading Government Efficiency Data")
    overall_progress = st.progress(0)
    
    for i, (dataset_name, config) in enumerate(API_CONFIGS.items()):
        st.markdown(f"#### Loading {dataset_name} Data...")
        
        try:
            datasets[dataset_name] = load_data(config)
            
            # Show dataset summary
            if not datasets[dataset_name].empty:
                df = datasets[dataset_name]
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Records", f"{len(df):,}")
                with col2:
                    st.metric("Columns", f"{len(df.columns)}")
                with col3:
                    # Calculate file size
                    if os.path.exists(config["cache"]):
                        size_mb = os.path.getsize(config["cache"]) / 1024 / 1024
                        st.metric("Cache Size", f"{size_mb:.1f} MB")
                    else:
                        st.metric("Cache Size", "N/A")
            else:
                st.warning(f"⚠️ No data loaded for {dataset_name}")
            
        except Exception as e:
            st.error(f"❌ Failed to load {dataset_name}: {e}")
            datasets[dataset_name] = pd.DataFrame()
        
        # Update overall progress
        overall_progress.progress((i + 1) / total_datasets)
    
    # Clear progress bar
    overall_progress.empty()
    
    # Summary of all loaded datasets
    st.markdown("---")
    st.markdown("### 📋 Data Loading Summary")
    
    summary_data = []
    total_records = 0
    
    for dataset_name, df in datasets.items():
        if not df.empty:
            cache_file = API_CONFIGS[dataset_name]["cache"]
            cache_age = "N/A"
            if os.path.exists(cache_file):
                cache_age = f"{(time.time() - os.path.getmtime(cache_file)) / 3600:.1f}h"
            
            summary_data.append({
                "Dataset": dataset_name,
                "Records": f"{len(df):,}",
                "Columns": len(df.columns),
                "Cache Age": cache_age,
                "Status": "✅ Loaded"
            })
            total_records += len(df)
        else:
            summary_data.append({
                "Dataset": dataset_name,
                "Records": "0",
                "Columns": "0",
                "Cache Age": "N/A",
                "Status": "❌ Failed"
            })
    
    # Display summary table
    summary_df = pd.DataFrame(summary_data)
    st.dataframe(summary_df, use_container_width=True)
    
    # Overall statistics
    st.success(f"🎉 Data loading complete! Total records: {total_records:,}")
    
    return datasets

def get_data_summary_for_homepage(datasets):
    """Generate summary statistics optimized for homepage display"""
    if not datasets:
        return None
    
    summary = {
        "last_updated": time.time(),
        "total_records": 0,
        "datasets_loaded": 0,
        "datasets_failed": 0,
        "cache_status": "unknown"
    }
    
    oldest_cache = 0
    
    for dataset_name, df in datasets.items():
        if not df.empty:
            summary["total_records"] += len(df)
            summary["datasets_loaded"] += 1
            
            # Check cache age
            cache_file = API_CONFIGS[dataset_name]["cache"]
            if os.path.exists(cache_file):
                cache_age = (time.time() - os.path.getmtime(cache_file)) / 3600
                oldest_cache = max(oldest_cache, cache_age)
        else:
            summary["datasets_failed"] += 1
    
    # Determine overall cache status
    if oldest_cache < 6:
        summary["cache_status"] = "fresh"
    elif oldest_cache < 12:
        summary["cache_status"] = "recent"
    else:
        summary["cache_status"] = "stale"
    
    summary["oldest_cache_hours"] = oldest_cache
    
    return summary
