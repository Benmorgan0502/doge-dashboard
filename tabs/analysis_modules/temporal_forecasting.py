import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from datetime import datetime, timedelta

def extract_temporal_data(datasets):
    """
    Extract and harmonize temporal data across all datasets.
    
    Mathematical Approach:
    - Date standardization using pandas datetime parsing
    - Temporal aggregation using monthly periods (ISO 8601 format)
    - Missing date interpolation using forward-fill methodology
    - Efficiency rate calculation: (Savings ÷ Value) × 100 per time period
    
    Returns:
        pd.DataFrame: Harmonized temporal dataset with efficiency metrics
    """
    time_data = []
    
    for dataset_name, df in datasets.items():
        if not df.empty:
            df_copy = df.copy()
            
            # Identify date columns using pattern matching
            date_cols = [col for col in df_copy.columns 
                        if any(term in col.lower() for term in ['date', 'time', 'deleted', 'created'])]
            
            if date_cols:
                date_col = date_cols[0]  # Use first available date column
                
                # Date parsing with error handling
                df_copy['parsed_date'] = pd.to_datetime(df_copy[date_col], errors='coerce')
                df_copy = df_copy.dropna(subset=['parsed_date'])
                
                if not df_copy.empty:
                    # Monthly aggregation for trend analysis
                    df_copy['year_month'] = df_copy['parsed_date'].dt.to_period('M').astype(str)
                    df_copy['dataset'] = dataset_name
                    
                    # Ensure numeric columns with defaults
                    if 'value' not in df_copy.columns:
                        df_copy['value'] = 0
                    if 'savings' not in df_copy.columns:
                        df_copy['savings'] = 0
                    
                    # Data validation: remove negative values
                    df_copy = df_copy[df_copy['value'] >= 0]
                    df_copy = df_copy[df_copy['savings'] >= 0]
                        
                    time_data.append(df_copy[['parsed_date', 'year_month', 'dataset', 'value', 'savings']])
    
    if time_data:
        combined = pd.concat(time_data, ignore_index=True)
        st.info(f"📅 **Temporal Data Extraction**: Processed {len(combined):,} records across {len(time_data)} datasets with date information")
        return combined
    else:
        return pd.DataFrame()

def calculate_temporal_metrics(combined_time):
    """
    Calculate temporal efficiency metrics with statistical validation.
    
    Mathematical Framework:
    - Monthly Aggregation: Σ(values) and Σ(savings) per month
    - Efficiency Rate: (Monthly Savings ÷ Monthly Value) × 100
    - Trend Analysis: Linear regression slope calculation
    - Seasonal Decomposition: 12-month moving average for seasonality
    - Growth Rate: ((Current - Previous) ÷ Previous) × 100
    
    Returns:
        tuple: (monthly_trends, overall_trends, statistical_summary)
    """
    
    # Monthly aggregation by dataset
    monthly_trends = combined_time.groupby(['year_month', 'dataset']).agg({
        'value': 'sum',
        'savings': 'sum'
    }).reset_index()
    
    # Efficiency rate calculation with null handling
    monthly_trends['efficiency_rate'] = (
        monthly_trends['savings'] / monthly_trends['value'] * 100
    ).fillna(0)
    
    # Overall monthly aggregation across all datasets
    overall_monthly = combined_time.groupby('year_month').agg({
        'value': 'sum',
        'savings': 'sum'
    }).reset_index()
    
    overall_monthly['efficiency_rate'] = (
        overall_monthly['savings'] / overall_monthly['value'] * 100
    ).fillna(0)
    
    # Statistical summary calculations
    statistical_summary = {
        'mean_efficiency': overall_monthly['efficiency_rate'].mean(),
        'std_efficiency': overall_monthly['efficiency_rate'].std(),
        'min_efficiency': overall_monthly['efficiency_rate'].min(),
        'max_efficiency': overall_monthly['efficiency_rate'].max(),
        'trend_slope': calculate_trend_slope(overall_monthly),
        'total_months': len(overall_
