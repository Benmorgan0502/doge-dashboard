import streamlit as st
import pandas as pd
import plotly.express as px

def perform_outlier_detection(df, dataset_name):
    """Perform outlier detection using Isolation Forest"""
    
    if "value" not in df.columns or "agency" not in df.columns:
        st.warning("⚠️ Required columns (value, agency) not found for outlier detection.")
        return
    
    try:
        from sklearn.ensemble import IsolationForest
        from sklearn.preprocessing import LabelEncoder
        
        analysis_df = df[["value", "agency"]].dropna()
        if analysis_df.empty:
            st.warning("⚠️ Not enough data to perform outlier detection.")
            return
        
        # Encode categorical variables
        le = LabelEncoder()
        analysis_df = analysis_df.copy()
        analysis_df["agency_enc"] = le.fit_transform(analysis_df["agency"])
        
        # Prepare features
        X = analysis_df[["value", "agency_enc"]]
        
        # Apply Isolation Forest
        model = IsolationForest(contamination=0.05, random_state=42)
        analysis_df["anomaly"] = model.fit_predict(X)
        analysis_df["is_outlier"] = analysis_df["anomaly"] == -1
        
        # Display results
        outlier_count = analysis_df["is_outlier"].sum()
        st.write(f"Identified {outlier_count} outliers out of {len(analysis_df)} records.")
        
        if outlier_count > 0:
            outliers = analysis_df[analysis_df["is_outlier"]]
            st.dataframe(outliers.head(1000))
            
            # Create scatter plot
            fig_outliers = px.scatter(
                analysis_df, 
                x="value", 
                y="agency_enc",
                color=analysis_df["is_outlier"].map({True: "Outlier", False: "Normal"}),
                title=f"Outlier Detection for {dataset_name} using Isolation Forest",
                labels={"agency_enc": "Agency (Encoded)"}
            )
            st.plotly_chart(fig_outliers, use_container_width=True)
        else:
            st.info("No outliers detected with current parameters.")
            
    except ImportError:
        st.error("⚠️ scikit-learn is required for outlier detection. Please install it with: pip install scikit-learn")
    except Exception as e:
        st.error(f"⚠️ Error performing outlier detection: {e}")