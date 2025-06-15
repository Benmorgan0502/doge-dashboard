import streamlit as st
import pandas as pd
import plotly.express as px
from utils.chart_utils import format_billions, format_millions, create_download_button
from models.outlier_detection import perform_outlier_detection

def render_contracts_tab(df):
    """Render the contracts tab with all its functionality"""
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        # Filters
        if not df.empty:
            agencies = sorted(df["agency"].dropna().unique().tolist())
            statuses = sorted(df["fpds_status"].dropna().unique().tolist())

            selected_agency = st.selectbox("Filter by Agency", ["All"] + agencies)
            selected_status = st.selectbox("Filter by Contract Status", ["All"] + statuses)
            value_min = st.number_input("Minimum Contract Value", value=float(df["value"].min()), step=1000.0)
            value_max = st.number_input("Maximum Contract Value", value=float(df["value"].max()), step=1000.0)

            # Apply filters
            filtered_df = df.copy()
            if selected_agency != "All":
                filtered_df = filtered_df[filtered_df["agency"] == selected_agency]
            if selected_status != "All":
                filtered_df = filtered_df[filtered_df["fpds_status"] == selected_status]
            filtered_df = filtered_df[(filtered_df["value"] >= value_min) & (filtered_df["value"] <= value_max)]
            
            # Download button
            create_download_button(filtered_df, "Contracts")
        else:
            filtered_df = df
    
    with col2:
        if filtered_df.empty:
            st.warning("No data available for Contracts.")
            return
        
        # Summary metrics
        st.markdown("### 📈 Contract Summary")
        summary_cols = st.columns(4)
        with summary_cols[0]:
            st.metric("🧾 Total Contracts", f"{len(filtered_df):,}")
        with summary_cols[1]:
            st.metric("💰 Total Value", format_billions(filtered_df["value"].sum()))
        with summary_cols[2]:
            st.metric("💸 Total Savings", format_billions(filtered_df["savings"].sum()))
        with summary_cols[3]:
            avg_val = filtered_df["value"].mean()
            st.metric("📊 Avg. Value", format_millions(avg_val))
        st.markdown("---")

        # Tab options
        tab_options = ["By Agency", "By Status", "Savings Over Time", "By Vendor", "Missing Descriptions", "Outlier Detection"]
        selected_tab = st.radio("📊 Explore Views", tab_options, horizontal=True, label_visibility="collapsed")

        if selected_tab == "By Agency":
            render_agency_view(filtered_df)
        elif selected_tab == "By Status":
            render_status_view(filtered_df)
        elif selected_tab == "Savings Over Time":
            render_savings_over_time_view(filtered_df)
        elif selected_tab == "By Vendor":
            render_vendor_view(filtered_df)
        elif selected_tab == "Missing Descriptions":
            render_missing_descriptions_view(filtered_df)
        elif selected_tab == "Outlier Detection":
            render_outlier_detection_view(filtered_df)

def render_agency_view(df):
    """Render top agencies by contract value"""
    st.markdown("### 🏢 Top Agencies by Contract Value")
    agency_summary = df.groupby("agency", as_index=False)["value"].sum().sort_values("value", ascending=False).head(10)
    fig_agency = px.bar(agency_summary, x="agency", y="value", title="Top 10 Agencies by Contract Value")
    st.plotly_chart(fig_agency, use_container_width=True)

def render_status_view(df):
    """Render contract status distribution"""
    st.markdown("### 📊 Contract Status Distribution")
    status_counts = df["fpds_status"].value_counts().reset_index()
    status_counts.columns = ["Status", "Count"]
    fig_status = px.pie(status_counts, values="Count", names="Status", title="Contract Status Distribution")
    st.plotly_chart(fig_status, use_container_width=True)

def render_savings_over_time_view(df):
    """Render savings over time chart"""
    st.markdown("### 💰 Savings Over Time")
    df_copy = df.copy()
    df_copy["deleted_date"] = pd.to_datetime(df_copy["deleted_date"], errors="coerce")
    savings_over_time = df_copy.dropna(subset=["deleted_date"]).groupby(df_copy["deleted_date"].dt.to_period("M")).sum(numeric_only=True).reset_index()
    savings_over_time["deleted_date"] = savings_over_time["deleted_date"].astype(str)
    fig_savings = px.line(savings_over_time, x="deleted_date", y="savings", title="Savings Over Time", markers=True)
    st.plotly_chart(fig_savings, use_container_width=True)

def render_vendor_view(df):
    """Render top vendors by contract value"""
    st.markdown("### 🏬 Top Vendors by Contract Value")
    vendor_summary = df.groupby("vendor", as_index=False)["value"].sum().sort_values("value", ascending=False).head(10)
    fig_vendor = px.bar(vendor_summary, x="vendor", y="value", title="Top 10 Vendors by Contract Value")
    st.plotly_chart(fig_vendor, use_container_width=True)

def render_missing_descriptions_view(df):
    """Render contracts with missing information"""
    st.markdown("### 🚨 Contracts with Missing or Vague Information")
    missing_info = df[
        (df["agency"].str.contains("usaid", case=False, na=False) & 
         df["vendor"].str.contains("undisclosed", case=False, na=False)) |
        (df["fpds_link"] == "https://fpds.gov")
    ]
    st.write(f"Found {len(missing_info)} records with missing descriptions or generic FPDS links.")
    st.dataframe(missing_info.head(1000))

def render_outlier_detection_view(df):
    """Render outlier detection analysis"""
    st.markdown("### 🧮 Outlier Detection on Contract Value")
    perform_outlier_detection(df, "Contracts")