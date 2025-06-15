import streamlit as st
import pandas as pd
import plotly.express as px
from utils.chart_utils import format_billions, format_millions, create_download_button
from models.grant_impact_model import render_grant_impact_model

def render_grants_tab(df):
    """Render the grants tab with all its functionality"""
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        # Filters
        if not df.empty:
            agencies = sorted(df["agency"].dropna().unique().tolist())
            selected_agency = st.selectbox("Filter by Agency", ["All"] + agencies, key="grants_agency_filter")
            value_min = st.number_input("Minimum Grant Value", value=float(df["value"].min()), step=1000.0, key="grants_min_val")
            value_max = st.number_input("Maximum Grant Value", value=float(df["value"].max()), step=1000.0, key="grants_max_val")
            
            # Apply filters
            filtered_df = df.copy()
            if selected_agency != "All":
                filtered_df = filtered_df[filtered_df["agency"] == selected_agency]
            filtered_df = filtered_df[(filtered_df["value"] >= value_min) & (filtered_df["value"] <= value_max)]
            
            # Download button
            create_download_button(filtered_df, "Grants", "grants")
        else:
            filtered_df = df
    
    with col2:
        if filtered_df.empty:
            st.warning("No data available for Grants.")
            return
        
        # Summary metrics
        st.markdown("### 📈 Grant Summary")
        summary_cols = st.columns(4)
        with summary_cols[0]:
            st.metric("📄 Total Grants", f"{len(filtered_df):,}")
        with summary_cols[1]:
            st.metric("💰 Total Value", format_billions(filtered_df["value"].sum()))
        with summary_cols[2]:
            st.metric("💸 Total Savings", format_billions(filtered_df["savings"].sum()))
        with summary_cols[3]:
            avg_val = filtered_df["value"].mean()
            st.metric("📊 Avg. Value", format_millions(avg_val))
        st.markdown("---")

        # Tab options
        tab_options = ["By Agency", "By Recipient", "Savings Over Time", "Missing Info", "Grant Impact Model"]
        selected_tab = st.radio("📊 Explore Views", tab_options, horizontal=True, key="grants_tabs", label_visibility="collapsed")

        if selected_tab == "By Agency":
            render_agency_view(filtered_df)
        elif selected_tab == "By Recipient":
            render_recipient_view(filtered_df)
        elif selected_tab == "Savings Over Time":
            render_savings_over_time_view(filtered_df)
        elif selected_tab == "Missing Info":
            render_missing_info_view(filtered_df)
        elif selected_tab == "Grant Impact Model":
            render_grant_impact_model(filtered_df)

        # Raw data table
        st.divider()
        with st.expander("View Raw Data Table"):
            st.dataframe(filtered_df.head(1000))

def render_agency_view(df):
    """Render top grant-issuing agencies"""
    st.markdown("### 🏢 Top Grant-Issuing Agencies")
    agency_summary = df.groupby("agency", as_index=False)["value"].sum().sort_values("value", ascending=False).head(10)
    fig_agency = px.bar(agency_summary, x="agency", y="value", title="Top Agencies by Total Grant Value")
    st.plotly_chart(fig_agency, use_container_width=True)

def render_recipient_view(df):
    """Render top grant recipients"""
    st.markdown("### 🏬 Top Grant Recipients")
    recipient_summary = df.groupby("recipient", as_index=False)["value"].sum().sort_values("value", ascending=False).head(10)
    fig_recip = px.bar(recipient_summary, x="recipient", y="value", title="Top Recipients by Total Grant Value")
    st.plotly_chart(fig_recip, use_container_width=True)

def render_savings_over_time_view(df):
    """Render savings over time chart"""
    st.markdown("### 💰 Savings Over Time")
    df_copy = df.copy()
    df_copy["date"] = pd.to_datetime(df_copy["date"], errors="coerce")
    savings_time = df_copy.dropna(subset=["date"]).groupby(df_copy["date"].dt.to_period("M")).sum(numeric_only=True).reset_index()
    savings_time["date"] = savings_time["date"].astype(str)
    fig_savings = px.line(savings_time, x="date", y="savings", title="Grant Savings Over Time", markers=True)
    st.plotly_chart(fig_savings, use_container_width=True)

def render_missing_info_view(df):
    """Render grants missing information"""
    st.markdown("### ❌ Grants Missing Descriptions or Links")
    missing = df[df["link"].isna() | df["description"].isna()]
    st.write(f"Found {len(missing)} grants missing key information.")
    st.dataframe(missing.head(1000))