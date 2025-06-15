import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from utils.chart_utils import format_billions, format_millions, create_download_button
from models.outlier_detection import perform_outlier_detection

def render_leases_tab(df):
    """Render the leases tab with comprehensive analysis optimized for DOGE lease data"""
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        # Filters
        if not df.empty:
            filtered_df = create_lease_filters(df)
            create_download_button(filtered_df, "Leases", "leases")
        else:
            filtered_df = df
    
    with col2:
        if filtered_df.empty:
            st.warning("No data available for Leases.")
            return
        
        # Summary metrics
        render_lease_summary(filtered_df)
        
        # Analysis tabs
        render_lease_analysis_tabs(filtered_df)
        
        # Raw data table
        st.divider()
        with st.expander("View Raw Data Table"):
            st.dataframe(filtered_df.head(1000))

def create_lease_filters(df):
    """Create filters specific to DOGE lease data structure"""
    
    # Agency filter
    agencies = sorted(df["agency"].dropna().unique().tolist())
    selected_agency = st.selectbox("Filter by Agency", ["All"] + agencies, key="leases_agency")
    
    # Location filter (state-based)
    # Extract states from location (format appears to be "CITY, STATE")
    df_copy = df.copy()
    df_copy['state'] = df_copy['location'].str.split(', ').str[-1]
    states = sorted(df_copy['state'].dropna().unique().tolist())
    selected_state = st.selectbox("Filter by State", ["All"] + states, key="leases_state")
    
    # Square footage range
    sq_ft_min = st.number_input("Minimum Square Footage", 
                               value=int(df["sq_ft"].min()), 
                               step=1000, 
                               key="leases_sqft_min")
    sq_ft_max = st.number_input("Maximum Square Footage", 
                               value=int(df["sq_ft"].max()), 
                               step=1000, 
                               key="leases_sqft_max")
    
    # Value range
    value_min = st.number_input("Minimum Lease Value ($)", 
                               value=int(df["value"].min()), 
                               step=10000, 
                               key="leases_value_min")
    value_max = st.number_input("Maximum Lease Value ($)", 
                               value=int(df["value"].max()), 
                               step=10000, 
                               key="leases_value_max")
    
    # Savings filter
    has_savings = st.checkbox("Only show leases with savings > $0", key="leases_has_savings")
    
    # Apply filters
    filtered_df = df.copy()
    
    if selected_agency != "All":
        filtered_df = filtered_df[filtered_df["agency"] == selected_agency]
    
    if selected_state != "All":
        filtered_df['state'] = filtered_df['location'].str.split(', ').str[-1]
        filtered_df = filtered_df[filtered_df['state'] == selected_state]
    
    filtered_df = filtered_df[
        (filtered_df["sq_ft"] >= sq_ft_min) & 
        (filtered_df["sq_ft"] <= sq_ft_max) &
        (filtered_df["value"] >= value_min) & 
        (filtered_df["value"] <= value_max)
    ]
    
    if has_savings:
        filtered_df = filtered_df[filtered_df["savings"] > 0]
    
    return filtered_df

def render_lease_summary(df):
    """Render comprehensive lease summary metrics"""
    st.markdown("### 🏢 Lease Termination Summary")
    
    # Calculate key metrics
    total_leases = len(df)
    total_value = df["value"].sum()
    total_savings = df["savings"].sum()
    total_sqft = df["sq_ft"].sum()
    avg_cost_per_sqft = total_value / total_sqft if total_sqft > 0 else 0
    savings_rate = (total_savings / total_value * 100) if total_value > 0 else 0
    
    # Display metrics in columns
    summary_cols = st.columns(6)
    
    with summary_cols[0]:
        st.metric("🏢 Total Leases", f"{total_leases:,}")
    
    with summary_cols[1]:
        st.metric("💰 Total Value", format_millions(total_value))
    
    with summary_cols[2]:
        st.metric("💸 Total Savings", format_millions(total_savings))
    
    with summary_cols[3]:
        st.metric("📐 Total Sq Ft", f"{total_sqft:,}")
    
    with summary_cols[4]:
        st.metric("📊 Avg $/Sq Ft", f"${avg_cost_per_sqft:.2f}")
    
    with summary_cols[5]:
        st.metric("⚡ Savings Rate", f"{savings_rate:.1f}%")
    
    # Additional insights
    leases_with_savings = len(df[df["savings"] > 0])
    st.markdown("---")
    st.markdown(f"**📈 Key Insights:** {leases_with_savings} out of {total_leases} lease terminations ({leases_with_savings/total_leases*100:.1f}%) generated cost savings")

def render_lease_analysis_tabs(df):
    """Render lease analysis tabs"""
    
    tab_options = [
        "By Agency", 
        "By Location", 
        "Cost Efficiency", 
        "Timeline Analysis", 
        "Savings Analysis",
        "Property Size Analysis",
        "Outlier Detection"
    ]
    
    selected_tab = st.radio("📊 Analysis Views", tab_options, horizontal=True, key="leases_analysis_tabs", label_visibility="collapsed")
    
    if selected_tab == "By Agency":
        render_agency_analysis(df)
    elif selected_tab == "By Location":
        render_location_analysis(df)
    elif selected_tab == "Cost Efficiency":
        render_cost_efficiency_analysis(df)
    elif selected_tab == "Timeline Analysis":
        render_timeline_analysis(df)
    elif selected_tab == "Savings Analysis":
        render_savings_analysis(df)
    elif selected_tab == "Property Size Analysis":
        render_property_size_analysis(df)
    elif selected_tab == "Outlier Detection":
        perform_outlier_detection(df, "Leases")

def render_agency_analysis(df):
    """Render analysis by agency"""
    st.markdown("### 🏢 Analysis by Agency")
    
    # Agency summary with multiple metrics
    agency_summary = df.groupby("agency").agg({
        'value': ['sum', 'count', 'mean'],
        'savings': 'sum',
        'sq_ft': 'sum'
    }).round(2)
    
    agency_summary.columns = ['Total Value', 'Lease Count', 'Avg Value', 'Total Savings', 'Total Sq Ft']
    agency_summary = agency_summary.reset_index()
    agency_summary['Cost per Sq Ft'] = agency_summary['Total Value'] / agency_summary['Total Sq Ft']
    agency_summary['Savings Rate'] = (agency_summary['Total Savings'] / agency_summary['Total Value'] * 100).round(1)
    agency_summary = agency_summary.sort_values('Total Value', ascending=False)
    
    # Top 15 agencies chart
    top_agencies = agency_summary.head(15)
    
    fig = px.bar(top_agencies, 
                 x="agency", 
                 y="Total Value",
                 title="Top 15 Agencies by Total Lease Value",
                 hover_data=['Lease Count', 'Total Savings', 'Savings Rate'])
    fig.update_layout(xaxis_tickangle=45)
    st.plotly_chart(fig, use_container_width=True)
    
    # Agency efficiency comparison
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🎯 Most Cost-Effective Agencies")
        cost_effective = agency_summary[agency_summary['Lease Count'] >= 3].nsmallest(10, 'Cost per Sq Ft')
        st.dataframe(cost_effective[['agency', 'Cost per Sq Ft', 'Lease Count', 'Savings Rate']])
    
    with col2:
        st.markdown("#### 💰 Highest Savings Rate Agencies")
        high_savings = agency_summary[agency_summary['Lease Count'] >= 3].nlargest(10, 'Savings Rate')
        st.dataframe(high_savings[['agency', 'Savings Rate', 'Total Savings', 'Lease Count']])

def render_location_analysis(df):
    """Render geographic analysis"""
    st.markdown("### 🗺️ Geographic Analysis")
    
    # Extract state information
    df_geo = df.copy()
    df_geo['state'] = df_geo['location'].str.split(', ').str[-1]
    df_geo['city'] = df_geo['location'].str.split(', ').str[0]
    
    # State-level analysis
    state_summary = df_geo.groupby('state').agg({
        'value': ['sum', 'count', 'mean'],
        'savings': 'sum',
        'sq_ft': 'sum'
    }).round(2)
    
    state_summary.columns = ['Total Value', 'Lease Count', 'Avg Value', 'Total Savings', 'Total Sq Ft']
    state_summary = state_summary.reset_index()
    state_summary['Savings Rate'] = (state_summary['Total Savings'] / state_summary['Total Value'] * 100).round(1)
    state_summary = state_summary.sort_values('Total Value', ascending=False)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Top states by value
        fig_states = px.bar(state_summary.head(20), 
                           x="state", 
                           y="Total Value",
                           title="Top 20 States by Total Lease Value",
                           hover_data=['Lease Count', 'Savings Rate'])
        fig_states.update_layout(xaxis_tickangle=45)
        st.plotly_chart(fig_states, use_container_width=True)
    
    with col2:
        # Savings rate by state (for states with multiple leases)
        state_savings = state_summary[state_summary['Lease Count'] >= 3].head(15)
        fig_savings = px.bar(state_savings, 
                            x="state", 
                            y="Savings Rate",
                            title="Savings Rate by State (3+ Leases)",
                            hover_data=['Total Savings', 'Lease Count'])
        fig_savings.update_layout(xaxis_tickangle=45)
        st.plotly_chart(fig_savings, use_container_width=True)
    
    # City-level analysis for top cities
    st.markdown("#### 🏙️ Top Cities by Lease Activity")
    city_summary = df_geo.groupby(['city', 'state']).agg({
        'value': ['sum', 'count'],
        'savings': 'sum',
        'sq_ft': 'sum'
    }).round(2)
    
    city_summary.columns = ['Total Value', 'Lease Count', 'Total Savings', 'Total Sq Ft']
    city_summary = city_summary.reset_index()
    city_summary['Location'] = city_summary['city'] + ', ' + city_summary['state']
    city_summary = city_summary.sort_values('Total Value', ascending=False).head(20)
    
    st.dataframe(city_summary[['Location', 'Total Value', 'Lease Count', 'Total Savings', 'Total Sq Ft']])

def render_cost_efficiency_analysis(df):
    """Render cost efficiency analysis"""
    st.markdown("### ⚡ Cost Efficiency Analysis")
    
    # Calculate cost per square foot
    df_eff = df.copy()
    df_eff['cost_per_sqft'] = df_eff['value'] / df_eff['sq_ft']
    df_eff = df_eff.dropna(subset=['cost_per_sqft'])
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Cost per sq ft distribution
        fig_hist = px.histogram(df_eff, 
                               x='cost_per_sqft', 
                               title="Cost per Square Foot Distribution",
                               nbins=30,
                               labels={'cost_per_sqft': 'Cost per Sq Ft ($)'})
        st.plotly_chart(fig_hist, use_container_width=True)
        
        # Summary stats
        st.markdown("#### 📊 Cost Efficiency Metrics")
        avg_cost = df_eff['cost_per_sqft'].mean()
        median_cost = df_eff['cost_per_sqft'].median()
        st.metric("Average Cost/Sq Ft", f"${avg_cost:.2f}")
        st.metric("Median Cost/Sq Ft", f"${median_cost:.2f}")
    
    with col2:
        # Scatter plot: Size vs Cost per sq ft
        fig_scatter = px.scatter(df_eff.sample(min(500, len(df_eff))), 
                                x='sq_ft', 
                                y='cost_per_sqft',
                                title="Property Size vs Cost Efficiency",
                                labels={'sq_ft': 'Square Footage', 'cost_per_sqft': 'Cost per Sq Ft ($)'},
                                hover_data=['agency', 'location'])
        st.plotly_chart(fig_scatter, use_container_width=True)
    
    # Most and least efficient leases
    col3, col4 = st.columns(2)
    
    with col3:
        st.markdown("#### 💸 Most Expensive (per sq ft)")
        expensive = df_eff.nlargest(10, 'cost_per_sqft')[['agency', 'location', 'sq_ft', 'value', 'cost_per_sqft']]
        expensive['cost_per_sqft'] = expensive['cost_per_sqft'].round(2)
        st.dataframe(expensive)
    
    with col4:
        st.markdown("#### 💰 Most Cost-Effective (per sq ft)")
        efficient = df_eff.nsmallest(10, 'cost_per_sqft')[['agency', 'location', 'sq_ft', 'value', 'cost_per_sqft']]
        efficient['cost_per_sqft'] = efficient['cost_per_sqft'].round(2)
        st.dataframe(efficient)

def render_timeline_analysis(df):
    """Render timeline analysis"""
    st.markdown("### 📅 Timeline Analysis")
    
    # Convert dates and analyze trends
    df_time = df.copy()
    df_time['date'] = pd.to_datetime(df_time['date'], errors='coerce')
    df_time = df_time.dropna(subset=['date']).sort_values('date')
    
    if len(df_time) == 0:
        st.warning("No valid dates found for timeline analysis.")
        return
    
    # Daily trends
    daily_summary = df_time.groupby(df_time['date'].dt.date).agg({
        'value': ['sum', 'count'],
        'savings': 'sum',
        'sq_ft': 'sum'
    })
    
    daily_summary.columns = ['Total Value', 'Lease Count', 'Total Savings', 'Total Sq Ft']
    daily_summary = daily_summary.reset_index()
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Lease terminations over time
        fig_count = px.line(daily_summary, 
                           x='date', 
                           y='Lease Count',
                           title="Daily Lease Terminations",
                           markers=True)
        st.plotly_chart(fig_count, use_container_width=True)
    
    with col2:
        # Value and savings over time
        fig_value = px.line(daily_summary, 
                           x='date', 
                           y=['Total Value', 'Total Savings'],
                           title="Daily Value and Savings",
                           markers=True)
        st.plotly_chart(fig_value, use_container_width=True)
    
    # Cumulative analysis
    daily_summary_cum = daily_summary.copy()
    daily_summary_cum['Cumulative Leases'] = daily_summary_cum['Lease Count'].cumsum()
    daily_summary_cum['Cumulative Value'] = daily_summary_cum['Total Value'].cumsum()
    daily_summary_cum['Cumulative Savings'] = daily_summary_cum['Total Savings'].cumsum()
    
    fig_cum = px.line(daily_summary_cum, 
                      x='date', 
                      y=['Cumulative Value', 'Cumulative Savings'],
                      title="Cumulative Lease Value and Savings Over Time",
                      markers=True)
    st.plotly_chart(fig_cum, use_container_width=True)

def render_savings_analysis(df):
    """Render savings-focused analysis"""
    st.markdown("### 💰 Savings Analysis")
    
    # Savings distribution
    savings_leases = df[df['savings'] > 0]
    no_savings_leases = df[df['savings'] == 0]
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Savings rate distribution
        df_savings = df.copy()
        df_savings['savings_rate'] = (df_savings['savings'] / df_savings['value'] * 100).clip(0, 1000)  # Cap at 1000%
        
        fig_savings_dist = px.histogram(df_savings[df_savings['savings_rate'] > 0], 
                                       x='savings_rate',
                                       title="Savings Rate Distribution (%)",
                                       nbins=30,
                                       labels={'savings_rate': 'Savings Rate (%)'})
        st.plotly_chart(fig_savings_dist, use_container_width=True)
        
        st.markdown("#### 📊 Savings Metrics")
        total_savings_rate = (df['savings'].sum() / df['value'].sum() * 100)
        st.metric("Overall Savings Rate", f"{total_savings_rate:.1f}%")
        st.metric("Leases with Savings", f"{len(savings_leases):,} ({len(savings_leases)/len(df)*100:.1f}%)")
        st.metric("Average Savings (when > 0)", f"${savings_leases['savings'].mean():,.0f}")
    
    with col2:
        # Savings vs Value scatter
        fig_scatter = px.scatter(df[df['savings'] > 0].sample(min(300, len(savings_leases))), 
                                x='value', 
                                y='savings',
                                title="Lease Value vs Savings",
                                labels={'value': 'Lease Value ($)', 'savings': 'Savings ($)'},
                                hover_data=['agency', 'location'])
        
        # Add break-even line
        max_val = df['value'].max()
        fig_scatter.add_trace(go.Scatter(x=[0, max_val], y=[0, max_val], 
                                        mode='lines', name='Break-even Line',
                                        line=dict(dash='dash', color='red')))
        st.plotly_chart(fig_scatter, use_container_width=True)
    
    # Top savings opportunities
    st.markdown("#### 🎯 Highest Savings Achievements")
    top_savings = df.nlargest(15, 'savings')[['agency', 'location', 'sq_ft', 'value', 'savings']]
    top_savings['savings_rate'] = (top_savings['savings'] / top_savings['value'] * 100).round(1)
    st.dataframe(top_savings)

def render_property_size_analysis(df):
    """Render property size analysis"""
    st.markdown("### 📐 Property Size Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Size distribution
        fig_size = px.histogram(df, 
                               x='sq_ft',
                               title="Property Size Distribution",
                               nbins=30,
                               labels={'sq_ft': 'Square Footage'})
        st.plotly_chart(fig_size, use_container_width=True)
        
        # Size categories
        df_size = df.copy()
        df_size['size_category'] = pd.cut(df_size['sq_ft'], 
                                         bins=[0, 1000, 5000, 10000, 50000, float('inf')],
                                         labels=['Small (<1K)', 'Medium (1K-5K)', 'Large (5K-10K)', 'Very Large (10K-50K)', 'Massive (50K+)'])
        
        size_summary = df_size.groupby('size_category').agg({
            'value': ['sum', 'count', 'mean'],
            'savings': 'sum',
            'sq_ft': 'sum'
        }).round(0)
        
        size_summary.columns = ['Total Value', 'Count', 'Avg Value', 'Total Savings', 'Total Sq Ft']
        size_summary = size_summary.reset_index()
        
        st.markdown("#### 📊 Size Category Summary")
        st.dataframe(size_summary)
    
    with col2:
        # Size vs Value relationship
        fig_value_size = px.scatter(df.sample(min(400, len(df))), 
                                   x='sq_ft', 
                                   y='value',
                                   title="Property Size vs Total Value",
                                   labels={'sq_ft': 'Square Footage', 'value': 'Total Value ($)'},
                                   hover_data=['agency', 'location'])
        st.plotly_chart(fig_value_size, use_container_width=True)
        
        # Largest properties
        st.markdown("#### 🏗️ Largest Properties Terminated")
        largest = df.nlargest(10, 'sq_ft')[['agency', 'location', 'sq_ft', 'value', 'savings']]
        largest['cost_per_sqft'] = (largest['value'] / largest['sq_ft']).round(2)
        st.dataframe(largest)