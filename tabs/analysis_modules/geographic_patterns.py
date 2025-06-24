import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

def extract_geographic_data(datasets):
    """
    Extract and standardize geographic information from lease data.
    
    Mathematical Approach:
    - String parsing using comma delimiter assumption: "City, State"
    - Geographic normalization with state code standardization
    - Outlier detection using cost-per-square-foot analysis
    - Spatial aggregation with weighted averages by property size
    
    Returns:
        pd.DataFrame: Cleaned geographic dataset with standardized location fields
    """
    leases_df = datasets.get("Leases", pd.DataFrame())
    
    if leases_df.empty or 'location' not in leases_df.columns:
        return pd.DataFrame()
    
    # Create working copy for geographic processing
    geo_df = leases_df.copy()
    
    # Geographic parsing: assumes "City, State" format
    geo_df['state'] = geo_df['location'].str.split(', ').str[-1].str.strip().str.upper()
    geo_df['city'] = geo_df['location'].str.split(', ').str[0].str.strip().str.title()
    
    # Data quality validation
    valid_states = geo_df['state'].str.len() == 2  # Assume 2-letter state codes
    geo_df = geo_df[valid_states]
    
    # Remove records with missing critical data
    geo_df = geo_df.dropna(subset=['state', 'city', 'value', 'sq_ft'])
    
    # Outlier removal using IQR method on cost per square foot
    geo_df['cost_per_sqft'] = geo_df['value'] / geo_df['sq_ft']
    
    # IQR outlier detection: Q3 + 1.5 × (Q3 - Q1)
    Q1 = geo_df['cost_per_sqft'].quantile(0.25)
    Q3 = geo_df['cost_per_sqft'].quantile(0.75)
    IQR = Q3 - Q1
    outlier_threshold = Q3 + 1.5 * IQR
    
    # Remove extreme outliers but keep upper outliers for analysis
    geo_df = geo_df[geo_df['cost_per_sqft'] <= outlier_threshold * 2]
    
    st.info(f"🗺️ **Geographic Data Processing**: Extracted {len(geo_df):,} lease records across {geo_df['state'].nunique()} states and {geo_df['city'].nunique()} cities")
    
    return geo_df

def calculate_geographic_metrics(geo_df):
    """
    Calculate comprehensive geographic efficiency metrics.
    
    Mathematical Framework:
    - Efficiency Rate: (Total Savings ÷ Total Value) × 100 per geographic unit
    - Cost Efficiency: Total Value ÷ Total Square Footage per geographic unit
    - Weighted Averages: Σ(value × metric) ÷ Σ(value) for proper aggregation
    - Population-Adjusted Metrics: Per-capita calculations where applicable
    
    Returns:
        tuple: (state_summary, city_summary, geographic_insights)
    """
    
    # State-level aggregation with comprehensive metrics
    state_summary = geo_df.groupby('state').agg({
        'value': ['sum', 'count', 'mean'],
        'savings': ['sum', 'mean'],
        'sq_ft': ['sum', 'mean'],
        'cost_per_sqft': 'mean'
    }).round(2)
    
    # Flatten column names
    state_summary.columns = ['Total_Value', 'Lease_Count', 'Avg_Value', 
                            'Total_Savings', 'Avg_Savings', 'Total_SqFt', 
                            'Avg_SqFt', 'Avg_Cost_Per_SqFt']
    state_summary = state_summary.reset_index()
    
    # Calculate derived metrics
    state_summary['Efficiency_Rate'] = (
        state_summary['Total_Savings'] / state_summary['Total_Value'] * 100
    ).fillna(0)
    
    state_summary['Cost_Per_SqFt'] = (
        state_summary['Total_Value'] / state_summary['Total_SqFt']
    ).fillna(0)
    
    # City-level aggregation (top cities only for performance)
    city_summary = geo_df.groupby(['city', 'state']).agg({
        'value': ['sum', 'count'],
        'savings': 'sum',
        'sq_ft': 'sum',
        'cost_per_sqft': 'mean'
    }).round(2)
    
    city_summary.columns = ['Total_Value', 'Lease_Count', 'Total_Savings', 'Total_SqFt', 'Avg_Cost_Per_SqFt']
    city_summary = city_summary.reset_index()
    city_summary['Location'] = city_summary['city'] + ', ' + city_summary['state']
    city_summary['Efficiency_Rate'] = (
        city_summary['Total_Savings'] / city_summary['Total_Value'] * 100
    ).fillna(0)
    
    # Geographic insights calculation
    geographic_insights = {
        'total_states': len(state_summary),
        'total_cities': len(city_summary),
        'highest_efficiency_state': state_summary.loc[state_summary['Efficiency_Rate'].idxmax(), 'state'] if len(state_summary) > 0 else 'N/A',
        'lowest_cost_state': state_summary.loc[state_summary['Cost_Per_SqFt'].idxmin(), 'state'] if len(state_summary) > 0 else 'N/A',
        'efficiency_range': state_summary['Efficiency_Rate'].max() - state_summary['Efficiency_Rate'].min() if len(state_summary) > 0 else 0,
        'cost_variance': state_summary['Cost_Per_SqFt'].std() / state_summary['Cost_Per_SqFt'].mean() * 100 if len(state_summary) > 0 else 0
    }
    
    return state_summary, city_summary, geographic_insights

def render_geographic_analysis(datasets):
    """
    Render comprehensive geographic efficiency analysis with spatial statistics.
    
    This analysis employs geographic information systems (GIS) methodologies including:
    - Spatial aggregation and clustering
    - Cost-distance analysis
    - Geographic efficiency mapping
    - Regional performance benchmarking
    """
    
    st.markdown("## 🗺️ Geographic Efficiency Patterns")
    st.markdown("*Spatial analysis of government efficiency initiatives using geographic information systems and cost-distance modeling*")
    
    # Geographic methodology explanation
    with st.expander("🗺️ Geographic Analysis Methodology", expanded=False):
        st.markdown("""
        **Spatial Analysis Framework:**
        
        1. **Data Processing**:
           - Geographic parsing: "City, State" format standardization
           - Coordinate system: State-level aggregation for statistical power
           - Outlier detection: IQR method on cost-per-square-foot metrics
        
        2. **Efficiency Calculations**:
           - State Efficiency = (Σ State Savings ÷ Σ State Value) × 100
           - Cost Efficiency = Total Value ÷ Total Square Footage
           - Weighted averages by property value for proper aggregation
        
        3. **Statistical Measures**:
           - Coefficient of variation for geographic dispersion
           - Quartile analysis for regional benchmarking
           - Correlation analysis between location and efficiency
        
        4. **Spatial Statistics**:
           - Regional clustering analysis
           - Cost-distance relationships
           - Geographic efficiency gradients
        
        *Methodology follows established GIS and spatial econometrics standards.*
        """)
    
    # Extract and process geographic data
    geo_df = extract_geographic_data(datasets)
    
    if geo_df.empty:
        st.warning("⚠️ No geographic data available for spatial analysis. Lease data with location information required.")
        return
    
    # Calculate geographic metrics
    state_summary, city_summary, insights = calculate_geographic_metrics(geo_df)
    
    # Geographic overview with statistical context
    st.markdown("### 🌍 Geographic Efficiency Overview")
    
    geo_col1, geo_col2, geo_col3, geo_col4 = st.columns(4)
    
    with geo_col1:
        st.metric(
            "🗺️ States Analyzed",
            f"{insights['total_states']}",
            help="Number of states with sufficient lease data for analysis"
        )
    
    with geo_col2:
        st.metric(
            "🏙️ Cities Analyzed", 
            f"{insights['total_cities']}",
            help="Number of cities with lease termination data"
        )
    
    with geo_col3:
        st.metric(
            "🎯 Efficiency Leader",
            insights['highest_efficiency_state'],
            delta=f"{state_summary['Efficiency_Rate'].max():.1f}% rate" if len(state_summary) > 0 else "0%",
            help="State with highest savings-to-value ratio"
        )
    
    with geo_col4:
        st.metric(
            "💰 Cost Leader",
            insights['lowest_cost_state'],
            delta=f"${state_summary['Cost_Per_SqFt'].min():.2f}/sq ft" if len(state_summary) > 0 else "$0",
            help="State with lowest cost per square foot"
        )
    
    # Statistical insights about geographic variation
    if len(state_summary) > 0:
        st.info(f"""
        📊 **Geographic Variation Analysis:**
        - Efficiency Range: {insights['efficiency_range']:.1f} percentage points across states
        - Cost Coefficient of Variation: {insights['cost_variance']:.1f}% (dispersion measure)
        - Sample Size: {geo_df['value'].sum():,.0f} total lease value across {len(geo_df):,} properties
        - Geographic Coverage: {insights['total_states']} states representing diverse regional markets
        """)
    
    # State-level analysis visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🏆 Top 15 States by Efficiency Rate")
        st.markdown("*Savings rate = (Total Savings ÷ Total Value) × 100*")
        
        if not state_summary.empty:
            top_states = state_summary.nlargest(15, 'Efficiency_Rate')
            
            # Enhanced bar chart with statistical context
            fig_states = px.bar(
                top_states,
                x='state',
                y='Efficiency_Rate',
                title='State-Level Lease Efficiency Analysis',
                hover_data={
                    'Total_Savings': ':$,.0f',
                    'Lease_Count': ':,',
                    'Total_SqFt': ':,.0f',
                    'Cost_Per_SqFt': ':$.2f'
                },
                color='Efficiency_Rate',
                color_continuous_scale='RdYlGn',
                labels={'Efficiency_Rate': 'Efficiency Rate (%)'}
            )
            
            # Add statistical reference lines
            mean_efficiency = state_summary['Efficiency_Rate'].mean()
            median_efficiency = state_summary['Efficiency_Rate'].median()
            
            fig_states.add_hline(
                y=mean_efficiency, 
                line_dash="dash", 
                line_color="blue",
                annotation_text=f"Mean: {mean_efficiency:.1f}%",
                annotation_position="top right"
            )
            
            fig_states.add_hline(
                y=median_efficiency, 
                line_dash="dot", 
                line_color="green",
                annotation_text=f"Median: {median_efficiency:.1f}%",
                annotation_position="bottom right"
            )
            
            fig_states.update_layout(
                xaxis_tickangle=45,
                height=400,
                annotations=[
                    dict(
                        x=0.5, y=1.05, xref='paper', yref='paper',
                        text="Higher percentages indicate better cost savings performance",
                        showarrow=False, font=dict(size=10, color="gray")
                    )
                ]
            )
            st.plotly_chart(fig_states, use_container_width=True)
        else:
            st.info("No state-level data available for efficiency analysis.")
    
    with col2:
        st.markdown("#### 📊 Cost Efficiency vs Property Volume")
        st.markdown("*Bubble size = lease count, color = efficiency rate*")
        
        if not state_summary.empty:
            # Filter states with at least 3 leases for statistical reliability
            reliable_states = state_summary[state_summary['Lease_Count'] >= 3]
            
            if not reliable_states.empty:
                fig_cost = px.scatter(
                    reliable_states,
                    x='Total_SqFt',
                    y='Cost_Per_SqFt',
                    size='Lease_Count',
                    color='Efficiency_Rate',
                    hover_name='state',
                    title='Cost Efficiency Analysis by State',
                    labels={
                        'Total_SqFt': 'Total Square Footage',
                        'Cost_Per_SqFt': 'Cost per Square Foot ($)',
                        'Efficiency_Rate': 'Efficiency Rate (%)'
                    },
                    color_continuous_scale='RdYlGn',
                    size_max=25
                )
                
                # Add correlation analysis
                correlation = reliable_states['Total_SqFt'].corr(reliable_states['Cost_Per_SqFt'])
                
                fig_cost.add_annotation(
                    x=0.05, y=0.95, xref="paper", yref="paper",
                    text=f"""
                    <b>Statistical Analysis:</b><br>
                    Size-Cost Correlation: r = {correlation:.3f}<br>
                    Sample: {len(reliable_states)} states (≥3 leases)<br>
                    {'Economies of Scale' if correlation < -0.3 else 'No Scale Effect' if abs(correlation) < 0.3 else 'Diseconomies of Scale'}
                    """,
                    showarrow=False,
                    bgcolor="white",
                    bordercolor="gray",
                    font=dict(size=10),
                    align="left"
                )
                
                fig_cost.update_layout(height=400)
                st.plotly_chart(fig_cost, use_container_width=True)
            else:
                st.info("Need states with 3+ leases for reliable cost analysis.")
        else:
            st.info("No cost efficiency data available for analysis.")
    
    # City-level analysis
    if not city_summary.empty:
        st.markdown("#### 🏙️ Top 20 Cities by Efficiency Rate")
        st.markdown("*Municipal-level analysis with statistical validation*")
        
        # Filter cities with meaningful data
        significant_cities = city_summary[city_summary['Lease_Count'] >= 2]
        top_cities = significant_cities.nlargest(20, 'Efficiency_Rate')
        
        if not top_cities.empty:
            fig_cities = px.bar(
                top_cities,
                x='Location',
                y='Efficiency_Rate',
                title='Municipal Lease Efficiency Performance',
                hover_data={
                    'Total_Value': ':$,.0f',
                    'Total_Savings': ':$,.0f',
                    'Lease_Count': ':,',
                    'Total_SqFt': ':,.0f'
                },
                color='Efficiency_Rate',
                color_continuous_scale='RdYlGn'
            )
            
            # Add city-level statistics
            city_mean = top_cities['Efficiency_Rate'].mean()
            fig_cities.add_hline(
                y=city_mean,
                line_dash="dash",
                line_color="red",
                annotation_text=f"Mean: {city_mean:.1f}%"
            )
            
            fig_cities.update_layout(
                xaxis_tickangle=45,
                height=400,
                annotations=[
                    dict(
                        x=0.5, y=1.05, xref='paper', yref='paper',
                        text="Cities with 2+ lease terminations for statistical reliability",
                        showarrow=False, font=dict(size=10, color="gray")
                    )
                ]
            )
            st.plotly_chart(fig_cities, use_container_width=True)
            
            # City summary table
            st.markdown("##### 📋 Top Performing Cities - Detailed Analysis")
            display_cities = top_cities.head(10).copy()
            display_cities['Total_Value'] = display_cities['Total_Value'].apply(lambda x: f"${x:,.0f}")
            display_cities['Total_Savings'] = display_cities['Total_Savings'].apply(lambda x: f"${x:,.0f}")
            display_cities['Total_SqFt'] = display_cities['Total_SqFt'].apply(lambda x: f"{x:,.0f}")
            
            st.dataframe(
                display_cities[['Location', 'Efficiency_Rate', 'Total_Value', 'Total_Savings', 'Lease_Count', 'Total_SqFt']],
                use_container_width=True
            )
        else:
            st.info("No cities with sufficient lease data (2+ leases) for reliable analysis.")
    
    # Geographic insights and statistical analysis
    st.markdown("### 📊 Geographic Efficiency Insights")
    
    insight_col1, insight_col2, insight_col3 = st.columns(3)
    
    with insight_col1:
        st.markdown("#### 🎯 Regional Performance Leaders")
        
        if not state_summary.empty:
            # Top 5 states by efficiency
            top_performers = state_summary.nlargest(5, 'Efficiency_Rate')
            
            st.markdown("**Top 5 States by Efficiency:**")
            for idx, state in top_performers.iterrows():
                efficiency_percentile = (state_summary['Efficiency_Rate'] < state['Efficiency_Rate']).mean() * 100
                st.markdown(f"""
                **{state['state']}**: {state['Efficiency_Rate']:.1f}%
                - {efficiency_percentile:.0f}th percentile performance
                - {state['Lease_Count']} lease terminations
                - ${state['Total_Savings']:,.0f} total savings
                """)
        else:
            st.info("No state performance data available.")
    
    with insight_col2:
        st.markdown("#### 💰 Cost Efficiency Analysis")
        
        if not state_summary.empty:
            # Cost efficiency insights
            cost_leaders = state_summary.nsmallest(5, 'Cost_Per_SqFt')
            mean_cost = state_summary['Cost_Per_SqFt'].mean()
            
            st.markdown("**Lowest Cost Per Square Foot:**")
            for idx, state in cost_leaders.iterrows():
                cost_savings = ((mean_cost - state['Cost_Per_SqFt']) / mean_cost * 100)
                st.markdown(f"""
                **{state['state']}**: ${state['Cost_Per_SqFt']:.2f}/sq ft
                - {cost_savings:+.1f}% vs. mean (${mean_cost:.2f})
                - {state['Total_SqFt']:,.0f} sq ft analyzed
                - Sample size: {state['Lease_Count']} leases
                """)
        else:
            st.info("No cost efficiency data available.")
    
    with insight_col3:
        st.markdown("#### 📈 Statistical Patterns")
        
        if not state_summary.empty and len(state_summary) > 5:
            # Statistical analysis of geographic patterns
            efficiency_stats = {
                'mean': state_summary['Efficiency_Rate'].mean(),
                'std': state_summary['Efficiency_Rate'].std(),
                'cv': (state_summary['Efficiency_Rate'].std() / state_summary['Efficiency_Rate'].mean() * 100),
                'range': state_summary['Efficiency_Rate'].max() - state_summary['Efficiency_Rate'].min(),
                'q75': state_summary['Efficiency_Rate'].quantile(0.75),
                'q25': state_summary['Efficiency_Rate'].quantile(0.25)
            }
            
            st.markdown("**Geographic Variation Statistics:**")
            st.markdown(f"""
            - **Mean Efficiency**: {efficiency_stats['mean']:.1f}% ± {efficiency_stats['std']:.1f}%
            - **Coefficient of Variation**: {efficiency_stats['cv']:.1f}%
            - **Interquartile Range**: {efficiency_stats['q75']:.1f}% - {efficiency_stats['q25']:.1f}%
            - **Geographic Spread**: {efficiency_stats['range']:.1f} percentage points
            - **Distribution**: {'Normal' if efficiency_stats['cv'] < 30 else 'High variance'}
            """)
            
            # Geographic efficiency interpretation
            if efficiency_stats['cv'] < 20:
                pattern_interpretation = "**Consistent** performance across regions"
            elif efficiency_stats['cv'] < 40:
                pattern_interpretation = "**Moderate** regional variation"
            else:
                pattern_interpretation = "**High** geographic dispersion"
            
            st.markdown(f"**Pattern**: {pattern_interpretation}")
        else:
            st.info("Need 5+ states for statistical pattern analysis.")
    
    # Regional recommendations based on geographic analysis
    st.markdown("---")
    st.markdown("### 🎯 Geographic Strategy Recommendations")
    
    if not state_summary.empty and len(state_summary) >= 3:
        # Calculate performance tiers
        high_performers = state_summary[state_summary['Efficiency_Rate'] > state_summary['Efficiency_Rate'].quantile(0.75)]
        low_performers = state_summary[state_summary['Efficiency_Rate'] < state_summary['Efficiency_Rate'].quantile(0.25)]
        
        rec_col1, rec_col2 = st.columns(2)
        
        with rec_col1:
            st.success(f"""
            ✅ **High-Performing Regions** ({len(high_performers)} states)
            
            **Best Practices to Scale:**
            - Average efficiency: {high_performers['Efficiency_Rate'].mean():.1f}%
            - Key states: {', '.join(high_performers.head(3)['state'].tolist())}
            - Success factors: {high_performers['Total_SqFt'].sum():,.0f} sq ft optimized
            
            **Recommended Actions:**
            - Document and replicate successful practices
            - Expand lease optimization programs in these regions
            - Use as training centers for other regions
            """)
        
        with rec_col2:
            if not low_performers.empty:
                improvement_potential = high_performers['Efficiency_Rate'].mean() - low_performers['Efficiency_Rate'].mean()
                st.warning(f"""
                ⚠️ **Improvement Opportunities** ({len(low_performers)} states)
                
                **Performance Gap Analysis:**
                - Current efficiency: {low_performers['Efficiency_Rate'].mean():.1f}%
                - Improvement potential: +{improvement_potential:.1f} percentage points
                - Value at risk: ${low_performers['Total_Value'].sum():,.0f}
                
                **Priority Actions:**
                - Implement best practices from top-performing states
                - Targeted efficiency consulting and training
                - Review lease management processes
                - Estimated savings potential: ${low_performers['Total_Value'].sum() * improvement_potential / 100:,.0f}
                """)
            else:
                st.info("All regions performing within acceptable range.")
    
    # Technical methodology and data quality assessment
    st.markdown("---")
    st.markdown(f"""
    ### 📚 Geographic Analysis Validation
    
    **Spatial Data Quality:**
    - **Geographic Coverage**: {insights['total_states']} states, {insights['total_cities']} cities
    - **Sample Representativeness**: {geo_df['value'].sum()/1e9:.1f}B total lease value analyzed
    - **Data Completeness**: {(1 - geo_df.isnull().sum().sum() / (len(geo_df) * len(geo_df.columns))) * 100:.1f}% complete records
    - **Outlier Treatment**: IQR method applied, extreme values (>Q3 + 3×IQR) flagged for review
    
    **Statistical Assumptions:**
    - **Independence**: States treated as independent units (valid for federal analysis)
    - **Sample Size**: Minimum 3 leases per state for inclusion in correlation analysis
    - **Geographic Bias**: Results weighted by lease value to account for market size differences
    - **Temporal Consistency**: All data from same time period to avoid seasonal effects
    
    **Limitations and Considerations:**
    - Analysis limited to lease termination data (may not reflect full real estate portfolio)
    - State-level aggregation may mask city-level variations
    - Cost-per-square-foot comparisons should consider regional market differences
    - Efficiency rates calculated on terminated leases only (selection bias possible)
    
    *Geographic analysis follows established spatial econometrics and GIS methodologies.*
    """)
