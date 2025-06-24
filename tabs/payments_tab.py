import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from utils.chart_utils import format_billions, format_millions, create_download_button
from models.outlier_detection import perform_outlier_detection

def render_payments_tab(df):
    """Enhanced payments tab with comprehensive analysis and better visualizations"""
    
    st.markdown("# 💳 Government Payments Analysis")
    st.markdown("*Comprehensive analysis of federal payment transactions with advanced analytics*")
    
    # Enhanced CSS for better visibility
    st.markdown("""
    <style>
    .payment-card {
        background: rgba(255, 255, 255, 0.95) !important;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        border-left: 5px solid #007bff;
    }
    
    .payment-card h4 {
        color: #333 !important;
        margin-bottom: 1rem;
    }
    
    .insight-box {
        background: #f8f9fa;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
        border-left: 4px solid #28a745;
    }
    
    .warning-box {
        background: #fff3cd;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
        border-left: 4px solid #ffc107;
    }
    
    .metric-highlight {
        font-size: 1.2rem;
        font-weight: bold;
        color: #007bff;
    }
    </style>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        # Enhanced filters
        if not df.empty:
            filtered_df = create_enhanced_payment_filters(df)
            create_download_button(filtered_df, "Payments", "enhanced_payments")
        else:
            filtered_df = df
    
    with col2:
        if filtered_df.empty:
            st.warning("⚠️ No payment data available for analysis.")
            return
        
        # Enhanced summary metrics
        render_enhanced_payment_summary(filtered_df)
        
        # Enhanced analysis tabs
        render_enhanced_payment_analysis_tabs(filtered_df)
        
        # Enhanced raw data table
        st.markdown("---")
        with st.expander("📋 View Enhanced Data Table", expanded=False):
            render_enhanced_data_table(filtered_df)

def create_enhanced_payment_filters(df):
    """Create enhanced filters with better organization"""
    
    st.markdown("### 🎛️ Analysis Filters")
    
    # Agency filter (if available)
    if 'agency_name' in df.columns:
        agencies = sorted(df['agency_name'].dropna().unique().tolist())
        selected_agencies = st.multiselect(
            "🏢 Select Agencies", 
            agencies, 
            default=agencies[:5] if len(agencies) > 5 else agencies,
            key="payments_agencies"
        )
    else:
        selected_agencies = []
    
    # Payment amount filter
    amount_col = None
    for col in df.columns:
        if any(term in col.lower() for term in ['amount', 'payment_amt', 'value']):
            if pd.api.types.is_numeric_dtype(df[col]):
                amount_col = col
                break
    
    if amount_col:
        min_val, max_val = df[amount_col].min(), df[amount_col].max()
        
        # Use slider for amount range
        amount_range = st.slider(
            f"💰 Payment Amount Range",
            min_value=float(min_val),
            max_value=float(max_val),
            value=(float(min_val), float(max_val)),
            format="$%.0f",
            key="payment_amount_range"
        )
    else:
        amount_range = None
    
    # Date filter (if available)
    date_cols = [col for col in df.columns if 'date' in col.lower()]
    if date_cols:
        date_col = date_cols[0]
        df_temp = df.copy()
        df_temp[date_col] = pd.to_datetime(df_temp[date_col], errors='coerce')
        valid_dates = df_temp[date_col].dropna()
        
        if len(valid_dates) > 0:
            date_range = st.date_input(
                "📅 Date Range",
                value=(valid_dates.min().date(), valid_dates.max().date()),
                min_value=valid_dates.min().date(),
                max_value=valid_dates.max().date(),
                key="payment_date_range"
            )
        else:
            date_range = None
    else:
        date_col = None
        date_range = None
    
    # Top N filter
    top_n = st.selectbox(
        "📊 Show Top N Results",
        [10, 20, 50, 100],
        index=1,
        key="payment_top_n"
    )
    
    # Apply filters
    filtered_df = df.copy()
    
    if selected_agencies and 'agency_name' in df.columns:
        filtered_df = filtered_df[filtered_df['agency_name'].isin(selected_agencies)]
    
    if amount_range and amount_col:
        filtered_df = filtered_df[
            (filtered_df[amount_col] >= amount_range[0]) & 
            (filtered_df[amount_col] <= amount_range[1])
        ]
    
    if date_range and date_col and len(date_range) == 2:
        filtered_df[date_col] = pd.to_datetime(filtered_df[date_col], errors='coerce')
        start_date, end_date = date_range
        filtered_df = filtered_df[
            (filtered_df[date_col].dt.date >= start_date) & 
            (filtered_df[date_col].dt.date <= end_date)
        ]
    
    # Store top_n for later use
    st.session_state.payment_top_n = top_n
    
    return filtered_df

def render_enhanced_payment_summary(df):
    """Render enhanced payment summary with better insights"""
    
    st.markdown("### 💳 Enhanced Payment Analytics Dashboard")
    
    # Find amount column
    amount_col = None
    for col in df.columns:
        if any(term in col.lower() for term in ['amount', 'payment_amt', 'value']):
            if pd.api.types.is_numeric_dtype(df[col]):
                amount_col = col
                break
    
    # Calculate comprehensive metrics
    total_payments = len(df)
    total_amount = df[amount_col].sum() if amount_col else 0
    avg_amount = df[amount_col].mean() if amount_col else 0
    median_amount = df[amount_col].median() if amount_col else 0
    unique_agencies = df['agency_name'].nunique() if 'agency_name' in df.columns else 0
    
    # Display enhanced metrics
    metric_col1, metric_col2, metric_col3, metric_col4, metric_col5 = st.columns(5)
    
    with metric_col1:
        st.metric(
            "💳 Total Payments", 
            f"{total_payments:,}",
            help="Total number of payment transactions analyzed"
        )
    
    with metric_col2:
        if total_amount > 1e9:
            amount_display = format_billions(total_amount)
        elif total_amount > 1e6:
            amount_display = format_millions(total_amount)
        else:
            amount_display = f"${total_amount:,.0f}"
        
        st.metric(
            "💰 Total Amount", 
            amount_display,
            help="Total value of all payment transactions"
        )
    
    with metric_col3:
        if avg_amount > 1e6:
            avg_display = format_millions(avg_amount)
        else:
            avg_display = f"${avg_amount:,.0f}"
        
        st.metric(
            "📊 Average Payment", 
            avg_display,
            delta=f"Median: ${median_amount:,.0f}",
            help="Average payment amount with median comparison"
        )
    
    with metric_col4:
        st.metric(
            "🏢 Agencies", 
            f"{unique_agencies}",
            help="Number of unique agencies making payments"
        )
    
    with metric_col5:
        # Calculate payment frequency (if date available)
        date_cols = [col for col in df.columns if 'date' in col.lower()]
        if date_cols and not df.empty:
            date_col = date_cols[0]
            df_temp = df.copy()
            df_temp[date_col] = pd.to_datetime(df_temp[date_col], errors='coerce')
            valid_dates = df_temp[date_col].dropna()
            
            if len(valid_dates) > 1:
                date_range_days = (valid_dates.max() - valid_dates.min()).days
                if date_range_days > 0:
                    daily_rate = len(valid_dates) / date_range_days
                    st.metric(
                        "📈 Daily Rate", 
                        f"{daily_rate:.1f}",
                        help="Average payments per day"
                    )
                else:
                    st.metric("📈 Daily Rate", "N/A")
            else:
                st.metric("📈 Daily Rate", "N/A")
        else:
            st.metric("📈 Processing Rate", "100%")
    
    # Enhanced insights
    st.markdown("---")
    
    if amount_col:
        # Statistical insights
        std_dev = df[amount_col].std()
        cv = (std_dev / avg_amount * 100) if avg_amount > 0 else 0
        
        insight_col1, insight_col2 = st.columns(2)
        
        with insight_col1:
            st.markdown(f"""
            <div class="insight-box">
                <h4>📊 Statistical Insights</h4>
                <ul>
                    <li><strong>Payment Variability:</strong> {cv:.1f}% coefficient of variation</li>
                    <li><strong>Distribution:</strong> {'Right-skewed' if avg_amount > median_amount else 'Left-skewed' if avg_amount < median_amount else 'Normal'}</li>
                    <li><strong>Largest Payment:</strong> ${df[amount_col].max():,.0f}</li>
                    <li><strong>Smallest Payment:</strong> ${df[amount_col].min():,.0f}</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        with insight_col2:
            # Risk indicators
            large_payments = df[df[amount_col] > df[amount_col].quantile(0.95)]
            risk_level = "High" if len(large_payments) > total_payments * 0.1 else "Medium" if cv > 200 else "Low"
            
            st.markdown(f"""
            <div class="{'warning-box' if risk_level != 'Low' else 'insight-box'}">
                <h4>⚠️ Risk Assessment</h4>
                <ul>
                    <li><strong>Risk Level:</strong> {risk_level}</li>
                    <li><strong>Large Payments (>95th percentile):</strong> {len(large_payments):,}</li>
                    <li><strong>Concentration:</strong> Top 5% = {(df[amount_col].nlargest(int(len(df)*0.05)).sum() / total_amount * 100):.1f}% of total</li>
                    <li><strong>Recommendation:</strong> {'Review large payments' if risk_level != 'Low' else 'Standard monitoring'}</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

def render_enhanced_payment_analysis_tabs(df):
    """Render enhanced analysis tabs with better visualizations"""
    
    # Determine available tabs based on data
    available_tabs = ["📊 Payment Overview", "🏢 Agency Analysis"]
    
    # Add tabs based on available columns
    if any('recipient' in col.lower() for col in df.columns):
        available_tabs.append("🎯 Recipient Analysis")
    
    if any('date' in col.lower() for col in df.columns):
        available_tabs.append("📅 Timeline Analysis")
    
    # Financial analysis
    amount_cols = [col for col in df.columns if any(term in col.lower() for term in ['amount', 'value', 'payment'])]
    if amount_cols:
        available_tabs.append("💰 Financial Analysis")
    
    # Text analysis for justifications
    if any('justification' in col.lower() for col in df.columns):
        available_tabs.append("📝 Justification Analysis")
    
    # Anomaly detection
    if len(df.select_dtypes(include=['number']).columns) >= 1:
        available_tabs.append("🔍 Anomaly Detection")
    
    # Create enhanced tabs
    selected_tab = st.radio(
        "🎯 Choose Analysis Focus", 
        available_tabs, 
        horizontal=True, 
        key="enhanced_payments_tabs"
    )
    
    if selected_tab == "📊 Payment Overview":
        render_payment_overview(df)
    elif selected_tab == "🏢 Agency Analysis":
        render_enhanced_agency_analysis(df)
    elif selected_tab == "🎯 Recipient Analysis":
        render_enhanced_recipient_analysis(df)
    elif selected_tab == "📅 Timeline Analysis":
        render_enhanced_timeline_analysis(df)
    elif selected_tab == "💰 Financial Analysis":
        render_enhanced_financial_analysis(df)
    elif selected_tab == "📝 Justification Analysis":
        render_justification_analysis(df)
    elif selected_tab == "🔍 Anomaly Detection":
        render_enhanced_anomaly_detection(df)

def render_payment_overview(df):
    """Render comprehensive payment overview"""
    
    st.markdown("#### 📊 Payment Distribution Analysis")
    
    amount_col = None
    for col in df.columns:
        if any(term in col.lower() for term in ['amount', 'payment_amt', 'value']):
            if pd.api.types.is_numeric_dtype(df[col]):
                amount_col = col
                break
    
    if amount_col:
        col1, col2 = st.columns(2)
        
        with col1:
            # Enhanced histogram
            fig_hist = px.histogram(
                df, 
                x=amount_col, 
                title="Payment Amount Distribution",
                nbins=30,
                labels={amount_col: 'Payment Amount ($)'},
                color_discrete_sequence=['#007bff']
            )
            
            # Add statistical lines
            mean_val = df[amount_col].mean()
            median_val = df[amount_col].median()
            
            fig_hist.add_vline(x=mean_val, line_dash="dash", line_color="red", 
                              annotation_text=f"Mean: ${mean_val:,.0f}")
            fig_hist.add_vline(x=median_val, line_dash="dot", line_color="green", 
                              annotation_text=f"Median: ${median_val:,.0f}")
            
            fig_hist.update_layout(height=400)
            st.plotly_chart(fig_hist, use_container_width=True)
        
        with col2:
            # Box plot for outlier identification
            fig_box = px.box(
                df, 
                y=amount_col, 
                title="Payment Amount Distribution (Box Plot)",
                labels={amount_col: 'Payment Amount ($)'}
            )
            fig_box.update_layout(height=400)
            st.plotly_chart(fig_box, use_container_width=True)
        
        # Payment size categories
        st.markdown("#### 💳 Payment Size Categories")
        
        # Define categories
        df_temp = df.copy()
        df_temp['amount_category'] = pd.cut(
            df_temp[amount_col],
            bins=[0, 1000, 10000, 100000, 1000000, float('inf')],
            labels=['Small (<$1K)', 'Medium ($1K-$10K)', 'Large ($10K-$100K)', 
                   'Very Large ($100K-$1M)', 'Massive (>$1M)'],
            include_lowest=True
        )
        
        category_summary = df_temp.groupby('amount_category').agg({
            amount_col: ['count', 'sum', 'mean']
        }).round(0)
        
        category_summary.columns = ['Count', 'Total_Amount', 'Avg_Amount']
        category_summary = category_summary.reset_index()
        category_summary['Percentage'] = (category_summary['Count'] / len(df) * 100).round(1)
        
        # Visualization
        fig_cat = px.bar(
            category_summary,
            x='amount_category',
            y='Count',
            title='Payment Count by Size Category',
            text='Percentage',
            labels={'amount_category': 'Payment Size Category'},
            color='Count',
            color_continuous_scale='Blues'
        )
        fig_cat.update_traces(texttemplate='%{text}%', textposition='outside')
        fig_cat.update_layout(height=400)
        st.plotly_chart(fig_cat, use_container_width=True)
        
        # Summary table
        st.dataframe(category_summary, use_container_width=True)

def render_enhanced_agency_analysis(df):
    """Enhanced agency analysis with better insights"""
    
    st.markdown("#### 🏢 Agency Payment Analysis")
    
    if 'agency_name' not in df.columns:
        st.warning("No agency information available for analysis.")
        return
    
    amount_col = next((col for col in df.columns if any(term in col.lower() for term in ['amount', 'payment_amt', 'value']) and pd.api.types.is_numeric_dtype(df[col])), None)
    
    if not amount_col:
        st.warning("No payment amount column found for analysis.")
        return
    
    # Get top N from session state
    top_n = getattr(st.session_state, 'payment_top_n', 20)
    
    # Agency summary
    agency_summary = df.groupby('agency_name').agg({
        amount_col: ['count', 'sum', 'mean', 'std']
    }).round(2)
    
    agency_summary.columns = ['Payment_Count', 'Total_Amount', 'Avg_Payment', 'Std_Dev']
    agency_summary = agency_summary.reset_index()
    agency_summary['Efficiency_Score'] = (agency_summary['Total_Amount'] / agency_summary['Payment_Count']).round(0)
    agency_summary = agency_summary.sort_values('Total_Amount', ascending=False)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Top agencies by total amount
        top_agencies = agency_summary.head(top_n)
        
        fig_agencies = px.bar(
            top_agencies,
            x='agency_name',
            y='Total_Amount',
            title=f'Top {top_n} Agencies by Total Payment Amount',
            hover_data=['Payment_Count', 'Avg_Payment'],
            labels={'agency_name': 'Agency', 'Total_Amount': 'Total Amount ($)'},
            color='Total_Amount',
            color_continuous_scale='Blues'
        )
        fig_agencies.update_layout(xaxis_tickangle=45, height=500)
        st.plotly_chart(fig_agencies, use_container_width=True)
    
    with col2:
        # Efficiency analysis
        fig_efficiency = px.scatter(
            agency_summary.head(top_n),
            x='Payment_Count',
            y='Avg_Payment',
            size='Total_Amount',
            hover_name='agency_name',
            title='Agency Efficiency Analysis',
            labels={
                'Payment_Count': 'Number of Payments',
                'Avg_Payment': 'Average Payment Amount ($)'
            },
            color='Efficiency_Score',
            color_continuous_scale='RdYlGn'
        )
        fig_efficiency.update_layout(height=500)
        st.plotly_chart(fig_efficiency, use_container_width=True)
    
    # Agency performance table
    st.markdown(f"#### 📋 Top {top_n} Agency Performance Summary")
    display_df = top_agencies.copy()
    display_df['Total_Amount'] = display_df['Total_Amount'].apply(lambda x: f"${x:,.0f}")
    display_df['Avg_Payment'] = display_df['Avg_Payment'].apply(lambda x: f"${x:,.0f}")
    st.dataframe(display_df, use_container_width=True)

def render_justification_analysis(df):
    """Analyze payment justifications"""
    
    st.markdown("#### 📝 Payment Justification Analysis")
    
    # Find justification columns
    justification_cols = [col for col in df.columns if 'justification' in col.lower()]
    
    if not justification_cols:
        st.warning("No justification columns found for analysis.")
        return
    
    # Analyze justifications
    for col in justification_cols:
        st.markdown(f"##### {col.replace('_', ' ').title()}")
        
        # Get top justifications by amount
        amount_col = next((c for c in df.columns if any(term in c.lower() for term in ['amount', 'payment_amt', 'value']) and pd.api.types.is_numeric_dtype(df[c])), None)
        
        if amount_col:
            justification_summary = df.groupby(col).agg({
                amount_col: ['sum', 'count', 'mean']
            }).round(0)
            
            justification_summary.columns = ['Total_Amount', 'Payment_Count', 'Avg_Amount']
            justification_summary = justification_summary.reset_index()
            justification_summary = justification_summary.sort_values('Total_Amount', ascending=False)
            
            # Get top N
            top_n = getattr(st.session_state, 'payment_top_n', 20)
            top_justifications = justification_summary.head(top_n)
            
            # Visualization
            fig_just = px.bar(
                top_justifications,
                x='Total_Amount',
                y=col,
                orientation='h',
                title=f'Top {top_n} {col.replace("_", " ").title()} by Total Amount',
                hover_data=['Payment_Count', 'Avg_Amount'],
                labels={col: 'Justification', 'Total_Amount': 'Total Amount ($)'},
                color='Total_Amount',
                color_continuous_scale='Viridis'
            )
            fig_just.update_layout(height=600)
            st.plotly_chart(fig_just, use_container_width=True)
            
            # Show detailed table
            with st.expander(f"📋 Detailed {col.replace('_', ' ').title()} Data"):
                display_df = top_justifications.copy()
                display_df['Total_Amount'] = display_df['Total_Amount'].apply(lambda x: f"${x:,.0f}")
                display_df['Avg_Amount'] = display_df['Avg_Amount'].apply(lambda x: f"${x:,.0f}")
                st.dataframe(display_df, use_container_width=True)

def render_enhanced_timeline_analysis(df):
    """Enhanced timeline analysis"""
    
    st.markdown("#### 📅 Payment Timeline Analysis")
    
    date_cols = [col for col in df.columns if 'date' in col.lower()]
    if not date_cols:
        st.warning("No date columns available for timeline analysis.")
        return
    
    date_col = st.selectbox("Select Date Column", date_cols, key="timeline_date_col")
    amount_col = next((col for col in df.columns if any(term in col.lower() for term in ['amount', 'payment_amt', 'value']) and pd.api.types.is_numeric_dtype(df[col])), None)
    
    df_time = df.copy()
    df_time[date_col] = pd.to_datetime(df_time[date_col], errors='coerce')
    df_time = df_time.dropna(subset=[date_col]).sort_values(date_col)
    
    if len(df_time) == 0:
        st.warning("No valid dates found for timeline analysis.")
        return
    
    # Determine grouping
    date_range = (df_time[date_col].max() - df_time[date_col].min()).days
    if date_range <= 90:
        period = 'D'
        title_period = "Daily"
    elif date_range <= 730:
        period = 'W'
        title_period = "Weekly"
    else:
        period = 'M'
        title_period = "Monthly"
    
    # Timeline aggregation
    if amount_col:
        time_summary = df_time.groupby(df_time[date_col].dt.to_period(period)).agg({
            amount_col: ['sum', 'count', 'mean']
        }).round(0)
        time_summary.columns = ['Total_Amount', 'Payment_Count', 'Avg_Amount']
        time_summary = time_summary.reset_index()
        time_summary[date_col] = time_summary[date_col].astype(str)
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig_amount = px.line(
                time_summary, 
                x=date_col, 
                y='Total_Amount',
                title=f'{title_period} Payment Amounts',
                markers=True,
                labels={date_col: 'Date', 'Total_Amount': 'Total Amount ($)'}
            )
            fig_amount.update_layout(height=400)
            st.plotly_chart(fig_amount, use_container_width=True)
        
        with col2:
            fig_count = px.line(
                time_summary, 
                x=date_col, 
                y='Payment_Count',
                title=f'{title_period} Payment Count',
                markers=True,
                labels={date_col: 'Date', 'Payment_Count': 'Payment Count'}
            )
            fig_count.update_layout(height=400)
            st.plotly_chart(fig_count, use_container_width=True)
        
        # Summary statistics
        st.markdown("#### 📊 Timeline Statistics")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Peak Payment Day", 
                     time_summary.loc[time_summary['Total_Amount'].idxmax(), date_col],
                     f"${time_summary['Total_Amount'].max():,.0f}")
        
        with col2:
            st.metric("Average per Period", 
                     f"${time_summary['Total_Amount'].mean():,.0f}")
        
        with col3:
            growth_rate = ((time_summary['Total_Amount'].iloc[-1] / time_summary['Total_Amount'].iloc[0]) - 1) * 100 if len(time_summary) > 1 else 0
            st.metric("Growth Rate", f"{growth_rate:+.1f}%")

def render_enhanced_financial_analysis(df):
    """Enhanced financial analysis"""
    
    st.markdown("#### 💰 Financial Analysis")
    
    amount_cols = [col for col in df.columns if any(term in col.lower() for term in ['amount', 'value', 'payment']) and pd.api.types.is_numeric_dtype(df[col])]
    
    if not amount_cols:
        st.warning("No financial columns available for analysis.")
        return
    
    main_amount_col = amount_cols[0]
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("##### 📊 Financial Metrics")
        
        total = df[main_amount_col].sum()
        mean_val = df[main_amount_col].mean()
        median_val = df[main_amount_col].median()
        std_val = df[main_amount_col].std()
        
        st.metric("Total Amount", f"${total:,.0f}")
        st.metric("Average Payment", f"${mean_val:,.0f}")
        st.metric("Median Payment", f"${median_val:,.0f}")
        st.metric("Standard Deviation", f"${std_val:,.0f}")
        
        # Concentration analysis
        top_5_pct = df[main_amount_col].nlargest(int(len(df) * 0.05)).sum()
        concentration = (top_5_pct / total) * 100
        st.metric("Top 5% Concentration", f"{concentration:.1f}%")
    
    with col2:
        # Financial distribution
        fig_dist = px.histogram(
            df,
            x=main_amount_col,
            title="Payment Amount Distribution",
            nbins=50,
            marginal="box"
        )
        fig_dist.update_layout(height=400)
        st.plotly_chart(fig_dist, use_container_width=True)
    
    # Largest payments analysis
    st.markdown("##### 💎 Largest Payments Analysis")
    top_n = getattr(st.session_state, 'payment_top_n', 20)
    
    largest_payments = df.nlargest(top_n, main_amount_col)
    
    # Display columns based on what's available
    display_cols = [main_amount_col]
    if 'agency_name' in df.columns:
        display_cols.insert(0, 'agency_name')
    if any('recipient' in col.lower() for col in df.columns):
        recipient_col = next(col for col in df.columns if 'recipient' in col.lower())
        display_cols.append(recipient_col)
    if any('justification' in col.lower() for col in df.columns):
        justification_col = next(col for col in df.columns if 'justification' in col.lower())
        display_cols.append(justification_col)
    
    st.dataframe(largest_payments[display_cols], use_container_width=True)

def render_enhanced_anomaly_detection(df):
    """Enhanced anomaly detection with multiple methods"""
    
    st.markdown("#### 🔍 Advanced Anomaly Detection")
    
    amount_col = next((col for col in df.columns if any(term in col.lower() for term in ['amount', 'payment_amt', 'value']) and pd.api.types.is_numeric_dtype(df[col])), None)
    
    if not amount_col:
        st.warning("No payment amount column found for anomaly detection.")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("##### 📊 Statistical Outliers (IQR Method)")
        
        Q1 = df[amount_col].quantile(0.25)
        Q3 = df[amount_col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = df[(df[amount_col] < lower_bound) | (df[amount_col] > upper_bound)]
        
        st.metric("Outliers Found", f"{len(outliers):,}")
        st.metric("Outlier Rate", f"{len(outliers)/len(df)*100:.1f}%")
        st.metric("Upper Threshold", f"${upper_bound:,.0f}")
        
        if len(outliers) > 0:
            # Show outlier distribution
            fig_outliers = px.scatter(
                df,
                x=df.index,
                y=amount_col,
                color=df[amount_col].apply(lambda x: 'Outlier' if x < lower_bound or x > upper_bound else 'Normal'),
                title="Payment Outlier Detection",
                labels={'x': 'Payment Index', 'y': 'Payment Amount ($)'},
                color_discrete_map={'Outlier': 'red', 'Normal': 'blue'}
            )
            
            # Add threshold lines
            fig_outliers.add_hline(y=upper_bound, line_dash="dash", line_color="red", 
                                  annotation_text=f"Upper Threshold: ${upper_bound:,.0f}")
            if lower_bound > 0:
                fig_outliers.add_hline(y=lower_bound, line_dash="dash", line_color="red", 
                                      annotation_text=f"Lower Threshold: ${lower_bound:,.0f}")
            
            fig_outliers.update_layout(height=400)
            st.plotly_chart(fig_outliers, use_container_width=True)
    
    with col2:
        st.markdown("##### 🎯 Top Anomalous Payments")
        
        if len(outliers) > 0:
            # Show top outliers
            top_outliers = outliers.nlargest(10, amount_col)
            
            display_cols = [amount_col]
            if 'agency_name' in df.columns:
                display_cols.insert(0, 'agency_name')
            
            st.dataframe(top_outliers[display_cols], use_container_width=True)
            
            # Outlier analysis
            st.markdown("##### 📈 Outlier Analysis")
            
            outlier_agencies = outliers['agency_name'].value_counts().head(5) if 'agency_name' in outliers.columns else pd.Series()
            
            if not outlier_agencies.empty:
                fig_agency_outliers = px.bar(
                    x=outlier_agencies.values,
                    y=outlier_agencies.index,
                    orientation='h',
                    title="Top 5 Agencies with Most Outliers",
                    labels={'x': 'Number of Outliers', 'y': 'Agency'}
                )
                fig_agency_outliers.update_layout(height=300)
                st.plotly_chart(fig_agency_outliers, use_container_width=True)
        else:
            st.success("✅ No statistical outliers detected using IQR method!")
            st.info("This suggests consistent payment patterns across the dataset.")
    
    # Machine Learning Anomaly Detection (if scikit-learn available)
    try:
        from sklearn.ensemble import IsolationForest
        from sklearn.preprocessing import StandardScaler
        
        st.markdown("##### 🤖 Machine Learning Anomaly Detection")
        
        # Prepare features for ML
        features = [amount_col]
        if 'agency_name' in df.columns:
            # Encode agencies
            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()
            df_ml = df.copy()
            df_ml['agency_encoded'] = le.fit_transform(df_ml['agency_name'].astype(str))
            features.append('agency_encoded')
        else:
            df_ml = df.copy()
        
        # Scale features
        scaler = StandardScaler()
        X = scaler.fit_transform(df_ml[features])
        
        # Apply Isolation Forest
        iso_forest = IsolationForest(contamination=0.05, random_state=42)
        anomaly_labels = iso_forest.fit_predict(X)
        
        df_ml['anomaly'] = anomaly_labels
        ml_anomalies = df_ml[df_ml['anomaly'] == -1]
        
        col3, col4 = st.columns(2)
        
        with col3:
            st.metric("ML Anomalies Found", f"{len(ml_anomalies):,}")
            st.metric("ML Anomaly Rate", f"{len(ml_anomalies)/len(df)*100:.1f}%")
            
            if len(ml_anomalies) > 0:
                avg_anomaly_amount = ml_anomalies[amount_col].mean()
                avg_normal_amount = df_ml[df_ml['anomaly'] == 1][amount_col].mean()
                st.metric("Avg Anomaly Amount", f"${avg_anomaly_amount:,.0f}", 
                         delta=f"vs Normal: ${avg_normal_amount:,.0f}")
        
        with col4:
            if len(ml_anomalies) > 0:
                # Show ML anomaly distribution
                fig_ml = px.scatter(
                    df_ml,
                    x=df_ml.index,
                    y=amount_col,
                    color=df_ml['anomaly'].map({1: 'Normal', -1: 'ML Anomaly'}),
                    title="Machine Learning Anomaly Detection",
                    labels={'x': 'Payment Index', 'y': 'Payment Amount ($)'},
                    color_discrete_map={'ML Anomaly': 'orange', 'Normal': 'lightblue'}
                )
                fig_ml.update_layout(height=300)
                st.plotly_chart(fig_ml, use_container_width=True)
    
    except ImportError:
        st.info("💡 Install scikit-learn for advanced ML-based anomaly detection: `pip install scikit-learn`")

def render_enhanced_data_table(df):
    """Render enhanced data table with better formatting"""
    
    # Format monetary columns
    amount_cols = [col for col in df.columns if any(term in col.lower() for term in ['amount', 'value', 'payment']) and pd.api.types.is_numeric_dtype(df[col])]
    
    display_df = df.copy()
    
    # Format amount columns
    for col in amount_cols:
        if col in display_df.columns:
            display_df[col] = display_df[col].apply(lambda x: f"${x:,.0f}" if pd.notnull(x) else "N/A")
    
    # Truncate long text fields
    text_cols = display_df.select_dtypes(include=['object']).columns
    for col in text_cols:
        if 'justification' in col.lower() or 'description' in col.lower():
            display_df[col] = display_df[col].astype(str).apply(
                lambda x: x[:100] + "..." if len(str(x)) > 100 else x
            )
    
    # Show search functionality
    search_term = st.text_input("🔍 Search payments:", key="payment_search")
    
    if search_term:
        # Search across all text columns
        mask = display_df.astype(str).apply(
            lambda x: x.str.contains(search_term, case=False, na=False)
        ).any(axis=1)
        display_df = display_df[mask]
        st.info(f"Found {len(display_df)} payments matching '{search_term}'")
    
    # Show top N records
    top_n = getattr(st.session_state, 'payment_top_n', 100)
    display_df = display_df.head(top_n)
    
    st.dataframe(display_df, use_container_width=True, height=400)
    
    # Export options
    if st.button("📥 Export Filtered Data"):
        st.success(f"✅ {len(display_df)} records ready for export!")
        # Note: Actual export functionality would be implemented here
