import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from utils.chart_utils import format_billions, format_millions, create_download_button
from models.outlier_detection import perform_outlier_detection

def render_payments_tab(df):
    """Render the payments tab with comprehensive analysis"""
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        # Filters
        if not df.empty:
            filtered_df = create_payment_filters(df)
            create_download_button(filtered_df, "Payments", "payments")
        else:
            filtered_df = df
    
    with col2:
        if filtered_df.empty:
            st.warning("No data available for Payments.")
            return
        
        # Summary metrics
        render_payment_summary(filtered_df)
        
        # Analysis tabs
        render_payment_analysis_tabs(filtered_df)
        
        # Raw data table
        st.divider()
        with st.expander("View Raw Data Table"):
            st.dataframe(filtered_df.head(1000))

def create_payment_filters(df):
    """Create dynamic filters based on available columns in payments data"""
    
    # Agency filter (if available)
    if "agency" in df.columns:
        agencies = sorted(df["agency"].dropna().unique().tolist())
        selected_agency = st.selectbox("Filter by Agency", ["All"] + agencies, key="payments_agency")
    else:
        selected_agency = "All"
    
    # Payment type filter (look for type-related columns)
    type_cols = [col for col in df.columns if any(term in col.lower() for term in ['type', 'category', 'method', 'kind'])]
    if type_cols:
        type_col = type_cols[0]
        types = sorted(df[type_col].dropna().unique().tolist())
        selected_type = st.selectbox(f"Filter by {type_col.title()}", ["All"] + types, key="payments_type")
    else:
        selected_type = "All"
        type_col = None
    
    # Value/Amount filter (look for monetary columns)
    amount_cols = [col for col in df.columns if any(term in col.lower() for term in ['amount', 'value', 'cost', 'payment', 'total'])]
    if amount_cols:
        amount_col = amount_cols[0]
        # Check if the column is actually numeric
        if pd.api.types.is_numeric_dtype(df[amount_col]):
            amount_min = st.number_input(f"Minimum {amount_col.title()}", 
                                       value=float(df[amount_col].min()), 
                                       step=1000.0, 
                                       key="payments_amount_min")
            amount_max = st.number_input(f"Maximum {amount_col.title()}", 
                                       value=float(df[amount_col].max()), 
                                       step=1000.0, 
                                       key="payments_amount_max")
        else:
            st.info(f"Column {amount_col} appears to contain non-numeric data. Skipping amount filter.")
            amount_col = None
            amount_min = 0
            amount_max = float('inf')
    else:
        amount_col = None
        amount_min = 0
        amount_max = float('inf')
    
    # Date range filter (if date columns exist)
    date_cols = [col for col in df.columns if any(term in col.lower() for term in ['date', 'time', 'created', 'processed'])]
    if date_cols:
        st.markdown("**Date Range Filter:**")
        date_col = date_cols[0]
        df_temp = df.copy()
        df_temp[date_col] = pd.to_datetime(df_temp[date_col], errors='coerce')
        valid_dates = df_temp[date_col].dropna()
        if len(valid_dates) > 0:
            date_range = st.date_input(
                f"Select Date Range ({date_col})",
                value=(valid_dates.min().date(), valid_dates.max().date()),
                min_value=valid_dates.min().date(),
                max_value=valid_dates.max().date(),
                key="payments_date_range"
            )
        else:
            date_range = None
    else:
        date_col = None
        date_range = None
    
    # Apply filters
    filtered_df = df.copy()
    
    if "agency" in df.columns and selected_agency != "All":
        filtered_df = filtered_df[filtered_df["agency"] == selected_agency]
    
    if type_col and selected_type != "All":
        filtered_df = filtered_df[filtered_df[type_col] == selected_type]
    
    if amount_col and pd.api.types.is_numeric_dtype(df[amount_col]):
        filtered_df = filtered_df[
            (pd.to_numeric(filtered_df[amount_col], errors='coerce') >= amount_min) & 
            (pd.to_numeric(filtered_df[amount_col], errors='coerce') <= amount_max)
        ]
    
    if date_col and date_range and len(date_range) == 2:
        filtered_df[date_col] = pd.to_datetime(filtered_df[date_col], errors='coerce')
        start_date, end_date = date_range
        filtered_df = filtered_df[
            (filtered_df[date_col].dt.date >= start_date) & 
            (filtered_df[date_col].dt.date <= end_date)
        ]
    
    return filtered_df

def render_payment_summary(df):
    """Render payment summary metrics"""
    st.markdown("### 💳 Payment Summary")
    
    # Identify key columns
    amount_col = None
    for col in df.columns:
        if any(term in col.lower() for term in ['amount', 'value', 'cost', 'payment', 'total']):
            if pd.api.types.is_numeric_dtype(df[col]):
                amount_col = col
                break
    
    savings_col = next((col for col in df.columns if 'saving' in col.lower() and pd.api.types.is_numeric_dtype(df[col])), None)
    
    # Calculate metrics
    total_payments = len(df)
    
    if amount_col:
        total_amount = df[amount_col].sum()
        avg_amount = df[amount_col].mean()
    else:
        total_amount = 0
        avg_amount = 0
    
    if savings_col:
        total_savings = df[savings_col].sum()
        savings_rate = (total_savings / total_amount * 100) if total_amount > 0 else 0
    else:
        total_savings = 0
        savings_rate = 0
    
    # Display metrics
    summary_cols = st.columns(5)
    
    with summary_cols[0]:
        st.metric("💳 Total Payments", f"{total_payments:,}")
    
    with summary_cols[1]:
        if amount_col:
            if total_amount > 1_000_000_000:
                st.metric("💰 Total Amount", format_billions(total_amount))
            else:
                st.metric("💰 Total Amount", format_millions(total_amount))
        else:
            st.metric("💰 Total Amount", "N/A")
    
    with summary_cols[2]:
        if amount_col:
            if avg_amount > 1_000_000:
                st.metric("📊 Avg Amount", format_millions(avg_amount))
            else:
                st.metric("📊 Avg Amount", f"${avg_amount:,.0f}")
        else:
            st.metric("📊 Avg Amount", "N/A")
    
    with summary_cols[3]:
        if savings_col:
            if total_savings > 1_000_000_000:
                st.metric("💸 Total Savings", format_billions(total_savings))
            else:
                st.metric("💸 Total Savings", format_millions(total_savings))
        else:
            st.metric("💸 Total Savings", "N/A")
    
    with summary_cols[4]:
        if savings_col and total_amount > 0:
            st.metric("⚡ Savings Rate", f"{savings_rate:.1f}%")
        else:
            st.metric("⚡ Processing Rate", "100%")
    
    st.markdown("---")

def render_payment_analysis_tabs(df):
    """Render payment analysis tabs"""
    
    # Determine available analysis options based on data
    available_tabs = ["Data Overview"]
    
    if "agency" in df.columns:
        available_tabs.append("By Agency")
    
    # Payment type analysis
    type_cols = [col for col in df.columns if any(term in col.lower() for term in ['type', 'category', 'method', 'kind'])]
    if type_cols:
        available_tabs.append("By Payment Type")
    
    # Recipient/Vendor analysis
    recipient_cols = [col for col in df.columns if any(term in col.lower() for term in ['recipient', 'vendor', 'payee', 'beneficiary'])]
    if recipient_cols:
        available_tabs.append("By Recipient")
    
    # Time-based analysis
    date_cols = [col for col in df.columns if any(term in col.lower() for term in ['date', 'time', 'created', 'processed'])]
    if date_cols:
        available_tabs.append("Timeline Analysis")
    
    # Financial analysis
    amount_cols = [col for col in df.columns if any(term in col.lower() for term in ['amount', 'value', 'cost', 'payment', 'saving'])]
    if len(amount_cols) >= 1:
        available_tabs.append("Financial Analysis")
    
    # Fraud/Anomaly detection
    if len(df.select_dtypes(include=['number']).columns) >= 2:
        available_tabs.append("Anomaly Detection")
    
    # Create tabs
    selected_tab = st.radio("📊 Analysis Views", available_tabs, horizontal=True, key="payments_analysis_tabs", label_visibility="collapsed")
    
    if selected_tab == "Data Overview":
        render_data_overview(df)
    elif selected_tab == "By Agency":
        render_agency_analysis(df)
    elif selected_tab == "By Payment Type":
        render_payment_type_analysis(df, type_cols[0])
    elif selected_tab == "By Recipient":
        render_recipient_analysis(df, recipient_cols[0])
    elif selected_tab == "Timeline Analysis":
        render_timeline_analysis(df, date_cols)
    elif selected_tab == "Financial Analysis":
        render_financial_analysis(df, amount_cols)
    elif selected_tab == "Anomaly Detection":
        render_anomaly_detection(df)

def render_data_overview(df):
    """Render basic data overview"""
    st.markdown("### 📊 Payment Data Overview")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Dataset Information:**")
        st.write(f"• **Total Records:** {len(df):,}")
        st.write(f"• **Total Columns:** {len(df.columns)}")
        st.write(f"• **Numerical Columns:** {len(df.select_dtypes(include=['number']).columns)}")
        st.write(f"• **Text Columns:** {len(df.select_dtypes(include=['object']).columns)}")
        
        st.write("**Available Columns:**")
        for i, col in enumerate(df.columns, 1):
            st.write(f"{i}. {col}")
    
    with col2:
        if len(df.select_dtypes(include=['number']).columns) > 0:
            st.markdown("### 📈 Numerical Summary")
            st.dataframe(df.describe())
        
        # Show sample of categorical data
        categorical_cols = df.select_dtypes(include=['object']).columns
        if len(categorical_cols) > 0:
            st.markdown("### 📝 Categorical Data Sample")
            sample_col = categorical_cols[0]
            value_counts = df[sample_col].value_counts().head(10)
            st.bar_chart(value_counts)

def render_agency_analysis(df):
    """Render analysis by agency"""
    st.markdown("### 🏢 Analysis by Agency")
    
    amount_col = None
    for col in df.columns:
        if any(term in col.lower() for term in ['amount', 'value', 'cost', 'payment', 'total']):
            if pd.api.types.is_numeric_dtype(df[col]):
                amount_col = col
                break
    
    if amount_col:
        # Agency summary with multiple metrics
        agency_summary = df.groupby("agency").agg({
            amount_col: ['sum', 'count', 'mean'],
        }).round(2)
        
        agency_summary.columns = ['Total Amount', 'Payment Count', 'Avg Amount']
        agency_summary = agency_summary.reset_index()
        agency_summary = agency_summary.sort_values('Total Amount', ascending=False)
        
        # Top agencies chart
        top_agencies = agency_summary.head(15)
        
        fig = px.bar(top_agencies, 
                     x="agency", 
                     y="Total Amount",
                     title="Top 15 Agencies by Total Payment Amount",
                     hover_data=['Payment Count', 'Avg Amount'])
        fig.update_layout(xaxis_tickangle=45)
        st.plotly_chart(fig, use_container_width=True)
        
        # Summary table
        st.dataframe(agency_summary.head(20))
        
    else:
        # Just count payments by agency
        agency_counts = df['agency'].value_counts().head(15).reset_index()
        agency_counts.columns = ['Agency', 'Payment Count']
        
        fig = px.bar(agency_counts, x="Agency", y="Payment Count", 
                    title="Top 15 Agencies by Payment Count")
        fig.update_layout(xaxis_tickangle=45)
        st.plotly_chart(fig, use_container_width=True)
        
        st.dataframe(agency_counts)

def render_payment_type_analysis(df, type_col):
    """Render analysis by payment type"""
    st.markdown(f"### 💸 Analysis by {type_col.title()}")
    
    # Find numeric amount column
    amount_col = None
    for col in df.columns:
        if any(term in col.lower() for term in ['amount', 'value', 'cost', 'payment', 'total']):
            if pd.api.types.is_numeric_dtype(df[col]):
                amount_col = col
                break
    
    # Type distribution
    type_counts = df[type_col].value_counts().reset_index()
    type_counts.columns = ['Payment Type', 'Count']
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Pie chart of payment type distribution
        fig_pie = px.pie(type_counts, values="Count", names="Payment Type", 
                         title=f"Distribution by {type_col.title()}")
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col2:
        if amount_col:
            # Amount by payment type
            type_summary = df.groupby(type_col).agg({
                amount_col: ['sum', 'count', 'mean']
            }).round(2)
            type_summary.columns = ['Total Amount', 'Payment Count', 'Avg Amount']
            type_summary = type_summary.reset_index()
            type_summary = type_summary.sort_values('Total Amount', ascending=False)
            
            fig_bar = px.bar(type_summary, x=type_col, y="Total Amount", 
                            title=f"Total Amount by {type_col.title()}",
                            hover_data=['Payment Count', 'Avg Amount'])
            fig_bar.update_layout(xaxis_tickangle=45)
            st.plotly_chart(fig_bar, use_container_width=True)
        else:
            # Just show counts if no numeric amount column
            fig_bar = px.bar(type_counts, x="Payment Type", y="Count", 
                            title=f"Payment Count by {type_col.title()}")
            fig_bar.update_layout(xaxis_tickangle=45)
            st.plotly_chart(fig_bar, use_container_width=True)
    
    # Summary table
    if amount_col:
        st.dataframe(type_summary)
    else:
        st.dataframe(type_counts)

def render_recipient_analysis(df, recipient_col):
    """Render analysis by recipient"""
    st.markdown(f"### 🎯 Analysis by {recipient_col.title()}")
    
    # Find numeric amount column
    amount_col = None
    for col in df.columns:
        if any(term in col.lower() for term in ['amount', 'value', 'cost', 'payment', 'total']):
            if pd.api.types.is_numeric_dtype(df[col]):
                amount_col = col
                break
    
    if amount_col:
        recipient_summary = df.groupby(recipient_col).agg({
            amount_col: ['sum', 'count', 'mean']
        }).round(2)
        recipient_summary.columns = ['Total Amount', 'Payment Count', 'Avg Amount']
        recipient_summary = recipient_summary.reset_index()
        recipient_summary = recipient_summary.sort_values('Total Amount', ascending=False)
        
        # Top recipients chart
        top_recipients = recipient_summary.head(20)
        
        fig = px.bar(top_recipients, 
                     x=recipient_col, 
                     y="Total Amount",
                     title=f"Top 20 {recipient_col.title()}s by Total Amount",
                     hover_data=['Payment Count', 'Avg Amount'])
        fig.update_layout(xaxis_tickangle=45)
        st.plotly_chart(fig, use_container_width=True)
        
        st.dataframe(recipient_summary.head(25))
    else:
        recipient_counts = df[recipient_col].value_counts().head(20).reset_index()
        recipient_counts.columns = ['Recipient', 'Payment Count']
        
        fig = px.bar(recipient_counts, x="Recipient", y="Payment Count", 
                    title=f"Top 20 {recipient_col.title()}s by Payment Count")
        fig.update_layout(xaxis_tickangle=45)
        st.plotly_chart(fig, use_container_width=True)
        
        st.dataframe(recipient_counts)

def render_timeline_analysis(df, date_cols):
    """Render timeline analysis"""
    st.markdown("### 📅 Timeline Analysis")
    
    date_col = st.selectbox("Select Date Column", date_cols, key="payment_date_col")
    
    # Find a numeric amount column for timeline analysis
    amount_col = None
    for col in df.columns:
        if any(term in col.lower() for term in ['amount', 'value', 'cost', 'payment', 'total']):
            if pd.api.types.is_numeric_dtype(df[col]):
                amount_col = col
                break
    
    df_time = df.copy()
    df_time[date_col] = pd.to_datetime(df_time[date_col], errors='coerce')
    df_time = df_time.dropna(subset=[date_col]).sort_values(date_col)
    
    if len(df_time) == 0:
        st.warning(f"No valid dates found in {date_col} column.")
        return
    
    # Determine appropriate time grouping
    date_range = (df_time[date_col].max() - df_time[date_col].min()).days
    if date_range <= 30:
        period = 'D'  # Daily
        title_period = "Daily"
    elif date_range <= 365:
        period = 'W'  # Weekly
        title_period = "Weekly"
    else:
        period = 'M'  # Monthly
        title_period = "Monthly"
    
    # Time series analysis
    if amount_col:
        # Create aggregation dictionary for numeric column only
        agg_dict = {amount_col: ['sum', 'count', 'mean']}
        
        time_summary = df_time.groupby(df_time[date_col].dt.to_period(period)).agg(agg_dict)
        time_summary.columns = ['Total Amount', 'Payment Count', 'Avg Amount']
        time_summary = time_summary.reset_index()
        time_summary[date_col] = time_summary[date_col].astype(str)
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig_amount = px.line(time_summary, x=date_col, y="Total Amount", 
                                title=f"{title_period} Payment Amounts",
                                markers=True)
            st.plotly_chart(fig_amount, use_container_width=True)
        
        with col2:
            fig_count = px.line(time_summary, x=date_col, y="Payment Count", 
                               title=f"{title_period} Payment Count",
                               markers=True)
            st.plotly_chart(fig_count, use_container_width=True)
        
        # Cumulative analysis
        time_summary['Cumulative Amount'] = time_summary['Total Amount'].cumsum()
        time_summary['Cumulative Count'] = time_summary['Payment Count'].cumsum()
        
        fig_cum = px.line(time_summary, x=date_col, y="Cumulative Amount", 
                         title="Cumulative Payment Amount Over Time",
                         markers=True)
        st.plotly_chart(fig_cum, use_container_width=True)
        
    else:
        # Just count payments over time
        time_counts = df_time.groupby(df_time[date_col].dt.to_period(period)).size().reset_index()
        time_counts.columns = [date_col, 'Payment Count']
        time_counts[date_col] = time_counts[date_col].astype(str)
        
        fig = px.line(time_counts, x=date_col, y="Payment Count", 
                     title=f"{title_period} Payment Count",
                     markers=True)
        st.plotly_chart(fig, use_container_width=True)
        
        # Show summary info
        st.info(f"Timeline analysis shows payment counts only. No numeric amount column found for value analysis.")

def render_financial_analysis(df, amount_cols):
    """Render financial analysis"""
    st.markdown("### 💰 Financial Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Financial Metrics")
        for col in amount_cols:
            if df[col].dtype in ['int64', 'float64']:
                total = df[col].sum()
                avg = df[col].mean()
                median = df[col].median()
                std = df[col].std()
                
                st.write(f"**{col.title()}:**")
                st.write(f"• Total: ${total:,.2f}")
                st.write(f"• Average: ${avg:,.2f}")
                st.write(f"• Median: ${median:,.2f}")
                st.write(f"• Std Dev: ${std:,.2f}")
                st.write("---")
    
    with col2:
        # Amount distribution
        main_amount_col = amount_cols[0]
        fig_hist = px.histogram(df, x=main_amount_col, 
                               title=f"{main_amount_col.title()} Distribution",
                               nbins=50)
        st.plotly_chart(fig_hist, use_container_width=True)
    
    # Top payments
    st.markdown("#### 💎 Largest Payments")
    main_amount_col = amount_cols[0]
    top_payments = df.nlargest(20, main_amount_col)
    
    # Show relevant columns
    display_cols = [main_amount_col]
    if "agency" in df.columns:
        display_cols.insert(0, "agency")
    if any(col in df.columns for col in ['recipient', 'vendor', 'payee']):
        recipient_col = next((col for col in ['recipient', 'vendor', 'payee'] if col in df.columns), None)
        display_cols.append(recipient_col)
    
    st.dataframe(top_payments[display_cols])

def render_anomaly_detection(df):
    """Render anomaly detection analysis"""
    st.markdown("### 🔍 Anomaly Detection")
    
    amount_col = next((col for col in df.columns if any(term in col.lower() for term in ['amount', 'value', 'cost', 'payment', 'total'])), None)
    
    if amount_col:
        col1, col2 = st.columns(2)
        
        with col1:
            # Statistical outliers (using IQR method)
            Q1 = df[amount_col].quantile(0.25)
            Q3 = df[amount_col].quantile(0.75)
            IQR = Q3 - Q1
            outlier_threshold = Q3 + 1.5 * IQR
            
            outliers = df[df[amount_col] > outlier_threshold]
            
            st.markdown("#### 📊 Statistical Outliers")
            st.write(f"Found {len(outliers)} payments above ${outlier_threshold:,.0f}")
            
            if len(outliers) > 0:
                st.dataframe(outliers.nlargest(10, amount_col))
        
        with col2:
            # Box plot to visualize outliers
            fig_box = px.box(df, y=amount_col, 
                            title=f"{amount_col.title()} Distribution (Box Plot)")
            st.plotly_chart(fig_box, use_container_width=True)
        
        # Use the existing outlier detection model
        st.markdown("#### 🤖 Machine Learning Anomaly Detection")
        perform_outlier_detection(df, "Payments")
    
    else:
        st.info("Anomaly detection requires numerical amount data to identify unusual payment patterns.")