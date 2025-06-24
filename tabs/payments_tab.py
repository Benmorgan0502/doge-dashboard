import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from utils.chart_utils import format_billions, format_millions, create_download_button

def render_payments_tab(df):
    """Simple payments tab with four key visualizations and filters"""
    
    st.markdown("# 💳 Government Payments Analysis")
    st.markdown("*Analysis of federal payment transactions for efficiency insights*")
    
    if df.empty:
        st.warning("⚠️ No payment data available for analysis.")
        st.info("The payments dataset appears to be empty. Please check the data loading process.")
        return
    
    # Display basic info about the dataset
    st.info(f"📊 Dataset contains {len(df):,} payment records with {len(df.columns)} data fields")
    
    # Create filters in sidebar
    with st.sidebar:
        st.markdown("### 🎛️ Payment Filters")
        
        # Agency filter (if available)
        if 'agency_name' in df.columns:
            agencies = ['All'] + sorted(df['agency_name'].dropna().unique().tolist())
            selected_agency = st.selectbox("Filter by Agency", agencies)
            
            if selected_agency != 'All':
                df = df[df['agency_name'] == selected_agency]
        
        # Payment amount filter (if available)
        amount_col = 'payment_amt' if 'payment_amt' in df.columns else None
        if amount_col and pd.api.types.is_numeric_dtype(df[amount_col]):
            min_val = float(df[amount_col].min())
            max_val = float(df[amount_col].max())
            
            amount_range = st.slider(
                "Payment Amount Range ($)",
                min_value=min_val,
                max_value=max_val,
                value=(min_val, max_val),
                format="$%.0f"
            )
            
            df = df[(df[amount_col] >= amount_range[0]) & (df[amount_col] <= amount_range[1])]
        
        # Date filter (if available)
        if 'payment_date' in df.columns:
            df['payment_date'] = pd.to_datetime(df['payment_date'], errors='coerce')
            valid_dates = df['payment_date'].dropna()
            
            if len(valid_dates) > 0:
                min_date = valid_dates.min().date()
                max_date = valid_dates.max().date()
                
                date_range = st.date_input(
                    "Date Range",
                    value=(min_date, max_date),
                    min_value=min_date,
                    max_value=max_date
                )
                
                if len(date_range) == 2:
                    start_date, end_date = date_range
                    df = df[
                        (df['payment_date'].dt.date >= start_date) & 
                        (df['payment_date'].dt.date <= end_date)
                    ]
        
        # Show results count
        st.markdown(f"**Filtered Results:** {len(df):,} payments")
        
        # Download button
        create_download_button(df, "Filtered_Payments", "payments_filtered")
    
    # Check if we have data after filtering
    if df.empty:
        st.warning("No data matches the selected filters. Please adjust your filter criteria.")
        return
    
    # Summary metrics at the top
    render_summary_metrics(df)
    
    # Four main charts
    st.markdown("---")
    st.markdown("### 📊 Payment Analysis Dashboard")
    
    # Chart 1 & 2 in first row
    col1, col2 = st.columns(2)
    
    with col1:
        render_payment_amounts_chart(df)
    
    with col2:
        render_agency_payments_chart(df)
    
    # Chart 3 & 4 in second row
    col3, col4 = st.columns(2)
    
    with col3:
        render_timeline_chart(df)
    
    with col4:
        render_payment_distribution_chart(df)
    
    # Optional: Raw data table
    with st.expander("📋 View Payment Records", expanded=False):
        st.dataframe(df.head(100), use_container_width=True)

def render_summary_metrics(df):
    """Display key summary metrics"""
    
    # Calculate metrics
    total_payments = len(df)
    
    # Find amount column
    amount_col = 'payment_amt' if 'payment_amt' in df.columns else None
    if amount_col and pd.api.types.is_numeric_dtype(df[amount_col]):
        total_amount = df[amount_col].sum()
        avg_amount = df[amount_col].mean()
        max_amount = df[amount_col].max()
    else:
        total_amount = avg_amount = max_amount = 0
    
    # Count unique agencies
    unique_agencies = df['agency_name'].nunique() if 'agency_name' in df.columns else 0
    
    # Display metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Payments", f"{total_payments:,}")
    
    with col2:
        if total_amount > 1e9:
            amount_display = format_billions(total_amount)
        elif total_amount > 1e6:
            amount_display = format_millions(total_amount)
        else:
            amount_display = f"${total_amount:,.0f}"
        st.metric("Total Amount", amount_display)
    
    with col3:
        if avg_amount > 1e6:
            avg_display = format_millions(avg_amount)
        else:
            avg_display = f"${avg_amount:,.0f}"
        st.metric("Average Payment", avg_display)
    
    with col4:
        st.metric("Unique Agencies", f"{unique_agencies}")

def render_payment_amounts_chart(df):
    """Chart 1: Top Recipients by Payment Value"""
    
    st.markdown("#### 🎯 Top Recipients by Total Payment Value")
    
    amount_col = 'payment_amt' if 'payment_amt' in df.columns else None
    recipient_col = 'org_name' if 'org_name' in df.columns else None
    
    if amount_col and recipient_col and pd.api.types.is_numeric_dtype(df[amount_col]):
        # Group by recipient and sum payment amounts
        recipient_totals = df.groupby(recipient_col)[amount_col].agg(['sum', 'count']).reset_index()
        recipient_totals.columns = ['Recipient', 'Total_Amount', 'Payment_Count']
        
        # Sort and get top 10
        top_recipients = recipient_totals.sort_values('Total_Amount', ascending=False).head(10)
        
        # Create horizontal bar chart
        fig = px.bar(
            top_recipients,
            x='Total_Amount',
            y='Recipient',
            orientation='h',
            title="Top 10 Recipients by Total Payment Value",
            labels={'Total_Amount': 'Total Payment Amount ($)', 'Recipient': 'Organization'},
            color='Total_Amount',
            color_continuous_scale='Blues',
            hover_data={'Payment_Count': True}
        )
        
        # Format and improve layout
        fig.update_layout(
            height=400,
            yaxis={'categoryorder': 'total ascending'},
            xaxis_tickformat='$,.0f'
        )
        
        # Truncate long organization names for better display
        fig.update_yaxis(tickmode='array', 
                        tickvals=list(range(len(top_recipients))),
                        ticktext=[name[:40] + "..." if len(name) > 40 else name 
                                for name in top_recipients['Recipient']])
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Show summary stats
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Top Recipient", top_recipients.iloc[0]['Recipient'][:30] + "..." if len(top_recipients.iloc[0]['Recipient']) > 30 else top_recipients.iloc[0]['Recipient'])
        with col2:
            st.metric("Highest Total", f"${top_recipients.iloc[0]['Total_Amount']:,.0f}")
        with col3:
            st.metric("Total Recipients", f"{len(recipient_totals):,}")
        
        st.markdown("""
        **Analysis:** This chart identifies the organizations receiving the highest total payment values from the government. 
        Understanding payment concentration helps identify key contractors, vendors, and service providers. 
        High-value recipients may benefit from streamlined payment processes, while the distribution pattern 
        reveals dependencies on major suppliers and potential areas for competitive bidding improvements.
        """)
    
    elif not recipient_col:
        st.info("Recipient organization data not available for this analysis.")
    elif not amount_col:
        st.info("Payment amount data not available for this analysis.")
    else:
        st.info("Required data columns not found for recipient analysis.")

def render_agency_payments_chart(df):
    """Chart 2: Top Agencies by Payment Volume"""
    
    st.markdown("#### 🏢 Top Agencies by Payment Count")
    
    if 'agency_name' in df.columns:
        # Count payments by agency
        agency_counts = df['agency_name'].value_counts().head(10).reset_index()
        agency_counts.columns = ['Agency', 'Payment_Count']
        
        # Create bar chart
        fig = px.bar(
            agency_counts,
            x='Payment_Count',
            y='Agency',
            orientation='h',
            title="Top 10 Agencies by Number of Payments",
            labels={'Payment_Count': 'Number of Payments', 'Agency': 'Agency'},
            color='Payment_Count',
            color_continuous_scale='Blues'
        )
        
        fig.update_layout(height=400, yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("""
        **Analysis:** This chart identifies which federal agencies are making the most payment transactions. 
        High transaction volume agencies may benefit from automated payment processing improvements, while 
        agencies with fewer transactions might need specialized attention for efficiency optimization.
        """)
    else:
        st.info("Agency data not available for visualization.")

def render_timeline_chart(df):
    """Chart 3: Payment Timeline Analysis"""
    
    st.markdown("#### 📅 Payments Over Time")
    
    if 'payment_date' in df.columns:
        # Ensure payment_date is datetime
        df['payment_date'] = pd.to_datetime(df['payment_date'], errors='coerce')
        df_time = df.dropna(subset=['payment_date'])
        
        if len(df_time) > 0:
            # Group by date and count payments
            daily_counts = df_time.groupby(df_time['payment_date'].dt.date).size().reset_index()
            daily_counts.columns = ['Date', 'Payment_Count']
            
            # Create line chart
            fig = px.line(
                daily_counts,
                x='Date',
                y='Payment_Count',
                title="Daily Payment Volume",
                labels={'Date': 'Date', 'Payment_Count': 'Number of Payments'},
                markers=True
            )
            
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("""
            **Analysis:** This timeline shows payment activity patterns over time. Spikes in payment volume 
            may correspond to budget cycles, fiscal year-end spending, or emergency appropriations. 
            Identifying these patterns helps predict resource needs and optimize payment processing workflows.
            """)
        else:
            st.info("No valid payment dates found for timeline analysis.")
    else:
        st.info("Payment date data not available for timeline visualization.")

def render_payment_distribution_chart(df):
    """Chart 4: Payment Size Categories"""
    
    st.markdown("#### 📊 Payment Size Categories")
    
    amount_col = 'payment_amt' if 'payment_amt' in df.columns else None
    
    if amount_col and pd.api.types.is_numeric_dtype(df[amount_col]):
        # Create payment size categories
        df_temp = df.copy()
        df_temp['Payment_Category'] = pd.cut(
            df_temp[amount_col],
            bins=[0, 1000, 10000, 100000, 1000000, float('inf')],
            labels=['Small\n(<$1K)', 'Medium\n($1K-$10K)', 'Large\n($10K-$100K)', 
                   'Very Large\n($100K-$1M)', 'Massive\n(>$1M)'],
            include_lowest=True
        )
        
        # Count by category
        category_counts = df_temp['Payment_Category'].value_counts().reset_index()
        category_counts.columns = ['Category', 'Count']
        category_counts['Percentage'] = (category_counts['Count'] / len(df_temp) * 100).round(1)
        
        # Create pie chart
        fig = px.pie(
            category_counts,
            values='Count',
            names='Category',
            title="Payment Distribution by Size Category",
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        
        # Add percentage labels
        fig.update_traces(textposition='inside', textinfo='percent+label')
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("""
        **Analysis:** This pie chart categorizes payments by size to understand spending patterns. 
        Small, routine payments may benefit from automated processing, while large payments typically 
        require additional oversight and approval workflows. This distribution helps identify 
        opportunities for process optimization based on payment size.
        """)
    else:
        st.info("Payment amount data not available for categorization.")
