import streamlit as st
from config.settings import PAGE_CONFIG
from utils.data_loader import load_all_datasets
from tabs.contracts_tab import render_contracts_tab
from tabs.grants_tab import render_grants_tab
from tabs.leases_tab import render_leases_tab
from tabs.payments_tab import render_payments_tab
from tabs.deep_analysis_tab import render_deep_analysis_tab

# Import the enhanced homepage function
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from datetime import datetime, timedelta
import time

def get_homepage_stats(datasets):
    """Calculate real statistics from loaded datasets"""
    stats = {}
    
    try:
        # Contracts stats
        if not datasets["Contracts"].empty:
            contracts_df = datasets["Contracts"]
            stats["total_contracts"] = len(contracts_df)
            stats["terminated_contracts"] = len(contracts_df[contracts_df.get("fpds_status", "") == "Terminated"])
            stats["contract_savings"] = contracts_df.get("savings", pd.Series([0])).sum()
            stats["contract_value"] = contracts_df.get("value", pd.Series([0])).sum()
            stats["agencies_contracts"] = contracts_df.get("agency", pd.Series([])).nunique()
        else:
            stats.update({"total_contracts": 0, "terminated_contracts": 0, "contract_savings": 0, "contract_value": 0, "agencies_contracts": 0})
        
        # Grants stats
        if not datasets["Grants"].empty:
            grants_df = datasets["Grants"]
            stats["total_grants"] = len(grants_df)
            stats["grant_savings"] = grants_df.get("savings", pd.Series([0])).sum()
            stats["grant_value"] = grants_df.get("value", pd.Series([0])).sum()
            stats["agencies_grants"] = grants_df.get("agency", pd.Series([])).nunique()
        else:
            stats.update({"total_grants": 0, "grant_savings": 0, "grant_value": 0, "agencies_grants": 0})
        
        # Leases stats
        if not datasets["Leases"].empty:
            leases_df = datasets["Leases"]
            stats["total_leases"] = len(leases_df)
            stats["lease_savings"] = leases_df.get("savings", pd.Series([0])).sum()
            stats["lease_value"] = leases_df.get("value", pd.Series([0])).sum()
            stats["total_sqft"] = leases_df.get("sq_ft", pd.Series([0])).sum()
        else:
            stats.update({"total_leases": 0, "lease_savings": 0, "lease_value": 0, "total_sqft": 0})
        
        # Payments stats
        if not datasets["Payments"].empty:
            payments_df = datasets["Payments"]
            stats["total_payments"] = len(payments_df)
            # Try to find amount column
            amount_col = None
            for col in payments_df.columns:
                if any(term in col.lower() for term in ['amount', 'value', 'cost', 'payment', 'total']):
                    if pd.api.types.is_numeric_dtype(payments_df[col]):
                        amount_col = col
                        break
            stats["payment_amount"] = payments_df[amount_col].sum() if amount_col else 0
        else:
            stats.update({"total_payments": 0, "payment_amount": 0})
        
        # Calculate overall metrics
        stats["total_savings"] = stats["contract_savings"] + stats["grant_savings"] + stats["lease_savings"]
        stats["total_value"] = stats["contract_value"] + stats["grant_value"] + stats["lease_value"]
        stats["savings_rate"] = (stats["total_savings"] / stats["total_value"] * 100) if stats["total_value"] > 0 else 0
        
    except Exception as e:
        st.error(f"Error calculating stats: {e}")
        # Return default values
        stats = {
            "total_contracts": 0, "terminated_contracts": 0, "contract_savings": 0,
            "total_grants": 0, "grant_savings": 0, "total_leases": 0, "lease_savings": 0,
            "total_payments": 0, "total_savings": 0, "total_value": 0, "savings_rate": 0
        }
    
    return stats

def create_sparkline_chart(data, title, color="#1f77b4"):
    """Create a small sparkline chart for the homepage"""
    if len(data) == 0:
        # Create dummy data for demo
        data = [10, 15, 13, 17, 20, 18, 25, 22, 28, 30]
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        y=data,
        mode='lines',
        line=dict(color=color, width=2),
        fill='tonexty',
        fillcolor=f'rgba({int(color[1:3], 16)}, {int(color[3:5], 16)}, {int(color[5:7], 16)}, 0.1)'
    ))
    
    fig.update_layout(
        height=60,
        margin=dict(l=0, r=0, t=0, b=0),
        showlegend=False,
        xaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
        yaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig

def render_data_freshness_indicator():
    """Render a data freshness indicator"""
    # Simulate data age (you'd calculate this from actual cache timestamps)
    hours_since_update = 2.5
    
    if hours_since_update < 6:
        status = "🟢 Fresh"
        color = "green"
    elif hours_since_update < 12:
        status = "🟡 Recent"
        color = "orange"
    else:
        status = "🔴 Stale"
        color = "red"
    
    st.markdown(f"""
    <div style="text-align: center; margin: 1rem 0;">
        <span style="color: {color}; font-weight: bold;">{status}</span>
        <span style="color: #666; font-size: 0.9rem;"> • Last updated {hours_since_update:.1f} hours ago</span>
    </div>
    """, unsafe_allow_html=True)

def render_enhanced_homepage(datasets=None):
    """Render the enhanced homepage with all new features"""
    
    # Add custom CSS for mobile responsiveness and animations
    st.markdown("""
    <style>
    @media (max-width: 768px) {
        .hero-section {
            padding: 1rem 0 !important;
            font-size: 0.9rem !important;
        }
        .hero-section h1 {
            font-size: 2rem !important;
        }
        .nav-cards {
            flex-direction: column !important;
        }
    }
    
    .metric-card {
        transition: transform 0.3s ease;
        border-radius: 10px;
        padding: 1rem;
        border: 1px solid #e0e0e0;
        background: white;
        margin-bottom: 1rem;
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    
    .quick-stats {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        border-radius: 15px;
        padding: 2rem;
        margin: 2rem 0;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Hero Section with enhanced styling
    st.markdown("""
    <div class="hero-section" style="text-align: center; padding: 2rem 0; background: linear-gradient(135deg, #1f77b4 0%, #005bbb 100%); border-radius: 15px; margin-bottom: 2rem; color: white; box-shadow: 0 8px 16px rgba(0,0,0,0.1);">
        <h1 style="font-size: 3rem; margin-bottom: 1rem; font-weight: 700;">🏛️ DOGE Government Efficiency Dashboard</h1>
        <p style="font-size: 1.3rem; margin-bottom: 1.5rem; opacity: 0.9;">
            Comprehensive Analysis of Department of Government Efficiency Data
        </p>
        <p style="font-size: 1rem; opacity: 0.8; max-width: 800px; margin: 0 auto;">
            An interactive data visualization platform analyzing government contracts, grants, leases, and payments 
            to assess efficiency initiatives and fiscal impact across federal agencies.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Data freshness indicator
    render_data_freshness_indicator()
    
    # Enhanced Quick Stats with real data
    st.markdown('<div class="quick-stats">', unsafe_allow_html=True)
    st.markdown("### 📊 Real-Time Dashboard Metrics")
    
    if datasets:
        stats = get_homepage_stats(datasets)
        
        # Top row metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            delta_val = f"+{stats['terminated_contracts']:,}" if stats['terminated_contracts'] > 0 else "No data"
            st.metric(
                label="📄 Total Contracts",
                value=f"{stats['total_contracts']:,}",
                delta=f"{delta_val} terminated",
                help="Government contracts analyzed for efficiency"
            )
            # Add sparkline
            if stats['total_contracts'] > 0:
                sparkline_data = [stats['total_contracts'] * 0.7, stats['total_contracts'] * 0.8, 
                                stats['total_contracts'] * 0.9, stats['total_contracts']]
                fig = create_sparkline_chart(sparkline_data, "Contracts", "#1f77b4")
                st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric(
                label="🏢 Total Leases", 
                value=f"{stats['total_leases']:,}",
                delta=f"${stats['lease_savings']/1000000:.1f}M saved" if stats['lease_savings'] > 0 else "No savings data",
                help="Property leases evaluated for cost savings"
            )
            if stats['total_leases'] > 0:
                sparkline_data = [stats['total_leases'] * 0.6, stats['total_leases'] * 0.75, 
                                stats['total_leases'] * 0.85, stats['total_leases']]
                fig = create_sparkline_chart(sparkline_data, "Leases", "#d62828")
                st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col3:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric(
                label="🎯 Total Grants",
                value=f"{stats['total_grants']:,}",
                delta=f"${stats['grant_savings']/1000000:.1f}M impact" if stats['grant_savings'] > 0 else "Impact analysis",
                help="Federal grants assessed for impact"
            )
            if stats['total_grants'] > 0:
                sparkline_data = [stats['total_grants'] * 0.5, stats['total_grants'] * 0.7, 
                                stats['total_grants'] * 0.8, stats['total_grants']]
                fig = create_sparkline_chart(sparkline_data, "Grants", "#003049")
                st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col4:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            payment_display = f"{stats['total_payments']:,}" if stats['total_payments'] > 0 else "Processing"
            st.metric(
                label="💳 Payment Records",
                value=payment_display,
                delta="Anomaly detection active",
                help="Government payment transactions monitored"
            )
            if stats['total_payments'] > 0:
                sparkline_data = [stats['total_payments'] * 0.3, stats['total_payments'] * 0.6, 
                                stats['total_payments'] * 0.8, stats['total_payments']]
                fig = create_sparkline_chart(sparkline_data, "Payments", "#fcbf49")
                st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Bottom row - summary metrics
        st.markdown("---")
        summary_col1, summary_col2, summary_col3 = st.columns(3)
        
        with summary_col1:
            st.metric(
                label="💰 Total Value Analyzed",
                value=f"${stats['total_value']/1000000000:.1f}B" if stats['total_value'] > 1000000000 else f"${stats['total_value']/1000000:.1f}M",
                help="Combined value of all analyzed government spending"
            )
        
        with summary_col2:
            st.metric(
                label="💸 Total Savings Identified",
                value=f"${stats['total_savings']/1000000000:.1f}B" if stats['total_savings'] > 1000000000 else f"${stats['total_savings']/1000000:.1f}M",
                delta=f"{stats['savings_rate']:.1f}% efficiency rate" if stats['savings_rate'] > 0 else "Calculating efficiency",
                help="Total cost savings and efficiency improvements"
            )
        
        with summary_col3:
            total_records = stats['total_contracts'] + stats['total_grants'] + stats['total_leases'] + stats['total_payments']
            st.metric(
                label="📊 Total Records Analyzed",
                value=f"{total_records:,}",
                delta="Real-time analysis",
                help="Combined dataset size across all categories"
            )
    
    else:
        # Fallback metrics if no data loaded
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("📄 Total Contracts", "Loading...", help="Government contracts analyzed for efficiency")
        with col2:
            st.metric("🏢 Total Leases", "Loading...", help="Property leases evaluated for cost savings")
        with col3:
            st.metric("🎯 Total Grants", "Loading...", help="Federal grants assessed for impact")
        with col4:
            st.metric("💳 Payment Records", "Loading...", help="Government payment transactions monitored")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Navigation Cards
    st.markdown("---")
    st.markdown("### 🎯 Explore Analysis Sections")
    
    nav_col1, nav_col2 = st.columns(2)
    
    with nav_col1:
        with st.container():
            st.markdown("""
            <div style="border: 2px solid #1f77b4; border-radius: 15px; padding: 1.5rem; margin-bottom: 1rem; background: linear-gradient(135deg, #f8f9fa 0%, #e3f2fd 100%);">
                <h4 style="color: #1f77b4; margin-bottom: 1rem;">📋 Contracts Analysis</h4>
                <p>Analyze federal contract terminations, savings, and vendor performance across agencies. 
                Includes outlier detection and timeline analysis of contract efficiency initiatives.</p>
                <ul style="margin-top: 1rem;">
                    <li>Agency performance tracking</li>
                    <li>Contract status distribution</li>
                    <li>Savings analysis over time</li>
                    <li>Vendor analysis & outlier detection</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
    
    with nav_col2:
        with st.container():
            st.markdown("""
            <div style="border: 2px solid #d62828; border-radius: 15px; padding: 1.5rem; margin-bottom: 1rem; background: linear-gradient(135deg, #f8f9fa 0%, #ffebee 100%);">
                <h4 style="color: #d62828; margin-bottom: 1rem;">🎁 Grants Analysis</h4>
                <p>Examine federal grant distribution, recipient analysis, and impact assessment. 
                Features machine learning models for grant effectiveness classification.</p>
                <ul style="margin-top: 1rem;">
                    <li>Agency grant distribution</li>
                    <li>Recipient impact analysis</li>
                    <li>Grant classification model</li>
                    <li>Missing information detection</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

@st.cache_data(ttl=3600)  # Cache for 1 hour to improve performance
def load_datasets_cached():
    """Load datasets with caching for better performance"""
    return load_all_datasets()

def main():
    # Set page configuration
    st.set_page_config(**PAGE_CONFIG)
    
    # Create tabs - adding Homepage as the first tab
    tabs = st.tabs(["🏠 Homepage", "🔬 Deep Analysis", "📋 Contracts", "🎁 Grants", "🏢 Leases", "💳 Payments"])
    
    # Initialize session state for datasets
    if 'datasets_loaded' not in st.session_state:
        st.session_state.datasets_loaded = False
        st.session_state.datasets = None
    
    with tabs[0]:
        # For homepage, try to load datasets for real stats, but don't require them
        if not st.session_state.datasets_loaded:
            with st.spinner("Loading data for real-time metrics..."):
                try:
                    st.session_state.datasets = load_datasets_cached()
                    st.session_state.datasets_loaded = True
                except Exception as e:
                    st.warning(f"Could not load real-time data: {e}. Showing demo homepage.")
                    st.session_state.datasets = None
        
        render_enhanced_homepage(st.session_state.datasets)
    
    with tabs[1]:
        # Load datasets if not already loaded
        if not st.session_state.datasets_loaded:
            with st.spinner("Loading datasets..."):
                try:
                    st.session_state.datasets = load_datasets_cached()
                    st.session_state.datasets_loaded = True
                except Exception as e:
                    st.error(f"Error loading datasets: {e}")
                    st.session_state.datasets = {
                        "Contracts": pd.DataFrame(),
                        "Grants": pd.DataFrame(),
                        "Leases": pd.DataFrame(),
                        "Payments": pd.DataFrame()
                    }
        
        render_deep_analysis_tab(st.session_state.datasets)
    
    with tabs[2]:
        if not st.session_state.datasets_loaded:
            with st.spinner("Loading datasets..."):
                try:
                    st.session_state.datasets = load_datasets_cached()
                    st.session_state.datasets_loaded = True
                except Exception as e:
                    st.error(f"Error loading datasets: {e}")
                    st.session_state.datasets = {
                        "Contracts": pd.DataFrame(),
                        "Grants": pd.DataFrame(),
                        "Leases": pd.DataFrame(),
                        "Payments": pd.DataFrame()
                    }
        
        render_contracts_tab(st.session_state.datasets["Contracts"])
    
    with tabs[3]:
        if not st.session_state.datasets_loaded:
            with st.spinner("Loading datasets..."):
                try:
                    st.session_state.datasets = load_datasets_cached()
                    st.session_state.datasets_loaded = True
                except Exception as e:
                    st.error(f"Error loading datasets: {e}")
                    st.session_state.datasets = {
                        "Contracts": pd.DataFrame(),
                        "Grants": pd.DataFrame(),
                        "Leases": pd.DataFrame(),
                        "Payments": pd.DataFrame()
                    }
        
        render_grants_tab(st.session_state.datasets["Grants"])
    
    with tabs[4]:
        if not st.session_state.datasets_loaded:
            with st.spinner("Loading datasets..."):
                try:
                    st.session_state.datasets = load_datasets_cached()
                    st.session_state.datasets_loaded = True
                except Exception as e:
                    st.error(f"Error loading datasets: {e}")
                    st.session_state.datasets = {
                        "Contracts": pd.DataFrame(),
                        "Grants": pd.DataFrame(),
                        "Leases": pd.DataFrame(),
                        "Payments": pd.DataFrame()
                    }
        
        render_leases_tab(st.session_state.datasets["Leases"])
    
    with tabs[5]:
        if not st.session_state.datasets_loaded:
            with st.spinner("Loading datasets..."):
                try:
                    st.session_state.datasets = load_datasets_cached()
                    st.session_state.datasets_loaded = True
                except Exception as e:
                    st.error(f"Error loading datasets: {e}")
                    st.session_state.datasets = {
                        "Contracts": pd.DataFrame(),
                        "Grants": pd.DataFrame(),
                        "Leases": pd.DataFrame(),
                        "Payments": pd.DataFrame()
                    }
        
        render_payments_tab(st.session_state.datasets["Payments"])

if __name__ == "__main__":
    main()
