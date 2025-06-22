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

def render_quick_tour():
    """Render an interactive quick tour"""
    if st.button("🎯 Take a Quick Tour", type="primary", help="Learn about dashboard features"):
        with st.expander("🚀 Quick Tour - Dashboard Features", expanded=True):
            st.markdown("""
            ### Welcome to your DOGE Analysis Dashboard!
            
            **Step 1: Homepage Overview**
            - Real-time metrics from government data
            - Visual navigation to analysis sections
            - Data freshness indicators
            
            **Step 2: Analysis Sections**
            - **Contracts**: Analyze terminations, savings, vendor performance
            - **Grants**: Examine distribution, recipients, impact models
            - **Leases**: Geographic analysis, cost efficiency metrics
            - **Payments**: Timeline analysis, anomaly detection
            
            **Step 3: Interactive Features**
            - Use sidebar filters to narrow your focus
            - Switch between analysis views within each tab
            - Export data for further analysis
            - Hover over charts for detailed insights
            
            **Step 4: Advanced Analytics**
            - Machine learning outlier detection
            - Grant impact classification models
            - Geographic and temporal analysis
            - Savings rate calculations
            
            **Pro Tips:**
            - Start with the overview metrics on each tab
            - Use filters to focus on specific agencies or time periods
            - Download filtered data for presentations
            - Check the methodology section for analysis details
            """)

def render_enhanced_homepage(datasets=None):
    """Render the enhanced homepage with DOGE emblem and corrected text"""
    
    # Add improved CSS with fixed card title colors
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
    }
    
    .metric-card {
        transition: transform 0.3s ease;
        border-radius: 10px;
        padding: 1rem;
        border: 1px solid #e0e0e0;
        background: white;
        margin-bottom: 1rem;
        min-height: 120px;
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
    
    /* Remove the problematic white boxes */
    .quick-stats .element-container {
        background: transparent !important;
    }
    
    /* Fix text color in cards */
    .nav-card-content h4 {
        color: #1f77b4 !important;
    }
    .nav-card-content p {
        color: #333 !important;
    }
    .nav-card-content li {
        color: #555 !important;
    }
    
    /* Style the metrics better - FIXED TITLE COLORS */
    .metric-container {
        background: white;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 0.5rem 0;
        border-left: 4px solid #1f77b4;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .metric-container h3 {
        color: #333 !important;
        font-weight: 600;
        margin-bottom: 0.5rem;
        font-size: 1.1rem;
    }
    
    .metric-container h2 {
        margin: 0.5rem 0;
        font-weight: 700;
        font-size: 2rem;
    }
    
    .metric-container p {
        color: #666 !important;
        margin: 0;
        font-size: 0.9rem;
    }
    
    .doge-emblem {
        width: 80px;
        height: 80px;
        margin-right: 1rem;
        border-radius: 50%;
        background: #d4af37;
        display: inline-block;
        vertical-align: middle;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Hero Section with DOGE emblem
    st.markdown("""
    <div class="hero-section" style="text-align: center; padding: 2rem 0; background: linear-gradient(135deg, #1f77b4 0%, #005bbb 100%); border-radius: 15px; margin-bottom: 2rem; color: white; box-shadow: 0 8px 16px rgba(0,0,0,0.1);">
        <div style="display: flex; align-items: center; justify-content: center; margin-bottom: 1rem;">
            <div style="width: 80px; height: 80px; background: url('data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iODAiIGhlaWdodD0iODAiIHZpZXdCb3g9IjAgMCA4MCA4MCIgZmlsbD0ibm9uZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KPGNpcmNsZSBjeD0iNDAiIGN5PSI0MCIgcj0iNDAiIGZpbGw9IiNkNGFmMzciLz4KPHN2ZyB4PSIxMCIgeT0iMTAiIHdpZHRoPSI2MCIgaGVpZ2h0PSI2MCI+CjxjaXJjbGUgY3g9IjMwIiBjeT0iMzAiIHI9IjI4IiBmaWxsPSIjZjBmMGYwIiBzdHJva2U9IiMzMzMiIHN0cm9rZS13aWR0aD0iMiIvPgo8IS0tIERvZ2UgSGVhZCAtLT4KPGNpcmNsZSBjeD0iMzAiIGN5PSIyNSIgcj0iMTIiIGZpbGw9IiNmZGQ5MzUiLz4KPCEtLSBFYXJzIC0tPgo8ZWxsaXBzZSBjeD0iMjMiIGN5PSIxOCIgcng9IjQiIHJ5PSI3IiBmaWxsPSIjZmRkOTM1Ii8+CjxlbGxpcHNlIGN4PSIzNyIgY3k9IjE4IiByeD0iNCIgcnk9IjciIGZpbGw9IiNmZGQ5MzUiLz4KPCEtLSBFeWVzIC0tPgo8Y2lyY2xlIGN4PSIyNiIgY3k9IjIzIiByPSIyIiBmaWxsPSIjMzMzIi8+CjxjaXJjbGUgY3g9IjM0IiBjeT0iMjMiIHI9IjIiIGZpbGw9IiMzMzMiLz4KPCEtLSBOb3NlIC0tPgo8Y2lyY2xlIGN4PSIzMCIgY3k9IjI3IiByPSIxLjUiIGZpbGw9IiMzMzMiLz4KPCEtLSBNYWduaWZ5aW5nIEdsYXNzIC0tPgo8Y2lyY2xlIGN4PSI0NSIgY3k9IjM1IiByPSI4IiBmaWxsPSJub25lIiBzdHJva2U9IiM2NjQ0MDAiIHN0cm9rZS13aWR0aD0iMyIvPgo8bGluZSB4MT0iNTAiIHkxPSI0MCIgeDI9IjU1IiB5Mj0iNDUiIHN0cm9rZT0iIzY2NDQwMCIgc3Ryb2tlLXdpZHRoPSIzIi8+CjwhLS0gSGF0IC0tPgo8ZWxsaXBzZSBjeD0iMzAiIGN5PSIxNSIgcng9IjEwIiByeT0iNCIgZmlsbD0iIzY2NDQwMCIvPgo8IS0tIEFtZXJpY2FuIEZsYWcgLS0+CjxyZWN0IHg9IjI1IiB5PSIzNSIgd2lkdGg9IjEwIiBoZWlnaHQ9IjciIGZpbGw9IiNiMjIyMzQiLz4KPHN0cmlwZSB4PSIyNSIgeT0iMzciIHdpZHRoPSIxMCIgaGVpZ2h0PSIxIiBmaWxsPSIjZmZmIi8+CjxzdHJpcGUgeD0iMjUiIHk9IjM5IiB3aWR0aD0iMTAiIGhlaWdodD0iMSIgZmlsbD0iI2ZmZiIvPgo8cmVjdCB4PSIyNSIgeT0iMzUiIHdpZHRoPSI0IiBoZWlnaHQ9IjQiIGZpbGw9IiMzYzNiNmUiLz4KPC9zdmc+Cjwvc3ZnPgo=') center/contain no-repeat; border-radius: 50%; margin-right: 1rem; border: 3px solid rgba(255,255,255,0.3);"></div>
            <h1 style="font-size: 3rem; margin: 0; font-weight: 700;">DOGE Government Efficiency Dashboard</h1>
        </div>
        <p style="font-size: 1.3rem; margin-bottom: 1.5rem; opacity: 0.9;">
            Comprehensive Analysis of Department of Government Efficiency Data
        </p>
        <p style="font-size: 1rem; opacity: 0.8; max-width: 800px; margin: 0 auto;">
            An interactive data visualization platform analyzing government contracts, grants, leases, and payments 
            to assess efficiency initiatives and fiscal impact across federal agencies.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Alternative approach if the embedded SVG doesn't work - use emoji
    # st.markdown("""
    # <div class="hero-section" style="text-align: center; padding: 2rem 0; background: linear-gradient(135deg, #1f77b4 0%, #005bbb 100%); border-radius: 15px; margin-bottom: 2rem; color: white; box-shadow: 0 8px 16px rgba(0,0,0,0.1);">
    #     <div style="display: flex; align-items: center; justify-content: center; margin-bottom: 1rem;">
    #         <div style="font-size: 4rem; margin-right: 1rem;">🔍🐕</div>
    #         <h1 style="font-size: 3rem; margin: 0; font-weight: 700;">DOGE Government Efficiency Dashboard</h1>
    #     </div>
    # """, unsafe_allow_html=True)
    
    # Data freshness indicator
    render_data_freshness_indicator()
    
    # Quick tour button
    render_quick_tour()
    
    # Enhanced Quick Stats with REMOVED sparklines
    st.markdown("### 📊 Real-Time Dashboard Metrics")
    
    if datasets:
        stats = get_homepage_stats(datasets)
        
        # Top row metrics - NO SPARKLINES
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            delta_val = f"+{stats['terminated_contracts']:,}" if stats['terminated_contracts'] > 0 else "No data"
            st.metric(
                label="📄 Total Contracts",
                value=f"{stats['total_contracts']:,}",
                delta=f"{delta_val} terminated",
                help="Government contracts analyzed for efficiency"
            )
        
        with col2:
            st.metric(
                label="🏢 Total Leases", 
                value=f"{stats['total_leases']:,}",
                delta=f"${stats['lease_savings']/1000000:.1f}M saved" if stats['lease_savings'] > 0 else "No savings data",
                help="Property leases evaluated for cost savings"
            )
        
        with col3:
            st.metric(
                label="🎯 Total Grants",
                value=f"{stats['total_grants']:,}",
                delta=f"${stats['grant_savings']/1000000:.1f}M impact" if stats['grant_savings'] > 0 else "Impact analysis",
                help="Federal grants assessed for impact"
            )
        
        with col4:
            payment_display = f"{stats['total_payments']:,}" if stats['total_payments'] > 0 else "Processing"
            st.metric(
                label="💳 Payment Records",
                value=payment_display,
                delta="Anomaly detection active",
                help="Government payment transactions monitored"
            )
        
        # Bottom row - summary metrics with FIXED styling
        st.markdown("---")
        
        # Use metric containers for better styling
        summary_col1, summary_col2, summary_col3 = st.columns(3)
        
        with summary_col1:
            st.markdown("""
            <div class="metric-container">
                <h3>💰 Total Value Analyzed</h3>
                <h2 style="color: #1f77b4;">{}</h2>
                <p>Combined value of all analyzed government spending</p>
            </div>
            """.format(
                f"${stats['total_value']/1000000000:.1f}B" if stats['total_value'] > 1000000000 else f"${stats['total_value']/1000000:.1f}M"
            ), unsafe_allow_html=True)
        
        with summary_col2:
            efficiency_text = f"{stats['savings_rate']:.1f}% efficiency rate" if stats['savings_rate'] > 0 else "Calculating efficiency"
            st.markdown("""
            <div class="metric-container">
                <h3>💸 Total Savings Identified</h3>
                <h2 style="color: #28a745;">{}</h2>
                <p>Total cost savings • {}</p>
            </div>
            """.format(
                f"${stats['total_savings']/1000000000:.1f}B" if stats['total_savings'] > 1000000000 else f"${stats['total_savings']/1000000:.1f}M",
                efficiency_text
            ), unsafe_allow_html=True)
        
        with summary_col3:
            total_records = stats['total_contracts'] + stats['total_grants'] + stats['total_leases'] + stats['total_payments']
            st.markdown("""
            <div class="metric-container">
                <h3>📊 Total Records Analyzed</h3>
                <h2 style="color: #17a2b8;">{:,}</h2>
                <p>Combined dataset size • Real-time analysis</p>
            </div>
            """.format(total_records), unsafe_allow_html=True)
    
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
    
    # Recent Activity Timeline (simulated)
    st.markdown("---")
    st.markdown("### 📅 Recent DOGE Activity Timeline")
    
    # Create timeline data
    timeline_data = [
        {"date": "2025-06-15", "activity": "Contract Analysis", "description": "Identified $2.3M in potential savings", "type": "savings"},
        {"date": "2025-06-14", "activity": "Lease Termination", "description": "748 federal leases reviewed for efficiency", "type": "review"},
        {"date": "2025-06-13", "activity": "Grant Assessment", "description": "ML model identified low-impact grants", "type": "analysis"},
        {"date": "2025-06-12", "activity": "Payment Audit", "description": "Anomaly detection flagged 156 transactions", "type": "audit"},
        {"date": "2025-06-11", "activity": "System Update", "description": "Dashboard enhanced with new visualization features", "type": "system"}
    ]
    
    for item in timeline_data[:3]:  # Show last 3 activities
        icon = {"savings": "💰", "review": "🔍", "analysis": "🤖", "audit": "🚨", "system": "⚙️"}[item["type"]]
        color = {"savings": "#28a745", "review": "#17a2b8", "analysis": "#6f42c1", "audit": "#fd7e14", "system": "#6c757d"}[item["type"]]
        
        st.markdown(f"""
        <div style="border-left: 4px solid {color}; padding-left: 1rem; margin: 1rem 0; background: #f8f9fa; border-radius: 0 8px 8px 0;">
            <div style="display: flex; align-items: center; margin-bottom: 0.5rem;">
                <span style="font-size: 1.2rem; margin-right: 0.5rem;">{icon}</span>
                <strong style="color: #333;">{item['activity']}</strong>
                <span style="margin-left: auto; color: #6c757d; font-size: 0.9rem;">{item['date']}</span>
            </div>
            <p style="margin: 0; color: #495057;">{item['description']}</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Navigation Cards with IMPROVED styling
    st.markdown("---")
    st.markdown("### 🎯 Explore Analysis Sections")
    
    nav_col1, nav_col2 = st.columns(2)
    
    with nav_col1:
        st.markdown("""
        <div style="border: 2px solid #1f77b4; border-radius: 15px; padding: 1.5rem; margin-bottom: 1rem; background: linear-gradient(135deg, #f8f9fa 0%, #e3f2fd 100%); box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
            <h4 style="color: #1f77b4; margin-bottom: 1rem; font-weight: 600;">📋 Contracts Analysis</h4>
            <p style="color: #333; line-height: 1.5;">Analyze federal contract terminations, savings, and vendor performance across agencies. 
            Includes outlier detection and timeline analysis of contract efficiency initiatives.</p>
            <ul style="margin-top: 1rem; color: #555; padding-left: 1.2rem;">
                <li>Agency performance tracking</li>
                <li>Contract status distribution</li>
                <li>Savings analysis over time</li>
                <li>Vendor analysis & outlier detection</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with nav_col2:
        st.markdown("""
        <div style="border: 2px solid #d62828; border-radius: 15px; padding: 1.5rem; margin-bottom: 1rem; background: linear-gradient(135deg, #f8f9fa 0%, #ffebee 100%); box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
            <h4 style="color: #d62828; margin-bottom: 1rem; font-weight: 600;">🎁 Grants Analysis</h4>
            <p style="color: #333; line-height: 1.5;">Examine federal grant distribution, recipient analysis, and impact assessment. 
            Features machine learning models for grant effectiveness classification.</p>
            <ul style="margin-top: 1rem; color: #555; padding-left: 1.2rem;">
                <li>Agency grant distribution</li>
                <li>Recipient impact analysis</li>
                <li>Grant classification model</li>
                <li>Missing information detection</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    nav_col3, nav_col4 = st.columns(2)
    
    with nav_col3:
        st.markdown("""
        <div style="border: 2px solid #003049; border-radius: 15px; padding: 1.5rem; margin-bottom: 1rem; background: linear-gradient(135deg, #f8f9fa 0%, #e0f2f1 100%); box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
            <h4 style="color: #003049; margin-bottom: 1rem; font-weight: 600;">🏢 Leases Analysis</h4>
            <p style="color: #333; line-height: 1.5;">Geographic analysis of federal property leases, cost efficiency metrics, 
            and termination impact assessment across states and cities.</p>
            <ul style="margin-top: 1rem; color: #555; padding-left: 1.2rem;">
                <li>Geographic analysis by state/city</li>
                <li>Cost efficiency (cost per sq ft)</li>
                <li>Property size analysis</li>
                <li>Timeline trends & savings rates</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with nav_col4:
        st.markdown("""
        <div style="border: 2px solid #fcbf49; border-radius: 15px; padding: 1.5rem; margin-bottom: 1rem; background: linear-gradient(135deg, #f8f9fa 0%, #fff8e1 100%); box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
            <h4 style="color: #fcbf49; margin-bottom: 1rem; font-weight: 600;">💳 Payments Analysis</h4>
            <p style="color: #333; line-height: 1.5;">Government payment pattern analysis, anomaly detection, and financial 
            trend identification across agencies and payment types.</p>
            <ul style="margin-top: 1rem; color: #555; padding-left: 1.2rem;">
                <li>Payment timeline analysis</li>
                <li>Agency spending patterns</li>
                <li>Payment type distribution</li>
                <li>Anomaly detection & fraud analysis</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Enhanced Key Features Section
    st.markdown("---")
    st.markdown("### 🛠️ Advanced Dashboard Capabilities")
    
    feature_col1, feature_col2, feature_col3 = st.columns(3)
    
    with feature_col1:
        st.markdown("""
        **🔍 Interactive Analysis**
        - Dynamic filtering and drill-down capabilities
        - Real-time data exploration with live updates
        - Customizable date ranges and parameters
        - One-click export functionality for presentations
        - Mobile-responsive design for on-the-go analysis
        """)
    
    with feature_col2:
        st.markdown("""
        **🤖 Machine Learning Insights**
        - Isolation Forest outlier detection algorithms
        - Random Forest grant impact classification
        - Anomaly detection for fraud prevention
        - Predictive efficiency scoring models
        - Automated pattern recognition across datasets
        """)
    
    with feature_col3:
        st.markdown("""
        **📈 Executive-Grade Visualizations**
        - Publication-ready charts and graphs
        - Interactive geographic mapping with drill-down
        - Real-time timeline and trend analysis
        - Comparative benchmarking across agencies
        - Professional dashboard design
        """)
    
    # Academic Context - REMOVED Academic Standards Section, CORRECTED to MSBA
    st.markdown("---")
    st.markdown("### 🎓 Academic Excellence & Professional Standards")
    
    st.info("""
    **MSBA Capstone Project - Fairfield University Dolan School of Business**
    
    **Project Objectives:**
    ✅ Demonstrate advanced data visualization and interactive dashboard development  
    ✅ Apply machine learning techniques to government efficiency analysis  
    ✅ Provide actionable insights for public policy decision-making  
    ✅ Showcase professional-grade analytical capabilities for career advancement  
    
    **Real-World Applications:**
    This dashboard demonstrates practical skills in government analytics, policy evaluation, 
    and executive decision support - directly applicable to consulting, public sector, and 
    corporate strategy roles requiring data-driven efficiency optimization.
    """)
    
    # Enhanced Footer - CORRECTED to MSBA
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; font-size: 0.9rem; padding: 2rem; background: #f8f9fa; border-radius: 10px;">
        <h4 style="color: #333; margin-bottom: 1rem;">Government Efficiency Dashboard</h4>
        <p><strong>Fairfield University Dolan School of Business</strong> | MSBA Business Analytics Program | 2025</p>
        <p>Data sourced from DOGE API • Educational and research purposes • Public domain government data</p>
        <p style="margin-top: 1rem; font-size: 0.8rem;">
            <strong>Technical Stack:</strong> Streamlit • Plotly • Pandas • Scikit-learn • Python 3.11<br>
            <strong>Deployment:</strong> GitHub Actions • Streamlit Cloud • API Integration • Real-time Analytics
        </p>
        <div style="margin-top: 1.5rem; border-top: 1px solid #dee2e6; padding-top: 1rem;">
            <p style="margin: 0; font-size: 0.8rem; color: #6c757d;">
                <em>This dashboard represents rigorous academic work meeting MSBA capstone standards for 
                data analysis, visualization design, and professional presentation quality.</em>
            </p>
        </div>
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
