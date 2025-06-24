import streamlit as st
import pandas as pd

# Direct imports from individual modules (avoiding __init__.py)
try:
    from tabs.analysis_modules.executive_summary import render_executive_summary
except ImportError as e:
    st.error(f"Could not import executive_summary: {e}")
    render_executive_summary = None

try:
    from tabs.analysis_modules.agency_benchmarking import render_cross_agency_benchmarking
except ImportError as e:
    st.error(f"Could not import agency_benchmarking: {e}")
    render_cross_agency_benchmarking = None

try:
    from tabs.analysis_modules.temporal_forecasting import render_temporal_trend_analysis
except ImportError as e:
    st.error(f"Could not import temporal_forecasting: {e}")
    render_temporal_trend_analysis = None

try:
    from tabs.analysis_modules.geographic_patterns import render_geographic_analysis
except ImportError as e:
    st.error(f"Could not import geographic_patterns: {e}")
    render_geographic_analysis = None

try:
    from tabs.analysis_modules.advanced_analytics import (
        render_savings_optimization,
        render_multidimensional_outliers,
        render_correlation_analysis,
        render_performance_scorecard,
        render_risk_assessment,
        render_cost_benefit_analysis,
        render_predictive_modeling
    )
except ImportError as e:
    st.error(f"Could not import advanced_analytics: {e}")
    render_savings_optimization = None
    render_multidimensional_outliers = None
    render_correlation_analysis = None
    render_performance_scorecard = None
    render_risk_assessment = None
    render_cost_benefit_analysis = None
    render_predictive_modeling = None

def render_placeholder(analysis_name):
    """Render a placeholder when an analysis module is not available"""
    st.markdown(f"## {analysis_name}")
    st.error("⚠️ This analysis module is currently unavailable due to import issues.")
    st.info("Please check the module implementation and try again.")

def render_deep_analysis_tab(datasets):
    """
    Main orchestrator for deep analysis tab with advanced government efficiency analytics.
    
    This tab provides executive-level insights through multiple analytical lenses,
    combining descriptive statistics, predictive modeling, and risk assessment
    to support data-driven decision making in government efficiency initiatives.
    
    Args:
        datasets (dict): Dictionary containing all government datasets
                        (Contracts, Grants, Leases, Payments)
    """
    
    st.markdown("# 🔬 Deep Analysis: Government Efficiency Insights")
    st.markdown("*Advanced analytics for data-driven government efficiency optimization*")
    
    # Data validation
    if not any(len(df) > 0 for df in datasets.values() if not df.empty):
        st.warning("⚠️ No data available for deep analysis. Please ensure data is loaded in other tabs first.")
        return
    
    # Calculate total records for context
    total_records = sum(len(df) for df in datasets.values() if not df.empty)
    st.info(f"📊 Analyzing {total_records:,} government efficiency records across {len([df for df in datasets.values() if not df.empty])} datasets")
    
    # Executive Summary Section - Always shown first
    if render_executive_summary:
        try:
            render_executive_summary(datasets)
        except Exception as e:
            st.error(f"Error in Executive Summary: {str(e)}")
            render_placeholder("Executive Summary")
    else:
        render_placeholder("Executive Summary")
    
    st.markdown("---")
    
    # Analysis Selection Interface
    st.markdown("### 🎯 Advanced Analytics Selection")
    st.markdown("*Choose from comprehensive analytical frameworks designed for government efficiency assessment*")
    
    # Analysis options with descriptions
    analysis_options = {
        "Cross-Agency Efficiency Benchmarking": {
            "description": "Statistical comparison of agency performance using multi-dimensional efficiency metrics",
            "methodology": "Composite scoring with weighted performance indicators",
            "function": render_cross_agency_benchmarking
        },
        "Temporal Trend Analysis & Forecasting": {
            "description": "Time-series analysis with predictive modeling for efficiency trends",
            "methodology": "Monthly aggregation with trend decomposition and forecasting",
            "function": render_temporal_trend_analysis
        },
        "Geographic Efficiency Patterns": {
            "description": "Spatial analysis of efficiency initiatives across states and cities",
            "methodology": "Geographic clustering with cost-per-square-foot optimization",
            "function": render_geographic_analysis
        },
        "Savings Rate Optimization Analysis": {
            "description": "Portfolio optimization for maximizing cost reduction impact",
            "methodology": "Efficiency rate calculations with heatmap correlation analysis",
            "function": render_savings_optimization
        },
        "Multi-Dimensional Outlier Detection": {
            "description": "Machine learning anomaly detection for fraud and waste identification",
            "methodology": "Isolation Forest algorithm with statistical outlier thresholds",
            "function": render_multidimensional_outliers
        },
        "Contract-Lease Correlation Analysis": {
            "description": "Cross-program efficiency relationships and portfolio optimization",
            "methodology": "Pearson correlation with agency-level efficiency comparison",
            "function": render_correlation_analysis
        },
        "Agency Performance Scorecard": {
            "description": "Comprehensive multi-criteria performance evaluation framework",
            "methodology": "Weighted composite scoring with consistency and scale metrics",
            "function": render_performance_scorecard
        },
        "Risk Assessment & Anomaly Patterns": {
            "description": "Predictive risk modeling and fraud detection analytics",
            "methodology": "Coefficient of variation analysis with risk factor weighting",
            "function": render_risk_assessment
        },
        "Cost-Benefit ROI Analysis": {
            "description": "Return on investment modeling for government efficiency initiatives",
            "methodology": "ROI percentage calculations with investment vs. savings analysis",
            "function": render_cost_benefit_analysis
        },
        "Predictive Efficiency Modeling": {
            "description": "Machine learning models for forecasting efficiency outcomes",
            "methodology": "Trend-based prediction with confidence interval estimation",
            "function": render_predictive_modeling
        }
    }
    
    # Create selection interface with methodology preview
    selected_analysis = st.selectbox(
        "Choose your analytical focus:",
        list(analysis_options.keys()),
        help="Each analysis provides unique insights using proven statistical and machine learning methodologies"
    )
    
    # Show methodology info for selected analysis
    if selected_analysis:
        analysis_info = analysis_options[selected_analysis]
        st.info(f"**{analysis_info['description']}**\n\n*Methodology: {analysis_info['methodology']}*")
    
    st.markdown("---")
    
    # Render selected analysis with error handling
    if selected_analysis and selected_analysis in analysis_options:
        analysis_function = analysis_options[selected_analysis]["function"]
        
        if analysis_function:
            try:
                analysis_function(datasets)
            except Exception as e:
                st.error(f"❌ Error in {selected_analysis}: {str(e)}")
                st.info("💡 Try refreshing the page or selecting a different analysis type.")
                render_placeholder(selected_analysis)
        else:
            render_placeholder(selected_analysis)
    
    # Footer with methodology notes
    st.markdown("---")
    st.markdown("""
    ### 📚 Analytical Methodology Notes
    
    All analyses follow established statistical and business intelligence practices:
    - **Descriptive Analytics**: Summary statistics with quartile analysis
    - **Diagnostic Analytics**: Root cause analysis through correlation and segmentation  
    - **Predictive Analytics**: Machine learning models with confidence intervals
    - **Prescriptive Analytics**: Optimization recommendations with action prioritization
    
    *Data quality checks and statistical significance testing applied throughout all calculations.*
    """)
