import streamlit as st
import pandas as pd
from .analysis_modules.executive_summary import render_executive_summary
from .analysis_modules.agency_benchmarking import render_cross_agency_benchmarking
from .analysis_modules.temporal_forecasting import render_temporal_trend_analysis
from .analysis_modules.geographic_patterns import render_geographic_analysis
from .analysis_modules.advanced_analytics import (
    render_savings_optimization,
    render_multidimensional_outliers,
    render_correlation_analysis,
    render_performance_scorecard,
    render_risk_assessment,
    render_cost_benefit_analysis,
    render_predictive_modeling
)

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
    render_executive_summary(datasets)
    
    st.markdown("---")
    
    # Analysis Selection Interface
    st.markdown("### 🎯 Advanced Analytics Selection")
    st.markdown("*Choose from comprehensive analytical frameworks designed for government efficiency assessment*")
    
    # Analysis options with descriptions
    analysis_options = {
        "Cross-Agency Efficiency Benchmarking": {
            "description": "Statistical comparison of agency performance using multi-dimensional efficiency metrics",
            "methodology": "Composite scoring with weighted performance indicators"
        },
        "Temporal Trend Analysis & Forecasting": {
            "description": "Time-series analysis with predictive modeling for efficiency trends",
            "methodology": "Monthly aggregation with trend decomposition and forecasting"
        },
        "Geographic Efficiency Patterns": {
            "description": "Spatial analysis of efficiency initiatives across states and cities",
            "methodology": "Geographic clustering with cost-per-square-foot optimization"
        },
        "Savings Rate Optimization Analysis": {
            "description": "Portfolio optimization for maximizing cost reduction impact",
            "methodology": "Efficiency rate calculations with heatmap correlation analysis"
        },
        "Multi-Dimensional Outlier Detection": {
            "description": "Machine learning anomaly detection for fraud and waste identification",
            "methodology": "Isolation Forest algorithm with statistical outlier thresholds"
        },
        "Contract-Lease Correlation Analysis": {
            "description": "Cross-program efficiency relationships and portfolio optimization",
            "methodology": "Pearson correlation with agency-level efficiency comparison"
        },
        "Agency Performance Scorecard": {
            "description": "Comprehensive multi-criteria performance evaluation framework",
            "methodology": "Weighted composite scoring with consistency and scale metrics"
        },
        "Risk Assessment & Anomaly Patterns": {
            "description": "Predictive risk modeling and fraud detection analytics",
            "methodology": "Coefficient of variation analysis with risk factor weighting"
        },
        "Cost-Benefit ROI Analysis": {
            "description": "Return on investment modeling for government efficiency initiatives",
            "methodology": "ROI percentage calculations with investment vs. savings analysis"
        },
        "Predictive Efficiency Modeling": {
            "description": "Machine learning models for forecasting efficiency outcomes",
            "methodology": "Trend-based prediction with confidence interval estimation"
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
    try:
        if selected_analysis == "Cross-Agency Efficiency Benchmarking":
            render_cross_agency_benchmarking(datasets)
        elif selected_analysis == "Temporal Trend Analysis & Forecasting":
            render_temporal_trend_analysis(datasets)
        elif selected_analysis == "Geographic Efficiency Patterns":
            render_geographic_analysis(datasets)
        elif selected_analysis == "Savings Rate Optimization Analysis":
            render_savings_optimization(datasets)
        elif selected_analysis == "Multi-Dimensional Outlier Detection":
            render_multidimensional_outliers(datasets)
        elif selected_analysis == "Contract-Lease Correlation Analysis":
            render_correlation_analysis(datasets)
        elif selected_analysis == "Agency Performance Scorecard":
            render_performance_scorecard(datasets)
        elif selected_analysis == "Risk Assessment & Anomaly Patterns":
            render_risk_assessment(datasets)
        elif selected_analysis == "Cost-Benefit ROI Analysis":
            render_cost_benefit_analysis(datasets)
        elif selected_analysis == "Predictive Efficiency Modeling":
            render_predictive_modeling(datasets)
            
    except Exception as e:
        st.error(f"❌ Error in {selected_analysis}: {str(e)}")
        st.info("💡 Try refreshing the page or selecting a different analysis type.")
    
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
