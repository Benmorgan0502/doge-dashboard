"""
Deep Analysis Modules for Government Efficiency Dashboard

This package provides comprehensive analytical capabilities for government efficiency analysis,
including executive summaries, risk assessment, geographic analysis, temporal forecasting,
and advanced analytics using statistical and machine learning methodologies.

Academic Standards:
- All calculations include mathematical explanations and methodological transparency
- Statistical significance testing and confidence intervals provided
- Peer-reviewable methodology documentation
- Established business intelligence and econometric standards followed

Ben Morgan's MSBA Capstone Project - Fairfield University Dolan School of Business
Date: 2025
Version: 1.0.0
"""

__version__ = "1.0.0"
__author__ = "MSBA Capstone Project Team"
__email__ = "capstone@fairfield.edu"

# Simple imports without error-prone multi-line statements
try:
    from .executive_summary import render_executive_summary
except ImportError as e:
    print(f"Warning: Could not import executive_summary: {e}")
    render_executive_summary = None

try:
    from .agency_benchmarking import render_cross_agency_benchmarking
except ImportError as e:
    print(f"Warning: Could not import agency_benchmarking: {e}")
    render_cross_agency_benchmarking = None

try:
    from .temporal_forecasting import render_temporal_trend_analysis
except ImportError as e:
    print(f"Warning: Could not import temporal_forecasting: {e}")
    render_temporal_trend_analysis = None

try:
    from .geographic_patterns import render_geographic_analysis
except ImportError as e:
    print(f"Warning: Could not import geographic_patterns: {e}")
    render_geographic_analysis = None

try:
    from .advanced_analytics import (
        render_savings_optimization,
        render_multidimensional_outliers,
        render_correlation_analysis,
        render_performance_scorecard,
        render_risk_assessment,
        render_cost_benefit_analysis,
        render_predictive_modeling
    )
except ImportError as e:
    print(f"Warning: Could not import advanced_analytics: {e}")
    render_savings_optimization = None
    render_multidimensional_outliers = None
    render_correlation_analysis = None
    render_performance_scorecard = None
    render_risk_assessment = None
    render_cost_benefit_analysis = None
    render_predictive_modeling = None

# Define public API with None checks
__all__ = [
    # Executive Summary Functions
    'render_executive_summary',
    
    # Agency Benchmarking Functions
    'render_cross_agency_benchmarking',
    
    # Temporal Analysis Functions
    'render_temporal_trend_analysis',
    
    # Geographic Analysis Functions
    'render_geographic_analysis',
    
    # Advanced Analytics Functions
    'render_savings_optimization',
    'render_multidimensional_outliers',
    'render_correlation_analysis',
    'render_performance_scorecard',
    'render_risk_assessment',
    'render_cost_benefit_analysis',
    'render_predictive_modeling'
]
