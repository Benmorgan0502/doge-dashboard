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

Modules:
- executive_summary: C-level dashboard with risk assessment and KPI calculation
- agency_benchmarking: Cross-agency performance comparison using statistical methods
- temporal_forecasting: Time-series analysis with linear regression forecasting
- geographic_patterns: Spatial analysis of efficiency patterns with GIS methods
- advanced_analytics: Portfolio optimization, outlier detection, and correlation analysis

Mathematical Frameworks:
- Descriptive Analytics: Summary statistics with quartile analysis
- Diagnostic Analytics: Root cause analysis through correlation and segmentation
- Predictive Analytics: Machine learning models with confidence intervals
- Prescriptive Analytics: Optimization recommendations with action prioritization

Ben Morgan's MSBA Capstone Project - Fairfield University Dolan School of Business
Date: 2025
Version: 1.0.0
"""

__version__ = "1.0.0"
__author__ = "MSBA Capstone Project Team"
__email__ = "capstone@fairfield.edu"

# Import all main functions for easy access
from .executive_summary import (
    render_executive_summary,
    calculate_comprehensive_stats,
    calculate_risk_metrics,
    render_risk_dashboard_fixed
)

from .agency_benchmarking import (
    render_cross_agency_benchmarking,
    combine_datasets_for_agency_analysis,
    calculate_agency_performance_metrics
)

from .temporal_forecasting import (
    render_temporal_trend_analysis,
    extract_temporal_data,
    calculate_temporal_metrics,
    calculate_trend_slope,
    create_forecast
)

from .geographic_patterns import (
    render_geographic_analysis,
    extract_geographic_data,
    calculate_geographic_metrics
)

from .advanced_analytics import (
    render_savings_optimization,
    render_multidimensional_outliers,
    render_correlation_analysis,
    render_performance_scorecard,
    render_risk_assessment,
    render_cost_benefit_analysis,
    render_predictive_modeling
)

# Define public API
__all__ = [
    # Executive Summary Functions
    'render_executive_summary',
    'calculate_comprehensive_stats',
    'calculate_risk_metrics',
    'render_risk_dashboard_fixed',
    
    # Agency Benchmarking Functions
    'render_cross_agency_benchmarking',
    'combine_datasets_for_agency_analysis',
    'calculate_agency_performance_metrics',
    
    # Temporal Analysis Functions
    'render_temporal_trend_analysis',
    'extract_temporal_data',
    'calculate_temporal_metrics',
    'calculate_trend_slope',
    'create_forecast',
    
    # Geographic Analysis Functions
    'render_geographic_analysis',
    'extract_geographic_data',
    'calculate_geographic_metrics',
    
    # Advanced Analytics Functions
    'render_savings_optimization',
    'render_multidimensional_outliers',
    'render_correlation_analysis',
    'render_performance_scorecard',
    'render_risk_assessment',
    'render_cost_benefit_analysis',
    'render_predictive_modeling'
]

# Package metadata
PACKAGE_INFO = {
    "name": "analysis_modules",
    "version": __version__,
    "description": "Advanced government efficiency analysis modules",
    "author": __author__,
    "academic_institution": "Fairfield University Dolan School of Business",
    "program": "MSBA - Master of Science in Business Analytics",
    "project_type": "Capstone Project",
    "year": "2025",
    "keywords": [
        "government efficiency",
        "business analytics",
        "statistical analysis",
        "machine learning",
        "data visualization",
        "executive dashboard",
        "performance optimization"
    ],
    "methodology_standards": [
        "Statistical significance testing (p < 0.05)",
        "95% confidence intervals",
        "Peer-reviewable documentation",
        "Established econometric methods",
        "Business intelligence best practices"
    ]
}

def get_package_info():
    """
    Return comprehensive package information for academic documentation.
    
    Returns:
        dict: Package metadata including academic standards and methodology
    """
    return PACKAGE_INFO

def validate_requirements():
    """
    Validate that all required dependencies are available for analysis modules.
    
    Returns:
        dict: Validation results with status and missing dependencies
    """
    required_packages = {
        'streamlit': 'Web application framework',
        'pandas': 'Data manipulation and analysis',
        'plotly': 'Interactive visualization',
        'numpy': 'Numerical computing',
        'scikit-learn': 'Machine learning (optional for some features)'
    }
    
    validation_results = {
        'status': 'success',
        'available': [],
        'missing': [],
        'optional_missing': []
    }
    
    for package, description in required_packages.items():
        try:
            if package == 'scikit-learn':
                import sklearn
                validation_results['available'].append(f"{package}: {description}")
            elif package == 'streamlit':
                import streamlit
                validation_results['available'].append(f"{package}: {description}")
            elif package == 'pandas':
                import pandas
                validation_results['available'].append(f"{package}: {description}")
            elif package == 'plotly':
                import plotly
                validation_results['available'].append(f"{package}: {description}")
            elif package == 'numpy':
                import numpy
                validation_results['available'].append(f"{package}: {description}")
        except ImportError:
            if package == 'scikit-learn':
                validation_results['optional_missing'].append(f"{package}: {description} (outlier detection limited)")
            else:
                validation_results['missing'].append(f"{package}: {description}")
                validation_results['status'] = 'error'
    
    return validation_results

# Academic documentation helper
def generate_methodology_report():
    """
    Generate comprehensive methodology report for academic review.
    
    Returns:
        str: Formatted methodology report suitable for academic submission
    """
    
    report = f"""
# Deep Analysis Methodology Report
## MSBA Capstone Project - Fairfield University Dolan School of Business

### Package Information
- **Version**: {__version__}
- **Analysis Modules**: {len(__all__)} comprehensive analytical functions
- **Academic Standards**: Peer-reviewable methodology with statistical validation

### Mathematical Frameworks Implemented

#### 1. Executive Summary Analysis
- **KPI Calculation**: Weighted composite scoring with statistical validation
- **Risk Assessment**: Multi-factor risk modeling using outlier detection
- **Efficiency Metrics**: (Savings ÷ Value) × 100 with confidence intervals

#### 2. Agency Benchmarking
- **Performance Scoring**: Composite methodology with empirically validated weights
- **Statistical Comparison**: Quartile analysis with significance testing
- **Correlation Analysis**: Pearson coefficients with p-value validation

#### 3. Temporal Forecasting
- **Trend Analysis**: Linear regression with least squares estimation
- **Forecasting**: Trend extrapolation with 95% confidence intervals
- **Seasonality**: 12-month moving averages for pattern identification

#### 4. Geographic Analysis
- **Spatial Statistics**: State and city-level aggregation with population weighting
- **Cost Efficiency**: Geographic cost-per-square-foot analysis
- **Regional Benchmarking**: Coefficient of variation for geographic dispersion

#### 5. Advanced Analytics
- **Portfolio Optimization**: Modern portfolio theory adapted for government efficiency
- **Outlier Detection**: Isolation Forest ML algorithm with statistical validation
- **Correlation Analysis**: Multi-dimensional relationship assessment

### Statistical Validation Standards
- **Confidence Level**: 95% for all inferential statistics
- **Significance Testing**: α = 0.05 threshold for hypothesis testing
- **Sample Size**: Minimum thresholds enforced for statistical power
- **Outlier Treatment**: IQR method with conservative 1.5× multiplier
- **Missing Data**: Documented handling with impact assessment

### Data Quality Assurance
- **Validation**: Multi-step data quality checks and validation
- **Documentation**: Complete audit trail for all calculations
- **Reproducibility**: Deterministic algorithms with seed values where applicable
- **Peer Review**: Methodology suitable for academic and professional review

### Academic Compliance
- **Methodology Transparency**: All calculations explained with mathematical formulas
- **Statistical Rigor**: Established econometric and business intelligence standards
- **Documentation**: Comprehensive docstrings and methodology explanations
- **Validation**: Cross-validation and significance testing throughout

---
*Generated automatically from analysis_modules package v{__version__}*
*Fairfield University Dolan School of Business - MSBA Program 2025*
    """
    
    return report.strip()
