import plotly.io as pio

# Streamlit page configuration
PAGE_CONFIG = {
    "page_title": "DOGE API Viewer",
    "layout": "wide"
}

# API configurations
API_CONFIGS = {
    "Contracts": {
        "url": "https://api.doge.gov/savings/contracts",
        "cache": "data/contracts_cache.csv",
        "result_key": "contracts"
    },
    "Grants": {
        "url": "https://api.doge.gov/savings/grants",
        "cache": "data/grants_cache.csv",
        "result_key": "grants"
    },
    "Leases": {
        "url": "https://api.doge.gov/savings/leases",
        "cache": "data/leases_cache.csv",
        "result_key": "leases"
    },
    "Payments": {
        "url": "https://api.doge.gov/payments",
        "cache": "data/payments_cache.csv",
        "result_key": "payments"
    }
}

# Cache duration in hours
CACHE_DURATION_HOURS = 24

# Plotly theme configuration
def setup_plotly_theme():
    """Set up custom Plotly theme"""
    pio.templates["custom_blue"] = pio.templates["plotly_white"]
    pio.templates["custom_blue"].layout.colorway = ["#d62828", "#005bbb", "#003049", "#fcbf49"]
    pio.templates.default = "custom_blue"

# Initialize theme
setup_plotly_theme()