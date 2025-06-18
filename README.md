# 🏛️ DOGE Government Efficiency Dashboard

A comprehensive interactive dashboard analyzing Department of Government Efficiency (DOGE) data including contracts, grants, leases, and payments.

## 🚀 Live Demo
[**View Dashboard**](https://your-actual-streamlit-url-here.streamlit.app)

## 🎓 Academic Project
MSBA Capstone Project - Fairfield University Business Analytics Program

## 📊 Features
- Real-time government efficiency data analysis
- Interactive visualizations and filtering  
- Machine learning outlier detection
- Professional executive-ready presentations
- Mobile-responsive design

## 🛠️ Technology Stack
- **Frontend**: Streamlit
- **Data Processing**: Pandas, NumPy
- **Visualizations**: Plotly
- **Machine Learning**: Scikit-learn
- **Data Source**: DOGE API

## 📈 Analysis Capabilities
- Contract termination analysis
- Grant impact assessment
- Lease cost efficiency evaluation
- Payment anomaly detection
- Cross-agency benchmarking

---
*This dashboard demonstrates advanced data analytics, visualization design, and professional presentation standards for government efficiency analysis.*

## 🏃‍♂️ Running Locally

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/doge-dashboard.git
   cd doge-dashboard
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the dashboard:**
   ```bash
   streamlit run app.py
   ```

4. **Open your browser** to `http://localhost:8501`

## 📁 Project Structure

```
doge_dashboard/
├── app.py                          # Main application
├── requirements.txt                # Dependencies
├── README.md                       # This file
├── config/
│   └── settings.py                 # Configuration settings
├── utils/
│   ├── data_loader.py              # Data loading utilities
│   └── chart_utils.py              # Chart and formatting utilities
├── tabs/
│   ├── contracts_tab.py            # Contracts analysis
│   ├── grants_tab.py               # Grants analysis
│   ├── leases_tab.py               # Leases analysis
│   └── payments_tab.py             # Payments analysis
├── models/
│   ├── outlier_detection.py        # ML outlier detection
│   └── grant_impact_model.py       # Grant classification model
└── data/                           # Cached data files
    ├── contracts_cache.csv
    ├── grants_cache.csv
    ├── leases_cache.csv
    └── payments_cache.csv
```

## 📈 Data Sources

Data is sourced from the DOGE API endpoints:
- Contracts: `https://api.doge.gov/savings/contracts`
- Grants: `https://api.doge.gov/savings/grants`  
- Leases: `https://api.doge.gov/savings/leases`
- Payments: `https://api.doge.gov/payments`

Data is cached locally for 24 hours to improve performance.

## 🔧 Configuration

### Optional: OpenAI Integration
For AI-powered chart commentary, set your OpenAI API key:
```bash
export OPENAI_API_KEY="your-api-key-here"
```

## 📝 License

This project is for educational purposes. Government data is public domain.

## 👨‍🎓 Academic Use

This dashboard was created by Ben Morgan for the Business Analytics Capstone Course at Fairfield University to demonstrate:
- Data visualization best practices
- Interactive dashboard development
- Government data analysis
- Machine learning applications in public policy

---
