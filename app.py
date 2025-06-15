import streamlit as st
from config.settings import PAGE_CONFIG
from utils.data_loader import load_all_datasets
from tabs.contracts_tab import render_contracts_tab
from tabs.grants_tab import render_grants_tab
from tabs.leases_tab import render_leases_tab
from tabs.payments_tab import render_payments_tab

def main():
    # Set page configuration
    st.set_page_config(**PAGE_CONFIG)
    st.title("DOGE Government Efficiency Dashboard")

    # Load all datasets
    datasets = load_all_datasets()
    
    # Create tabs
    tabs = st.tabs(["Contracts", "Grants", "Leases", "Payments"])
    
    with tabs[0]:
        render_contracts_tab(datasets["Contracts"])
    
    with tabs[1]:
        render_grants_tab(datasets["Grants"])
    
    with tabs[2]:
        render_leases_tab(datasets["Leases"])
    
    with tabs[3]:
        render_payments_tab(datasets["Payments"])

if __name__ == "__main__":
    main()