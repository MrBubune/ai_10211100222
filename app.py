import streamlit as st
from sections import home, regression, clustering, neural_network, llm_multimodal

st.set_page_config(
    page_title="ML & AI Explorer",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.sidebar.title("Navigation")
section = st.sidebar.selectbox("Choose a section:", (
    "Home", "Regression", "Clustering", "Neural Network", "LLM Q&A"
))

if section == "Home":
    home.home_section()
elif section == "Regression":
    regression.regression_section()
elif section == "Clustering":
    clustering.clustering_section()
elif section == "Neural Network":
    neural_network.neural_network_section()
elif section == "LLM Q&A":
    llm_multimodal.llm_multimodal_section()