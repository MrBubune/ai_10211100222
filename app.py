import streamlit as st
from sections import regression, clustering, neural_net

st.set_page_config(page_title="ML Explorer", layout="wide")
st.title("🔍 Machine Learning & AI Explorer")

menu = st.sidebar.radio("Choose Task", ["Regression", "Clustering", "Neural Network"])

if menu == "Regression":
    regression.show()
elif menu == "Clustering":
    clustering.show()
elif menu == "Neural Network":
    neural_net.show()