import streamlit as st

def home_section():
    st.title("🧠 ML & AI Explorer")
    st.markdown("""
    Welcome to the **ML & AI Explorer** app — an interactive tool to explore and solve diverse Machine Learning and AI problems using a unified dashboard.  
    This app demonstrates key ML/AI tasks with visual explanations and custom predictions.

    ### 📦 Features
    - **Regression**: Build linear models to predict continuous values.
    - **Clustering**: Group similar data points using K-Means with interactive 2D/3D visualizations.
    - **Neural Networks**: Train classification models with live accuracy/loss graphs.
    - **LLM Q&A**: Ask questions about documents, images, or text using Gemini AI.

    ### 🚀 How To Use
    1. Navigate via the sidebar.
    2. Upload your dataset.
    3. Follow instructions for each task (training, prediction, visualization).
    4. For LLMs, try uploading a PDF, image, or input plain text for Q&A.

    ### 🛠️ Technologies
    - Python + Streamlit
    - Scikit-learn, TensorFlow
    - Google Gemini API
    - Plotly for visualization

    ---
    **Built for Academic City University**
    """)
