import os
import streamlit as st
import pandas as pd
import PyPDF2
from PIL import Image
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    st.error("❌ Gemini API key not found. Please add GEMINI_API_KEY to your .env file.")
else:
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel("gemini-pro-vision")

def extract_text_from_pdf(file_path):
    """Extracts text from a local PDF file."""
    try:
        with open(file_path, "rb") as file:
            reader = PyPDF2.PdfReader(file)
            text = ""
            for page in reader.pages:
                text += page.extract_text() or ""
            return text
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
        return ""

def load_csv_data(file_path):
    """Reads a local CSV file and returns a DataFrame."""
    try:
        return pd.read_csv(file_path)
    except Exception as e:
        st.error(f"Error loading CSV: {e}")
        return None

def llm_multimodal_section():
    st.header("💬 LLM Q&A with Gemini AI")
    st.write("Ask questions about documents and data using Gemini AI.")

    # Dataset selection
    st.subheader("📂 Select a Preloaded Dataset")
    datasets = {
        "Ghana Election Results (CSV)": "datasets/Ghana_Election_Result.csv",
        "2025 Budget Statement (PDF)": "datasets/2025-Budget-Statement-and-Economic-Policy_v4.pdf",
        "Academic City Student Handbook (PDF)": "datasets/handbook.pdf"
    }

    dataset_name = st.selectbox("Choose a dataset:", list(datasets.keys()))
    selected_path = datasets[dataset_name]
    content = None

    # Display and extract content
    if selected_path.endswith(".csv"):
        df = load_csv_data(selected_path)
        if df is not None:
            st.subheader("📊 CSV Preview")
            st.dataframe(df.head())
            content = df.to_string(index=False)

    elif selected_path.endswith(".pdf"):
        st.subheader("📄 PDF Content Preview")
        content = extract_text_from_pdf(selected_path)
        st.text(content[:1000] + "..." if len(content) > 1000 else content)

    # Ask a question
    st.subheader("❓ Ask a Question")
    question = st.text_input("Enter your question:")
    if st.button("Ask Gemini"):
        if content and question:
            with st.spinner("Thinking..."):
                try:
                    response = model.generate_content([content, question])
                    st.success("🧠 Gemini's Answer:")
                    st.write(response.text)
                except Exception as e:
                    st.error(f"Error generating answer: {e}")
        else:
            st.warning("Please select a dataset and enter a question.")
