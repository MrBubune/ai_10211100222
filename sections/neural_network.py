import os
import streamlit as st
import pandas as pd
import PyPDF2
from PIL import Image
import google.generativeai as genai  # ✅ Correct import
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    st.error("Gemini API key not found. Please set the GEMINI_API_KEY in your .env file.")
else:
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel("gemini-pro-vision")

# Function to extract text from PDF
def extract_text_from_pdf(file_path):
    with open(file_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
        return text

# Function to load CSV data
def load_csv_data(file_path):
    return pd.read_csv(file_path)

def llm_multimodal_section():
    st.header("💬 LLM Q&A with Gemini AI")
    st.write("Ask questions based on the selected dataset using Gemini AI.")

    # Dataset selection
    dataset_options = {
        "Ghana Election Results": "datasets/Ghana_Election_Result.csv",
        "2025 Budget Statement": "datasets/2025-Budget-Statement-and-Economic-Policy_v4.pdf",
        "Academic City Student Handbook": "datasets/handbook.pdf"
    }
    selected_dataset = st.selectbox("Select a dataset:", list(dataset_options.keys()))

    # Load and display the selected dataset
    file_path = dataset_options[selected_dataset]
    content = None

    if file_path.endswith(".csv"):
        df = load_csv_data(file_path)
        st.subheader("Dataset Preview")
        st.dataframe(df.head())
        content = df.to_string(index=False)
    elif file_path.endswith(".pdf"):
        content = extract_text_from_pdf(file_path)
        st.subheader("Document Preview")
        st.text(content[:1000] + "..." if len(content) > 1000 else content)

    # Question input
    question = st.text_input("Enter your question about the selected dataset:")
    if st.button("Get Answer"):
        if content and question:
            with st.spinner("Generating answer..."):
                response = model.generate_content([content, question])
                st.success("Answer:")
                st.write(response.text)
        else:
            st.warning("Please select a dataset and enter a question.")
