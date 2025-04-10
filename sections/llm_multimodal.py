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
    st.error("Gemini API key not found. Please set the GEMINI_API_KEY in your environment.")
else:
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel("gemini-pro-vision")


def extract_text_from_pdf(uploaded_file):
    try:
        reader = PyPDF2.PdfReader(uploaded_file)
        return "\n".join([page.extract_text() or "" for page in reader.pages])
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
        return ""

def llm_multimodal_section():
    st.header("💬 LLM Q&A with Gemini AI")
    st.write("Ask questions about text, documents, or images using Gemini AI.")

    approach = st.radio("Choose Mode:", ("Text (RAG)", "Document/Image (Multimodal)"), horizontal=True)

    if approach == "Text (RAG)":
        context = st.text_area("Enter context (text)", height=200)
        question = st.text_input("Enter your question")
        if st.button("Ask Gemini"):
            if context and question:
                with st.spinner("Generating answer..."):
                    response = model.generate_content([context, question])
                    st.success("Answer:")
                    st.write(response.text)
            else:
                st.warning("Please provide both context and question.")

    else:
        input_type = st.radio("Input Type", ("PDF/TXT/CSV", "Image"))
        content = None

        if input_type == "PDF/TXT/CSV":
            file = st.file_uploader("Upload a document", type=["pdf", "txt", "csv"])
            if file:
                if file.name.endswith(".pdf"):
                    content = extract_text_from_pdf(file)
                elif file.name.endswith(".txt"):
                    content = file.read().decode("utf-8")
                elif file.name.endswith(".csv"):
                    try:
                        df = pd.read_csv(file)
                        content = df.to_string(index=False)
                    except Exception as e:
                        st.error(f"Error reading CSV: {e}")

                if content:
                    st.subheader("Preview Extracted Content")
                    st.text(content[:1000] + "..." if len(content) > 1000 else content)

        elif input_type == "Image":
            image_file = st.file_uploader("Upload an image", type=["jpg", "png"])
            if image_file:
                try:
                    image = Image.open(image_file)
                    content = image
                    st.image(image, caption="Uploaded Image", use_container_width=True)
                except Exception as e:
                    st.error(f"Image error: {e}")

        if content:
            question = st.text_input("Enter your question about the content")
            if st.button("Ask Gemini AI"):
                with st.spinner("Thinking..."):
                    try:
                        response = model.generate_content([content, question])
                        st.success("Answer:")
                        st.write(response.text)
                    except Exception as e:
                        st.error(f"Gemini Error: {e}")