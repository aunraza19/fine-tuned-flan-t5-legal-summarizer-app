import streamlit as st
from PyPDF2 import PdfReader
import io

def extract_text_from_pdf(pdf_file_obj) -> str | None:
    """
    Extracts text from a PDF file object.

    Args:
        pdf_file_obj: A file-like object representing the PDF.

    Returns:
        The extracted text as a string, or None if an error occurs.
    """
    try:
        reader = PdfReader(pdf_file_obj)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text
    except Exception as e:
        st.error(f"Error extracting text from PDF: {e}")
        return None

# You can add other utility functions here in the future,
# e.g., for text cleaning, chunking for very long documents, etc.