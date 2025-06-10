import streamlit as st
import io

# Import functions from your custom modules
from utils import extract_text_from_pdf
from model_inference import generate_summary, perform_ner

# --- Streamlit UI ---
st.set_page_config(page_title="Legal Document Summarizer & NER", layout="wide")

st.title("⚖️ Legal Document Summarizer & NER App")
st.markdown("""
This application summarizes legal documents using a fine-tuned Google Flan-T5 model
and performs Named Entity Recognition (NER) to extract key entities.
""")

st.subheader("Input Document")

input_choice = st.radio("Choose Input Method:", ("Plain Text", "Upload PDF File"))

document_text = ""

if input_choice == "Plain Text":
    document_text = st.text_area("Paste your legal document here:", height=300)
elif input_choice == "Upload PDF File":
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
    if uploaded_file is not None:
        # Read the PDF file as bytes
        pdf_bytes = io.BytesIO(uploaded_file.read())
        document_text = extract_text_from_pdf(pdf_bytes)
        if document_text:
            st.success("PDF text extracted successfully!")
            with st.expander("View Extracted Text"):
                # Display only a part of the text if it's very long
                display_text = document_text[:2000] + "..." if len(document_text) > 2000 else document_text
                st.text(display_text)

if st.button("Generate Summary & Perform NER"):
    if not document_text.strip():
        st.warning("Please provide some input text or upload a PDF.")
    else:
        st.subheader("Summarization Result")
        with st.spinner("Generating summary..."):
            summary = generate_summary(document_text)
            st.success("Summary Generated!")
            st.write(summary)

        st.subheader("Named Entity Recognition (NER) Results")
        with st.spinner("Performing NER..."):
            ner_results = perform_ner(document_text)

            if ner_results:
                st.success("NER Completed!")
                # Group entities by type for better display
                entities_by_type = {}
                for entity in ner_results:
                    entity_type = entity['entity_group']
                    if entity_type not in entities_by_type:
                        entities_by_type[entity_type] = []
                    entities_by_type[entity_type].append(entity['word'])

                for entity_type, words in entities_by_type.items():
                    st.markdown(f"**{entity_type}:**")
                    # Use set to remove duplicate words for cleaner display
                    st.write(", ".join(sorted(list(set(words)))))
            else:
                st.info("No entities found or NER could not be performed.")

st.markdown("---")
st.markdown("Developed with ❤️ using Streamlit, Hugging Face Transformers, and PyPDF2.")