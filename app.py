import streamlit as st
st.set_page_config(page_title="Legal Document Summarizer & NER", layout="wide")
import io
import os
from utils import extract_text_from_pdf
from model_inference import generate_summary, perform_ner



st.title("⚖️ Legal Document Summarizer & NER App")
st.markdown("""
This application summarizes legal documents using a fine-tuned Google Flan-T5 model
and performs Named Entity Recognition (NER) to extract key entities.
""")

#  Input Section
st.subheader("📄 Input Document")
input_choice = st.radio("Choose Input Method:", ("Plain Text", "Upload PDF File", "Use Example PDF"))

document_text = ""
filename = ""

# Option 1: Paste Text
if input_choice == "Plain Text":
    document_text = st.text_area("Paste your legal document here:", height=300)

# Option 2: Upload PDF
elif input_choice == "Upload PDF File":
    uploaded_file = st.file_uploader("Upload a legal PDF", type="pdf")
    if uploaded_file is not None:
        filename = uploaded_file.name
        pdf_bytes = io.BytesIO(uploaded_file.read())
        document_text = extract_text_from_pdf(pdf_bytes)
        if document_text:
            st.success("PDF text extracted successfully!")
            with st.expander("View Extracted Text"):
                preview = document_text[:2000] + "..." if len(document_text) > 2000 else document_text
                st.text(preview)

# Option 3: Use Example PDF
elif input_choice == "Use Example PDF":
    example_path = os.path.join("example_docs", "example.pdf")
    if os.path.exists(example_path):
        with open(example_path, "rb") as example_file:
            filename = "example.pdf"
            st.info(f"Using built-in example: {filename}")
            document_text = extract_text_from_pdf(example_file)
            with st.expander("View Extracted Text"):
                preview = document_text[:2000] + "..." if len(document_text) > 2000 else document_text
                st.text(preview)
    else:
        st.error("Example PDF not found. Please ensure it's in the 'example_docs' folder.")

# Process Document
if st.button("Generate Summary & Perform NER"):
    if not document_text.strip():
        st.warning("⚠️ Please provide some input text or upload a PDF.")
    else:
        # Summarization
        st.subheader("Summarization Result")
        with st.spinner("Generating summary..."):
            summary = generate_summary(document_text)
            st.success("Summary Generated!")
            st.write(summary)

        #Named Entity Recognition
        st.subheader("Named Entity Recognition (NER)")
        with st.spinner("Extracting entities..."):
            ner_results = perform_ner(document_text)

        if ner_results:
            st.success("NER Completed!")
            entities_by_type = {}
            for entity in ner_results:
                label = entity['entity_group']
                word = entity['word']
                entities_by_type.setdefault(label, set()).add(word)

            for label, words in entities_by_type.items():
                st.markdown(f"**{label}:**")
                st.write(", ".join(sorted(words)))
        else:
            st.info("No entities found or NER could not be performed.")

        # Download Output
        output_text = f"--- SUMMARY ---\n{summary}\n\n--- NAMED ENTITIES ---\n"
        for label, words in entities_by_type.items():
            output_text += f"\n{label}:\n" + ", ".join(sorted(words)) + "\n"

        st.download_button(
            label="📥 Download Summary & NER Results",
            data=output_text,
            file_name="summary_and_entities.txt",
            mime="text/plain"
        )

# Footer
st.markdown("---")
st.markdown("<small style='color: gray;'>Developed with ❤️ using Streamlit, Hugging Face, and Internet:) </small>", unsafe_allow_html=True)
