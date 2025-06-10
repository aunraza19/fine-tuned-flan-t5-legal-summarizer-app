import streamlit as st
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline

# --- Configuration ---
# Your fine-tuned summarization model on Hugging Face
SUMMARIZATION_MODEL_NAME = "aun09/flan-t5-legal-summary"

# A suitable legal domain NER model from Hugging Face
# Source: https://huggingface.co/dslim/bert-base-NER-legal
NER_MODEL_NAME = "dslim/bert-base-NER"

# --- Model Loading (Cached for performance) ---
@st.cache_resource
def load_summarization_model_and_tokenizer():
    """Loads the fine-tuned Flan-T5 summarization model and its tokenizer."""
    print(f"Loading summarization model: {SUMMARIZATION_MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(SUMMARIZATION_MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(SUMMARIZATION_MODEL_NAME)
    print("Summarization model loaded!")
    return tokenizer, model

 # Replace with your actual token
@st.cache_resource
def load_ner_pipeline():
    """Loads the legal NER pipeline."""
    print(f"Loading NER model: {NER_MODEL_NAME}...")
    # Using pipeline for NER is convenient
    ner_pipeline = pipeline("ner", model=NER_MODEL_NAME, tokenizer=NER_MODEL_NAME, aggregation_strategy="simple")
    print("NER model loaded!")
    return ner_pipeline

# Load models once when the app starts
summarizer_tokenizer, summarizer_model = load_summarization_model_and_tokenizer()
ner_pipeline = load_ner_pipeline()


# --- Inference Functions ---
def generate_summary(text: str) -> str:
    """Generates a summary using the fine-tuned Flan-T5 model."""
    if not text.strip():
        return "No text provided for summarization."

    inputs = summarizer_tokenizer.encode("summarize: " + text, return_tensors="pt", max_length=1024, truncation=True)

    # You might need to adjust generation parameters based on your fine-tuning.
    # For legal summaries, typically longer summaries might be desired.
    summary_ids = summarizer_model.generate(
        inputs,
        max_length=250,  # Max length of the generated summary
        min_length=50,  # Min length of the generated summary
        length_penalty=2.0,  # Encourage longer summaries
        num_beams=4,  # For better quality summary
        early_stopping=True
    )
    summary = summarizer_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary


def perform_ner(text: str):
    """Performs Named Entity Recognition using the legal NER model."""
    if not text.strip():
        return []

    # The NER model might have input length limitations (typically around 512 tokens for BERT-based models).
    # For very long documents, you might need to chunk the text and run NER on each chunk,
    # then combine results. This simple implementation processes the whole text.
    max_ner_length_words = 512
    if len(text.split()) > max_ner_length_words:
        print(
            f"Input text is very long ({len(text.split())} words). NER model might truncate or perform less accurately. Consider processing in chunks for production.")
        # For robust production use, implement proper text chunking and NER inference per chunk.

    try:
        ner_results = ner_pipeline(text)
        return ner_results
    except Exception as e:
        print(f"Error performing NER: {e}")
        return []