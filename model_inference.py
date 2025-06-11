import streamlit as st
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline

#Configuration
# My fine-tuned summarization model on Hugging Face
SUMMARIZATION_MODEL_NAME = "aun09/flan-t5-legal-summary"

# A suitable legal domain NER model from Hugging Face
# Source: https://huggingface.co/dslim/bert-base-NER-legal
NER_MODEL_NAME = "dslim/bert-base-NER"

# Model Loading (Cached for performance)
@st.cache_resource
def load_summarization_model_and_tokenizer():
    """Loads the fine-tuned Flan-T5 summarization model and its tokenizer."""
    st.info(f"Loading summarization model: {SUMMARIZATION_MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(SUMMARIZATION_MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(SUMMARIZATION_MODEL_NAME)
    model.eval()
    st.success("Summarization model loaded!")
    return tokenizer, model

@st.cache_resource
def load_ner_pipeline():
    """Loads the legal NER pipeline."""
    st.info(f"Loading NER model: {NER_MODEL_NAME}...")
    ner_pipeline = pipeline("ner", model=NER_MODEL_NAME, tokenizer=NER_MODEL_NAME, aggregation_strategy="simple")
    st.success("NER model loaded!")
    return ner_pipeline

# Loading models once when the app starts
summarizer_tokenizer, summarizer_model = load_summarization_model_and_tokenizer()
ner_pipeline = load_ner_pipeline()


#Inference Functions
def generate_summary(text: str) -> str:
    """Generates a summary using the fine-tuned Flan-T5 model."""
    if not text.strip():
        return "No text provided for summarization."

    inputs = summarizer_tokenizer.encode("summarize: " + text, return_tensors="pt", max_length=1024, truncation=True, padding= "max_length")

    summary_ids = summarizer_model.generate(
        inputs,
        max_length=500,  # Max length of the generated summary
        min_length=30,  # Min length of the generated summary
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


    max_ner_length_words = 512
    if len(text.split()) > max_ner_length_words:
        print(
            f"Input text is very long ({len(text.split())} words). NER model might truncate or perform less accurately. Consider processing in chunks for production.")

    try:
        ner_results = ner_pipeline(text)
        return ner_results
    except Exception as e:
        st.error(f"Error performing NER: {e}")
        return []