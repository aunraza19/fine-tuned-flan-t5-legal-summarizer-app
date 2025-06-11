# ⚖️ Legal Document Summarizer & NER App

This project is a complete Natural Language Processing (NLP) pipeline for **summarizing legal documents** and performing **Named Entity Recognition (NER)** using a fine-tuned large language model (LLM). It features an interactive web interface built with **Streamlit**, supports both **PDF and plain text input**, and lets users **download results** for legal analysis or reporting.

---

## 📌 Project Overview

Legal documents are often long, dense, and filled with domain-specific language that general-purpose summarization models fail to handle well. This project addresses that by fine-tuning a large language model specifically for the **legal summarization domain** and pairing it with an **NER system** to identify people, organizations, laws, locations, and more.

---

## 🧠 Model Architecture

### 🔹 Base Model
- [`google/flan-t5-large`](https://huggingface.co/google/flan-t5-large) — a powerful encoder-decoder model pretrained using instruction tuning across diverse tasks.

### 🔹 Fine-tuned Model
- Model Name: `aun09/flan-t5-legal-summary`
- Hosted on Hugging Face: [View Model](https://huggingface.co/aun09/flan-t5-legal-summary)
- Fine-tuned on legal bill summaries to make it more accurate for law-related documents.

---

## 📚 Dataset

- **Dataset:** [`FiscalNote/billsum`](https://huggingface.co/datasets/FiscalNote/billsum)
- **Use Case:** Legal bill summarization
- **Split:**  
  - Train: ~18,949 examples  
  - Test: ~3,269 examples  
- **Fields used:** `text`, `summary`

---

## 🚀 Features

| Feature                         | Description                                                                 |
|----------------------------------|-----------------------------------------------------------------------------|
| 🧾 Input Options                 | Users can paste legal text or upload a PDF                                  |
| 🧪 Summarization                 | Generates a clear summary of legal text using fine-tuned Flan-T5            |
| 🧠 Named Entity Recognition      | Highlights named entities (persons, orgs, laws, locations, dates, etc.)     |
| 📎 Built-in Example PDF          | Preview functionality without uploading your own file                       |
| 📥 Download Output               | Users can download summary + NER results in `.txt` format                   |
| 🎨 Clean Web Interface           | Built with Streamlit, styled with user-friendly layout and sections         |

---

## 🧑‍💻 Repository Structure

- `app.py`  
  → Main Streamlit app for legal summarization and NER

- `model_inference.py`  
  → Loads the fine-tuned FLAN-T5 model from Hugging Face and performs summarization

- `utils.py`  
  → Handles PDF text extraction (via PyMuPDF) and spaCy-based NER

- `requirements.txt`  
  → List of Python dependencies for running the app

- `example_docs/`  
  → Folder containing an example legal PDF
  - `Court Ruling - BC-2024-087543.pdf` — Built-in PDF for testing without uploading

- `training/`  
  → Contains the Kaggle notebook used to fine-tune the summarization model
  - `flan_t5_finetuning_kaggle.ipynb` — Model training code on `FiscalNote/billsum` dataset

- `README.md`  
  → You're here! Full documentation of the project

---

## 🧠 Why Fine-Tune a Legal Summarizer?

Generic LLMs struggle with:

- Legal jargon
- Structured contract language
- Long, nested clauses
- Section-specific relevance

By fine-tuning a model on real legal bill summaries (`FiscalNote/billsum`), we get:

- ✅ Better comprehension of legal syntax
- ✅ More faithful and concise summaries
- ✅ Increased relevance and context retention
- ✅ Domain-specific abstraction

---

## 🔧 How It Works

1. **User uploads or pastes legal text**
2. `extract_text_from_pdf()` pulls clean text using `PyMuPDF`
3. `generate_summary()` sends the text to the Hugging Face-hosted fine-tuned model
4. `perform_ner()` runs spaCy NER pipeline on the same text
5. Streamlit displays both:
   - 📌 Summary
   - 🔍 Grouped entities by type
6. Option to download results as `.txt`

---

## 💻 Running the App Locally

### Step 1: Clone the Repository
```bash
git clone https://github.com/aunraza19/Fine_tuned_Flan_T5.git
cd legal-summarizer-app
```

### Step 2: Install Requirements
```bash
pip install -r requirements.txt
```

### Step 3: Run Streamlit App
```bash
streamlit run app.py
```

## 🙌Acknowledgments
Hugging Face for transformers and model hosting

Streamlit for simple UI

Kaggle for free GPU during fine-tuning

FiscalNote for the billsum dataset

## 📄 License
MIT License. Feel free to fork and build upon this for research or academic use.

## 👤 Author
Made by Aun Raza
If you use this project or the model, feel free to ⭐ star it or cite the Hugging Face model.

