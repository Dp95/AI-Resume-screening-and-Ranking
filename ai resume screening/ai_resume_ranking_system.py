import streamlit as st
import pandas as pd
import pdfplumber
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.title('AI Resume Screening')
st.write('Welcome to the AI Resume Screening Application!')

# Load Pretrained NLP Model
nlp = spacy.load("en_core_web_sm")

# Extract Text from PDFs
def extract_text_from_pdf(pdf_file):
    text = ""
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            text += page.extract_text() + " "
    return text.strip()

# Preprocess Text (Cleaning & Tokenization)
def preprocess_text(text):
    doc = nlp(text.lower())
    tokens = [token.lemma_ for token in doc if not token.is_stop and token.is_alpha]
    return " ".join(tokens)

# Get User Inputs (Job Description & Resumes)
job_desc = st.sidebar.text_area("Paste Job Description Here")
uploaded_files = st.sidebar.file_uploader("Upload Resumes (PDF)", accept_multiple_files=True)

# Process & Rank Resumes on Button Click
if st.sidebar.button("Analyze Resumes"):
    if not job_desc:
        st.warning("Please enter a job description.")
    elif not uploaded_files:
        st.warning("Please upload at least one resume.")
    else:
        # Preprocess job description
        job_desc_processed = preprocess_text(job_desc)

        # Extract and preprocess resumes
        resume_texts = []
        resume_names = []
        for file in uploaded_files:
            text = extract_text_from_pdf(file)
            processed_text = preprocess_text(text)
            resume_texts.append(processed_text)
            resume_names.append(file.name)

        # Vectorization & Similarity Calculation
        vectorizer = TfidfVectorizer()
        all_texts = [job_desc_processed] + resume_texts
        tfidf_matrix = vectorizer.fit_transform(all_texts)
        similarity_scores = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1:]).flatten()

        # Display results
        results_df = pd.DataFrame({"Resume": resume_names, "Score": similarity_scores})
        results_df = results_df.sort_values(by="Score", ascending=False)
        st.write("### Ranked Resumes")
        st.dataframe(results_df)
