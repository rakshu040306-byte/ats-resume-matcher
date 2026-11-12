import io
import re
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

STOPWORDS = {
    "and", "the", "to", "of", "in", "a", "for", "with", "on", "as", "is",
    "are", "be", "this", "that", "or", "by", "an", "at", "from", "we", "you",
    "your", "our", "will", "have", "has", "can", "all", "more", "about", "who"
}

def extract_text_from_pdf(file_bytes: bytes) -> str:
    """Extracts raw text from all pages of an uploaded PDF file."""
    reader = PdfReader(io.BytesIO(file_bytes))
    text = ""
    for page in reader.pages:
        extracted = page.extract_text()
        if extracted:
            text += extracted + " "
    return text.strip()

def clean_and_tokenize(text: str) -> set:
    """Normalizes text to lowercase words and removes stopwords."""
    words = re.findall(r"\b[a-zA-Z0-9+#.-]{2,}\b", text.lower())
    return {w for w in words if w not in STOPWORDS and not w.isdigit()}

def analyze_resume(pdf_bytes: bytes, job_description: str) -> dict:
    """Computes ATS match percentage and identifies missing keywords."""
    resume_text = extract_text_from_pdf(pdf_bytes)
    
    if not resume_text:
        return {"error": "Could not extract text from PDF. Please upload a text-based PDF."}

    # 1. Calculate TF-IDF Cosine Similarity Score
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform([resume_text, job_description])
    similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
    match_score = round(float(similarity[0][0]) * 100, 1)

    # 2. Find Matched & Missing Keywords using Set Operations
    resume_words = clean_and_tokenize(resume_text)
    jd_words = clean_and_tokenize(job_description)

    matched_keywords = sorted(list(resume_words.intersection(jd_words)))[:15]
    missing_keywords = sorted(list(jd_words - resume_words))[:15]

    return {
        "match_score": match_score,
        "matched_keywords": matched_keywords,
        "missing_keywords": missing_keywords
    }
