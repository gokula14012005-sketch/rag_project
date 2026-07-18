"""
Simple RAG (Retrieval-Augmented Generation) pipeline.

Steps:
1. Read text from a PDF (notes.pdf)
2. Split text into chunks
3. Create embeddings for each chunk (TF-IDF based, no API needed for retrieval)
4. On a user question, find the most relevant chunks
5. Send question + relevant chunks to Claude API to get a final answer

Install dependencies first (see requirements below or run in VS Code terminal):
    pip install pypdf scikit-learn google-generativeai

Set your Gemini API key as an environment variable before running:
    Windows (PowerShell): $env:GEMINI_API_KEY="your-key-here"
    Mac/Linux:             export GEMINI_API_KEY="your-key-here"

Get a free Gemini API key here: https://aistudio.google.com/app/apikey
"""

import os
import sys
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

try:
    import google.generativeai as genai
except ImportError:
    genai = None


# ---------- STEP 1: Load PDF text ----------
def load_pdf_text(pdf_path: str) -> str:
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    reader = PdfReader(pdf_path)
    full_text = ""
    for page in reader.pages:
        text = page.extract_text() or ""
        full_text += text + "\n"
    return full_text


# ---------- STEP 2: Split text into chunks ----------
def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list:
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


# ---------- STEP 3: Build a simple TF-IDF based retriever ----------
class SimpleRetriever:
    def __init__(self, chunks: list):
        self.chunks = chunks
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.chunk_vectors = self.vectorizer.fit_transform(chunks)

    def get_top_chunks(self, query: str, top_k: int = 3) -> list:
        query_vector = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vector, self.chunk_vectors).flatten()
        top_indices = similarities.argsort()[::-1][:top_k]
        return [self.chunks[i] for i in top_indices]


# ---------- STEP 4: Ask Gemini using retrieved context ----------
def ask_gemini(question: str, context_chunks: list) -> str:
    if genai is None:
        return (
            "[google-generativeai package not installed] Here are the top "
            "matching chunks found for your question instead:\n\n"
            + "\n---\n".join(context_chunks)
        )

    api_key = os.environ.get("your api key")
    if not api_key:
        return (
            "GEMINI_API_KEY environment variable not set. "
            "Showing retrieved context instead:\n\n" + "\n---\n".join(context_chunks)
        )

    genai.configure(api_key=api_key)

    context_text = "\n\n".join(
        f"[Chunk {i+1}]\n{chunk}" for i, chunk in enumerate(context_chunks)
    )

    prompt = f"""You are a helpful assistant. Answer the question using ONLY the
context provided below. If the answer is not in the context, say you don't know.

Context:
{context_text}

Question: {question}

Answer:"""

    model = genai.GenerativeModel("gemini-2.0-flash")
    response = model.generate_content(prompt)

    return response.text


# ---------- MAIN ----------
def main():
    pdf_path = os.path.join(os.path.dirname(__file__), "notes.pdf")

    print(f"Loading PDF from: {pdf_path}")
    text = load_pdf_text(pdf_path)

    print("Splitting text into chunks...")
    chunks = chunk_text(text)
    print(f"Total chunks created: {len(chunks)}")

    print("Building retriever...")
    retriever = SimpleRetriever(chunks)

    print("\nRAG system ready! Type your question below (type 'exit' to quit).\n")

    while True:
        question = input("Your question: ").strip()
        if question.lower() in ("exit", "quit"):
            print("Bye!")
            break
        if not question:
            continue

        top_chunks = retriever.get_top_chunks(question, top_k=3)
        answer = ask_gemini(question, top_chunks)

        print("\n--- Answer ---")
        print(answer)
        print("--------------\n")


if __name__ == "__main__":
    sys.exit(main() or 0)
