RAG Project (Retrieval-Augmented Generation)

A simple RAG (Retrieval-Augmented Generation) pipeline built in Python that answers questions using content from a PDF file, powered by the Gemini API.

How it works


Load PDF – Reads text from notes.pdf using pypdf.
Chunk text – Splits the extracted text into smaller overlapping chunks.
Retrieve – Uses TF-IDF + cosine similarity (scikit-learn) to find the most relevant chunks for a user's question.
Generate – Sends the question along with the retrieved chunks to Google's Gemini API, which generates a final answer grounded in the PDF content.


Files


rag.py – Main RAG pipeline script
notes.pdf – Sample document used as the knowledge source


Setup

bashpip install pypdf scikit-learn google-genai python-dotenv

Create a .env file in the project folder with your Gemini API key:

GEMINI_API_KEY=your-key-here

Get a free API key at Google AI Studio.

Run

bashpython rag.py

Then type your question when prompted, for example:

Your question: What is RAG?

Type exit to quit.
