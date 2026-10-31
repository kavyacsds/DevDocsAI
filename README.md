# DevDocs AI

A local RAG-based technical document assistant.

## Features

- Upload PDF documents
- Extract and split document content
- Generate local embeddings
- Store embeddings in FAISS
- Retrieve relevant document chunks
- Rerank retrieved chunks
- Generate answers using a local Ollama LLM
- Return document and page sources
- Conversational question support
- FastAPI backend
- React frontend

## Architecture

```text
React
   ↓
FastAPI
   ↓
Document Ingestion
   ↓
Chunking
   ↓
HuggingFace Embeddings
   ↓
FAISS
   ↓
Retriever
   ↓
Reranker
   ↓
Ollama
   ↓
Answer + Sources
## Image frontend
<img width="439" height="419" alt="Screenshot 2026-10-31 131124" src="https://github.com/user-attachments/assets/db5657d4-b515-4c45-b228-12d431c73259" />
# 1. Start Ollama (Terminal 1)
ollama serve

# 2. Start FastAPI Backend (Terminal 2)
cd C:\DevDocs
.\venv\Scripts\Activate.ps1
uvicorn app.main:app --reload

# 3. Start React Frontend (Terminal 3)
cd C:\DevDocs\frontend
npm run dev

# Install Python dependencies
pip install -r requirements.txt

# Install frontend dependencies
cd frontend
npm install

# Download Llama 3.2 model
ollama pull llama3.2

Frontend: http://localhost:5173
Backend: http://127.0.0.1:8000
API Documentation: http://127.0.0.1:8000/docs

