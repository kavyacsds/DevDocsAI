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

