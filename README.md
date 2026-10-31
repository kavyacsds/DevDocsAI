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
<img width="439" height="419" alt="image" src="https://github.com/user-attachments/assets/4975967f-e0d1-4aa4-847b-fa9c064e3976" />
