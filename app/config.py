import os


UPLOAD_DIR = os.getenv(
    "UPLOAD_DIR",
    "data/documents"
)

VECTORSTORE_DIR = os.getenv(
    "VECTORSTORE_DIR",
    "faiss_index"
)

EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "sentence-transformers/all-MiniLM-L6-v2"
)

OLLAMA_MODEL = os.getenv(
    "OLLAMA_MODEL",
    "llama3.2"
)

CHUNK_SIZE = int(
    os.getenv(
        "CHUNK_SIZE",
        "1000"
    )
)

CHUNK_OVERLAP = int(
    os.getenv(
        "CHUNK_OVERLAP",
        "200"
    )
)

RETRIEVAL_K = int(
    os.getenv(
        "RETRIEVAL_K",
        "10"
    )
)

RERANK_TOP_K = int(
    os.getenv(
        "RERANK_TOP_K",
        "3"
    )
)