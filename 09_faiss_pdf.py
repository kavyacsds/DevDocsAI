from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

import faiss
import numpy as np



# 1. LOAD PDF


pdf_path = "data/documents/python.pdf"

loader = PyMuPDFLoader(pdf_path)
documents = loader.load()

print("Number of pages:", len(documents))



# 2. SPLIT DOCUMENTS INTO CHUNKS


text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = text_splitter.split_documents(documents)

print("Number of chunks:", len(chunks))



# 3. LOAD EMBEDDING MODEL


model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)



# 4. CONVERT CHUNKS INTO TEXT


texts = [chunk.page_content for chunk in chunks]


# 5. CREATE EMBEDDINGS


embeddings = model.encode(
    texts,
    show_progress_bar=True
)

embeddings = np.array(
    embeddings,
    dtype="float32"
)

print("Embedding shape:", embeddings.shape)


# 6. CREATE FAISS INDEX


dimension = embeddings.shape[1]

index = faiss.IndexFlatL2(dimension)

index.add(embeddings)

print("Number of vectors in FAISS:", index.ntotal)


# 7. ASK A QUESTION


query = "What is a Python list?"


# 8. EMBED THE QUERY


query_embedding = model.encode([query])

query_embedding = np.array(
    query_embedding,
    dtype="float32"
)


# 9. SEARCH FAISS


distances, indices = index.search(
    query_embedding,
    k=3
)



# 10. DISPLAY RETRIEVED CHUNKS


print("\n========================================")
print("QUERY")
print("========================================")

print(query)


print("\n========================================")
print("RETRIEVED CHUNKS")
print("========================================")


for distance, idx in zip(distances[0], indices[0]):

    chunk = chunks[idx]

    page = chunk.metadata.get("page", -1)

    print("\n----------------------------------------")
    print("Distance:", distance)
    print("Page:", page + 1)
    print("----------------------------------------")

    print(chunk.page_content)