from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# 1. Load embedding model
model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

# 2. Our example documents
documents = [
    "Python is a programming language.",
    "Python is used to develop software.",
    "Dogs are domestic animals."
]

# 3. Convert documents into vectors
document_embeddings = model.encode(documents)

# 4. Convert to float32 because FAISS expects this format
document_embeddings = np.array(
    document_embeddings,
    dtype="float32"
)

# 5. Get vector dimensions
dimension = document_embeddings.shape[1]

print("Vector dimensions:", dimension)

# 6. Create FAISS index
index = faiss.IndexFlatL2(dimension)

# 7. Add vectors to the index
index.add(document_embeddings)
# 8. Create a query
query = "What is Python?"

# 9. Convert the query into a vector
query_embedding = model.encode([query])

# 10. Convert query vector to float32
query_embedding = np.array(
    query_embedding,
    dtype="float32"
)

# 11. Search FAISS
distances, indices = index.search(
    query_embedding,
    k=2
)

print("\nQuery:", query)

print("\nResults:")

for distance, index_position in zip(distances[0], indices[0]):
    print(
        f"Distance: {distance:.4f} - "
        f"{documents[index_position]}"
    )

print("Number of vectors in FAISS:", index.ntotal)