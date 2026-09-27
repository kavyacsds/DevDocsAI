from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

documents = [
    "Python is a programming language.",
    "Python is used to develop software.",
    "Dogs are domestic animals."
]

query = "What is Python?"

document_embeddings = model.encode(documents)
query_embedding = model.encode(query)


def cosine_similarity(a, b):
    return np.dot(a, b) / (
        np.linalg.norm(a) * np.linalg.norm(b)
    )


print("Query:", query)

print("\nSimilarity scores:")

for document, embedding in zip(documents, document_embeddings):
    score = cosine_similarity(query_embedding, embedding)

    print(f"{score:.4f} - {document}")