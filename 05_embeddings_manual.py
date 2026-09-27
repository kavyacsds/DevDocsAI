from sentence_transformers import SentenceTransformer


model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)


texts = [
    "Python is a programming language.",
    "Python is used to develop software.",
    "Dogs are domestic animals."
]


embeddings = model.encode(texts)


for text, embedding in zip(texts, embeddings):
    print("\nText:")
    print(text)

    print("\nVector:")
    print(embedding)

    print("\nVector dimensions:")
    print(len(embedding))