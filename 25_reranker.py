from sentence_transformers import CrossEncoder

from app.embeddings import get_embeddings
from app.vectorstore import load_vectorstore


# ==================================================
# 1. LOAD EMBEDDINGS
# ==================================================

embeddings = get_embeddings()


# ==================================================
# 2. LOAD FAISS
# ==================================================

vector_store = load_vectorstore(
    "faiss_index",
    embeddings
)


# ==================================================
# 3. LOAD RETRIEVER
# ==================================================

retriever = vector_store.as_retriever(
    search_kwargs={
        "k": 10
    }
)


# ==================================================
# 4. LOAD RERANKER
# ==================================================

reranker = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)


# ==================================================
# 5. QUESTION
# ==================================================

query = "What is a Python list?"


# ==================================================
# 6. FIRST-STAGE RETRIEVAL
# ==================================================

documents = retriever.invoke(
    query
)


print("\nRetrieved candidates:", len(documents))


# ==================================================
# 7. CREATE QUESTION-DOCUMENT PAIRS
# ==================================================

pairs = [
    (query, document.page_content)
    for document in documents
]


# ==================================================
# 8. CALCULATE RERANKING SCORES
# ==================================================

scores = reranker.predict(
    pairs
)


# ==================================================
# 9. COMBINE DOCUMENTS + SCORES
# ==================================================

ranked_documents = sorted(
    zip(documents, scores),
    key=lambda x: x[1],
    reverse=True
)


# ==================================================
# 10. DISPLAY RESULTS
# ==================================================

print("\n========================================")
print("RERANKED RESULTS")
print("========================================")


for i, (document, score) in enumerate(
    ranked_documents
):

    page = document.metadata.get(
        "page",
        -1
    ) + 1

    print("\n----------------------------------------")

    print("Rank:", i + 1)
    print("Score:", float(score))
    print("Page:", page)

    print("----------------------------------------")

    print(
        document.page_content[:500]
    )