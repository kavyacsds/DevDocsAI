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
# 3. QUESTION
# ==================================================

query = "What is a Python list?"


# ==================================================
# 4. SEARCH WITH SCORES
# ==================================================

results = vector_store.similarity_search_with_score(
    query,
    k=5
)


# ==================================================
# 5. DISPLAY RESULTS
# ==================================================

print("\n========================================")
print("QUERY")
print("========================================")

print(query)


print("\n========================================")
print("RESULTS")
print("========================================")


for i, (document, score) in enumerate(results):

    page = document.metadata.get(
        "page",
        -1
    ) + 1

    print("\n----------------------------------------")

    print("Result:", i + 1)
    print("Score:", score)
    print("Page:", page)

    print("----------------------------------------")

    print(document.page_content[:500])