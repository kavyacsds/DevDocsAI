from app.embeddings import get_embeddings
from app.vectorstore import load_vectorstore
from app.retriever import get_mmr_retriever


# ==================================================
# LOAD
# ==================================================

embeddings = get_embeddings()

vector_store = load_vectorstore(
    "faiss_index",
    embeddings
)


# ==================================================
# CREATE MMR RETRIEVER
# ==================================================

retriever = get_mmr_retriever(
    vector_store
)


# ==================================================
# QUERY
# ==================================================

query = "What is a Python list?"


# ==================================================
# RETRIEVE
# ==================================================

results = retriever.invoke(
    query
)


# ==================================================
# DISPLAY
# ==================================================

print("\nQuery:")
print(query)

print("\nMMR Results:")


for i, document in enumerate(results):

    page = document.metadata.get(
        "page",
        -1
    ) + 1

    print("\n--------------------------------")

    print("Result:", i + 1)
    print("Page:", page)

    print("--------------------------------")

    print(document.page_content[:500])