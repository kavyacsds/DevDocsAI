from app.embeddings import get_embeddings
from app.vectorstore import load_vectorstore
from app.retriever import get_filtered_retriever


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
# 3. CREATE FILTERED RETRIEVER
# ==================================================

retriever = get_filtered_retriever(
    vector_store,
    "python.pdf"
)


# ==================================================
# 4. SEARCH
# ==================================================

query = "What is a Python list?"

results = retriever.invoke(
    query
)


# ==================================================
# 5. DISPLAY
# ==================================================

print("\nQuery:")
print(query)

print("\nResults:")

for i, document in enumerate(results):

    print("\n------------------------------")

    print("Result:", i + 1)

    print(
        "Source:",
        document.metadata.get(
            "source",
            "unknown"
        )
    )

    print(
        "Page:",
        document.metadata.get(
            "page",
            -1
        ) + 1
    )

    print("\nContent:")
    print(document.page_content)