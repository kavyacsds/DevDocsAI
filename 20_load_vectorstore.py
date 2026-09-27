from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


# ==================================================
# 1. CREATE SAME EMBEDDING MODEL
# ==================================================

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# ==================================================
# 2. LOAD SAVED FAISS INDEX
# ==================================================

vector_store = FAISS.load_local(
    "faiss_index",
    embeddings,
    allow_dangerous_deserialization=True
)

print("FAISS index loaded!")


# ==================================================
# 3. CREATE RETRIEVER
# ==================================================

retriever = vector_store.as_retriever(
    search_kwargs={"k": 3}
)


# ==================================================
# 4. ASK QUESTION
# ==================================================

question = "What is a Python list?"


# ==================================================
# 5. RETRIEVE
# ==================================================

results = retriever.invoke(question)


# ==================================================
# 6. DISPLAY RESULTS
# ==================================================

print("\nQuestion:")
print(question)

print("\nResults:")

for i, document in enumerate(results):

    page = document.metadata.get("page", -1) + 1

    print("\n------------------------------")
    print("Result:", i + 1)
    print("Page:", page)
    print("------------------------------")

    print(document.page_content)