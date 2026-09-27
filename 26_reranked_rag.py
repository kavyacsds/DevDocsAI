from sentence_transformers import CrossEncoder

from app.embeddings import get_embeddings
from app.vectorstore import load_vectorstore
from app.llm import get_llm

from langchain_core.prompts import ChatPromptTemplate


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
# 3. RETRIEVER
# ==================================================

retriever = vector_store.as_retriever(
    search_kwargs={
        "k": 10
    }
)


# ==================================================
# 4. RERANKER
# ==================================================

reranker = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)


# ==================================================
# 5. LLM
# ==================================================

llm = get_llm()


# ==================================================
# 6. PROMPT
# ==================================================

prompt = ChatPromptTemplate.from_template(
    """
You are a document question-answering assistant.

Answer using ONLY the provided context.

If the answer is not present in the context, say:
"I don't know based on the provided document."

Context:
{context}

Question:
{question}

Answer:
"""
)


# ==================================================
# 7. QUESTION
# ==================================================

question = "What is a Python list?"


# ==================================================
# 8. FIRST RETRIEVAL
# ==================================================

documents = retriever.invoke(
    question
)


# ==================================================
# 9. RERANK
# ==================================================

pairs = [
    (question, document.page_content)
    for document in documents
]

scores = reranker.predict(
    pairs
)

ranked_documents = sorted(
    zip(documents, scores),
    key=lambda x: x[1],
    reverse=True
)


# ==================================================
# 10. KEEP TOP 3
# ==================================================

top_documents = [
    document
    for document, score
    in ranked_documents[:3]
]


# ==================================================
# 11. BUILD CONTEXT
# ==================================================

context = "\n\n".join(
    document.page_content
    for document in top_documents
)


# ==================================================
# 12. CREATE PROMPT
# ==================================================

formatted_prompt = prompt.invoke(
    {
        "context": context,
        "question": question
    }
)


# ==================================================
# 13. GENERATE ANSWER
# ==================================================

response = llm.invoke(
    formatted_prompt
)


# ==================================================
# 14. DISPLAY
# ==================================================

print("\n========================================")
print("QUESTION")
print("========================================")

print(question)


print("\n========================================")
print("ANSWER")
print("========================================")

print(response.content)


# ==================================================
# 15. SOURCES
# ==================================================

print("\n========================================")
print("SOURCES")
print("========================================")

seen_pages = set()

for document in top_documents:

    page = document.metadata.get(
        "page",
        -1
    ) + 1

    if page not in seen_pages:

        print(
            f"- Page {page}"
        )

        seen_pages.add(page)