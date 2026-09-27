from app.embeddings import get_embeddings
from app.vectorstore import load_vectorstore
from app.retriever import get_retriever
from app.llm import get_llm

from langchain_core.prompts import ChatPromptTemplate


# ==================================================
# 1. TEST QUESTIONS
# ==================================================

evaluation_data = [
    {
        "question": "What is a Python list?",
        "expected_keywords": ["list"]
    },
    {
        "question": "What is a tuple in Python?",
        "expected_keywords": ["tuple"]
    },
    {
        "question": "What is a dictionary in Python?",
        "expected_keywords": ["dictionary"]
    }
]


# ==================================================
# 2. LOAD RAG COMPONENTS
# ==================================================

embeddings = get_embeddings()

vector_store = load_vectorstore(
    "faiss_index",
    embeddings
)

retriever = get_retriever(
    vector_store
)

llm = get_llm()


# ==================================================
# 3. PROMPT
# ==================================================

prompt = ChatPromptTemplate.from_template(
    """
Answer the question using ONLY the provided context.

If the answer is not present in the context,
say:

"I don't know based on the provided document."

Context:
{context}

Question:
{question}

Answer:
"""
)


# ==================================================
# 4. RUN EVALUATION
# ==================================================

passed = 0
total = len(evaluation_data)


for item in evaluation_data:

    question = item["question"]

    expected_keywords = item["expected_keywords"]


    # --------------------------------------------------
    # RETRIEVE
    # --------------------------------------------------

    documents = retriever.invoke(
        question
    )


    # --------------------------------------------------
    # CREATE CONTEXT
    # --------------------------------------------------

    context = "\n\n".join(
        document.page_content
        for document in documents
    )


    # --------------------------------------------------
    # ASK LLM
    # --------------------------------------------------

    formatted_prompt = prompt.invoke(
        {
            "context": context,
            "question": question
        }
    )

    response = llm.invoke(
        formatted_prompt
    )

    answer = response.content


    # --------------------------------------------------
    # SIMPLE CHECK
    # --------------------------------------------------

    answer_lower = answer.lower()

    matched = all(
        keyword.lower() in answer_lower
        for keyword in expected_keywords
    )


    if matched:
        passed += 1


    # --------------------------------------------------
    # DISPLAY
    # --------------------------------------------------

    print("\n========================================")

    print("Question:")
    print(question)

    print("\nAnswer:")
    print(answer)

    print("\nResult:")

    if matched:
        print("PASS")
    else:
        print("FAIL")


# ==================================================
# 5. SUMMARY
# ==================================================

accuracy = (
    passed / total
) * 100


print("\n========================================")
print("EVALUATION SUMMARY")
print("========================================")

print("Passed:", passed)
print("Total:", total)

print(
    f"Keyword-based score: {accuracy:.2f}%"
)