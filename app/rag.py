from langchain_core.prompts import ChatPromptTemplate

from app.retriever import get_retriever
from app.llm import get_llm
from app.reranker import (
    get_reranker,
    rerank_documents
)


# ==================================================
# QUERY REWRITING PROMPT
# ==================================================

REWRITE_PROMPT = ChatPromptTemplate.from_template(
    """
Rewrite the latest question as a standalone question.

Use the conversation history to understand references
such as "it", "they", "that", or "what about it".

Do NOT answer the question.

Conversation history:
{chat_history}

Latest question:
{question}

Standalone question:
"""
)


# ==================================================
# ANSWER PROMPT
# ==================================================

ANSWER_PROMPT = ChatPromptTemplate.from_template(
    """
You are a document question-answering assistant.

Answer the question using ONLY the provided context.

If the answer is not present in the context, say:

"I don't know based on the provided document."

Do not use outside knowledge.

Context:
{context}

Question:
{question}

Answer:
"""
)


# ==================================================
# CREATE RAG
# ==================================================

def create_rag_from_vectorstore(
    vector_store
):

    retriever = get_retriever(
        vector_store
    )

    reranker = get_reranker()

    llm = get_llm()

    return retriever, reranker, llm


# ==================================================
# ASK QUESTION
# ==================================================

def ask_question(
    question: str,
    chat_history,
    retriever,
    reranker,
    llm
):

    # --------------------------------------------------
    # 1. Convert history to text
    # --------------------------------------------------

    history_text = "\n".join(
        f"{message.role}: {message.content}"
        for message in chat_history
    )


    # --------------------------------------------------
    # 2. Rewrite question
    # --------------------------------------------------

    rewrite_input = REWRITE_PROMPT.invoke(
        {
            "chat_history": history_text,
            "question": question
        }
    )

    rewrite_response = llm.invoke(
        rewrite_input
    )

    standalone_question = (
        rewrite_response.content.strip()
    )


    # --------------------------------------------------
    # 3. Retrieve
    # --------------------------------------------------

    documents = retriever.invoke(
        standalone_question
    )


    # --------------------------------------------------
    # 4. Rerank
    # --------------------------------------------------

    documents = rerank_documents(
        standalone_question,
        documents,
        reranker,
        top_k=3
    )


    # --------------------------------------------------
    # 5. Create context
    # --------------------------------------------------

    context = "\n\n".join(
        document.page_content
        for document in documents
    )


    # --------------------------------------------------
    # 6. Create answer prompt
    # --------------------------------------------------

    answer_input = ANSWER_PROMPT.invoke(
        {
            "context": context,
            "question": standalone_question
        }
    )


    # --------------------------------------------------
    # 7. Generate answer
    # --------------------------------------------------

    response = llm.invoke(
        answer_input
    )


    return (
        response.content,
        documents,
        standalone_question
    )