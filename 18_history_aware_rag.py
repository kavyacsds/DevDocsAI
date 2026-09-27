from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate


# ==================================================
# 1. LOAD PDF
# ==================================================

loader = PyMuPDFLoader(
    "data/documents/python.pdf"
)

documents = loader.load()


# ==================================================
# 2. SPLIT INTO CHUNKS
# ==================================================

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = text_splitter.split_documents(documents)


# ==================================================
# 3. EMBEDDINGS
# ==================================================

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# ==================================================
# 4. FAISS
# ==================================================

vector_store = FAISS.from_documents(
    chunks,
    embeddings
)


# ==================================================
# 5. NORMAL RETRIEVER
# ==================================================

retriever = vector_store.as_retriever(
    search_kwargs={"k": 3}
)


# ==================================================
# 6. OLLAMA
# ==================================================

llm = ChatOllama(
    model="llama3.2",
    temperature=0
)


# ==================================================
# 7. QUERY REWRITING PROMPT
# ==================================================

rewrite_prompt = ChatPromptTemplate.from_template(
    """
Given the conversation history and the latest user question,
rewrite the latest question as a standalone question.

The rewritten question must be understandable without
the conversation history.

Do NOT answer the question.

Conversation history:
{chat_history}

Latest question:
{question}

Standalone question:
"""
)


# ==================================================
# 8. ANSWER PROMPT
# ==================================================

answer_prompt = ChatPromptTemplate.from_template(
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
# 9. CHAT HISTORY
# ==================================================

chat_history = []


# ==================================================
# 10. CHAT LOOP
# ==================================================

print("\n========================================")
print("DEV DOCS AI")
print("========================================")

print("Type 'exit' to stop.\n")


while True:

    question = input("You: ")

    if question.lower() == "exit":
        break


    # ==================================================
    # CREATE HISTORY TEXT
    # ==================================================

    history_text = "\n".join(
        f"User: {user}\nAI: {ai}"
        for user, ai in chat_history
    )


    # ==================================================
    # REWRITE QUESTION
    # ==================================================

    rewrite_input = rewrite_prompt.invoke(
        {
            "chat_history": history_text,
            "question": question
        }
    )

    rewritten_response = llm.invoke(
        rewrite_input
    )

    standalone_question = rewritten_response.content.strip()


    # ==================================================
    # SHOW REWRITTEN QUESTION
    # ==================================================

    print("\nRewritten query:")
    print(standalone_question)


    # ==================================================
    # RETRIEVE USING REWRITTEN QUESTION
    # ==================================================

    retrieved_documents = retriever.invoke(
        standalone_question
    )


    # ==================================================
    # CREATE CONTEXT
    # ==================================================

    context = "\n\n".join(
        document.page_content
        for document in retrieved_documents
    )


    # ==================================================
    # GENERATE ANSWER
    # ==================================================

    answer_input = answer_prompt.invoke(
        {
            "context": context,
            "question": standalone_question
        }
    )

    response = llm.invoke(
        answer_input
    )

    answer = response.content


    # ==================================================
    # SAVE HISTORY
    # ==================================================

    chat_history.append(
        (question, answer)
    )


    # ==================================================
    # DISPLAY ANSWER
    # ==================================================

    print("\nAI:")
    print(answer)


    # ==================================================
    # DISPLAY SOURCES
    # ==================================================

    print("\nSources:")

    seen_pages = set()

    for document in retrieved_documents:

        page = document.metadata.get(
            "page",
            -1
        ) + 1

        if page not in seen_pages:

            print(
                f"- python.pdf, Page {page}"
            )

            seen_pages.add(page)

    print()