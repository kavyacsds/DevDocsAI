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
# 2. SPLIT PDF INTO CHUNKS
# ==================================================

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = text_splitter.split_documents(documents)


# ==================================================
# 3. CREATE EMBEDDINGS
# ==================================================

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# ==================================================
# 4. CREATE FAISS VECTOR STORE
# ==================================================

vector_store = FAISS.from_documents(
    chunks,
    embeddings
)


# ==================================================
# 5. CREATE RETRIEVER
# ==================================================

retriever = vector_store.as_retriever(
    search_kwargs={"k": 3}
)


# ==================================================
# 6. CREATE OLLAMA
# ==================================================

llm = ChatOllama(
    model="llama3.2",
    temperature=0
)


# ==================================================
# 7. CHAT HISTORY
# ==================================================

chat_history = []


# ==================================================
# 8. PROMPT
# ==================================================

prompt = ChatPromptTemplate.from_template(
    """
You are a document question-answering assistant.

Use the conversation history and the provided context
to answer the user's question.

Answer ONLY using information from the provided context.

If the answer is not present in the context, say:
"I don't know based on the provided document."

Conversation history:
{chat_history}

Context:
{context}

Current question:
{question}

Answer:
"""
)


# ==================================================
# 9. CHAT LOOP
# ==================================================

print("\n========================================")
print("DEV DOCS AI - CONVERSATIONAL RAG")
print("========================================")

print("Ask questions about the Python PDF.")
print("Type 'exit' to stop.\n")


while True:

    question = input("You: ")

    if question.lower() == "exit":
        break


    # ==================================================
    # RETRIEVE DOCUMENTS
    # ==================================================

    retrieved_documents = retriever.invoke(
        question
    )


    # ==================================================
    # CREATE CONTEXT
    # ==================================================

    context = "\n\n".join(
        document.page_content
        for document in retrieved_documents
    )


    # ==================================================
    # CONVERT HISTORY TO TEXT
    # ==================================================

    history_text = "\n".join(
        f"User: {user}\nAI: {ai}"
        for user, ai in chat_history
    )


    # ==================================================
    # CREATE PROMPT
    # ==================================================

    formatted_prompt = prompt.invoke(
        {
            "chat_history": history_text,
            "context": context,
            "question": question
        }
    )


    # ==================================================
    # GENERATE ANSWER
    # ==================================================

    response = llm.invoke(
        formatted_prompt
    )

    answer = response.content


    # ==================================================
    # SAVE CONVERSATION
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