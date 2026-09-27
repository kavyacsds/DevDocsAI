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
# 3. CREATE EMBEDDINGS
# ==================================================

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# ==================================================
# 4. CREATE VECTOR STORE
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
# 6. CREATE LOCAL LLM
# ==================================================

llm = ChatOllama(
    model="llama3.2",
    temperature=0
)


# ==================================================
# 7. CREATE PROMPT
# ==================================================

prompt = ChatPromptTemplate.from_template(
    """
You are a document question-answering assistant.

Answer the question using ONLY the provided context.

If the answer is not present in the context, say:
"I don't know based on the provided document."

Do NOT mention page numbers.
Do NOT create sources.
Just answer the question.

Context:
{context}

Question:
{question}

Answer:
"""
)


# ==================================================
# 8. QUESTION
# ==================================================

question = "What is a Python list?"


# ==================================================
# 9. RETRIEVE DOCUMENTS
# ==================================================

retrieved_documents = retriever.invoke(question)


# ==================================================
# 10. BUILD CONTEXT
# ==================================================

context = "\n\n".join(
    document.page_content
    for document in retrieved_documents
)


# ==================================================
# 11. SEND TO LLM
# ==================================================

formatted_prompt = prompt.invoke(
    {
        "context": context,
        "question": question
    }
)

response = llm.invoke(formatted_prompt)


# ==================================================
# 12. DISPLAY ANSWER
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
# 13. GET SOURCES FROM RETRIEVED DOCUMENTS
# ==================================================

print("\n========================================")
print("SOURCES")
print("========================================")

seen_pages = set()

for document in retrieved_documents:

    page = document.metadata.get("page", -1) + 1

    if page not in seen_pages:

        print(
            f"- python.pdf, Page {page}"
        )

        seen_pages.add(page)