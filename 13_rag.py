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

print("Pages loaded:", len(documents))


# ==================================================
# 2. SPLIT INTO CHUNKS
# ==================================================

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = text_splitter.split_documents(documents)

print("Chunks created:", len(chunks))


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

print("Vector store created")


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
You are a helpful assistant.

Answer the question using ONLY the context provided below.

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
# 8. ASK QUESTION
# ==================================================

question = "What is a Python list?"


# ==================================================
# 9. RETRIEVE RELEVANT CHUNKS
# ==================================================

retrieved_documents = retriever.invoke(question)


# ==================================================
# 10. COMBINE CHUNKS INTO CONTEXT
# ==================================================

context = "\n\n".join(
    document.page_content
    for document in retrieved_documents
)


# ==================================================
# 11. CREATE PROMPT
# ==================================================

formatted_prompt = prompt.invoke(
    {
        "context": context,
        "question": question
    }
)


# ==================================================
# 12. SEND TO LOCAL LLM
# ==================================================

response = llm.invoke(formatted_prompt)


# ==================================================
# 13. DISPLAY ANSWER
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
# 14. DISPLAY SOURCES
# ==================================================

print("\n========================================")
print("SOURCES")
print("========================================")

for document in retrieved_documents:

    page = document.metadata.get("page", -1)

    print(
        "Page:",
        page + 1
    )