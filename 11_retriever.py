from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


# ==================================================
# 1. LOAD PDF
# ==================================================

loader = PyMuPDFLoader(
    "data/documents/python.pdf"
)

documents = loader.load()

print("Number of pages:", len(documents))


# ==================================================
# 2. SPLIT INTO CHUNKS
# ==================================================

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = text_splitter.split_documents(documents)

print("Number of chunks:", len(chunks))


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

print("Vector store created!")


# ==================================================
# 5. CREATE RETRIEVER
# ==================================================

retriever = vector_store.as_retriever(
    search_kwargs={"k": 3}
)


# ==================================================
# 6. ASK QUESTION
# ==================================================

query = "What is a Python list?"


# ==================================================
# 7. RETRIEVE RELEVANT DOCUMENTS
# ==================================================

results = retriever.invoke(query)


# ==================================================
# 8. DISPLAY RESULTS
# ==================================================

print("\n========================================")
print("QUERY")
print("========================================")

print(query)


print("\n========================================")
print("RETRIEVED DOCUMENTS")
print("========================================")


for i, document in enumerate(results):

    print("\n----------------------------------------")
    print("Result:", i + 1)

    page = document.metadata.get("page", -1)
    print("Page:", page + 1)

    print("----------------------------------------")

    print(document.page_content)