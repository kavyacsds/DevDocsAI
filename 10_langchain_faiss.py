from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


# ==================================================
# 1. LOAD PDF
# ==================================================

pdf_path = "data/documents/python.pdf"

loader = PyMuPDFLoader(pdf_path)
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
# 3. CREATE EMBEDDING MODEL
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

print("FAISS vector store created!")


# ==================================================
# 5. ASK A QUESTION
# ==================================================

query = "What is a Python list?"


# ==================================================
# 6. SEARCH FOR SIMILAR DOCUMENTS
# ==================================================

results = vector_store.similarity_search(
    query,
    k=3
)


# ==================================================
# 7. DISPLAY RESULTS
# ==================================================

print("\n========================================")
print("QUERY")
print("========================================")

print(query)


print("\n========================================")
print("RETRIEVED CHUNKS")
print("========================================")


for i, document in enumerate(results):

    print("\n----------------------------------------")
    print("Result:", i + 1)
    print("Page:", document.metadata.get("page", -1) + 1)
    print("----------------------------------------")

    print(document.page_content)