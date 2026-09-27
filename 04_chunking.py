from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


# 1. Load the PDF
pdf_path = "data/documents/python.pdf"

loader = PyMuPDFLoader(pdf_path)

documents = loader.load()

print("Number of pages:", len(documents))


# 2. Create the text splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1500,
    chunk_overlap=300
)


# 3. Split the documents
chunks = text_splitter.split_documents(documents)


# 4. Print results
# print("Number of chunks:", len(chunks))

# print("\nFirst chunk:")
# print(chunks[0].page_content)

# print("\nFirst chunk metadata:")
# print(chunks[0].metadata)

# print("\nFirst chunk length:")
# print(len(chunks[0].page_content))
for i, chunk in enumerate(chunks[:5]):

    print("\n==============================")
    print("Chunk:", i)
    print("Length:", len(chunk.page_content))
    print("Metadata:", chunk.metadata)
    print("Content:")
    print(chunk.page_content)