from langchain_community.document_loaders import PyMuPDFLoader


pdf_path = "data/documents/python.pdf"

loader = PyMuPDFLoader(pdf_path)

documents = loader.load()


print("Number of documents:", len(documents))

print("\nFirst document:")
print(documents[0])

print("\nFirst document content:")
print(documents[0].page_content[:1000])

print("\nFirst document metadata:")
print(documents[0].metadata)


for i, document in enumerate(documents[:3]):
    print("\n==============================")
    print("Document index:", i)
    print("Metadata:", document.metadata)
    print("Characters:", len(document.page_content))
    print("Content preview:")
    print(document.page_content[:300])