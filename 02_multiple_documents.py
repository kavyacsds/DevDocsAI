from langchain_core.documents import Document


documents = [
    Document(
        page_content="Python is a high-level programming language.",
        metadata={
            "source": "python.pdf",
            "page": 1
        }
    ),

    Document(
        page_content="JavaScript is commonly used for web development.",
        metadata={
            "source": "javascript.pdf",
            "page": 5
        }
    ),

    Document(
        page_content="React is a JavaScript library for building user interfaces.",
        metadata={
            "source": "react.pdf",
            "page": 10
        }
    )
]


print("Number of documents:", len(documents))

for document in documents:
    print("\n--------------------")
    print("Content:", document.page_content)
    print("Metadata:", document.metadata)
print("\nFirst document:")
print(documents[0])