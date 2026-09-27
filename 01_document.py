from langchain_core.documents import Document


document = Document(
    page_content="LangChain is a framework for building applications powered by language models.",
    metadata={
        "source": "langchain.pdf",
        "page": 25
    }
)


print("Content:")
print(document.page_content)

print("\nMetadata:")
print(document.metadata)

print(type(document))