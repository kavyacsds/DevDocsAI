from app.ingestion import load_and_split_pdf


chunks = load_and_split_pdf(
    "data/documents/python.pdf"
)


for chunk in chunks[:5]:

    print("\n==============================")

    print("Metadata:")
    print(chunk.metadata)

    print("\nContent:")
    print(chunk.page_content)