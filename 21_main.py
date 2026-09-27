from app.rag import create_rag, ask_question


retriever, llm = create_rag()


print("\n========================================")
print("DEV DOCS AI")
print("========================================")

print("Ask questions about your PDF.")
print("Type 'exit' to stop.\n")


while True:

    question = input("You: ")

    if question.lower() == "exit":
        break

    answer, documents = ask_question(
        question,
        retriever,
        llm
    )

    print("\nAI:")
    print(answer)

    print("\nSources:")

    seen_pages = set()

    for document in documents:

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