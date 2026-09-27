from langchain_community.vectorstores import FAISS


def create_vectorstore(chunks, embeddings):

    return FAISS.from_documents(
        chunks,
        embeddings
    )


def add_to_vectorstore(
    vector_store,
    chunks
):

    vector_store.add_documents(
        chunks
    )

    return vector_store


def save_vectorstore(
    vector_store,
    path: str
):

    vector_store.save_local(
        path
    )


def load_vectorstore(
    path: str,
    embeddings
):

    return FAISS.load_local(
        path,
        embeddings,
        allow_dangerous_deserialization=True
    )