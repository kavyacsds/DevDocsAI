from app.config import RETRIEVAL_K


def get_retriever(vector_store):

    return vector_store.as_retriever(
        search_kwargs={
            "k": RETRIEVAL_K
        }
    )


def get_filtered_retriever(
    vector_store,
    source: str
):

    return vector_store.as_retriever(
        search_kwargs={
            "k": RETRIEVAL_K,
            "filter": {
                "source": source
            }
        }
    )


def get_mmr_retriever(vector_store):

    return vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": RETRIEVAL_K,
            "fetch_k": 10
        }
    )