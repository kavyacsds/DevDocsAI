from sentence_transformers import CrossEncoder

from app.config import RERANK_TOP_K


def get_reranker():

    return CrossEncoder(
        "cross-encoder/ms-marco-MiniLM-L-6-v2"
    )


def rerank_documents(
    question: str,
    documents,
    reranker,
    top_k: int = RERANK_TOP_K
):

    pairs = [
        (
            question,
            document.page_content
        )
        for document in documents
    ]

    scores = reranker.predict(
        pairs
    )

    ranked = sorted(
        zip(documents, scores),
        key=lambda item: item[1],
        reverse=True
    )

    return [
        document
        for document, score
        in ranked[:top_k]
    ]