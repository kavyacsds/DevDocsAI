from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter
)

import os

from app.config import (
    CHUNK_SIZE,
    CHUNK_OVERLAP
)


def load_and_split_pdf(
    pdf_path: str
):

    loader = PyMuPDFLoader(
        pdf_path
    )

    documents = loader.load()

    filename = os.path.basename(
        pdf_path
    )

    for document in documents:

        document.metadata["source"] = (
            filename
        )

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )

    chunks = text_splitter.split_documents(
        documents
    )

    return chunks