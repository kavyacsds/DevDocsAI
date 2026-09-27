from fastapi import (
    FastAPI,
    UploadFile,
    File,
    HTTPException
)

from fastapi.middleware.cors import CORSMiddleware

from pydantic import (
    BaseModel,
    Field
)

import os


from app.config import (
    UPLOAD_DIR,
    VECTORSTORE_DIR
)

from app.ingestion import (
    load_and_split_pdf
)

from app.embeddings import (
    get_embeddings
)

from app.vectorstore import (
    create_vectorstore,
    add_to_vectorstore,
    save_vectorstore,
    load_vectorstore
)

from app.rag import (
    create_rag_from_vectorstore,
    ask_question
)


# ==================================================
# CREATE FASTAPI APP
# ==================================================

app = FastAPI(
    title="DevDocs AI"
)


# ==================================================
# CORS
# ==================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==================================================
# CREATE UPLOAD DIRECTORY
# ==================================================

os.makedirs(
    UPLOAD_DIR,
    exist_ok=True
)


# ==================================================
# REQUEST MODELS
# ==================================================

class ChatMessage(BaseModel):

    role: str

    content: str


class ChatRequest(BaseModel):

    question: str

    chat_history: list[ChatMessage] = Field(
        default_factory=list
    )


# ==================================================
# STARTUP
# ==================================================

embeddings = get_embeddings()

vector_store = None

retriever = None

reranker = None

llm = None


# ==================================================
# LOAD EXISTING VECTOR STORE
# ==================================================

if os.path.exists(
    VECTORSTORE_DIR
):

    vector_store = load_vectorstore(
        VECTORSTORE_DIR,
        embeddings
    )

    retriever, reranker, llm = (
        create_rag_from_vectorstore(
            vector_store
        )
    )


# ==================================================
# ROOT ENDPOINT
# ==================================================

@app.get("/")
def root():

    return {
        "message": "DevDocs AI API is running"
    }


# ==================================================
# UPLOAD PDF
# ==================================================

@app.post("/upload")
async def upload_pdf(
    file: UploadFile = File(...)
):

    global vector_store
    global retriever
    global reranker
    global llm


    # --------------------------------------------------
    # VALIDATE FILE
    # --------------------------------------------------

    if not file.filename:

        raise HTTPException(
            status_code=400,
            detail="No file selected."
        )


    if not file.filename.lower().endswith(".pdf"):

        raise HTTPException(
            status_code=400,
            detail="Only PDF files are allowed."
        )


    # --------------------------------------------------
    # SAVE FILE
    # --------------------------------------------------

    file_path = os.path.join(
        UPLOAD_DIR,
        file.filename
    )


    contents = await file.read()


    with open(
        file_path,
        "wb"
    ) as f:

        f.write(contents)


    # --------------------------------------------------
    # LOAD + SPLIT
    # --------------------------------------------------

    try:

        chunks = load_and_split_pdf(
            file_path
        )

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"Failed to process PDF: {str(e)}"
        )


    # --------------------------------------------------
    # CREATE / UPDATE VECTOR STORE
    # --------------------------------------------------

    try:

        if vector_store is None:

            vector_store = create_vectorstore(
                chunks,
                embeddings
            )

        else:

            vector_store = add_to_vectorstore(
                vector_store,
                chunks
            )

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"Failed to create vector store: {str(e)}"
        )


    # --------------------------------------------------
    # SAVE VECTOR STORE
    # --------------------------------------------------

    try:

        save_vectorstore(
            vector_store,
            VECTORSTORE_DIR
        )

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"Failed to save vector store: {str(e)}"
        )


    # --------------------------------------------------
    # UPDATE ACTIVE RAG COMPONENTS
    # --------------------------------------------------

    try:

        retriever, reranker, llm = (
            create_rag_from_vectorstore(
                vector_store
            )
        )

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"Failed to initialize RAG: {str(e)}"
        )


    # --------------------------------------------------
    # RESPONSE
    # --------------------------------------------------

    return {
        "message": "PDF added successfully.",
        "filename": file.filename,
        "chunks_added": len(chunks)
    }


# ==================================================
# CHAT
# ==================================================

@app.post("/chat")
def chat(
    request: ChatRequest
):

    # --------------------------------------------------
    # VALIDATE QUESTION
    # --------------------------------------------------

    if not request.question.strip():

        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty."
        )


    # --------------------------------------------------
    # CHECK DOCUMENTS
    # --------------------------------------------------

    if retriever is None:

        raise HTTPException(
            status_code=400,
            detail="No documents have been uploaded yet."
        )


    # --------------------------------------------------
    # ASK QUESTION
    # --------------------------------------------------

    try:

        answer, documents, standalone_question = (
            ask_question(
                request.question,
                request.chat_history,
                retriever,
                reranker,
                llm
            )
        )

    except Exception:

        raise HTTPException(
            status_code=500,
            detail="Failed to generate the answer."
        )


    # --------------------------------------------------
    # BUILD SOURCES
    # --------------------------------------------------

    sources = []

    seen_sources = set()


    for document in documents:

        source = document.metadata.get(
            "source",
            "unknown"
        )


        page = document.metadata.get(
            "page",
            -1
        ) + 1


        key = (
            source,
            page
        )


        if key not in seen_sources:

            sources.append(
                {
                    "file": source,
                    "page": page
                }
            )

            seen_sources.add(
                key
            )


    # --------------------------------------------------
    # RESPONSE
    # --------------------------------------------------

    return {
        "question": request.question,
        "rewritten_question": standalone_question,
        "answer": answer,
        "sources": sources
    }