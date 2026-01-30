from __future__ import annotations
"""FastAPI server for the RAG Document Intelligence System."""

import shutil
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from src.api.models import (
    QuestionRequest,
    IngestRequest,
    AnswerResponse,
    IngestResponse,
    StatsResponse,
    HealthResponse,
    UploadResponse,
)
from src.config import settings
from src.ingestion.loader import load_pdf, load_directory, chunk_documents
from src.ingestion.embedder import ingest_documents, get_collection_stats
from src.search.qa import ask

app = FastAPI(
    title="RAG Document Intelligence",
    description="Hybrid semantic + keyword search with re-ranking over your documents",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse()


@app.get("/stats", response_model=StatsResponse)
async def collection_stats():
    """Get vector store statistics."""
    return StatsResponse(**get_collection_stats())


@app.post("/ingest", response_model=IngestResponse)
async def ingest_directory(request: IngestRequest):
    """Ingest all PDFs from a directory."""
    dir_path = Path(request.directory)
    if not dir_path.exists():
        raise HTTPException(status_code=404, detail=f"Directory not found: {request.directory}")

    pdfs = list(dir_path.glob("*.pdf"))
    if not pdfs:
        raise HTTPException(status_code=400, detail="No PDF files found in directory")

    documents = load_directory(dir_path)
    chunks = chunk_documents(documents)
    stats = ingest_documents(chunks)

    return IngestResponse(
        **stats,
        message=f"Ingested {len(pdfs)} PDF(s) from {request.directory}",
    )


@app.post("/upload", response_model=UploadResponse)
async def upload_pdf(file: UploadFile = File(...)):
    """Upload and ingest a single PDF."""
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted")

    # Save upload
    upload_path = settings.upload_path / file.filename
    with open(upload_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # Process
    documents = load_pdf(upload_path)
    chunks = chunk_documents(documents)
    stats = ingest_documents(chunks)

    return UploadResponse(
        filename=file.filename,
        total_chunks=stats["total_chunks"],
        new_chunks=stats["new_chunks"],
        message=f"Uploaded and ingested {file.filename}",
    )


@app.post("/ask", response_model=AnswerResponse)
async def ask_question(request: QuestionRequest):
    """Ask a question against ingested documents."""
    if not settings.openai_api_key:
        raise HTTPException(
            status_code=500,
            detail="OPENAI_API_KEY not configured. Set it in your .env file.",
        )

    try:
        result = ask(
            question=request.question,
            top_k=request.top_k,
            rerank_k=request.rerank_k,
        )
        return AnswerResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/search")
async def search_documents(request: QuestionRequest):
    """Search documents without generating an answer (retrieval only)."""
    from src.search.hybrid import hybrid_search

    results = hybrid_search(
        query=request.question,
        top_k=request.top_k,
        rerank_k=request.rerank_k,
    )

    return {
        "query": request.question,
        "num_results": len(results),
        "results": [
            {
                "content": doc.page_content[:500],
                "metadata": doc.metadata,
            }
            for doc in results
        ],
    }
