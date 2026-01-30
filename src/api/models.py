"""Pydantic models for API request and response validation."""

from __future__ import annotations

from typing import List, Union

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Requests
# ---------------------------------------------------------------------------


class QuestionRequest(BaseModel):
    """Payload for the /ask and /search endpoints."""

    question: str = Field(
        ..., min_length=1, max_length=2000, description="Question to ask"
    )
    top_k: int = Field(
        default=10, ge=1, le=50, description="Number of candidates to retrieve"
    )
    rerank_k: int = Field(
        default=5, ge=1, le=20, description="Number of results after re-ranking"
    )


class IngestRequest(BaseModel):
    """Payload for the /ingest endpoint."""

    directory: str = Field(
        default="./docs", description="Directory containing PDFs to ingest"
    )


# ---------------------------------------------------------------------------
# Responses
# ---------------------------------------------------------------------------


class SourceInfo(BaseModel):
    """A single cited source (filename + page number)."""

    filename: str
    page: Union[int, str]


class AnswerResponse(BaseModel):
    """Response from the /ask endpoint with a generated answer and sources."""

    answer: str
    sources: List[SourceInfo]
    num_sources: int


class IngestResponse(BaseModel):
    """Response from the /ingest endpoint with ingestion statistics."""

    total_chunks: int
    new_chunks: int
    duplicates_skipped: int
    message: str


class StatsResponse(BaseModel):
    """Response from the /stats endpoint."""

    total_documents: int
    persist_dir: str


class HealthResponse(BaseModel):
    """Response from the /health endpoint."""

    status: str = "healthy"
    version: str = "1.0.0"


class UploadResponse(BaseModel):
    """Response from the /upload endpoint after PDF processing."""

    filename: str
    total_chunks: int
    new_chunks: int
    message: str
