"""Pydantic models for API request/response."""

from typing import List, Union
from pydantic import BaseModel, Field


# --- Requests ---

class QuestionRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000, description="Question to ask")
    top_k: int = Field(default=10, ge=1, le=50, description="Number of candidates to retrieve")
    rerank_k: int = Field(default=5, ge=1, le=20, description="Number of results after re-ranking")


class IngestRequest(BaseModel):
    directory: str = Field(default="./docs", description="Directory containing PDFs to ingest")


# --- Responses ---

class SourceInfo(BaseModel):
    filename: str
    page: Union[int, str]


class AnswerResponse(BaseModel):
    answer: str
    sources: List[SourceInfo]
    num_sources: int


class IngestResponse(BaseModel):
    total_chunks: int
    new_chunks: int
    duplicates_skipped: int
    message: str


class StatsResponse(BaseModel):
    total_documents: int
    persist_dir: str


class HealthResponse(BaseModel):
    status: str = "healthy"
    version: str = "1.0.0"


class UploadResponse(BaseModel):
    filename: str
    total_chunks: int
    new_chunks: int
    message: str
