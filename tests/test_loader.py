"""Tests for document loading and chunking."""

import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
from langchain.schema import Document

from src.ingestion.loader import chunk_documents


@pytest.fixture
def sample_documents():
    """Create sample documents for testing."""
    return [
        Document(
            page_content="This is a test document about machine learning. " * 50,
            metadata={"source": "/tmp/test.pdf", "page": 0},
        ),
        Document(
            page_content="Neural networks are computational models inspired by biological brains. " * 50,
            metadata={"source": "/tmp/test.pdf", "page": 1},
        ),
    ]


def test_chunk_documents_produces_chunks(sample_documents):
    """Chunking should produce multiple smaller chunks."""
    chunks = chunk_documents(sample_documents, chunk_size=200, chunk_overlap=50)
    assert len(chunks) > len(sample_documents)
    for chunk in chunks:
        assert len(chunk.page_content) <= 250  # Allow some flex


def test_chunk_metadata_enrichment(sample_documents):
    """Chunks should have enriched metadata."""
    chunks = chunk_documents(sample_documents, chunk_size=200, chunk_overlap=50)
    for chunk in chunks:
        assert "chunk_id" in chunk.metadata
        assert "char_count" in chunk.metadata
        assert "filename" in chunk.metadata
        assert chunk.metadata["filename"] == "test.pdf"


def test_chunk_empty_input():
    """Empty input should return empty output."""
    chunks = chunk_documents([])
    assert chunks == []
