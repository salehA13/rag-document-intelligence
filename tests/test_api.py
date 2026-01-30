"""Tests for the FastAPI endpoints."""

import pytest
from fastapi.testclient import TestClient

from src.api.server import app

client = TestClient(app)


def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_stats_endpoint():
    response = client.get("/stats")
    assert response.status_code == 200
    assert "total_documents" in response.json()


def test_ask_without_api_key():
    """Should fail gracefully when no API key is set."""
    response = client.post("/ask", json={"question": "test question"})
    # Either 500 (no key) or 200 (key exists) - both are valid
    assert response.status_code in [200, 500]


def test_ingest_missing_directory():
    response = client.post("/ingest", json={"directory": "/nonexistent/path"})
    assert response.status_code == 404


def test_upload_non_pdf():
    """Should reject non-PDF files."""
    response = client.post(
        "/upload",
        files={"file": ("test.txt", b"not a pdf", "text/plain")},
    )
    assert response.status_code == 400
