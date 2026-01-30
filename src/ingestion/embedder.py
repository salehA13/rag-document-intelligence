from __future__ import annotations
"""Vector store management with ChromaDB."""

import hashlib
from pathlib import Path

from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.schema import Document

from src.config import settings


def get_embeddings() -> OpenAIEmbeddings:
    """Create OpenAI embeddings instance."""
    return OpenAIEmbeddings(
        model=settings.embedding_model,
        openai_api_key=settings.openai_api_key,
    )


def get_vector_store() -> Chroma:
    """Get or create the ChromaDB vector store."""
    return Chroma(
        collection_name="documents",
        embedding_function=get_embeddings(),
        persist_directory=str(settings.chroma_path),
    )


def _doc_hash(doc: Document) -> str:
    """Generate a stable hash for deduplication."""
    content = doc.page_content + str(doc.metadata.get("source", ""))
    return hashlib.sha256(content.encode()).hexdigest()[:16]


def ingest_documents(chunks: list[Document]) -> dict:
    """Embed and store document chunks. Returns ingestion stats."""
    store = get_vector_store()

    # Deduplicate by content hash
    ids = [_doc_hash(c) for c in chunks]
    existing = set()
    try:
        collection = store._collection
        existing_ids = collection.get(ids=ids)["ids"]
        existing = set(existing_ids)
    except Exception:
        pass

    new_chunks = []
    new_ids = []
    for chunk, doc_id in zip(chunks, ids):
        if doc_id not in existing:
            new_chunks.append(chunk)
            new_ids.append(doc_id)

    if new_chunks:
        store.add_documents(new_chunks, ids=new_ids)

    return {
        "total_chunks": len(chunks),
        "new_chunks": len(new_chunks),
        "duplicates_skipped": len(chunks) - len(new_chunks),
    }


def get_collection_stats() -> dict:
    """Return stats about the current vector store."""
    try:
        store = get_vector_store()
        count = store._collection.count()
    except Exception:
        count = 0
    return {"total_documents": count, "persist_dir": str(settings.chroma_path)}
