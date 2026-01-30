from __future__ import annotations
"""PDF loading and text chunking pipeline."""

from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

from src.config import settings


def load_pdf(file_path: str | Path) -> list[Document]:
    """Load a single PDF and return raw page documents."""
    loader = PyPDFLoader(str(file_path))
    return loader.load()


def load_directory(dir_path: str | Path, glob: str = "*.pdf") -> list[Document]:
    """Load all PDFs from a directory."""
    path = Path(dir_path)
    docs = []
    for pdf_file in sorted(path.glob(glob)):
        docs.extend(load_pdf(pdf_file))
    return docs


def chunk_documents(
    documents: list[Document],
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[Document]:
    """Split documents into chunks with metadata preserved."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size or settings.chunk_size,
        chunk_overlap=chunk_overlap or settings.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )
    chunks = splitter.split_documents(documents)

    # Enrich metadata
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = i
        chunk.metadata["char_count"] = len(chunk.page_content)
        source = chunk.metadata.get("source", "")
        chunk.metadata["filename"] = Path(source).name if source else "unknown"

    return chunks
