#!/usr/bin/env python3
"""CLI tool to ingest PDFs into the vector store."""

import argparse
import sys
from pathlib import Path

from src.ingestion.loader import load_pdf, load_directory, chunk_documents
from src.ingestion.embedder import ingest_documents, get_collection_stats


def main():
    parser = argparse.ArgumentParser(description="Ingest PDFs into the RAG system")
    parser.add_argument(
        "path",
        type=str,
        help="Path to a PDF file or directory of PDFs",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=None,
        help="Override chunk size (default from .env)",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=None,
        help="Override chunk overlap (default from .env)",
    )
    parser.add_argument("--stats", action="store_true", help="Show collection stats and exit")

    args = parser.parse_args()

    if args.stats:
        stats = get_collection_stats()
        print(f"📊 Collection Stats:")
        print(f"   Documents: {stats['total_documents']}")
        print(f"   Location:  {stats['persist_dir']}")
        return

    target = Path(args.path)

    if target.is_file() and target.suffix.lower() == ".pdf":
        print(f"📄 Loading: {target.name}")
        documents = load_pdf(target)
    elif target.is_dir():
        pdfs = list(target.glob("*.pdf"))
        print(f"📁 Found {len(pdfs)} PDF(s) in {target}")
        if not pdfs:
            print("❌ No PDF files found")
            sys.exit(1)
        documents = load_directory(target)
    else:
        print(f"❌ Invalid path: {args.path}")
        sys.exit(1)

    print(f"   Pages loaded: {len(documents)}")

    # Chunk
    chunks = chunk_documents(
        documents,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )
    print(f"   Chunks created: {len(chunks)}")

    # Ingest
    print("🔄 Embedding and storing...")
    stats = ingest_documents(chunks)
    print(f"✅ Done!")
    print(f"   New chunks:    {stats['new_chunks']}")
    print(f"   Duplicates:    {stats['duplicates_skipped']}")

    # Show total
    total = get_collection_stats()
    print(f"   Total in store: {total['total_documents']}")


if __name__ == "__main__":
    main()
