from __future__ import annotations
"""Hybrid search: semantic (ChromaDB) + keyword (BM25) with RRF re-ranking."""

import numpy as np
from rank_bm25 import BM25Okapi
from langchain.schema import Document

from src.config import settings
from src.ingestion.embedder import get_vector_store


def _tokenize(text: str) -> list[str]:
    """Simple whitespace + lowercase tokenizer."""
    return text.lower().split()


def semantic_search(query: str, k: int | None = None) -> list[Document]:
    """Pure vector similarity search via ChromaDB."""
    store = get_vector_store()
    return store.similarity_search(query, k=k or settings.top_k)


def keyword_search(query: str, documents: list[Document], k: int | None = None) -> list[Document]:
    """BM25 keyword search over a set of documents."""
    k = k or settings.top_k
    if not documents:
        return []

    corpus = [_tokenize(doc.page_content) for doc in documents]
    bm25 = BM25Okapi(corpus)
    scores = bm25.get_scores(_tokenize(query))

    # Get top-k indices
    top_indices = np.argsort(scores)[::-1][:k]
    return [documents[i] for i in top_indices if scores[i] > 0]


def reciprocal_rank_fusion(
    result_lists: list[list[Document]],
    k: int = 60,
) -> list[Document]:
    """
    Reciprocal Rank Fusion (RRF) to merge multiple ranked lists.

    RRF score = Σ 1 / (k + rank_i) for each list where the doc appears.
    Default k=60 as per the original paper (Cormack et al., 2009).
    """
    scores: dict[str, float] = {}
    doc_map: dict[str, Document] = {}

    for result_list in result_lists:
        for rank, doc in enumerate(result_list):
            # Use content hash as key for dedup across lists
            doc_key = hash(doc.page_content)
            doc_map[doc_key] = doc
            scores[doc_key] = scores.get(doc_key, 0.0) + 1.0 / (k + rank + 1)

    # Sort by fused score descending
    sorted_keys = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
    return [doc_map[k] for k in sorted_keys]


def hybrid_search(query: str, top_k: int | None = None, rerank_k: int | None = None) -> list[Document]:
    """
    Full hybrid search pipeline:
    1. Semantic search (ChromaDB embeddings)
    2. Keyword search (BM25) over the semantic results' broader context
    3. RRF re-ranking to fuse both result sets
    """
    k = top_k or settings.top_k
    final_k = rerank_k or settings.rerank_top_k

    # Step 1: Semantic search (cast a wide net)
    semantic_results = semantic_search(query, k=k)

    if not semantic_results:
        return []

    # Step 2: BM25 keyword search over the same candidate pool
    # We pull a larger set from the vector store and re-rank with BM25
    all_candidates = semantic_search(query, k=k * 2)
    keyword_results = keyword_search(query, all_candidates, k=k)

    # Step 3: Reciprocal Rank Fusion
    fused = reciprocal_rank_fusion([semantic_results, keyword_results])

    return fused[:final_k]
