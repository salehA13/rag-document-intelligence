"""Tests for hybrid search and RRF."""

import pytest
from langchain.schema import Document

from src.search.hybrid import reciprocal_rank_fusion, keyword_search, _tokenize


@pytest.fixture
def sample_docs():
    return [
        Document(page_content="Machine learning is a subset of artificial intelligence", metadata={"id": 1}),
        Document(page_content="Deep learning uses neural networks with many layers", metadata={"id": 2}),
        Document(page_content="Natural language processing handles text data", metadata={"id": 3}),
        Document(page_content="Computer vision processes image and video data", metadata={"id": 4}),
    ]


def test_tokenize():
    tokens = _tokenize("Hello World Test")
    assert tokens == ["hello", "world", "test"]


def test_keyword_search(sample_docs):
    results = keyword_search("machine learning artificial intelligence", sample_docs, k=2)
    assert len(results) <= 2
    assert any("machine learning" in r.page_content.lower() for r in results)


def test_rrf_merges_lists():
    doc_a = Document(page_content="Document A", metadata={})
    doc_b = Document(page_content="Document B", metadata={})
    doc_c = Document(page_content="Document C", metadata={})

    list1 = [doc_a, doc_b]
    list2 = [doc_b, doc_c]

    fused = reciprocal_rank_fusion([list1, list2])
    # doc_b appears in both lists, so it should rank higher
    assert fused[0].page_content == "Document B"
    assert len(fused) == 3


def test_rrf_empty_lists():
    fused = reciprocal_rank_fusion([[], []])
    assert fused == []


def test_keyword_search_empty():
    results = keyword_search("test query", [], k=5)
    assert results == []
