<p align="center">
  <h1 align="center">🔍 RAG Document Intelligence</h1>
  <p align="center">
    <img src="https://github.com/salehA13/rag-document-intelligence/actions/workflows/ci.yml/badge.svg" alt="CI">
  </p>
  <p align="center">
    Production-grade Retrieval-Augmented Generation system with hybrid semantic + keyword search and re-ranking
  </p>
</p>

<p align="center">
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white" alt="Python"></a>
  <a href="https://fastapi.tiangolo.com"><img src="https://img.shields.io/badge/FastAPI-0.115-009688?logo=fastapi&logoColor=white" alt="FastAPI"></a>
  <a href="https://www.langchain.com"><img src="https://img.shields.io/badge/LangChain-0.3-1C3C3C?logo=langchain&logoColor=white" alt="LangChain"></a>
  <a href="https://www.trychroma.com"><img src="https://img.shields.io/badge/ChromaDB-1.0-FF6F00" alt="ChromaDB"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License: MIT"></a>
</p>

---

## The Problem

Traditional keyword search fails when users phrase questions differently from the source text. Pure vector search misses exact term matches. **Neither approach alone is reliable for document Q&A.**

RAG Document Intelligence solves this with a **hybrid retrieval pipeline** that combines both strategies and fuses the results using Reciprocal Rank Fusion (RRF) — the same technique used by production search engines — to deliver consistently relevant answers grounded in your documents.

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────────────┐
│                      RAG Document Intelligence                          │
│                                                                          │
│   ┌─────────────────────┐         ┌──────────────────────────────────┐  │
│   │   📄 Ingestion       │         │   🔍 Query Pipeline              │  │
│   │                      │         │                                  │  │
│   │   PDF Upload/Dir     │         │   User Question                  │  │
│   │        │             │         │        │                         │  │
│   │   PyPDF Loader       │         │        ├──────────┐             │  │
│   │        │             │         │        ▼          ▼             │  │
│   │   Recursive Text     │         │   ┌─────────┐ ┌─────────┐     │  │
│   │   Splitter           │         │   │Semantic │ │ Keyword │     │  │
│   │   (1000 char chunks  │         │   │ Search  │ │ Search  │     │  │
│   │    200 overlap)      │         │   │(ChromaDB)│ │ (BM25)  │     │  │
│   │        │             │         │   └────┬────┘ └────┬────┘     │  │
│   │   OpenAI Embeddings  │         │        │           │           │  │
│   │   (text-embedding-   │         │        └─────┬─────┘           │  │
│   │    3-small)          │         │              ▼                  │  │
│   │        │             │         │   Reciprocal Rank Fusion        │  │
│   │   ChromaDB + SHA-256 │         │   (k=60, per Cormack 2009)     │  │
│   │   Dedup Store        │         │              │                  │  │
│   │                      │         │       Re-ranked Results         │  │
│   └─────────────────────┘         │              │                  │  │
│                                    │       GPT-4o-mini + Context     │  │
│                                    │              │                  │  │
│                                    │    Answer + Source Citations     │  │
│                                    └──────────────────────────────────┘  │
│                                                                          │
│   ┌──────────────────────────────────────────────────────────────────┐  │
│   │  🌐 FastAPI REST API              📊 Streamlit Frontend         │  │
│   │  POST /ask    POST /upload        Interactive Document Q&A      │  │
│   │  POST /search POST /ingest        PDF Upload & Ingestion        │  │
│   │  GET  /stats  GET  /health        Source Attribution View       │  │
│   └──────────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## Key Features

| Feature | Description |
|---------|-------------|
| **Hybrid Search** | Combines dense vector retrieval (ChromaDB) with sparse BM25 keyword matching for robust recall |
| **Reciprocal Rank Fusion** | Merges ranked lists using RRF (`1/(k+rank)`) — outperforms single-strategy retrieval without tuning |
| **Source Attribution** | Every answer cites the exact source filename and page number |
| **Content Deduplication** | SHA-256 content hashing prevents duplicate embeddings across re-ingestions |
| **REST API** | Full FastAPI backend with OpenAPI docs, file upload, and typed request/response models |
| **Interactive UI** | Streamlit frontend for drag-and-drop PDF upload, Q&A, and retrieval-only search |
| **CLI Ingestion** | Batch-process entire directories of PDFs from the command line |
| **Configurable** | All parameters (chunk size, overlap, top-k, models) via environment variables |

---

## Why Hybrid Search + RRF?

| Strategy | Strength | Weakness |
|----------|----------|----------|
| **Semantic only** | Understands meaning, synonyms, paraphrasing | Misses exact terms, acronyms, proper nouns |
| **Keyword only** | Precise term matching, fast | No understanding of meaning or context |
| **Hybrid + RRF** | **Best of both** — high recall and precision | Slightly more compute (negligible in practice) |

RRF is parameter-free (only `k=60` constant) and consistently improves retrieval quality without requiring training data or score normalization across methods.

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| LLM | OpenAI GPT-4o-mini |
| Embeddings | OpenAI `text-embedding-3-small` (1536 dims) |
| Vector Store | ChromaDB (persistent, local) |
| Framework | LangChain 0.3 |
| API | FastAPI + Uvicorn |
| Frontend | Streamlit |
| Keyword Search | BM25Okapi (`rank-bm25`) |
| Re-ranking | Reciprocal Rank Fusion |
| PDF Processing | PyPDF |
| Config | Pydantic Settings |

---

## Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/salehA13/rag-document-intelligence.git
cd rag-document-intelligence

python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

### 2. Configure

```bash
cp .env.example .env
# Add your OpenAI API key to .env
```

### 3. Ingest Documents

```bash
# Ingest the sample PDFs
python ingest.py ./docs

# Ingest a single file
python ingest.py path/to/paper.pdf

# Check store stats
python ingest.py --stats .
```

### 4. Start the API

```bash
uvicorn src.api.server:app --reload --port 8000
```

Interactive docs at: **http://localhost:8000/docs**

### 5. Launch the UI (optional)

```bash
streamlit run src/frontend/app.py
```

---

## API Reference

### `GET /health`

Health check.

```json
{ "status": "healthy", "version": "1.0.0" }
```

### `GET /stats`

Vector store statistics.

```json
{ "total_documents": 142, "persist_dir": "./data/chroma" }
```

### `POST /ask`

Full RAG pipeline — retrieve, re-rank, generate answer with citations.

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is the self-attention mechanism?",
    "top_k": 10,
    "rerank_k": 5
  }'
```

```json
{
  "answer": "The self-attention mechanism allows each position in a sequence to attend to all other positions...",
  "sources": [
    { "filename": "transformer_survey.pdf", "page": 3 },
    { "filename": "transformer_survey.pdf", "page": 7 }
  ],
  "num_sources": 5
}
```

### `POST /search`

Retrieval-only — returns ranked document chunks without LLM generation.

```bash
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{"question": "attention mechanisms", "top_k": 10, "rerank_k": 5}'
```

### `POST /upload`

Upload and ingest a single PDF.

```bash
curl -X POST http://localhost:8000/upload \
  -F "file=@paper.pdf"
```

### `POST /ingest`

Batch-ingest all PDFs from a directory.

```bash
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{"directory": "./docs"}'
```

---

## Testing

```bash
pytest tests/ -v
```

All **13 tests** cover the ingestion pipeline, hybrid search logic, RRF correctness, and API endpoints:

```
tests/test_loader.py   — chunking, metadata enrichment, edge cases
tests/test_search.py   — BM25, RRF merging, tokenization
tests/test_api.py      — health, stats, upload validation, error handling
```

---

## Project Structure

```
rag-document-intelligence/
├── src/
│   ├── config.py                  # Centralized settings (pydantic-settings)
│   ├── ingestion/
│   │   ├── loader.py              # PDF loading & recursive text chunking
│   │   └── embedder.py            # ChromaDB vector store + SHA-256 dedup
│   ├── search/
│   │   ├── hybrid.py              # Hybrid search: semantic + BM25 + RRF
│   │   └── qa.py                  # QA chain with source attribution
│   ├── api/
│   │   ├── models.py              # Typed Pydantic request/response schemas
│   │   └── server.py              # FastAPI application + CORS
│   └── frontend/
│       └── app.py                 # Streamlit interactive UI
├── tests/
│   ├── test_loader.py             # Ingestion pipeline tests
│   ├── test_search.py             # Search & RRF tests
│   └── test_api.py                # API endpoint tests
├── docs/                          # Sample PDFs for demo
├── ingest.py                      # CLI ingestion tool
├── requirements.txt
├── .env.example
├── .gitignore
├── LICENSE
└── README.md
```

---

## Configuration

All settings are managed via environment variables (`.env` file) with sensible defaults:

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | — | Your OpenAI API key (required) |
| `OPENAI_MODEL` | `gpt-4o-mini` | LLM for answer generation |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | Embedding model |
| `CHROMA_PERSIST_DIR` | `./data/chroma` | ChromaDB storage path |
| `CHUNK_SIZE` | `1000` | Characters per chunk |
| `CHUNK_OVERLAP` | `200` | Overlap between chunks |
| `TOP_K` | `10` | Retrieval candidates |
| `RERANK_TOP_K` | `5` | Final results after RRF |

---

## License

[MIT](LICENSE)

---

Built by [Saleh](https://github.com/salehA13)
