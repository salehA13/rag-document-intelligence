# 🔍 RAG Document Intelligence

A production-grade **Retrieval-Augmented Generation** system that ingests PDF documents into vector embeddings and answers questions using hybrid semantic + keyword search with re-ranking.

Built with **LangChain**, **ChromaDB**, **OpenAI**, **FastAPI**, and **Streamlit**.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    RAG Document Intelligence                     │
├──────────────────────┬──────────────────────────────────────────┤
│                      │                                          │
│   📄 Ingestion       │   🔍 Query Pipeline                     │
│                      │                                          │
│   PDF Upload/Dir ──► │   User Question                          │
│         │            │        │                                  │
│   PyPDF Loader       │        ▼                                  │
│         │            │   ┌──────────┐    ┌──────────┐           │
│   Recursive          │   │ Semantic │    │ Keyword  │           │
│   Text Splitter      │   │ Search   │    │ Search   │           │
│         │            │   │(ChromaDB)│    │ (BM25)   │           │
│   OpenAI             │   └────┬─────┘    └────┬─────┘           │
│   Embeddings         │        │               │                  │
│         │            │        └───────┬───────┘                  │
│   ChromaDB           │                │                          │
│   Vector Store       │     Reciprocal Rank Fusion                │
│                      │                │                          │
│                      │         Re-ranked Results                 │
│                      │                │                          │
│                      │        GPT-4o-mini + Context              │
│                      │                │                          │
│                      │     Answer with Source Citations           │
├──────────────────────┴──────────────────────────────────────────┤
│                                                                  │
│   🌐 FastAPI REST API          📊 Streamlit Frontend            │
│   POST /ask                    Interactive Document Q&A          │
│   POST /upload                 PDF Upload & Ingestion            │
│   POST /search                 Search Controls & Results         │
│   POST /ingest                 Source Attribution View           │
│   GET  /stats                                                    │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

## ✨ Features

- **Hybrid Search** — Combines semantic vector search (ChromaDB) with BM25 keyword search for robust retrieval
- **Reciprocal Rank Fusion** — Merges results from multiple retrieval methods using the RRF algorithm
- **Source Attribution** — Every answer includes cited sources with filename and page numbers
- **Deduplication** — Content-hash based dedup prevents duplicate embeddings
- **REST API** — Full FastAPI backend with OpenAPI docs, file upload, and search endpoints
- **Interactive UI** — Streamlit frontend for document upload, Q&A, and retrieval-only search
- **CLI Ingestion** — Command-line tool for batch document processing
- **Configurable** — All parameters (chunk size, top-k, models) via environment variables

## Tech Stack

| Component | Technology |
|-----------|-----------|
| LLM | OpenAI GPT-4o-mini |
| Embeddings | OpenAI text-embedding-3-small |
| Vector DB | ChromaDB |
| Framework | LangChain |
| API | FastAPI + Uvicorn |
| Frontend | Streamlit |
| Search | BM25 (rank-bm25) + Cosine Similarity |
| Re-ranking | Reciprocal Rank Fusion |
| PDF Processing | PyPDF |

## 🚀 Quick Start

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
# Edit .env and add your OpenAI API key
```

### 3. Ingest Documents

```bash
# Ingest sample docs
python ingest.py ./docs

# Ingest a single PDF
python ingest.py path/to/document.pdf

# Check stats
python ingest.py --stats .
```

### 4. Start the API

```bash
uvicorn src.api.server:app --reload --port 8000
```

API docs available at: http://localhost:8000/docs

### 5. Launch the UI

```bash
streamlit run src/frontend/app.py
```

## 📡 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Health check |
| `GET` | `/stats` | Vector store statistics |
| `POST` | `/ask` | Ask a question (full RAG pipeline) |
| `POST` | `/search` | Search documents (retrieval only) |
| `POST` | `/upload` | Upload and ingest a PDF |
| `POST` | `/ingest` | Ingest all PDFs from a directory |

### Example: Ask a Question

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the self-attention mechanism?", "top_k": 10, "rerank_k": 5}'
```

```json
{
  "answer": "The self-attention mechanism is the core innovation of transformers...",
  "sources": [
    {"filename": "transformer_survey.pdf", "page": 0}
  ],
  "num_sources": 5
}
```

## 🧪 Testing

```bash
pip install pytest
pytest tests/ -v
```

## 📂 Project Structure

```
rag-document-intelligence/
├── src/
│   ├── config.py              # Centralized settings (pydantic-settings)
│   ├── ingestion/
│   │   ├── loader.py          # PDF loading & text chunking
│   │   └── embedder.py        # Vector store management
│   ├── search/
│   │   ├── hybrid.py          # Hybrid search + RRF re-ranking
│   │   └── qa.py              # QA chain with source attribution
│   ├── api/
│   │   ├── models.py          # Pydantic request/response models
│   │   └── server.py          # FastAPI application
│   └── frontend/
│       └── app.py             # Streamlit UI
├── tests/
│   ├── test_loader.py
│   ├── test_search.py
│   └── test_api.py
├── docs/                      # Sample PDFs
├── ingest.py                  # CLI ingestion tool
├── requirements.txt
├── .env.example
└── README.md
```

## Screenshots

> _Launch the Streamlit UI to see the interactive document Q&A interface._

| Feature | Description |
|---------|-------------|
| 📄 Document Upload | Drag & drop PDFs in the sidebar |
| 💬 Q&A Interface | Ask questions with adjustable search parameters |
| 🔎 Search Mode | Retrieval-only mode with expandable results |
| 📑 Source Citations | Every answer shows source documents and pages |

## How It Works

1. **Ingestion** — PDFs are loaded with PyPDF, split into overlapping chunks using LangChain's RecursiveCharacterTextSplitter, embedded with OpenAI, and stored in ChromaDB
2. **Retrieval** — Queries trigger both semantic search (cosine similarity on embeddings) and BM25 keyword search, then results are fused using Reciprocal Rank Fusion
3. **Generation** — Top-ranked chunks are formatted with source metadata and sent to GPT-4o-mini with a system prompt enforcing grounded, cited answers

## License

MIT

---

Built by [Saleh](https://github.com/salehA13)
