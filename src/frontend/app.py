from __future__ import annotations
"""Streamlit frontend for RAG Document Intelligence."""

import requests
import streamlit as st

API_URL = "http://localhost:8000"

st.set_page_config(
    page_title="RAG Document Intelligence",
    page_icon="🔍",
    layout="wide",
)


def check_api():
    """Check if the FastAPI backend is running."""
    try:
        r = requests.get(f"{API_URL}/health", timeout=3)
        return r.status_code == 200
    except requests.ConnectionError:
        return False


def get_stats():
    """Fetch collection stats."""
    try:
        r = requests.get(f"{API_URL}/stats", timeout=5)
        return r.json()
    except Exception:
        return {"total_documents": 0, "persist_dir": "N/A"}


# --- Sidebar ---
with st.sidebar:
    st.title("📚 Document Intelligence")
    st.markdown("---")

    api_live = check_api()
    st.markdown(f"**API Status:** {'🟢 Online' if api_live else '🔴 Offline'}")

    if api_live:
        stats = get_stats()
        st.metric("Documents Indexed", stats["total_documents"])

    st.markdown("---")
    st.subheader("📄 Upload PDF")
    uploaded_file = st.file_uploader("Choose a PDF", type="pdf")

    if uploaded_file and api_live:
        if st.button("🚀 Ingest Document", use_container_width=True):
            with st.spinner("Processing document..."):
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
                try:
                    r = requests.post(f"{API_URL}/upload", files=files, timeout=120)
                    if r.status_code == 200:
                        data = r.json()
                        st.success(f"✅ {data['message']}")
                        st.info(f"Chunks: {data['total_chunks']} total, {data['new_chunks']} new")
                    else:
                        st.error(f"Error: {r.json().get('detail', 'Unknown error')}")
                except Exception as e:
                    st.error(f"Upload failed: {e}")

    st.markdown("---")
    st.subheader("📁 Bulk Ingest")
    ingest_dir = st.text_input("PDF Directory", value="./docs")
    if st.button("📥 Ingest Directory", use_container_width=True) and api_live:
        with st.spinner("Ingesting documents..."):
            try:
                r = requests.post(f"{API_URL}/ingest", json={"directory": ingest_dir}, timeout=300)
                if r.status_code == 200:
                    data = r.json()
                    st.success(f"✅ {data['message']}")
                else:
                    st.error(f"Error: {r.json().get('detail', 'Unknown error')}")
            except Exception as e:
                st.error(f"Ingestion failed: {e}")

    st.markdown("---")
    st.markdown("Built with LangChain + ChromaDB + FastAPI")

# --- Main Area ---
st.title("🔍 Document Q&A")
st.markdown("Ask questions about your ingested documents. Uses hybrid semantic + keyword search with re-ranking.")

if not api_live:
    st.warning("⚠️ Backend API is not running. Start it with: `uvicorn src.api.server:app --reload`")
    st.stop()

# Search parameters
col1, col2 = st.columns(2)
with col1:
    top_k = st.slider("Retrieval candidates (top_k)", 1, 50, 10)
with col2:
    rerank_k = st.slider("Re-ranked results (rerank_k)", 1, 20, 5)

st.markdown("---")

# Question input
question = st.text_area(
    "💬 Ask a question",
    placeholder="e.g., What are the key findings about transformer architectures?",
    height=100,
)

col_ask, col_search = st.columns(2)

with col_ask:
    ask_btn = st.button("🤖 Ask (with AI answer)", use_container_width=True, type="primary")

with col_search:
    search_btn = st.button("🔎 Search (retrieval only)", use_container_width=True)

# Handle Ask
if ask_btn and question:
    with st.spinner("Searching and generating answer..."):
        try:
            r = requests.post(
                f"{API_URL}/ask",
                json={"question": question, "top_k": top_k, "rerank_k": rerank_k},
                timeout=60,
            )
            if r.status_code == 200:
                data = r.json()

                st.markdown("### 💡 Answer")
                st.markdown(data["answer"])

                if data["sources"]:
                    st.markdown("### 📑 Sources")
                    for src in data["sources"]:
                        st.markdown(f"- **{src['filename']}** — Page {src['page']}")
            else:
                st.error(f"Error: {r.json().get('detail', 'Unknown error')}")
        except Exception as e:
            st.error(f"Request failed: {e}")

# Handle Search
if search_btn and question:
    with st.spinner("Searching documents..."):
        try:
            r = requests.post(
                f"{API_URL}/search",
                json={"question": question, "top_k": top_k, "rerank_k": rerank_k},
                timeout=30,
            )
            if r.status_code == 200:
                data = r.json()
                st.markdown(f"### 🔎 Search Results ({data['num_results']} found)")

                for i, result in enumerate(data["results"], 1):
                    meta = result["metadata"]
                    with st.expander(
                        f"Result {i} — {meta.get('filename', 'unknown')} (Page {meta.get('page', '?')})",
                        expanded=i <= 3,
                    ):
                        st.markdown(result["content"])
                        st.json(meta)
            else:
                st.error(f"Error: {r.json().get('detail', 'Unknown error')}")
        except Exception as e:
            st.error(f"Request failed: {e}")
