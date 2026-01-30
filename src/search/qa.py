from __future__ import annotations
"""Question-answering chain with source attribution."""

from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import Document

from src.config import settings
from src.search.hybrid import hybrid_search


SYSTEM_PROMPT = """You are a precise document analyst. Answer questions based ONLY on the provided context.

Rules:
- If the context doesn't contain enough information, say so explicitly
- Cite specific sections when possible
- Be concise but thorough
- Use bullet points for multi-part answers
- Never fabricate information not in the context"""

QA_PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", """Context from documents:
---
{context}
---

Question: {question}

Answer based on the above context:"""),
])


def format_context(documents: list[Document]) -> str:
    """Format retrieved documents into a context string."""
    parts = []
    for i, doc in enumerate(documents, 1):
        source = doc.metadata.get("filename", "unknown")
        page = doc.metadata.get("page", "?")
        parts.append(f"[Source {i}: {source}, Page {page}]\n{doc.page_content}")
    return "\n\n".join(parts)


def ask(
    question: str,
    top_k: int | None = None,
    rerank_k: int | None = None,
) -> dict:
    """
    End-to-end RAG pipeline:
    1. Hybrid search for relevant chunks
    2. Format context with source attribution
    3. Generate answer with GPT
    """
    # Retrieve
    documents = hybrid_search(question, top_k=top_k, rerank_k=rerank_k)

    if not documents:
        return {
            "answer": "No relevant documents found. Please ingest some documents first.",
            "sources": [],
            "num_sources": 0,
        }

    # Build context
    context = format_context(documents)

    # Generate
    llm = ChatOpenAI(
        model=settings.openai_model,
        temperature=0.1,
        openai_api_key=settings.openai_api_key,
    )

    chain = QA_PROMPT | llm
    response = chain.invoke({"context": context, "question": question})

    # Extract source info
    sources = []
    seen = set()
    for doc in documents:
        filename = doc.metadata.get("filename", "unknown")
        page = doc.metadata.get("page", "?")
        key = f"{filename}:{page}"
        if key not in seen:
            sources.append({"filename": filename, "page": page})
            seen.add(key)

    return {
        "answer": response.content,
        "sources": sources,
        "num_sources": len(documents),
    }
