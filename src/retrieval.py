from __future__ import annotations

import re

from langchain_chroma import Chroma
from langchain_core.documents import Document

from src.schemas import (
    ChunkMetadata,
    RetrievalFilters,
    RetrievalRequest,
    RetrievedChunk,
    RetrievalResult,
)


NON_WORD_PATTERN = re.compile(r"[^a-z0-9\s]")
WHITESPACE_PATTERN = re.compile(r"\s+")

STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "for",
    "from",
    "help",
    "how",
    "i",
    "in",
    "is",
    "me",
    "of",
    "on",
    "please",
    "should",
    "the",
    "to",
    "what",
    "with",
}

TOPIC_KEYWORDS = {
    "langchain": {"langchain", "chain", "chains", "retriever", "retrieval chain"},
    "rag": {"rag", "grounding", "grounded", "chunking", "chunks"},
    "chroma": {"chroma", "vector store", "vectorstore", "embedding store"},
    "streamlit": {"streamlit", "chat ui", "sidebar", "session state"},
    "tool_calling": {"tool calling", "tools", "function calling"},
    "prompting": {"prompt", "prompting", "guardrails", "instructions"},
}

LIBRARY_KEYWORDS = {
    "langchain": {"langchain"},
    "chroma": {"chroma"},
    "streamlit": {"streamlit"},
    "openai": {"openai", "embedding model", "api key"},
}

DOC_TYPE_KEYWORDS = {
    "troubleshooting": {"error", "bug", "debug", "fix", "issue", "failing"},
    "how_to": {"how to", "configure", "build", "use", "persist"},
    "example": {"example", "sample", "pattern"},
    "concept": {"concept", "overview", "fundamentals", "why"},
}

ERROR_FAMILY_KEYWORDS = {
    "imports": {"import", "imports", "module not found"},
    "api": {"api", "authentication", "auth", "rate limit"},
    "retrieval": {"retrieval", "recall", "similarity search", "filters"},
    "ui": {"ui", "render", "chat input", "session state"},
    "persistence": {"persist", "persistence", "database", "stored", "rebuild"},
}


class RetrievalError(ValueError):
    pass


def retrieve_chunks(
    *,
    vector_store: Chroma,
    request: RetrievalRequest,
) -> RetrievalResult:
    _validate_vector_store(vector_store)
    rewritten_query = rewrite_query(request.query)
    filters = infer_metadata_filters(request.query)

    filtered_documents: list[Document] = []
    chroma_filter = filters.as_chroma_filter()
    if chroma_filter:
        filtered_documents = vector_store.similarity_search(
            rewritten_query,
            k=request.top_k,
            filter=chroma_filter,
        )

    used_fallback = False
    documents = filtered_documents
    if not documents:
        used_fallback = bool(chroma_filter)
        documents = vector_store.similarity_search(
            rewritten_query,
            k=request.top_k,
        )

    retrieved_chunks = [_to_retrieved_chunk(document) for document in documents]
    return RetrievalResult(
        rewritten_query=rewritten_query,
        applied_filters=filters,
        used_fallback=used_fallback,
        chunks=retrieved_chunks,
        sources=format_sources(retrieved_chunks),
    )


def rewrite_query(query: str) -> str:
    request = RetrievalRequest(query=query)
    normalized = NON_WORD_PATTERN.sub(" ", request.query.lower())
    tokens = [
        token
        for token in WHITESPACE_PATTERN.split(normalized)
        if token and token not in STOP_WORDS
    ]
    if not tokens:
        return request.query
    return " ".join(dict.fromkeys(tokens))


def infer_metadata_filters(query: str) -> RetrievalFilters:
    normalized = _normalize_query(query)
    return RetrievalFilters(
        topic=_infer_single_match(normalized, TOPIC_KEYWORDS),
        library=_infer_single_match(normalized, LIBRARY_KEYWORDS),
        doc_type=_infer_single_match(normalized, DOC_TYPE_KEYWORDS),
        error_family=_infer_single_match(normalized, ERROR_FAMILY_KEYWORDS),
    )


def format_sources(chunks: list[RetrievedChunk]) -> list[str]:
    sources: list[str] = []
    for chunk in chunks:
        metadata = chunk.metadata
        source = (
            f"{metadata.title} | topic={metadata.topic} | library={metadata.library} | "
            f"doc_type={metadata.doc_type} | difficulty={metadata.difficulty} | "
            f"source={metadata.source_path} | chunk={metadata.chunk_index}"
        )
        if metadata.error_family is not None:
            source += f" | error_family={metadata.error_family}"
        sources.append(source)
    return sources


def _validate_vector_store(vector_store: Chroma) -> None:
    try:
        existing_ids = vector_store.get()["ids"]
    except Exception as exc:
        raise RetrievalError(
            "Chroma vector store is unavailable. Build the local index before retrieval."
        ) from exc

    if not existing_ids:
        raise RetrievalError(
            "Chroma vector store is empty. Build the local index before retrieval."
        )


def _to_retrieved_chunk(document: Document) -> RetrievedChunk:
    return RetrievedChunk(
        content=document.page_content,
        metadata=ChunkMetadata.model_validate(document.metadata),
    )


def _normalize_query(query: str) -> str:
    request = RetrievalRequest(query=query)
    normalized = NON_WORD_PATTERN.sub(" ", request.query.lower())
    return WHITESPACE_PATTERN.sub(" ", normalized).strip()


def _infer_single_match(
    normalized_query: str,
    options: dict[str, set[str]],
) -> str | None:
    matched_values: list[str] = []
    for value, keywords in options.items():
        if any(keyword in normalized_query for keyword in keywords):
            matched_values.append(value)

    if len(matched_values) == 1:
        return matched_values[0]
    return None
