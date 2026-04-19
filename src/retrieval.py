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
    "concept": {"concept", "overview", "fundamentals"},
}

ERROR_FAMILY_KEYWORDS = {
    "imports": {"import", "imports", "module not found"},
    "api": {"api", "authentication", "auth"},
    "retrieval": {"retrieval", "recall", "similarity search", "filters"},
    "ui": {"ui", "render", "chat input", "session state"},
    "persistence": {"persist", "persistence", "database", "stored", "rebuild"},
}

DOMAIN_QUERY_KEYWORDS = {
    "analytics",
    "api",
    "api key",
    "chat",
    "chat input",
    "chroma",
    "chunk",
    "chunking",
    "chunks",
    "cost",
    "csv",
    "diagnose",
    "diagnostics",
    "docs",
    "documentation",
    "embedding",
    "embeddings",
    "error",
    "errors",
    "evaluation",
    "export",
    "exports",
    "filter",
    "filters",
    "grounded",
    "grounding",
    "index",
    "indexing",
    "json",
    "knowledge base",
    "langchain",
    "mcp",
    "metadata",
    "model",
    "models",
    "official docs",
    "openai",
    "pdf",
    "persist",
    "persistence",
    "prompt",
    "prompting",
    "rag",
    "rebuild",
    "rerun",
    "retrieval",
    "session",
    "session state",
    "sidebar",
    "source",
    "sources",
    "streamlit",
    "token",
    "tokens",
    "tool",
    "tools",
    "ui",
    "usage",
    "vector",
    "vector store",
    "vectorstore",
}
DOMAIN_QUERY_TOKENS = {
    keyword for keyword in DOMAIN_QUERY_KEYWORDS if " " not in keyword
}
DOMAIN_QUERY_PHRASES = {
    keyword for keyword in DOMAIN_QUERY_KEYWORDS if " " in keyword
}


class RetrievalError(ValueError):
    pass


def retrieve_chunks(
    *,
    vector_store: Chroma,
    request: RetrievalRequest,
) -> RetrievalResult:
    _validate_vector_store(vector_store)
    if _is_clearly_out_of_domain(request.query):
        return RetrievalResult(
            rewritten_query=rewrite_query(request.query),
            applied_filters=RetrievalFilters(),
            used_fallback=False,
            chunks=[],
            sources=[],
        )

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

    documents = _filter_usable_documents(
        documents=documents,
        rewritten_query=rewritten_query,
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
    topic = _infer_single_match(normalized, TOPIC_KEYWORDS)
    library = _infer_single_match(normalized, LIBRARY_KEYWORDS)
    doc_type = _infer_single_match(normalized, DOC_TYPE_KEYWORDS)
    error_family = _infer_single_match(normalized, ERROR_FAMILY_KEYWORDS)
    if topic is None and library is None and doc_type is None:
        error_family = None

    return RetrievalFilters(
        topic=topic,
        library=library,
        doc_type=doc_type,
        error_family=error_family,
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


def _filter_usable_documents(
    *,
    documents: list[Document],
    rewritten_query: str,
) -> list[Document]:
    query_tokens = _meaningful_tokens(rewritten_query)
    if not query_tokens:
        return documents

    usable_documents: list[Document] = []
    for document in documents:
        document_tokens = _document_tokens(document)
        overlap_count = len(query_tokens & document_tokens)
        minimum_overlap = 1 if len(query_tokens) <= 2 else 2
        if overlap_count >= minimum_overlap:
            usable_documents.append(document)
    return usable_documents


def _document_tokens(document: Document) -> set[str]:
    metadata = document.metadata
    metadata_text = " ".join(
        str(metadata.get(key, ""))
        for key in ("title", "topic", "library", "doc_type", "error_family")
        if metadata.get(key)
    )
    return _meaningful_tokens(f"{document.page_content} {metadata_text}")


def _meaningful_tokens(text: str) -> set[str]:
    normalized = _normalize_query(text)
    return {
        token
        for token in normalized.split()
        if token not in STOP_WORDS and len(token) > 2
    }


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


def _is_clearly_out_of_domain(query: str) -> bool:
    normalized = _normalize_query(query)
    if not normalized:
        return False

    query_tokens = set(normalized.split())
    return not (
        any(phrase in normalized for phrase in DOMAIN_QUERY_PHRASES)
        or any(token in query_tokens for token in DOMAIN_QUERY_TOKENS)
    )
