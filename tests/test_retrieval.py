from pathlib import Path

import pytest
from langchain_core.embeddings.fake import FakeEmbeddings

from src.config import Settings
from src.knowledge_base import build_index
from src.retrieval import (
    RetrievalError,
    format_sources,
    infer_metadata_filters,
    retrieve_chunks,
    rewrite_query,
)
from src.schemas import RetrievalRequest, RetrievedChunk


def test_rewrite_query_removes_noise_words() -> None:
    rewritten = rewrite_query("How do I persist Chroma data in the local database?")

    assert rewritten == "do persist chroma data local database"


def test_infer_metadata_filters_uses_only_strong_signals() -> None:
    filters = infer_metadata_filters("How do I persist Chroma data locally?")

    assert filters.topic == "chroma"
    assert filters.library == "chroma"
    assert filters.doc_type == "how_to"
    assert filters.error_family == "persistence"


def test_infer_metadata_filters_skips_weak_query() -> None:
    filters = infer_metadata_filters("Can you explain this setup?")

    assert filters.as_chroma_filter() == {}


def test_infer_metadata_filters_avoids_weak_why_and_standalone_error_filters() -> None:
    filters = infer_metadata_filters(
        "Why should metadata fields make filtered retrieval easier to implement?"
    )

    assert filters.as_chroma_filter() == {}


def test_retrieve_chunks_uses_filtered_retrieval(indexed_vector_store) -> None:
    result = retrieve_chunks(
        vector_store=indexed_vector_store,
        request=RetrievalRequest(query="How do I persist Chroma data locally?", top_k=2),
    )

    assert result.applied_filters.library == "chroma"
    assert result.used_fallback is False
    assert len(result.chunks) == 1
    assert result.chunks[0].metadata.title == "Chroma Persistence Guide"


def test_retrieve_chunks_falls_back_when_filtered_search_is_empty(indexed_vector_store) -> None:
    result = retrieve_chunks(
        vector_store=indexed_vector_store,
        request=RetrievalRequest(
            query="Show a Streamlit persistence debugging example",
            top_k=3,
        ),
    )

    assert result.applied_filters.topic == "streamlit"
    assert result.applied_filters.error_family == "persistence"
    assert result.used_fallback is True
    assert len(result.chunks) >= 1
    assert any(chunk.metadata.library == "streamlit" for chunk in result.chunks)


def test_format_sources_includes_metadata_fields() -> None:
    chunk = RetrievedChunk.model_validate(
        {
            "content": "Example content",
            "metadata": {
                "doc_id": "streamlit-chat-patterns",
                "source_path": "data/raw/streamlit_chat_patterns.md",
                "title": "Streamlit Chat Patterns",
                "topic": "streamlit",
                "library": "streamlit",
                "doc_type": "example",
                "difficulty": "intermediate",
                "error_family": "ui",
                "chunk_index": 0,
            },
        }
    )

    sources = format_sources([chunk])

    assert sources == [
        "Streamlit Chat Patterns | topic=streamlit | library=streamlit | "
        "doc_type=example | difficulty=intermediate | "
        "source=data/raw/streamlit_chat_patterns.md | chunk=0 | error_family=ui"
    ]


def test_retrieve_chunks_rejects_empty_vector_store(tmp_path: Path) -> None:
    settings = Settings(
        CHROMA_PERSIST_DIR=tmp_path / "empty_chroma",
        CHROMA_COLLECTION_NAME="empty_collection",
    )

    from langchain_chroma import Chroma

    vector_store = Chroma(
        collection_name=settings.chroma_collection_name,
        persist_directory=str(settings.chroma_persist_dir),
        embedding_function=FakeEmbeddings(size=16),
    )

    with pytest.raises(RetrievalError, match="Build the local index"):
        retrieve_chunks(
            vector_store=vector_store,
            request=RetrievalRequest(query="Chroma persistence"),
        )


def test_retrieve_chunks_returns_no_usable_chunks_for_off_domain_query(indexed_vector_store) -> None:
    result = retrieve_chunks(
        vector_store=indexed_vector_store,
        request=RetrievalRequest(query="What is the capital of France?", top_k=2),
    )

    assert result.chunks == []
    assert result.sources == []
    assert result.used_fallback is False


def test_retrieve_chunks_returns_no_context_for_undercovered_rate_limit_query(
    indexed_vector_store,
) -> None:
    result = retrieve_chunks(
        vector_store=indexed_vector_store,
        request=RetrievalRequest(
            query="How should I implement rate limiting for this assistant?",
            top_k=3,
        ),
    )

    assert result.chunks == []
    assert result.sources == []
    assert result.used_fallback is False


@pytest.fixture
def indexed_vector_store(tmp_path: Path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    chroma_dir = tmp_path / "chroma_db"

    (raw_dir / "chroma_persistence.md").write_text(
        """---
title: Chroma Persistence Guide
topic: chroma
library: chroma
doc_type: how_to
difficulty: intro
error_family: persistence
---
Use a persistent Chroma directory when you want local retrieval data to survive between runs.
Rebuild the collection if you need a clean index without duplicate chunks.
""",
        encoding="utf-8",
    )
    (raw_dir / "streamlit_debug.md").write_text(
        """---
title: Streamlit Debugging Example
topic: streamlit
library: streamlit
doc_type: example
difficulty: intermediate
error_family: ui
---
Streamlit chat interfaces should show source metadata next to the answer.
Use session state carefully when debugging reruns in a retrieval app.
""",
        encoding="utf-8",
    )
    (raw_dir / "retrieval_debugging_playbook.md").write_text(
        """---
title: Retrieval Debugging Playbook
topic: rag
library: general
doc_type: troubleshooting
difficulty: advanced
error_family: retrieval
---
Similarity search can still return weak context.
That matters for out-of-domain questions like What is the capital of France?
Those questions should still return no usable chunks and no grounded answer.
""",
        encoding="utf-8",
    )

    settings = Settings(
        RAW_DATA_DIR=raw_dir,
        CHROMA_PERSIST_DIR=chroma_dir,
        CHROMA_COLLECTION_NAME="retrieval_test_collection",
        CHUNK_SIZE=250,
        CHUNK_OVERLAP=20,
    )
    return build_index(
        settings=settings,
        embeddings=FakeEmbeddings(size=32),
    )
