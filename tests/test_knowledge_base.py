from pathlib import Path

from langchain_core.embeddings.fake import FakeEmbeddings

from src.config import Settings
from src.knowledge_base import build_index


def test_build_index_indexes_and_retrieves_markdown_chunks(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    chroma_dir = tmp_path / "chroma_db"

    (raw_dir / "chroma_persistence.md").write_text(
        """---
title: Chroma Persistence Basics
topic: chroma
library: chroma
doc_type: how_to
difficulty: intro
error_family: persistence
---
# Chroma Persistence Basics

Use a persistent directory when you want Chroma to keep embeddings across runs.
For a local project, point the vector store to a stable path like `data/chroma_db`.
Resetting and rebuilding the collection is a safe way to avoid duplicate chunks.
""",
        encoding="utf-8",
    )

    settings = Settings(
        RAW_DATA_DIR=raw_dir,
        CHROMA_PERSIST_DIR=chroma_dir,
        CHROMA_COLLECTION_NAME="test_knowledge_base",
        CHUNK_SIZE=250,
        CHUNK_OVERLAP=20,
    )
    vector_store = build_index(
        settings=settings,
        embeddings=FakeEmbeddings(size=32),
    )

    assert len(vector_store.get()["ids"]) > 0

    results = vector_store.similarity_search(
        "How should I persist Chroma data locally?",
        k=1,
    )

    assert len(results) == 1
    assert results[0].metadata["title"] == "Chroma Persistence Basics"
    assert results[0].metadata["topic"] == "chroma"
    assert results[0].metadata["error_family"] == "persistence"
