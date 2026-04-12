import os
from pathlib import Path

from src.config import Settings
from src.kb_status import (
    KB_MANIFEST_FILENAME,
    get_kb_status,
    write_kb_manifest,
)


def test_get_kb_status_returns_missing_when_index_is_absent(tmp_path: Path) -> None:
    settings = _build_settings(tmp_path)
    _write_raw_markdown(settings.raw_data_dir)

    status = get_kb_status(settings)

    assert status.state == "missing"
    assert status.rebuild_command == "python build_index.py"


def test_get_kb_status_returns_up_to_date_when_manifest_matches_snapshot(
    tmp_path: Path,
) -> None:
    settings = _build_settings(tmp_path)
    _write_raw_markdown(settings.raw_data_dir)
    _write_index_artifact(settings.chroma_persist_dir)
    write_kb_manifest(settings=settings, indexed_chunk_count=3)

    status = get_kb_status(settings)

    assert status.state == "up_to_date"
    assert status.rebuild_command is None


def test_get_kb_status_returns_outdated_after_raw_file_modification(
    tmp_path: Path,
) -> None:
    settings = _build_settings(tmp_path)
    raw_file = _write_raw_markdown(settings.raw_data_dir)
    _write_index_artifact(settings.chroma_persist_dir)
    write_kb_manifest(settings=settings, indexed_chunk_count=3)

    raw_file.write_text(raw_file.read_text(encoding="utf-8") + "\nUpdated.", encoding="utf-8")
    stat = raw_file.stat()
    os.utime(raw_file, ns=(stat.st_atime_ns, stat.st_mtime_ns + 5_000_000))

    status = get_kb_status(settings)

    assert status.state == "outdated"
    assert "changed" in status.detail.lower()


def test_get_kb_status_returns_outdated_when_manifest_is_missing_but_index_exists(
    tmp_path: Path,
) -> None:
    settings = _build_settings(tmp_path)
    _write_raw_markdown(settings.raw_data_dir)
    _write_index_artifact(settings.chroma_persist_dir)

    status = get_kb_status(settings)

    assert status.state == "outdated"
    assert "manifest" in status.detail.lower()


def test_get_kb_status_returns_outdated_for_malformed_manifest(tmp_path: Path) -> None:
    settings = _build_settings(tmp_path)
    _write_raw_markdown(settings.raw_data_dir)
    _write_index_artifact(settings.chroma_persist_dir)
    settings.chroma_persist_dir.mkdir(parents=True, exist_ok=True)
    (settings.chroma_persist_dir / KB_MANIFEST_FILENAME).write_text(
        "{not valid json",
        encoding="utf-8",
    )

    status = get_kb_status(settings)

    assert status.state == "outdated"
    assert "manifest" in status.detail.lower()


def _build_settings(tmp_path: Path) -> Settings:
    return Settings(
        RAW_DATA_DIR=tmp_path / "raw",
        CHROMA_PERSIST_DIR=tmp_path / "chroma_db",
        CHROMA_COLLECTION_NAME="kb_status_test_collection",
    )


def _write_raw_markdown(raw_dir: Path) -> Path:
    raw_dir.mkdir(parents=True, exist_ok=True)
    raw_file = raw_dir / "document.md"
    raw_file.write_text(
        """---
title: Test Document
topic: rag
library: general
doc_type: concept
difficulty: intro
---
Test content for the knowledge base status checks.
""",
        encoding="utf-8",
    )
    return raw_file


def _write_index_artifact(persist_dir: Path) -> None:
    persist_dir.mkdir(parents=True, exist_ok=True)
    (persist_dir / "chroma.sqlite3").write_text("indexed", encoding="utf-8")
