from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Literal

from src.config import Settings


KBState = Literal["missing", "up_to_date", "outdated"]
KB_MANIFEST_FILENAME = "kb_manifest.json"
REBUILD_COMMAND = "python build_index.py"


@dataclass(frozen=True)
class KBStatusResult:
    state: KBState
    summary: str
    detail: str
    rebuild_command: str | None = None


def build_raw_source_snapshot(raw_data_dir: Path) -> list[dict[str, int | str]]:
    if not raw_data_dir.exists():
        return []

    snapshot: list[dict[str, int | str]] = []
    for path in sorted(raw_data_dir.rglob("*.md")):
        if not path.is_file():
            continue
        stat = path.stat()
        snapshot.append(
            {
                "relative_path": path.relative_to(raw_data_dir).as_posix(),
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
            }
        )
    return snapshot


def build_source_fingerprint(snapshot: list[dict[str, int | str]]) -> str:
    payload = json.dumps(
        snapshot,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def get_manifest_path(persist_dir: Path) -> Path:
    return persist_dir / KB_MANIFEST_FILENAME


def write_kb_manifest(*, settings: Settings, indexed_chunk_count: int) -> Path:
    snapshot = build_raw_source_snapshot(settings.raw_data_dir)
    manifest = {
        "built_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "collection_name": settings.chroma_collection_name,
        "indexed_chunk_count": indexed_chunk_count,
        "raw_file_count": len(snapshot),
        "source_fingerprint": build_source_fingerprint(snapshot),
    }

    settings.chroma_persist_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = get_manifest_path(settings.chroma_persist_dir)
    manifest_path.write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest_path


def get_kb_status(settings: Settings) -> KBStatusResult:
    persist_dir = settings.chroma_persist_dir
    manifest_path = get_manifest_path(persist_dir)
    has_index_artifacts = _has_index_artifacts(persist_dir)
    manifest_exists = manifest_path.is_file()
    manifest = _load_manifest(manifest_path)

    if not has_index_artifacts:
        if manifest_exists:
            return KBStatusResult(
                state="outdated",
                summary="Knowledge base is outdated.",
                detail="The build manifest exists, but the local Chroma index looks incomplete.",
                rebuild_command=REBUILD_COMMAND,
            )
        return KBStatusResult(
            state="missing",
            summary="Knowledge base is missing.",
            detail="No usable local Chroma index was found for the current knowledge base.",
            rebuild_command=REBUILD_COMMAND,
        )

    if manifest is None:
        return KBStatusResult(
            state="outdated",
            summary="Knowledge base is outdated.",
            detail="Indexed files exist, but the build manifest is missing or invalid.",
            rebuild_command=REBUILD_COMMAND,
        )

    if manifest["collection_name"] != settings.chroma_collection_name:
        return KBStatusResult(
            state="outdated",
            summary="Knowledge base is outdated.",
            detail="The local index was built for a different Chroma collection name.",
            rebuild_command=REBUILD_COMMAND,
        )

    snapshot = build_raw_source_snapshot(settings.raw_data_dir)
    raw_file_count = len(snapshot)
    source_fingerprint = build_source_fingerprint(snapshot)

    if manifest["raw_file_count"] != raw_file_count:
        return KBStatusResult(
            state="outdated",
            summary="Knowledge base is outdated.",
            detail="The raw markdown file set has changed since the last successful index build.",
            rebuild_command=REBUILD_COMMAND,
        )

    if manifest["source_fingerprint"] != source_fingerprint:
        return KBStatusResult(
            state="outdated",
            summary="Knowledge base is outdated.",
            detail="One or more raw markdown files changed after the last successful index build.",
            rebuild_command=REBUILD_COMMAND,
        )

    return KBStatusResult(
        state="up_to_date",
        summary="Knowledge base is up to date.",
        detail="The local Chroma index matches the current raw markdown snapshot.",
        rebuild_command=None,
    )


def _has_index_artifacts(persist_dir: Path) -> bool:
    if not persist_dir.exists():
        return False

    manifest_path = get_manifest_path(persist_dir)
    return any(
        path.is_file() and path != manifest_path
        for path in persist_dir.rglob("*")
    )


def _load_manifest(manifest_path: Path) -> dict[str, int | str] | None:
    if not manifest_path.is_file():
        return None

    try:
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None

    if not isinstance(data, Mapping):
        return None

    built_at = data.get("built_at")
    collection_name = data.get("collection_name")
    indexed_chunk_count = data.get("indexed_chunk_count")
    raw_file_count = data.get("raw_file_count")
    source_fingerprint = data.get("source_fingerprint")

    if not isinstance(built_at, str) or not built_at.strip():
        return None
    if not isinstance(collection_name, str) or not collection_name.strip():
        return None
    if not isinstance(indexed_chunk_count, int) or indexed_chunk_count < 1:
        return None
    if not isinstance(raw_file_count, int) or raw_file_count < 1:
        return None
    if not isinstance(source_fingerprint, str) or not source_fingerprint.strip():
        return None

    return {
        "built_at": built_at,
        "collection_name": collection_name,
        "indexed_chunk_count": indexed_chunk_count,
        "raw_file_count": raw_file_count,
        "source_fingerprint": source_fingerprint,
    }
