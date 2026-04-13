from __future__ import annotations

import json
import re
from collections.abc import Mapping
from pathlib import Path

from src.config import Settings, get_settings
from src.schemas import (
    OfficialDocsDocument,
    OfficialDocsLookupRequest,
    OfficialDocsLookupResult,
    OfficialDocsSnippet,
)


FALLBACK_DOC_LIMIT = 3


def lookup_streamlit_official_docs(
    *,
    request: OfficialDocsLookupRequest,
    settings: Settings | None = None,
    manifest_path: Path | None = None,
) -> OfficialDocsLookupResult:
    return _lookup_fallback_official_docs(
        request=request,
        expected_library="streamlit",
        settings=settings,
        manifest_path=manifest_path,
    )


def lookup_chroma_official_docs(
    *,
    request: OfficialDocsLookupRequest,
    settings: Settings | None = None,
    manifest_path: Path | None = None,
) -> OfficialDocsLookupResult:
    return _lookup_fallback_official_docs(
        request=request,
        expected_library="chroma",
        settings=settings,
        manifest_path=manifest_path,
    )


def _lookup_fallback_official_docs(
    *,
    request: OfficialDocsLookupRequest,
    expected_library: str,
    settings: Settings | None,
    manifest_path: Path | None,
) -> OfficialDocsLookupResult:
    if request.library != expected_library:
        raise ValueError(
            f"Fallback adapter expected library '{expected_library}', got '{request.library}'."
        )

    resolved_settings = settings or get_settings()
    resolved_manifest_path = (
        manifest_path or resolved_settings.official_docs_fallback_manifest_path
    )
    manifest_entries = _load_manifest_entries(resolved_manifest_path)

    query_tokens = _tokenize(request.query)
    ranked_entries: list[tuple[int, str, Mapping[str, object]]] = []
    for entry in manifest_entries:
        if entry.get("library") != expected_library:
            continue

        score = _score_manifest_entry(query_tokens, entry)
        if score <= 0:
            continue

        title = str(entry.get("title", "")).strip().lower()
        ranked_entries.append((score, title, entry))

    ranked_entries.sort(key=lambda item: (-item[0], item[1]))
    documents = [
        _build_fallback_document(entry)
        for _, _, entry in ranked_entries[:FALLBACK_DOC_LIMIT]
    ]
    if not documents:
        raise ValueError(f"No fallback official docs matched the query for {expected_library}.")

    return OfficialDocsLookupResult(
        library=expected_library,
        documents=documents,
    )


def _load_manifest_entries(manifest_path: Path) -> list[Mapping[str, object]]:
    manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest_payload, Mapping):
        raise ValueError("Official docs fallback manifest must be a JSON object.")

    entries = manifest_payload.get("entries")
    if not isinstance(entries, list):
        raise ValueError("Official docs fallback manifest must include an entries list.")

    return [entry for entry in entries if isinstance(entry, Mapping)]


def _score_manifest_entry(
    query_tokens: set[str],
    entry: Mapping[str, object],
) -> int:
    entry_tokens: set[str] = set()
    for key in ("title", "url"):
        value = entry.get(key)
        if isinstance(value, str):
            entry_tokens.update(_tokenize(value))

    keywords = entry.get("keywords")
    if isinstance(keywords, list):
        for keyword in keywords:
            if isinstance(keyword, str):
                entry_tokens.update(_tokenize(keyword))

    snippets = entry.get("snippets")
    if isinstance(snippets, list):
        for snippet in snippets:
            if isinstance(snippet, str):
                entry_tokens.update(_tokenize(snippet))

    return len(query_tokens & entry_tokens)


def _build_fallback_document(entry: Mapping[str, object]) -> OfficialDocsDocument:
    title = str(entry.get("title", "")).strip()
    url = str(entry.get("url", "")).strip()
    raw_snippets = entry.get("snippets")
    snippet_texts = [
        snippet.strip()
        for snippet in raw_snippets
        if isinstance(snippet, str) and snippet.strip()
    ] if isinstance(raw_snippets, list) else []

    return OfficialDocsDocument(
        title=title,
        url=url,
        provider_mode="official_fallback",
        snippets=[
            OfficialDocsSnippet(text=snippet_text, rank=index)
            for index, snippet_text in enumerate(snippet_texts, start=1)
        ],
    )


def _tokenize(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9]+", text.lower())
        if len(token) >= 3
    }
