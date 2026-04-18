from __future__ import annotations

from rendering.structured_display import (
    _clean_display_text,
    _format_tool_field_lines,
    format_official_docs_library_label,
    format_official_docs_provider_label,
    format_tool_field_label,
    format_tool_name_label,
)


def build_official_docs_display_data(
    official_docs_result: object,
) -> dict[str, object] | None:
    if not isinstance(official_docs_result, dict):
        return None

    lookup_result = official_docs_result.get("lookup_result")
    raw_documents = lookup_result.get("documents") if isinstance(lookup_result, dict) else []
    documents: list[dict[str, object]] = []
    if isinstance(raw_documents, list):
        for raw_document in raw_documents:
            if not isinstance(raw_document, dict):
                continue

            raw_snippets = raw_document.get("snippets")
            snippets = []
            if isinstance(raw_snippets, list):
                for raw_snippet in raw_snippets:
                    if not isinstance(raw_snippet, dict):
                        continue
                    snippet_text = str(raw_snippet.get("text", "")).strip()
                    if snippet_text:
                        snippets.append(snippet_text)

            documents.append(
                {
                    "title": str(raw_document.get("title") or "Untitled document").strip(),
                    "url": _clean_display_text(raw_document.get("url")),
                    "provider_label": format_official_docs_provider_label(
                        raw_document.get("provider_mode")
                    ),
                    "snippets": snippets,
                }
            )

    return {
        "library": format_official_docs_library_label(official_docs_result.get("library")),
        "documents": documents,
    }


def build_tool_result_display_data(tool_result: object) -> dict[str, object] | None:
    if not isinstance(tool_result, dict):
        return None

    return {
        "tool_name": format_tool_name_label(tool_result.get("tool_name")) or "Tool",
        "raw_query": _clean_display_text(tool_result.get("raw_query")),
        "input_fields": build_tool_field_display_rows(tool_result.get("tool_input")),
        "output_fields": build_tool_field_display_rows(tool_result.get("tool_output")),
        "error": _clean_display_text(tool_result.get("tool_error")),
    }


def build_tool_field_display_rows(data: object) -> list[dict[str, object]]:
    if not isinstance(data, dict):
        return []

    rows: list[dict[str, object]] = []
    for key, value in data.items():
        lines = _format_tool_field_lines(key, value)
        if not lines:
            continue
        rows.append(
            {
                "label": format_tool_field_label(key),
                "lines": lines,
            }
        )
    return rows
