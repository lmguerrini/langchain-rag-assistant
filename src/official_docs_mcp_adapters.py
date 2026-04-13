from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Protocol
from urllib import error, request as urllib_request

from src.config import Settings, get_settings
from src.schemas import (
    OfficialDocsDocument,
    OfficialDocsLookupRequest,
    OfficialDocsLookupResult,
    OfficialDocsSnippet,
)


LANGCHAIN_OFFICIAL_MCP_TOOL_NAME = "search_docs_by_lang_chain"
OPENAI_OFFICIAL_MCP_TOOL_NAME = "search_openai_docs"


class MCPToolCallFn(Protocol):
    def __call__(
        self,
        *,
        server_url: str,
        tool_name: str,
        arguments: dict[str, object],
        timeout_seconds: float,
    ) -> dict[str, object]:
        ...


def send_mcp_tools_call_request(
    *,
    server_url: str,
    tool_name: str,
    arguments: dict[str, object],
    timeout_seconds: float,
) -> dict[str, object]:
    payload = {
        "jsonrpc": "2.0",
        "id": "official-docs-source",
        "method": "tools/call",
        "params": {
            "name": tool_name,
            "arguments": arguments,
        },
    }
    body = json.dumps(payload).encode("utf-8")
    http_request = urllib_request.Request(
        server_url,
        data=body,
        headers={
            "Accept": "application/json",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        with urllib_request.urlopen(http_request, timeout=timeout_seconds) as response:
            response_body = response.read().decode("utf-8").strip()
    except (error.URLError, TimeoutError) as exc:
        raise RuntimeError(f"Official docs MCP request failed: {exc}") from exc

    try:
        payload = json.loads(response_body)
    except json.JSONDecodeError as exc:
        raise ValueError("Official docs MCP response was not valid JSON.") from exc

    if not isinstance(payload, Mapping):
        raise ValueError("Official docs MCP response must be a JSON object.")

    response_error = payload.get("error")
    if isinstance(response_error, Mapping):
        message = response_error.get("message", "Unknown MCP error")
        raise RuntimeError(f"Official docs MCP returned an error: {message}")

    result = payload.get("result")
    if not isinstance(result, Mapping):
        raise ValueError("Official docs MCP response did not include a result object.")
    return dict(result)


def lookup_langchain_official_docs(
    *,
    request: OfficialDocsLookupRequest,
    settings: Settings | None = None,
    mcp_call_fn: MCPToolCallFn = send_mcp_tools_call_request,
) -> OfficialDocsLookupResult:
    resolved_settings = settings or get_settings()
    result_payload = mcp_call_fn(
        server_url=resolved_settings.official_langchain_docs_mcp_url,
        tool_name=LANGCHAIN_OFFICIAL_MCP_TOOL_NAME,
        arguments={"query": request.query},
        timeout_seconds=resolved_settings.official_docs_timeout_seconds,
    )
    documents = _build_langchain_documents(result_payload)
    if not documents:
        raise ValueError("LangChain official docs MCP returned no usable documents.")
    return OfficialDocsLookupResult(
        library="langchain",
        documents=documents,
    )


def lookup_openai_official_docs(
    *,
    request: OfficialDocsLookupRequest,
    settings: Settings | None = None,
    mcp_call_fn: MCPToolCallFn = send_mcp_tools_call_request,
) -> OfficialDocsLookupResult:
    resolved_settings = settings or get_settings()
    result_payload = mcp_call_fn(
        server_url=resolved_settings.official_openai_docs_mcp_url,
        tool_name=OPENAI_OFFICIAL_MCP_TOOL_NAME,
        arguments={"query": request.query},
        timeout_seconds=resolved_settings.official_docs_timeout_seconds,
    )
    documents = _build_openai_documents(result_payload)
    if not documents:
        raise ValueError("OpenAI official docs MCP returned no usable documents.")
    return OfficialDocsLookupResult(
        library="openai",
        documents=documents,
    )


def _build_langchain_documents(result_payload: Mapping[str, object]) -> list[OfficialDocsDocument]:
    structured_content = result_payload.get("structuredContent")
    if isinstance(structured_content, Mapping):
        docs = structured_content.get("documents")
        if isinstance(docs, list):
            documents = _build_documents_from_direct_entries(
                docs,
                provider_mode="official_mcp",
            )
            if documents:
                return documents

        docs = structured_content.get("docs")
        if isinstance(docs, list):
            documents = _build_documents_from_direct_entries(
                docs,
                provider_mode="official_mcp",
            )
            if documents:
                return documents

    content = result_payload.get("content")
    if isinstance(content, list):
        documents = _build_langchain_documents_from_content_blocks(content)
        if documents:
            return documents

    return []


def _build_langchain_documents_from_content_blocks(
    content_blocks: list[object],
) -> list[OfficialDocsDocument]:
    grouped_blocks: dict[tuple[str, str], list[str]] = {}
    ordered_keys: list[tuple[str, str]] = []

    for block in content_blocks:
        if not isinstance(block, Mapping):
            continue

        block_text = block.get("text")
        if not isinstance(block_text, str):
            continue

        title = _extract_flattened_field_value(block_text, "Title")
        url = _extract_flattened_field_value(block_text, "Link")
        snippet = _extract_flattened_field_value(block_text, "Content")
        if title is None or url is None or snippet is None:
            continue

        key = (title, url)
        if key not in grouped_blocks:
            grouped_blocks[key] = []
            ordered_keys.append(key)
        if snippet not in grouped_blocks[key]:
            grouped_blocks[key].append(snippet)

    documents: list[OfficialDocsDocument] = []
    for title, url in ordered_keys:
        documents.append(
            _build_document(
                title=title,
                url=url,
                provider_mode="official_mcp",
                snippet_texts=grouped_blocks[(title, url)],
            )
        )
    return documents


def _build_openai_documents(result_payload: Mapping[str, object]) -> list[OfficialDocsDocument]:
    structured_content = result_payload.get("structuredContent")
    if isinstance(structured_content, Mapping):
        documents = _build_openai_documents_from_payload(structured_content)
        if documents:
            return documents

    content = result_payload.get("content")
    parsed_payload = _parse_json_payload_from_content_blocks(content)
    if isinstance(parsed_payload, Mapping):
        documents = _build_openai_documents_from_payload(parsed_payload)
        if documents:
            return documents

    return []


def _build_openai_documents_from_payload(
    payload: Mapping[str, object],
) -> list[OfficialDocsDocument]:
    for key in ("documents", "hits", "results", "items"):
        raw_entries = payload.get(key)
        if not isinstance(raw_entries, list):
            continue

        documents = _build_documents_from_openai_entries(raw_entries)
        if documents:
            return documents

    return []


def _build_documents_from_openai_entries(
    raw_entries: list[object],
) -> list[OfficialDocsDocument]:
    documents: list[OfficialDocsDocument] = []

    for entry in raw_entries:
        if not isinstance(entry, Mapping):
            continue

        title = _extract_openai_title(entry)
        url = _first_text(entry, ("url_without_anchor", "url"))
        snippet_texts = _extract_openai_snippet_texts(entry)
        if title is None or url is None or not snippet_texts:
            continue

        documents.append(
            _build_document(
                title=title,
                url=url,
                provider_mode="official_mcp",
                snippet_texts=snippet_texts,
            )
        )

    return documents


def _build_documents_from_direct_entries(
    raw_entries: list[object],
    *,
    provider_mode: str,
) -> list[OfficialDocsDocument]:
    documents: list[OfficialDocsDocument] = []

    for entry in raw_entries:
        if not isinstance(entry, Mapping):
            continue

        title = _first_text(entry, ("title", "name"))
        url = _first_text(entry, ("url", "link"))
        snippet_texts = _extract_direct_snippet_texts(entry)
        if title is None or url is None or not snippet_texts:
            continue

        documents.append(
            _build_document(
                title=title,
                url=url,
                provider_mode=provider_mode,
                snippet_texts=snippet_texts,
            )
        )

    return documents


def _build_document(
    *,
    title: str,
    url: str,
    provider_mode: str,
    snippet_texts: list[str],
) -> OfficialDocsDocument:
    return OfficialDocsDocument(
        title=title,
        url=url,
        provider_mode=provider_mode,
        snippets=[
            OfficialDocsSnippet(text=snippet_text, rank=index)
            for index, snippet_text in enumerate(snippet_texts, start=1)
        ],
    )


def _extract_direct_snippet_texts(entry: Mapping[str, object]) -> list[str]:
    snippet_texts = _extract_snippet_list(entry.get("snippets"))
    if snippet_texts:
        return snippet_texts

    snippet = _first_text(
        entry,
        ("page_content", "content", "summary", "excerpt", "text", "description"),
    )
    return [snippet] if snippet is not None else []


def _extract_openai_title(entry: Mapping[str, object]) -> str | None:
    title = _first_text(entry, ("title", "heading"))
    if title is not None:
        return title

    hierarchy = entry.get("hierarchy")
    if isinstance(hierarchy, Mapping):
        for key in ("lvl3", "lvl2", "lvl1", "lvl0"):
            value = hierarchy.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()

    return None


def _extract_openai_snippet_texts(entry: Mapping[str, object]) -> list[str]:
    snippet_texts = _extract_snippet_list(entry.get("snippets"))
    if snippet_texts:
        return snippet_texts

    snippet = _first_text(entry, ("content", "text", "summary", "excerpt"))
    if snippet is not None:
        return [snippet]

    snippet_result = entry.get("_snippetResult")
    if isinstance(snippet_result, Mapping):
        content = snippet_result.get("content")
        if isinstance(content, Mapping):
            value = content.get("value")
            if isinstance(value, str) and value.strip():
                return [value.strip()]

    return []


def _extract_snippet_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []

    snippets: list[str] = []
    for item in value:
        if isinstance(item, str) and item.strip():
            snippets.append(item.strip())
        elif isinstance(item, Mapping):
            text = item.get("text")
            if isinstance(text, str) and text.strip():
                snippets.append(text.strip())
    return snippets


def _parse_json_payload_from_content_blocks(content_blocks: object) -> dict[str, object] | None:
    if not isinstance(content_blocks, list):
        return None

    for block in content_blocks:
        if not isinstance(block, Mapping):
            continue
        block_text = block.get("text")
        if not isinstance(block_text, str):
            continue

        try:
            payload = json.loads(block_text)
        except json.JSONDecodeError:
            continue

        if isinstance(payload, Mapping):
            return dict(payload)

    return None


def _extract_flattened_field_value(text: str, field_name: str) -> str | None:
    prefix = f"{field_name}:"
    for line in text.splitlines():
        if not line.startswith(prefix):
            continue
        value = line.removeprefix(prefix).strip()
        if value:
            return value
    return None


def _first_text(entry: Mapping[str, object], keys: tuple[str, ...]) -> str | None:
    for key in keys:
        value = entry.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None
