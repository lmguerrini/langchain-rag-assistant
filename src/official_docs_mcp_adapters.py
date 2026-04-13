from __future__ import annotations

import json
import re
from collections.abc import Mapping
from html import unescape
from typing import Protocol
from urllib import error as urllib_error
from urllib import request as urllib_request

from src.config import Settings, get_settings
from src.schemas import (
    OfficialDocsDocument,
    OfficialDocsLookupRequest,
    OfficialDocsLookupResult,
    OfficialDocsSnippet,
)


LANGCHAIN_OFFICIAL_MCP_TOOL_NAME = "search_docs_by_lang_chain"
OPENAI_OFFICIAL_MCP_TOOL_NAME = "search_openai_docs"
LANGCHAIN_ALLOWED_TOOL_NAMES = (LANGCHAIN_OFFICIAL_MCP_TOOL_NAME,)
OPENAI_ALLOWED_TOOL_NAMES = (OPENAI_OFFICIAL_MCP_TOOL_NAME,)
REMOTE_MCP_UNAVAILABLE_MESSAGE = "Remote MCP not available"
LANGCHAIN_MAX_DOCUMENTS = 3
LANGCHAIN_MAX_SNIPPETS_PER_DOCUMENT = 2
LANGCHAIN_MAX_SNIPPET_LENGTH = 240
LANGCHAIN_QUERY_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "documentation",
    "docs",
    "for",
    "from",
    "how",
    "langchain",
    "official",
    "the",
    "to",
    "what",
    "with",
}
OPENAI_MAX_DOCUMENTS = 3
OPENAI_MAX_SNIPPETS_PER_DOCUMENT = 2
OPENAI_MAX_SNIPPET_LENGTH = 240
OPENAI_QUERY_STOPWORDS = {
    "a",
    "an",
    "and",
    "according",
    "are",
    "documentation",
    "docs",
    "for",
    "from",
    "how",
    "official",
    "the",
    "to",
    "what",
    "with",
}


class MCPRequestFn(Protocol):
    def __call__(
        self,
        *,
        server_url: str,
        method: str,
        params: dict[str, object] | None,
        timeout_seconds: float,
    ) -> dict[str, object]:
        ...


def send_mcp_jsonrpc_request(
    *,
    server_url: str,
    method: str,
    params: dict[str, object] | None,
    timeout_seconds: float,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "jsonrpc": "2.0",
        "id": f"official-docs-{method}",
        "method": method,
    }
    if params is not None:
        payload["params"] = params

    body = json.dumps(payload).encode("utf-8")
    http_request = urllib_request.Request(
        server_url,
        data=body,
        headers={
            "Accept": "application/json, text/event-stream",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        with urllib_request.urlopen(http_request, timeout=timeout_seconds) as response:
            response_body = response.read().decode("utf-8").strip()
    except (urllib_error.URLError, TimeoutError) as exc:
        raise RuntimeError(f"Official docs MCP request failed: {exc}") from exc

    rpc_payload = _parse_jsonrpc_response_body(response_body)

    response_error = rpc_payload.get("error")
    if isinstance(response_error, Mapping):
        error_message = response_error.get("message", "Unknown MCP error")
        raise RuntimeError(f"Official docs MCP returned an error: {error_message}")

    result = rpc_payload.get("result")
    if not isinstance(result, Mapping):
        raise RuntimeError("Official docs MCP response did not include a valid result object.")

    return dict(result)


def _parse_jsonrpc_response_body(response_body: str) -> dict[str, object]:
    direct_payload = _parse_json_object(response_body)
    if direct_payload is not None:
        return direct_payload

    sse_payload = _parse_json_object_from_sse_body(response_body)
    if sse_payload is not None:
        return sse_payload

    raise RuntimeError("Official docs MCP response was not valid JSON.")


def _parse_json_object(response_body: str) -> dict[str, object] | None:
    try:
        parsed_payload = json.loads(response_body)
    except json.JSONDecodeError:
        return None

    if not isinstance(parsed_payload, Mapping):
        return None

    return dict(parsed_payload)


def _parse_json_object_from_sse_body(response_body: str) -> dict[str, object] | None:
    data_lines: list[str] = []

    for raw_line in response_body.splitlines():
        stripped_line = raw_line.strip()
        if not stripped_line:
            if data_lines:
                parsed_payload = _parse_json_object("\n".join(data_lines))
                if parsed_payload is not None:
                    return parsed_payload
                data_lines = []
            continue

        if stripped_line.startswith("data:"):
            data_lines.append(stripped_line.removeprefix("data:").strip())

    if data_lines:
        return _parse_json_object("\n".join(data_lines))

    return None


def lookup_langchain_official_docs(
    *,
    request: OfficialDocsLookupRequest,
    settings: Settings | None = None,
    mcp_call_fn: MCPRequestFn = send_mcp_jsonrpc_request,
) -> OfficialDocsLookupResult:
    resolved_settings = settings or get_settings()

    tools_result = mcp_call_fn(
        server_url=resolved_settings.official_langchain_docs_mcp_url,
        method="tools/list",
        params={},
        timeout_seconds=resolved_settings.official_docs_timeout_seconds,
    )
    tool_name = _select_langchain_search_tool_name(tools_result)

    tool_call_result = mcp_call_fn(
        server_url=resolved_settings.official_langchain_docs_mcp_url,
        method="tools/call",
        params={
            "name": tool_name,
            "arguments": {"query": request.query},
        },
        timeout_seconds=resolved_settings.official_docs_timeout_seconds,
    )

    documents = _build_langchain_documents(tool_call_result)
    documents = _shape_langchain_documents(documents, query=request.query)
    if not documents:
        raise ValueError("LangChain official docs MCP returned no usable documents.")

    return OfficialDocsLookupResult(
        library="langchain",
        documents=documents,
    )


def _select_langchain_search_tool_name(tools_result: Mapping[str, object]) -> str:
    raw_tools = tools_result.get("tools")
    if not isinstance(raw_tools, list):
        raise RuntimeError("Official docs MCP response did not include a valid tools list.")

    tool_names = [
        tool.get("name")
        for tool in raw_tools
        if isinstance(tool, Mapping) and isinstance(tool.get("name"), str)
    ]
    allowed_matches = [
        tool_name
        for tool_name in tool_names
        if tool_name in LANGCHAIN_ALLOWED_TOOL_NAMES
    ]
    if len(allowed_matches) == 1:
        return allowed_matches[0]
    if len(allowed_matches) > 1:
        raise ValueError("LangChain official docs MCP exposed multiple matching search tools.")
    if len(tool_names) == 1:
        return tool_names[0]

    inferred_matches = [
        tool_name
        for tool_name in tool_names
        if "search" in tool_name.lower()
        and ("doc" in tool_name.lower() or "langchain" in tool_name.lower())
    ]
    if len(inferred_matches) == 1:
        return inferred_matches[0]

    raise ValueError("LangChain official docs MCP did not expose a suitable search tool.")


def _build_langchain_documents(result_payload: Mapping[str, object]) -> list[OfficialDocsDocument]:
    structured_content = result_payload.get("structuredContent")
    if isinstance(structured_content, Mapping):
        documents = _build_langchain_documents_from_structured_content(structured_content)
        if documents is not None:
            return documents

    content = result_payload.get("content")
    if isinstance(content, list):
        return _build_langchain_documents_from_content_blocks(content)

    raise ValueError("LangChain official docs MCP returned an unsupported response shape.")


def _shape_langchain_documents(
    documents: list[OfficialDocsDocument],
    *,
    query: str,
) -> list[OfficialDocsDocument]:
    deduped_documents = _dedupe_langchain_documents(documents)
    query_tokens = _tokenize_langchain_query(query)

    scored_documents: list[tuple[int, int, OfficialDocsDocument]] = []
    for index, document in enumerate(deduped_documents):
        score = _score_langchain_document(document, query_tokens=query_tokens)
        scored_documents.append((score, index, document))

    if any(score > 0 for score, _, _ in scored_documents):
        selected_documents = [
            document
            for score, _, document in sorted(
                (item for item in scored_documents if item[0] > 0),
                key=lambda item: (-item[0], item[1]),
            )[:LANGCHAIN_MAX_DOCUMENTS]
        ]
    else:
        selected_documents = [
            document
            for _, _, document in scored_documents[:LANGCHAIN_MAX_DOCUMENTS]
        ]

    return [_compact_langchain_document(document) for document in selected_documents]


def _dedupe_langchain_documents(documents: list[OfficialDocsDocument]) -> list[OfficialDocsDocument]:
    merged_documents: dict[str, OfficialDocsDocument] = {}
    ordered_urls: list[str] = []

    for document in documents:
        canonical_url = _canonicalize_langchain_url(document.url)
        existing_document = merged_documents.get(canonical_url)
        if existing_document is None:
            merged_documents[canonical_url] = document
            ordered_urls.append(canonical_url)
            continue

        merged_documents[canonical_url] = OfficialDocsDocument(
            title=existing_document.title,
            url=existing_document.url,
            provider_mode=existing_document.provider_mode,
            snippets=_merge_langchain_snippets(
                existing_document.snippets,
                document.snippets,
            ),
        )

    return [merged_documents[url] for url in ordered_urls]


def _merge_langchain_snippets(
    left_snippets: list[OfficialDocsSnippet],
    right_snippets: list[OfficialDocsSnippet],
) -> list[OfficialDocsSnippet]:
    merged_snippets: list[OfficialDocsSnippet] = []
    seen_texts: set[str] = set()

    for snippet in [*left_snippets, *right_snippets]:
        if snippet.text in seen_texts:
            continue
        merged_snippets.append(
            OfficialDocsSnippet(
                text=snippet.text,
                rank=len(merged_snippets) + 1,
            )
        )
        seen_texts.add(snippet.text)

    return merged_snippets


def _score_langchain_document(
    document: OfficialDocsDocument,
    *,
    query_tokens: set[str],
) -> int:
    if not query_tokens:
        return 0

    title_overlap = query_tokens & _tokenize_langchain_query(document.title)
    url_overlap = query_tokens & _tokenize_langchain_query(document.url)
    snippet_overlap = query_tokens & _tokenize_langchain_query(
        " ".join(snippet.text for snippet in document.snippets)
    )
    return (len(title_overlap) * 3) + (len(url_overlap) * 2) + len(snippet_overlap)


def _compact_langchain_document(document: OfficialDocsDocument) -> OfficialDocsDocument:
    compact_snippets: list[OfficialDocsSnippet] = []
    seen_texts: set[str] = set()

    for snippet in document.snippets:
        trimmed_text = _trim_langchain_snippet_text(snippet.text)
        if trimmed_text in seen_texts:
            continue
        compact_snippets.append(
            OfficialDocsSnippet(
                text=trimmed_text,
                rank=len(compact_snippets) + 1,
            )
        )
        seen_texts.add(trimmed_text)
        if len(compact_snippets) >= LANGCHAIN_MAX_SNIPPETS_PER_DOCUMENT:
            break

    return OfficialDocsDocument(
        title=document.title,
        url=document.url,
        provider_mode="official_mcp",
        snippets=compact_snippets,
    )


def _trim_langchain_snippet_text(text: str) -> str:
    if len(text) <= LANGCHAIN_MAX_SNIPPET_LENGTH:
        return text

    truncated_text = text[: LANGCHAIN_MAX_SNIPPET_LENGTH - 3].rstrip()
    if " " in truncated_text:
        truncated_text = truncated_text.rsplit(" ", 1)[0]
    return f"{truncated_text}..."


def _canonicalize_langchain_url(url: str) -> str:
    return url.split("#", 1)[0].rstrip("/").lower()


def _tokenize_langchain_query(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9]+", text.lower())
        if len(token) > 2 and token not in LANGCHAIN_QUERY_STOPWORDS
    }


def _build_langchain_documents_from_structured_content(
    structured_content: Mapping[str, object],
) -> list[OfficialDocsDocument] | None:
    for key in ("documents", "docs"):
        raw_entries = structured_content.get(key)
        if raw_entries is None:
            continue
        if not isinstance(raw_entries, list):
            raise ValueError("LangChain official docs MCP returned malformed documents.")
        if not raw_entries:
            return []

        documents = _build_langchain_documents_from_entries(raw_entries)
        if documents:
            return documents

        raise ValueError("LangChain official docs MCP returned malformed documents.")

    return None


def _build_langchain_documents_from_entries(
    raw_entries: list[object],
) -> list[OfficialDocsDocument]:
    documents: list[OfficialDocsDocument] = []

    for entry in raw_entries:
        if not isinstance(entry, Mapping):
            continue

        title = _first_text(entry, ("title", "name"))
        url = _first_text(entry, ("url", "link"))
        snippet_texts = _extract_langchain_snippet_texts(entry)
        if title is None or url is None or not snippet_texts:
            continue

        documents.append(
            OfficialDocsDocument(
                title=title,
                url=url,
                provider_mode="official_mcp",
                snippets=[
                    OfficialDocsSnippet(text=snippet_text, rank=index)
                    for index, snippet_text in enumerate(snippet_texts, start=1)
                ],
            )
        )

    return documents


def _extract_langchain_snippet_texts(entry: Mapping[str, object]) -> list[str]:
    snippet_list = _extract_snippet_list(entry.get("snippets"))
    if snippet_list:
        return snippet_list

    direct_content = _first_text(
        entry,
        ("page_content", "content", "summary", "excerpt", "text", "description"),
    )
    if direct_content is None:
        return []
    return [direct_content]


def _build_langchain_documents_from_content_blocks(
    content_blocks: list[object],
) -> list[OfficialDocsDocument]:
    if not content_blocks:
        return []

    grouped_entries: dict[tuple[str, str], list[str]] = {}
    ordered_keys: list[tuple[str, str]] = []

    for block in content_blocks:
        if not isinstance(block, Mapping):
            continue
        block_text = block.get("text")
        if not isinstance(block_text, str):
            continue

        title = _extract_langchain_flattened_field_value(block_text, "Title")
        url = _extract_langchain_flattened_field_value(block_text, "Link")
        snippet = _extract_langchain_flattened_field_value(block_text, "Content")
        if title is None or url is None or snippet is None:
            continue

        key = (title, url)
        if key not in grouped_entries:
            grouped_entries[key] = []
            ordered_keys.append(key)
        if snippet not in grouped_entries[key]:
            grouped_entries[key].append(snippet)

    if not ordered_keys:
        raise ValueError("LangChain official docs MCP returned malformed content blocks.")

    return [
        OfficialDocsDocument(
            title=title,
            url=url,
            provider_mode="official_mcp",
            snippets=[
                OfficialDocsSnippet(text=snippet_text, rank=index)
                for index, snippet_text in enumerate(grouped_entries[(title, url)], start=1)
            ],
        )
        for title, url in ordered_keys
    ]


def _extract_langchain_flattened_field_value(text: str, field_name: str) -> str | None:
    prefix = f"{field_name}:"
    for raw_line in text.splitlines():
        stripped_line = raw_line.strip()
        if not stripped_line.startswith(prefix):
            continue
        cleaned_value = _clean_openai_text(stripped_line.removeprefix(prefix))
        if cleaned_value is not None:
            return cleaned_value
    return None


def lookup_openai_official_docs(
    *,
    request: OfficialDocsLookupRequest,
    settings: Settings | None = None,
    mcp_call_fn: MCPRequestFn = send_mcp_jsonrpc_request,
) -> OfficialDocsLookupResult:
    resolved_settings = settings or get_settings()

    tools_result = mcp_call_fn(
        server_url=resolved_settings.official_openai_docs_mcp_url,
        method="tools/list",
        params={},
        timeout_seconds=resolved_settings.official_docs_timeout_seconds,
    )
    tool_name = _select_openai_search_tool_name(tools_result)

    tool_call_result = mcp_call_fn(
        server_url=resolved_settings.official_openai_docs_mcp_url,
        method="tools/call",
        params={
            "name": tool_name,
            "arguments": {"query": request.query},
        },
        timeout_seconds=resolved_settings.official_docs_timeout_seconds,
    )

    documents = _build_openai_documents(tool_call_result)
    documents = _shape_openai_documents(documents, query=request.query)
    if not documents:
        raise ValueError("OpenAI official docs MCP returned no usable documents.")

    return OfficialDocsLookupResult(
        library="openai",
        documents=documents,
    )


def _select_openai_search_tool_name(tools_result: Mapping[str, object]) -> str:
    raw_tools = tools_result.get("tools")
    if not isinstance(raw_tools, list):
        raise RuntimeError("Official docs MCP response did not include a valid tools list.")

    tool_names = [
        tool.get("name")
        for tool in raw_tools
        if isinstance(tool, Mapping) and isinstance(tool.get("name"), str)
    ]
    allowed_matches = [
        tool_name
        for tool_name in tool_names
        if tool_name in OPENAI_ALLOWED_TOOL_NAMES
    ]
    if len(allowed_matches) == 1:
        return allowed_matches[0]
    if len(allowed_matches) > 1:
        raise ValueError("OpenAI official docs MCP exposed multiple matching search tools.")
    if len(tool_names) == 1:
        return tool_names[0]
    raise ValueError("OpenAI official docs MCP did not expose a suitable search tool.")


def _build_openai_documents(result_payload: Mapping[str, object]) -> list[OfficialDocsDocument]:
    structured_content = result_payload.get("structuredContent")
    if isinstance(structured_content, Mapping):
        documents = _build_openai_documents_from_search_payload(structured_content)
        if documents is not None:
            return documents

    embedded_payload = _parse_json_object_from_content_blocks(result_payload.get("content"))
    if embedded_payload is not None:
        documents = _build_openai_documents_from_search_payload(embedded_payload)
        if documents is not None:
            return documents

    raise ValueError("OpenAI official docs MCP returned an unsupported response shape.")


def _shape_openai_documents(
    documents: list[OfficialDocsDocument],
    *,
    query: str,
) -> list[OfficialDocsDocument]:
    deduped_documents = _dedupe_openai_documents(documents)
    query_tokens = _tokenize_openai_query(query)

    scored_documents: list[tuple[int, int, OfficialDocsDocument]] = []
    for index, document in enumerate(deduped_documents):
        score = _score_openai_document(document, query_tokens=query_tokens)
        scored_documents.append((score, index, document))

    if any(score > 0 for score, _, _ in scored_documents):
        selected_documents = [
            document
            for score, _, document in sorted(
                (item for item in scored_documents if item[0] > 0),
                key=lambda item: (-item[0], item[1]),
            )[:OPENAI_MAX_DOCUMENTS]
        ]
    else:
        selected_documents = [
            document
            for _, _, document in scored_documents[:OPENAI_MAX_DOCUMENTS]
        ]

    return [_compact_openai_document(document) for document in selected_documents]


def _dedupe_openai_documents(documents: list[OfficialDocsDocument]) -> list[OfficialDocsDocument]:
    merged_documents: dict[str, OfficialDocsDocument] = {}
    ordered_urls: list[str] = []

    for document in documents:
        canonical_url = _canonicalize_openai_url(document.url)
        existing_document = merged_documents.get(canonical_url)
        if existing_document is None:
            merged_documents[canonical_url] = document
            ordered_urls.append(canonical_url)
            continue

        merged_documents[canonical_url] = OfficialDocsDocument(
            title=existing_document.title,
            url=existing_document.url,
            provider_mode=existing_document.provider_mode,
            snippets=_merge_openai_snippets(
                existing_document.snippets,
                document.snippets,
            ),
        )

    return [merged_documents[url] for url in ordered_urls]


def _merge_openai_snippets(
    left_snippets: list[OfficialDocsSnippet],
    right_snippets: list[OfficialDocsSnippet],
) -> list[OfficialDocsSnippet]:
    merged_snippets: list[OfficialDocsSnippet] = []
    seen_texts: set[str] = set()

    for snippet in [*left_snippets, *right_snippets]:
        if snippet.text in seen_texts:
            continue
        merged_snippets.append(
            OfficialDocsSnippet(
                text=snippet.text,
                rank=len(merged_snippets) + 1,
            )
        )
        seen_texts.add(snippet.text)

    return merged_snippets


def _score_openai_document(
    document: OfficialDocsDocument,
    *,
    query_tokens: set[str],
) -> int:
    if not query_tokens:
        return 0

    title_overlap = query_tokens & _tokenize_openai_query(document.title)
    url_overlap = query_tokens & _tokenize_openai_query(document.url)
    snippet_overlap = query_tokens & _tokenize_openai_query(
        " ".join(snippet.text for snippet in document.snippets)
    )
    return (len(title_overlap) * 3) + (len(url_overlap) * 2) + len(snippet_overlap)


def _compact_openai_document(document: OfficialDocsDocument) -> OfficialDocsDocument:
    compact_snippets: list[OfficialDocsSnippet] = []
    seen_texts: set[str] = set()

    for snippet in document.snippets:
        trimmed_text = _trim_openai_snippet_text(snippet.text)
        if trimmed_text in seen_texts:
            continue
        compact_snippets.append(
            OfficialDocsSnippet(
                text=trimmed_text,
                rank=len(compact_snippets) + 1,
            )
        )
        seen_texts.add(trimmed_text)
        if len(compact_snippets) >= OPENAI_MAX_SNIPPETS_PER_DOCUMENT:
            break

    return OfficialDocsDocument(
        title=document.title,
        url=document.url,
        provider_mode="official_mcp",
        snippets=compact_snippets,
    )


def _trim_openai_snippet_text(text: str) -> str:
    if len(text) <= OPENAI_MAX_SNIPPET_LENGTH:
        return text

    truncated_text = text[: OPENAI_MAX_SNIPPET_LENGTH - 3].rstrip()
    if " " in truncated_text:
        truncated_text = truncated_text.rsplit(" ", 1)[0]
    return f"{truncated_text}..."


def _canonicalize_openai_url(url: str) -> str:
    return url.split("#", 1)[0].rstrip("/").lower()


def _build_openai_documents_from_search_payload(
    payload: Mapping[str, object],
) -> list[OfficialDocsDocument] | None:
    for key in ("documents", "hits", "results"):
        raw_entries = payload.get(key)
        if raw_entries is None:
            continue
        if not isinstance(raw_entries, list):
            raise ValueError("OpenAI official docs MCP returned malformed search results.")
        if not raw_entries:
            return []

        documents: list[OfficialDocsDocument] = []
        for entry in raw_entries:
            document = _build_openai_document(entry)
            if document is not None:
                documents.append(document)

        if documents:
            return documents

        raise ValueError("OpenAI official docs MCP returned malformed search results.")

    return None


def _build_openai_document(entry: object) -> OfficialDocsDocument | None:
    if not isinstance(entry, Mapping):
        return None

    title = _extract_openai_title(entry)
    url = _first_text(entry, ("url_without_anchor", "url"))
    snippet_texts = _extract_openai_snippet_texts(entry)
    if title is None or url is None or not snippet_texts:
        return None

    return OfficialDocsDocument(
        title=title,
        url=url,
        provider_mode="official_mcp",
        snippets=[
            OfficialDocsSnippet(text=snippet_text, rank=index)
            for index, snippet_text in enumerate(snippet_texts, start=1)
        ],
    )


def _extract_openai_title(entry: Mapping[str, object]) -> str | None:
    explicit_title = _first_text(entry, ("title", "heading"))
    if explicit_title is not None:
        return explicit_title

    hierarchy = entry.get("hierarchy")
    if not isinstance(hierarchy, Mapping):
        return None

    for key in ("lvl3", "lvl2", "lvl1", "lvl0"):
        value = hierarchy.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()

    return None


def _extract_openai_snippet_texts(entry: Mapping[str, object]) -> list[str]:
    snippet_list = _extract_snippet_list(entry.get("snippets"))
    if snippet_list:
        return snippet_list

    direct_content = _first_text(entry, ("content", "summary", "text", "excerpt"))
    if direct_content is not None:
        return [direct_content]

    snippet_result = entry.get("_snippetResult")
    if isinstance(snippet_result, Mapping):
        content_entry = snippet_result.get("content")
        if isinstance(content_entry, Mapping):
            snippet_value = _clean_openai_text(content_entry.get("value"))
            if snippet_value is not None:
                return [snippet_value]

    highlight_result = entry.get("_highlightResult")
    if isinstance(highlight_result, Mapping):
        content_entry = highlight_result.get("content")
        if isinstance(content_entry, Mapping):
            highlight_value = _clean_openai_text(content_entry.get("value"))
            if highlight_value is not None:
                return [highlight_value]

    return []


def _extract_snippet_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []

    snippets: list[str] = []
    for item in value:
        if isinstance(item, str):
            cleaned = _clean_openai_text(item)
            if cleaned is not None:
                snippets.append(cleaned)
        elif isinstance(item, Mapping):
            cleaned = _clean_openai_text(item.get("text"))
            if cleaned is not None:
                snippets.append(cleaned)
    return snippets


def _parse_json_object_from_content_blocks(content_blocks: object) -> dict[str, object] | None:
    if not isinstance(content_blocks, list):
        return None

    for block in content_blocks:
        if not isinstance(block, Mapping):
            continue
        block_text = block.get("text")
        if not isinstance(block_text, str):
            continue
        try:
            parsed_payload = json.loads(block_text)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed_payload, Mapping):
            return dict(parsed_payload)

    return None


def _first_text(entry: Mapping[str, object], keys: tuple[str, ...]) -> str | None:
    for key in keys:
        cleaned = _clean_openai_text(entry.get(key))
        if cleaned is not None:
            return cleaned
    return None


def _clean_openai_text(value: object) -> str | None:
    if not isinstance(value, str):
        return None

    cleaned = unescape(re.sub(r"<[^>]+>", " ", value))
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if not cleaned:
        return None
    return cleaned


def _tokenize_openai_query(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9]+", text.lower())
        if len(token) > 2 and token not in OPENAI_QUERY_STOPWORDS
    }
