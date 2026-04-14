from __future__ import annotations

import re
from collections.abc import Mapping
from html import unescape
from typing import Protocol

from src.config import Settings, get_settings
from src.official_docs_openai_adapter import run_openai_official_docs_lookup
from src.official_docs_mcp_transport import send_mcp_jsonrpc_request
from src.schemas import (
    OfficialDocsDocument,
    OfficialDocsLookupRequest,
    OfficialDocsLookupResult,
    OfficialDocsSnippet,
)


LANGCHAIN_OFFICIAL_MCP_TOOL_NAME = "search_docs_by_lang_chain"
LANGCHAIN_ALLOWED_TOOL_NAMES = (LANGCHAIN_OFFICIAL_MCP_TOOL_NAME,)
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
    return run_openai_official_docs_lookup(
        request=request,
        settings=resolved_settings,
        mcp_call_fn=mcp_call_fn,
    )


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
