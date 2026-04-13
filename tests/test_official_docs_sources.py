import json

import pytest

from src.official_docs_fallback_adapters import (
    lookup_chroma_official_docs,
    lookup_streamlit_official_docs,
)
from src.official_docs_mcp_adapters import (
    REMOTE_MCP_UNAVAILABLE_MESSAGE,
    lookup_langchain_official_docs,
    lookup_openai_official_docs,
    send_mcp_jsonrpc_request,
)
from src.official_docs_sources import (
    lookup_official_docs_documents,
    select_official_docs_source_adapter,
)
from src.schemas import OfficialDocsLookupRequest, OfficialDocsLookupResult


def test_select_official_docs_source_adapter_returns_requested_adapter() -> None:
    adapter = lambda *, request: OfficialDocsLookupResult(library="langchain", documents=[])

    selected_adapter = select_official_docs_source_adapter(
        library="langchain",
        adapters={"langchain": adapter},
    )

    assert selected_adapter is adapter


def test_lookup_langchain_official_docs_raises_when_remote_mcp_is_unavailable() -> None:
    request = OfficialDocsLookupRequest(
        query="How should I start a RAG app?",
        library="langchain",
    )
    called = False

    def mcp_call_fn(*, server_url, tool_name, arguments, timeout_seconds):
        nonlocal called
        called = True
        return {}

    with pytest.raises(NotImplementedError, match=REMOTE_MCP_UNAVAILABLE_MESSAGE):
        lookup_langchain_official_docs(
            request=request,
            mcp_call_fn=mcp_call_fn,
        )

    assert called is False


def test_lookup_openai_official_docs_returns_normalized_documents() -> None:
    request = OfficialDocsLookupRequest(
        query="How do streaming responses work?",
        library="openai",
    )
    calls: list[tuple[str, dict[str, object] | None]] = []

    payload = {
        "hits": [
            {
                "url": "https://developers.openai.com/docs/guides/streaming-responses#chunk",
                "url_without_anchor": "https://developers.openai.com/docs/guides/streaming-responses",
                "content": "Use streaming to receive partial response events as they are generated.",
                "hierarchy": {
                    "lvl0": "Documentation",
                    "lvl1": "Streaming responses",
                },
            }
        ]
    }

    def mcp_call_fn(*, server_url, method, params, timeout_seconds):
        assert server_url == "https://developers.openai.com/mcp"
        assert timeout_seconds == 15.0
        calls.append((method, params))
        if method == "tools/list":
            return {
                "tools": [
                    {
                        "name": "search_openai_docs",
                        "description": "Search the OpenAI developer docs.",
                    }
                ]
            }
        if method == "tools/call":
            assert params == {
                "name": "search_openai_docs",
                "arguments": {"query": "How do streaming responses work?"},
            }
            return {
                "content": [
                    {
                        "type": "text",
                        "text": json.dumps(payload),
                    }
                ]
            }
        raise AssertionError(f"Unexpected method: {method}")

    result = lookup_openai_official_docs(
        request=request,
        mcp_call_fn=mcp_call_fn,
    )

    assert calls == [
        ("tools/list", {}),
        (
            "tools/call",
            {
                "name": "search_openai_docs",
                "arguments": {"query": "How do streaming responses work?"},
            },
        ),
    ]
    assert result.library == "openai"
    assert result.documents[0].provider_mode == "official_mcp"
    assert result.documents[0].title == "Streaming responses"
    assert result.documents[0].url == "https://developers.openai.com/docs/guides/streaming-responses"
    assert "partial response events" in result.documents[0].snippets[0].text


def test_send_mcp_jsonrpc_request_uses_streamable_accept_header_and_parses_plain_json(
    monkeypatch,
) -> None:
    captured_headers: dict[str, str] = {}

    class StubResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self) -> bytes:
            return b'{"jsonrpc":"2.0","id":"request-1","result":{"tools":[]}}'

    def fake_urlopen(request, timeout):
        nonlocal captured_headers
        captured_headers = dict(request.header_items())
        assert timeout == 15.0
        return StubResponse()

    monkeypatch.setattr("src.official_docs_mcp_adapters.urllib_request.urlopen", fake_urlopen)

    result = send_mcp_jsonrpc_request(
        server_url="https://developers.openai.com/mcp",
        method="tools/list",
        params={},
        timeout_seconds=15.0,
    )

    assert result == {"tools": []}
    assert captured_headers["Accept"] == "application/json, text/event-stream"
    assert captured_headers["Content-type"] == "application/json"


def test_send_mcp_jsonrpc_request_parses_sse_json_body(monkeypatch) -> None:
    class StubResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self) -> bytes:
            return (
                b'event: message\n'
                b'data: {"jsonrpc":"2.0","id":"request-1","result":{"tools":[]}}\n\n'
            )

    def fake_urlopen(request, timeout):
        assert timeout == 15.0
        return StubResponse()

    monkeypatch.setattr("src.official_docs_mcp_adapters.urllib_request.urlopen", fake_urlopen)

    result = send_mcp_jsonrpc_request(
        server_url="https://developers.openai.com/mcp",
        method="tools/list",
        params={},
        timeout_seconds=15.0,
    )

    assert result == {"tools": []}


def test_send_mcp_jsonrpc_request_rejects_malformed_sse_body(monkeypatch) -> None:
    class StubResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self) -> bytes:
            return b"event: message\ndata: not-json\n\n"

    def fake_urlopen(request, timeout):
        assert timeout == 15.0
        return StubResponse()

    monkeypatch.setattr("src.official_docs_mcp_adapters.urllib_request.urlopen", fake_urlopen)

    with pytest.raises(RuntimeError, match="Official docs MCP response was not valid JSON."):
        send_mcp_jsonrpc_request(
            server_url="https://developers.openai.com/mcp",
            method="tools/list",
            params={},
            timeout_seconds=15.0,
        )


def test_lookup_openai_official_docs_prefers_search_tool_when_multiple_tools_are_present() -> None:
    request = OfficialDocsLookupRequest(
        query="How do streaming responses work?",
        library="openai",
    )
    tool_call_params: dict[str, object] | None = None

    def mcp_call_fn(*, server_url, method, params, timeout_seconds):
        nonlocal tool_call_params
        if method == "tools/list":
            return {
                "tools": [
                    {"name": "list_openai_docs"},
                    {"name": "search_openai_docs"},
                    {"name": "fetch_openai_doc"},
                ]
            }
        if method == "tools/call":
            tool_call_params = params
            return {
                "structuredContent": {
                    "hits": [
                        {
                            "url_without_anchor": (
                                "https://developers.openai.com/docs/guides/streaming-responses"
                            ),
                            "content": "Use streaming to receive partial response events.",
                            "hierarchy": {"lvl1": "Streaming responses"},
                        }
                    ]
                }
            }
        raise AssertionError(f"Unexpected method: {method}")

    result = lookup_openai_official_docs(
        request=request,
        mcp_call_fn=mcp_call_fn,
    )

    assert tool_call_params == {
        "name": "search_openai_docs",
        "arguments": {"query": "How do streaming responses work?"},
    }
    assert result.library == "openai"
    assert result.documents[0].title == "Streaming responses"


def test_lookup_openai_official_docs_keeps_only_top_relevant_documents() -> None:
    request = OfficialDocsLookupRequest(
        query="How do streaming responses work?",
        library="openai",
    )

    def mcp_call_fn(*, server_url, method, params, timeout_seconds):
        if method == "tools/list":
            return {"tools": [{"name": "search_openai_docs"}]}
        return {
            "structuredContent": {
                "hits": [
                    {
                        "url_without_anchor": (
                            "https://developers.openai.com/docs/guides/streaming-responses"
                        ),
                        "content": "Use streaming responses to receive partial output events.",
                        "hierarchy": {"lvl1": "Streaming responses"},
                    },
                    {
                        "url_without_anchor": (
                            "https://developers.openai.com/docs/api-reference/responses"
                        ),
                        "content": "The Responses API supports streaming output.",
                        "hierarchy": {"lvl1": "Responses API"},
                    },
                    {
                        "url_without_anchor": (
                            "https://developers.openai.com/docs/guides/batch"
                        ),
                        "content": "Batch processing lets you submit many requests offline.",
                        "hierarchy": {"lvl1": "Batch API"},
                    },
                    {
                        "url_without_anchor": (
                            "https://developers.openai.com/docs/guides/realtime"
                        ),
                        "content": "Realtime sessions support low-latency audio interactions.",
                        "hierarchy": {"lvl1": "Realtime API"},
                    },
                ]
            }
        }

    result = lookup_openai_official_docs(
        request=request,
        mcp_call_fn=mcp_call_fn,
    )

    assert [document.title for document in result.documents] == [
        "Streaming responses",
        "Responses API",
    ]


def test_lookup_openai_official_docs_dedupes_by_canonical_url() -> None:
    request = OfficialDocsLookupRequest(
        query="How do streaming responses work?",
        library="openai",
    )

    def mcp_call_fn(*, server_url, method, params, timeout_seconds):
        if method == "tools/list":
            return {"tools": [{"name": "search_openai_docs"}]}
        return {
            "structuredContent": {
                "hits": [
                    {
                        "url_without_anchor": (
                            "https://developers.openai.com/docs/guides/streaming-responses"
                        ),
                        "content": "Use streaming responses to receive partial output events.",
                        "hierarchy": {"lvl1": "Streaming responses"},
                    },
                    {
                        "url_without_anchor": (
                            "https://developers.openai.com/docs/guides/streaming-responses"
                        ),
                        "content": "Streaming responses emit events as tokens are generated.",
                        "hierarchy": {"lvl1": "Streaming responses"},
                    },
                ]
            }
        }

    result = lookup_openai_official_docs(
        request=request,
        mcp_call_fn=mcp_call_fn,
    )

    assert len(result.documents) == 1
    assert result.documents[0].url == (
        "https://developers.openai.com/docs/guides/streaming-responses"
    )
    assert len(result.documents[0].snippets) == 2


def test_lookup_openai_official_docs_drops_zero_overlap_docs_when_stronger_matches_exist() -> None:
    request = OfficialDocsLookupRequest(
        query="How do streaming responses work?",
        library="openai",
    )

    def mcp_call_fn(*, server_url, method, params, timeout_seconds):
        if method == "tools/list":
            return {"tools": [{"name": "search_openai_docs"}]}
        return {
            "structuredContent": {
                "hits": [
                    {
                        "url_without_anchor": (
                            "https://developers.openai.com/docs/guides/streaming-responses"
                        ),
                        "content": "Use streaming responses to receive partial output events.",
                        "hierarchy": {"lvl1": "Streaming responses"},
                    },
                    {
                        "url_without_anchor": (
                            "https://developers.openai.com/docs/guides/images"
                        ),
                        "content": "Generate or edit images with the Images API.",
                        "hierarchy": {"lvl1": "Images"},
                    },
                ]
            }
        }

    result = lookup_openai_official_docs(
        request=request,
        mcp_call_fn=mcp_call_fn,
    )

    assert [document.title for document in result.documents] == ["Streaming responses"]


def test_lookup_openai_official_docs_preserves_small_fallback_set_when_all_docs_are_weak() -> None:
    request = OfficialDocsLookupRequest(
        query="How do I optimize latency?",
        library="openai",
    )

    def mcp_call_fn(*, server_url, method, params, timeout_seconds):
        if method == "tools/list":
            return {"tools": [{"name": "search_openai_docs"}]}
        return {
            "structuredContent": {
                "hits": [
                    {
                        "url_without_anchor": "https://developers.openai.com/docs/guides/batch",
                        "content": "Batch processing lets you submit many requests offline.",
                        "hierarchy": {"lvl1": "Batch API"},
                    },
                    {
                        "url_without_anchor": "https://developers.openai.com/docs/guides/images",
                        "content": "Generate images with the Images API.",
                        "hierarchy": {"lvl1": "Images"},
                    },
                    {
                        "url_without_anchor": "https://developers.openai.com/docs/guides/realtime",
                        "content": "Realtime sessions support audio interactions.",
                        "hierarchy": {"lvl1": "Realtime API"},
                    },
                    {
                        "url_without_anchor": "https://developers.openai.com/docs/guides/tools",
                        "content": "Tools let models call external functions.",
                        "hierarchy": {"lvl1": "Tool calling"},
                    },
                ]
            }
        }

    result = lookup_openai_official_docs(
        request=request,
        mcp_call_fn=mcp_call_fn,
    )

    assert [document.title for document in result.documents] == [
        "Batch API",
        "Images",
        "Realtime API",
    ]


def test_lookup_openai_official_docs_trims_long_snippets_and_limits_snippet_count() -> None:
    request = OfficialDocsLookupRequest(
        query="How do streaming responses work?",
        library="openai",
    )
    long_snippet = (
        "Streaming responses emit partial output events while a response is being generated "
        "so applications can render text progressively and react before the final response "
        "has completed. " * 4
    )

    def mcp_call_fn(*, server_url, method, params, timeout_seconds):
        if method == "tools/list":
            return {"tools": [{"name": "search_openai_docs"}]}
        return {
            "structuredContent": {
                "documents": [
                    {
                        "title": "Streaming responses",
                        "url": "https://developers.openai.com/docs/guides/streaming-responses",
                        "snippets": [
                            {"text": long_snippet},
                            {"text": "Streaming events can be consumed incrementally."},
                            {"text": "This third snippet should be dropped by the adapter."},
                        ],
                    }
                ]
            }
        }

    result = lookup_openai_official_docs(
        request=request,
        mcp_call_fn=mcp_call_fn,
    )

    assert len(result.documents) == 1
    assert len(result.documents[0].snippets) == 2
    assert len(result.documents[0].snippets[0].text) <= 240
    assert result.documents[0].snippets[0].text.endswith("...")
    assert result.documents[0].snippets[1].text == (
        "Streaming events can be consumed incrementally."
    )


def test_lookup_streamlit_official_docs_uses_deterministic_fallback_manifest(tmp_path) -> None:
    manifest_path = tmp_path / "source_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "entries": [
                    {
                        "library": "streamlit",
                        "title": "st.session_state",
                        "url": "https://docs.streamlit.io/session-state",
                        "snippets": [
                            "st.session_state stores values across reruns."
                        ],
                        "keywords": ["streamlit", "session", "state", "reruns", "chat"],
                    },
                    {
                        "library": "streamlit",
                        "title": "st.chat_message",
                        "url": "https://docs.streamlit.io/chat-message",
                        "snippets": [
                            "st.chat_message renders chat containers."
                        ],
                        "keywords": ["streamlit", "chat", "message", "ui"],
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    request = OfficialDocsLookupRequest(
        query="How do I keep chat history across reruns in Streamlit?",
        library="streamlit",
    )

    result = lookup_streamlit_official_docs(
        request=request,
        manifest_path=manifest_path,
    )

    assert result.library == "streamlit"
    assert result.documents[0].provider_mode == "official_fallback"
    assert result.documents[0].title == "st.session_state"
    assert "reruns" in result.documents[0].snippets[0].text


def test_lookup_chroma_official_docs_uses_deterministic_fallback_manifest(tmp_path) -> None:
    manifest_path = tmp_path / "source_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "entries": [
                    {
                        "library": "chroma",
                        "title": "Persistent Client",
                        "url": "https://docs.trychroma.com/persistent-client",
                        "snippets": [
                            "Use a persistent client to keep collections on disk."
                        ],
                        "keywords": ["chroma", "persistent", "client", "disk", "restarts"],
                    },
                    {
                        "library": "chroma",
                        "title": "Metadata Filtering",
                        "url": "https://docs.trychroma.com/metadata-filtering",
                        "snippets": [
                            "Metadata filters narrow retrieval by attributes."
                        ],
                        "keywords": ["chroma", "metadata", "filtering", "retrieval"],
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    request = OfficialDocsLookupRequest(
        query="How do I persist Chroma collections across restarts?",
        library="chroma",
    )

    result = lookup_chroma_official_docs(
        request=request,
        manifest_path=manifest_path,
    )

    assert result.library == "chroma"
    assert result.documents[0].provider_mode == "official_fallback"
    assert result.documents[0].title == "Persistent Client"
    assert "collections on disk" in result.documents[0].snippets[0].text


def test_lookup_official_docs_documents_rejects_empty_adapter_output() -> None:
    request = OfficialDocsLookupRequest(
        query="How do I start?",
        library="langchain",
    )

    def empty_adapter(*, request):
        return OfficialDocsLookupResult(
            library="langchain",
            documents=[],
        )

    with pytest.raises(ValueError, match="returned no documents"):
        lookup_official_docs_documents(
            request=request,
            adapters={"langchain": empty_adapter},
        )


def test_lookup_openai_official_docs_rejects_malformed_payload() -> None:
    request = OfficialDocsLookupRequest(
        query="How do streaming responses work?",
        library="openai",
    )

    def mcp_call_fn(*, server_url, method, params, timeout_seconds):
        if method == "tools/list":
            return {"tools": [{"name": "search_openai_docs"}]}
        return {"structuredContent": {"hits": "bad"}}

    with pytest.raises(ValueError, match="malformed search results"):
        lookup_openai_official_docs(
            request=request,
            mcp_call_fn=mcp_call_fn,
        )


def test_lookup_openai_official_docs_rejects_empty_results() -> None:
    request = OfficialDocsLookupRequest(
        query="How do streaming responses work?",
        library="openai",
    )

    def mcp_call_fn(*, server_url, method, params, timeout_seconds):
        if method == "tools/list":
            return {"tools": [{"name": "search_openai_docs"}]}
        return {"structuredContent": {"hits": []}}

    with pytest.raises(ValueError, match="returned no usable documents"):
        lookup_openai_official_docs(
            request=request,
            mcp_call_fn=mcp_call_fn,
        )


def test_lookup_openai_official_docs_raises_transport_failure() -> None:
    request = OfficialDocsLookupRequest(
        query="How do streaming responses work?",
        library="openai",
    )

    def mcp_call_fn(*, server_url, method, params, timeout_seconds):
        raise RuntimeError("Official docs MCP request failed: timed out")

    with pytest.raises(RuntimeError, match="Official docs MCP request failed: timed out"):
        lookup_openai_official_docs(
            request=request,
            mcp_call_fn=mcp_call_fn,
        )


def test_lookup_openai_official_docs_rejects_missing_search_tool() -> None:
    request = OfficialDocsLookupRequest(
        query="How do streaming responses work?",
        library="openai",
    )

    def mcp_call_fn(*, server_url, method, params, timeout_seconds):
        assert method == "tools/list"
        return {
            "tools": [
                {"name": "fetch_openai_doc"},
                {"name": "list_openai_docs"},
            ]
        }

    with pytest.raises(ValueError, match="did not expose a suitable search tool"):
        lookup_openai_official_docs(
            request=request,
            mcp_call_fn=mcp_call_fn,
        )
