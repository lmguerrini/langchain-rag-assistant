import json

import pytest

from src.config import Settings
from src.official_docs_fallback_adapters import (
    lookup_chroma_official_docs,
    lookup_streamlit_official_docs,
)
from src.official_docs_mcp_adapters import (
    lookup_langchain_official_docs,
    lookup_openai_official_docs,
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


def test_lookup_langchain_official_docs_returns_normalized_documents() -> None:
    request = OfficialDocsLookupRequest(
        query="How should I start a RAG app?",
        library="langchain",
    )

    def mcp_call_fn(*, server_url, tool_name, arguments, timeout_seconds):
        assert server_url == "https://docs.langchain.com/mcp"
        assert tool_name == "search_docs_by_lang_chain"
        assert arguments == {"query": "How should I start a RAG app?"}
        return {
            "structuredContent": {
                "docs": [
                    {
                        "title": "Build a RAG agent with LangChain",
                        "link": "https://docs.langchain.com/guides/rag",
                        "page_content": "Start with a simple retrieval pipeline.",
                    }
                ]
            }
        }

    result = lookup_langchain_official_docs(
        request=request,
        settings=Settings(),
        mcp_call_fn=mcp_call_fn,
    )

    assert result.library == "langchain"
    assert result.documents[0].provider_mode == "official_mcp"
    assert result.documents[0].title == "Build a RAG agent with LangChain"
    assert result.documents[0].url == "https://docs.langchain.com/guides/rag"
    assert result.documents[0].snippets[0].text == "Start with a simple retrieval pipeline."


def test_lookup_openai_official_docs_returns_normalized_documents() -> None:
    request = OfficialDocsLookupRequest(
        query="How do streaming responses work?",
        library="openai",
    )

    payload = {
        "hits": [
            {
                "url": "https://developers.openai.com/docs/guides/streaming-responses#chunk",
                "url_without_anchor": "https://developers.openai.com/docs/guides/streaming-responses",
                "content": "Use streaming to receive partial response events as they are generated.",
                "hierarchy": {
                    "lvl0": "Documentation",
                    "lvl1": "Streaming responses"
                }
            }
        ]
    }

    def mcp_call_fn(*, server_url, tool_name, arguments, timeout_seconds):
        assert server_url == "https://developers.openai.com/mcp"
        assert tool_name == "search_openai_docs"
        assert arguments == {"query": "How do streaming responses work?"}
        return {
            "content": [
                {
                    "type": "text",
                    "text": json.dumps(payload),
                }
            ]
        }

    result = lookup_openai_official_docs(
        request=request,
        settings=Settings(),
        mcp_call_fn=mcp_call_fn,
    )

    assert result.library == "openai"
    assert result.documents[0].provider_mode == "official_mcp"
    assert result.documents[0].title == "Streaming responses"
    assert result.documents[0].url == "https://developers.openai.com/docs/guides/streaming-responses"
    assert "partial response events" in result.documents[0].snippets[0].text


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

    def mcp_call_fn(*, server_url, tool_name, arguments, timeout_seconds):
        return {"content": [{"type": "text", "text": "{\"unexpected\": true}"}]}

    with pytest.raises(ValueError, match="returned no usable documents"):
        lookup_openai_official_docs(
            request=request,
            settings=Settings(),
            mcp_call_fn=mcp_call_fn,
        )
