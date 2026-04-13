from pathlib import Path

from project_tools_mcp_server import (
    SEARCH_INTERNAL_DOCS_TOOL_NAME,
    handle_mcp_jsonrpc_request,
)


def test_project_tools_mcp_server_tools_list() -> None:
    response = handle_mcp_jsonrpc_request(
        {
            "jsonrpc": "2.0",
            "id": "request-1",
            "method": "tools/list",
        }
    )

    assert response["jsonrpc"] == "2.0"
    assert response["id"] == "request-1"
    assert response["result"]["tools"] == [
        {
            "name": SEARCH_INTERNAL_DOCS_TOOL_NAME,
            "description": (
                "Search local project markdown docs and return compact structured matches."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        }
    ]


def test_project_tools_mcp_server_tools_call_search_internal_docs_success(tmp_path: Path) -> None:
    docs_dir = tmp_path / "raw"
    docs_dir.mkdir()
    _write_markdown_doc(
        docs_dir / "chroma_persistence_guide.md",
        title="Chroma Persistence and Reindexing Guide",
        topic="chroma",
        library="chroma",
        doc_type="how_to",
        body=(
            "# Chroma Persistence and Reindexing Guide\n\n"
            "Chroma should persist to a stable local directory so collections survive restarts."
        ),
    )
    _write_markdown_doc(
        docs_dir / "streamlit_chat_patterns.md",
        title="Streamlit Chat Patterns for RAG Interfaces",
        topic="streamlit",
        library="streamlit",
        doc_type="example",
        body=(
            "# Streamlit Chat Patterns for RAG Interfaces\n\n"
            "Streamlit chat apps should keep the interaction simple."
        ),
    )

    response = handle_mcp_jsonrpc_request(
        {
            "jsonrpc": "2.0",
            "id": "request-2",
            "method": "tools/call",
            "params": {
                "name": SEARCH_INTERNAL_DOCS_TOOL_NAME,
                "arguments": {"query": "How do I persist Chroma collections?"},
            },
        },
        docs_dir=docs_dir,
    )

    assert response["jsonrpc"] == "2.0"
    assert response["id"] == "request-2"
    assert response["result"]["content"] == [
        {"type": "text", "text": "Found 1 internal doc matches."}
    ]
    assert response["result"]["structuredContent"]["match_count"] == 1
    assert response["result"]["structuredContent"]["query"] == (
        "How do I persist Chroma collections?"
    )
    match = response["result"]["structuredContent"]["matches"][0]
    assert match["title"] == "Chroma Persistence and Reindexing Guide"
    assert match["source_path"] == "raw/chroma_persistence_guide.md"
    assert match["doc_id"] == "chroma_persistence_guide"
    assert match["topic"] == "chroma"
    assert match["library"] == "chroma"
    assert match["doc_type"] == "how_to"
    assert "persist to a stable local directory" in match["excerpt"]


def test_project_tools_mcp_server_tools_call_unknown_tool() -> None:
    response = handle_mcp_jsonrpc_request(
        {
            "jsonrpc": "2.0",
            "id": "request-3",
            "method": "tools/call",
            "params": {
                "name": "unknown_tool",
                "arguments": {"query": "anything"},
            },
        }
    )

    assert response["error"]["code"] == -32602
    assert response["error"]["message"] == "Unknown tool requested."


def test_project_tools_mcp_server_tools_call_invalid_arguments() -> None:
    response = handle_mcp_jsonrpc_request(
        {
            "jsonrpc": "2.0",
            "id": "request-4",
            "method": "tools/call",
            "params": {
                "name": SEARCH_INTERNAL_DOCS_TOOL_NAME,
                "arguments": {},
            },
        }
    )

    assert response["error"]["code"] == -32602
    assert response["error"]["message"] == "Invalid internal docs tool arguments."
    assert "validation_error" in response["error"]["data"]


def test_project_tools_mcp_server_invalid_method() -> None:
    response = handle_mcp_jsonrpc_request(
        {
            "jsonrpc": "2.0",
            "id": "request-5",
            "method": "resources/list",
        }
    )

    assert response["error"]["code"] == -32601
    assert response["error"]["message"] == "Unsupported MCP method: resources/list"


def test_project_tools_mcp_server_empty_search_result_behavior(tmp_path: Path) -> None:
    docs_dir = tmp_path / "raw"
    docs_dir.mkdir()
    _write_markdown_doc(
        docs_dir / "langchain_retrieval_patterns.md",
        title="LangChain Retrieval Patterns for Small RAG Apps",
        topic="langchain",
        library="langchain",
        doc_type="concept",
        body=(
            "# LangChain Retrieval Patterns for Small RAG Apps\n\n"
            "Use a focused retriever and keep source metadata attached."
        ),
    )

    response = handle_mcp_jsonrpc_request(
        {
            "jsonrpc": "2.0",
            "id": "request-6",
            "method": "tools/call",
            "params": {
                "name": SEARCH_INTERNAL_DOCS_TOOL_NAME,
                "arguments": {"query": "kubernetes deployment autoscaling"},
            },
        },
        docs_dir=docs_dir,
    )

    assert response["result"]["content"] == [
        {"type": "text", "text": "No internal docs matched the query."}
    ]
    assert response["result"]["structuredContent"] == {
        "query": "kubernetes deployment autoscaling",
        "match_count": 0,
        "matches": [],
    }


def _write_markdown_doc(
    path: Path,
    *,
    title: str,
    topic: str,
    library: str,
    doc_type: str,
    body: str,
) -> None:
    path.write_text(
        "\n".join(
            [
                "---",
                f"title: {title}",
                f"topic: {topic}",
                f"library: {library}",
                f"doc_type: {doc_type}",
                "---",
                body,
                "",
            ]
        ),
        encoding="utf-8",
    )
