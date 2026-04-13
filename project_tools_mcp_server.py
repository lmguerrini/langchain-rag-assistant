from __future__ import annotations

import json
import re
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel, ValidationError, field_validator


SEARCH_INTERNAL_DOCS_TOOL_NAME = "search_internal_docs"
SEARCH_INTERNAL_DOCS_TOOL_DEFINITION = {
    "name": SEARCH_INTERNAL_DOCS_TOOL_NAME,
    "description": "Search local project markdown docs and return compact structured matches.",
    "inputSchema": {
        "type": "object",
        "properties": {
            "query": {"type": "string"},
        },
        "required": ["query"],
    },
}
DEFAULT_DOCS_DIR = Path("data/raw")
MAX_SEARCH_MATCHES = 3
MAX_EXCERPT_LENGTH = 220
QUERY_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "docs",
    "documentation",
    "for",
    "how",
    "the",
    "to",
    "what",
    "with",
}


class SearchInternalDocsArgs(BaseModel):
    query: str

    @field_validator("query")
    @classmethod
    def validate_query(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("Search query must not be empty.")
        return cleaned


@dataclass(frozen=True)
class InternalDocRecord:
    title: str
    source_path: str
    doc_id: str
    topic: str
    library: str
    doc_type: str
    body: str


def handle_mcp_jsonrpc_request(
    payload: object,
    *,
    docs_dir: Path = DEFAULT_DOCS_DIR,
) -> dict[str, object]:
    if not isinstance(payload, Mapping):
        return _build_error_response(
            None,
            code=-32600,
            message="Invalid MCP request payload.",
        )

    request_id = payload.get("id")
    method = payload.get("method")

    if method == "tools/list":
        return _build_success_response(
            request_id,
            {"tools": [SEARCH_INTERNAL_DOCS_TOOL_DEFINITION]},
        )

    if method != "tools/call":
        return _build_error_response(
            request_id,
            code=-32601,
            message=f"Unsupported MCP method: {method}",
        )

    params = payload.get("params")
    if not isinstance(params, Mapping):
        return _build_error_response(
            request_id,
            code=-32602,
            message="tools/call requires an object params payload.",
        )

    if params.get("name") != SEARCH_INTERNAL_DOCS_TOOL_NAME:
        return _build_error_response(
            request_id,
            code=-32602,
            message="Unknown tool requested.",
        )

    arguments = params.get("arguments", {})
    if not isinstance(arguments, Mapping):
        return _build_error_response(
            request_id,
            code=-32602,
            message="Tool arguments must be an object.",
        )

    try:
        search_args = SearchInternalDocsArgs.model_validate(arguments)
    except ValidationError as exc:
        return _build_error_response(
            request_id,
            code=-32602,
            message="Invalid internal docs tool arguments.",
            data={"validation_error": str(exc)},
        )

    try:
        matches = search_internal_docs(query=search_args.query, docs_dir=docs_dir)
    except Exception as exc:
        return _build_error_response(
            request_id,
            code=-32000,
            message="Internal docs search failed.",
            data={"tool_error": str(exc)},
        )

    structured_content = {
        "query": search_args.query,
        "match_count": len(matches),
        "matches": matches,
    }
    summary = (
        f"Found {len(matches)} internal doc matches."
        if matches
        else "No internal docs matched the query."
    )
    return _build_success_response(
        request_id,
        {
            "structuredContent": structured_content,
            "content": [{"type": "text", "text": summary}],
        },
    )


def search_internal_docs(
    *,
    query: str,
    docs_dir: Path = DEFAULT_DOCS_DIR,
) -> list[dict[str, str]]:
    query_tokens = _tokenize_query(query)
    if not query_tokens:
        return []

    documents = _load_internal_docs(docs_dir)
    scored_matches: list[tuple[int, int, dict[str, str]]] = []

    for index, document in enumerate(documents):
        score = _score_document(document, query_tokens=query_tokens)
        if score <= 0:
            continue
        scored_matches.append(
            (
                score,
                index,
                {
                    "title": document.title,
                    "source_path": document.source_path,
                    "doc_id": document.doc_id,
                    "topic": document.topic,
                    "library": document.library,
                    "doc_type": document.doc_type,
                    "excerpt": _build_excerpt(document.body, query_tokens=query_tokens),
                },
            )
        )

    sorted_matches = sorted(scored_matches, key=lambda item: (-item[0], item[1]))
    return [match for _, _, match in sorted_matches[:MAX_SEARCH_MATCHES]]


def _load_internal_docs(docs_dir: Path) -> list[InternalDocRecord]:
    documents: list[InternalDocRecord] = []
    for path in sorted(docs_dir.rglob("*.md")):
        metadata, body = _parse_markdown_document(path)
        relative_path = path.relative_to(docs_dir).as_posix()
        source_path = _format_source_path(path, docs_dir)
        documents.append(
            InternalDocRecord(
                title=metadata.get("title") or _extract_title_from_body(body) or path.stem,
                source_path=source_path,
                doc_id=relative_path.removesuffix(".md").replace("/", ":"),
                topic=metadata.get("topic", "unknown"),
                library=metadata.get("library", "general"),
                doc_type=metadata.get("doc_type", "concept"),
                body=body,
            )
        )
    return documents


def _parse_markdown_document(path: Path) -> tuple[dict[str, str], str]:
    text = path.read_text(encoding="utf-8")
    if not text.startswith("---\n"):
        return {}, text.strip()

    _, _, remainder = text.partition("---\n")
    frontmatter, separator, body = remainder.partition("\n---\n")
    if not separator:
        return {}, text.strip()

    metadata: dict[str, str] = {}
    for line in frontmatter.splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        metadata[key.strip()] = value.strip()
    return metadata, body.strip()


def _extract_title_from_body(body: str) -> str | None:
    for line in body.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            return stripped.lstrip("#").strip()
    return None


def _format_source_path(path: Path, docs_dir: Path) -> str:
    resolved_path = path.resolve()
    try:
        return resolved_path.relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return f"{docs_dir.name}/{path.relative_to(docs_dir).as_posix()}"


def _score_document(document: InternalDocRecord, *, query_tokens: set[str]) -> int:
    title_overlap = query_tokens & _tokenize_text(document.title)
    metadata_overlap = query_tokens & _tokenize_text(
        " ".join((document.doc_id, document.topic, document.library, document.doc_type))
    )
    body_overlap = query_tokens & _tokenize_text(document.body)
    return (len(title_overlap) * 3) + (len(metadata_overlap) * 2) + len(body_overlap)


def _build_excerpt(body: str, *, query_tokens: set[str]) -> str:
    paragraphs = [
        _collapse_whitespace(part)
        for part in re.split(r"\n\s*\n", body)
        if _collapse_whitespace(part)
    ]
    if not paragraphs:
        return ""

    best_excerpt = paragraphs[0]
    best_score = -1
    for paragraph in paragraphs:
        score = len(query_tokens & _tokenize_text(paragraph))
        if score > best_score:
            best_excerpt = paragraph
            best_score = score

    return _trim_text(best_excerpt, max_length=MAX_EXCERPT_LENGTH)


def _trim_text(text: str, *, max_length: int) -> str:
    if len(text) <= max_length:
        return text

    truncated = text[: max_length - 3].rstrip()
    if " " in truncated:
        truncated = truncated.rsplit(" ", 1)[0]
    return f"{truncated}..."


def _collapse_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _tokenize_query(text: str) -> set[str]:
    tokens = {
        token
        for token in re.findall(r"[a-z0-9]+", text.lower())
        if len(token) > 2 and token not in QUERY_STOPWORDS
    }
    if tokens:
        return tokens
    return {
        token
        for token in re.findall(r"[a-z0-9]+", text.lower())
        if len(token) > 1
    }


def _tokenize_text(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9]+", text.lower())
        if len(token) > 2 and token not in QUERY_STOPWORDS
    }


def _build_success_response(
    request_id: object,
    result: Mapping[str, object],
) -> dict[str, object]:
    return {
        "jsonrpc": "2.0",
        "id": request_id,
        "result": dict(result),
    }


def _build_error_response(
    request_id: object,
    *,
    code: int,
    message: str,
    data: Mapping[str, object] | None = None,
) -> dict[str, object]:
    error_payload: dict[str, object] = {
        "code": code,
        "message": message,
    }
    if data is not None:
        error_payload["data"] = dict(data)

    return {
        "jsonrpc": "2.0",
        "id": request_id,
        "error": error_payload,
    }


def main() -> int:
    raw_payload = sys.stdin.read()
    try:
        decoded_payload: object = json.loads(raw_payload)
    except json.JSONDecodeError:
        decoded_payload = None

    response = handle_mcp_jsonrpc_request(decoded_payload)
    sys.stdout.write(json.dumps(response))
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
