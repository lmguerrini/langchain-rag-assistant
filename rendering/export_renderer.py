from __future__ import annotations

import csv
import io
import json

import streamlit as st

from rendering.pdf_renderer import (
    PDF_TEXT_REPLACEMENTS,
    build_conversation_pdf,
    build_pdf_detail_lines,
    clean_markdown_text_for_pdf,
    normalize_text_for_pdf,
)
from rendering.response_labels import get_response_type_label
from rendering.tool_export import (
    _build_markdown_tool_field_lines,
    _build_text_tool_field_lines,
    build_tool_result_markdown_lines,
    build_tool_result_text_lines,
)


EXPORT_FORMAT_OPTIONS = ("Markdown", "JSON", "CSV", "PDF")


def build_conversation_markdown(conversation_history: list[dict[str, object]]) -> str:
    lines = ["# Conversation Export", ""]
    if not conversation_history:
        lines.append("_No conversation history available._")
        return "\n".join(lines)

    for index, turn in enumerate(conversation_history, start=1):
        lines.extend(
            [
                f"## Turn {index}",
                "",
                f"**User question:** {turn['query']}",
                "",
                f"**Response type:** {get_response_type_label(turn)}",
                "",
                f"**Assistant answer:** {turn['answer']}",
                "",
            ]
        )

        if turn["sources"]:
            lines.append("**Sources:**")
            for source in turn["sources"]:
                lines.append(f"- {source}")
            lines.append("")

        if turn["tool_result"]:
            lines.append("**Tool result:**")
            lines.extend(build_tool_result_markdown_lines(turn["tool_result"]))
            lines.append("")

    return "\n".join(lines).strip() + "\n"


def build_conversation_snapshot(conversation_history: list[dict[str, object]]) -> str:
    return json.dumps(
        conversation_history,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def build_conversation_json(conversation_history: list[dict[str, object]]) -> str:
    payload = {
        "export_format": "json",
        "turn_count": len(conversation_history),
        "turns": [
            {
                "query": turn["query"],
                "answer": turn["answer"],
                "response_type": get_response_type_label(turn),
                "used_context": turn["used_context"],
                "sources": turn["sources"],
                "tool_result": turn["tool_result"],
                "usage": turn["usage"],
            }
            for turn in conversation_history
        ],
    }
    return json.dumps(payload, indent=2) + "\n"


def build_conversation_csv(conversation_history: list[dict[str, object]]) -> str:
    output = io.StringIO()
    writer = csv.DictWriter(
        output,
        fieldnames=[
            "turn_index",
            "query",
            "answer",
            "response_type",
            "used_context",
            "sources",
            "tool_name",
            "tool_result_json",
            "usage_model_name",
            "usage_input_tokens",
            "usage_output_tokens",
            "usage_total_tokens",
            "usage_estimated_cost_usd",
        ],
    )
    writer.writeheader()

    for index, turn in enumerate(conversation_history, start=1):
        tool_result = turn.get("tool_result")
        usage = turn.get("usage")
        writer.writerow(
            {
                "turn_index": index,
                "query": turn["query"],
                "answer": turn["answer"],
                "response_type": get_response_type_label(turn),
                "used_context": turn["used_context"],
                "sources": " | ".join(turn["sources"]),
                "tool_name": (
                    tool_result.get("tool_name")
                    if isinstance(tool_result, dict)
                    else ""
                ),
                "tool_result_json": (
                    json.dumps(tool_result, separators=(",", ":"))
                    if isinstance(tool_result, dict)
                    else ""
                ),
                "usage_model_name": (
                    usage.get("model_name")
                    if isinstance(usage, dict)
                    else ""
                ),
                "usage_input_tokens": (
                    usage.get("input_tokens")
                    if isinstance(usage, dict)
                    else ""
                ),
                "usage_output_tokens": (
                    usage.get("output_tokens")
                    if isinstance(usage, dict)
                    else ""
                ),
                "usage_total_tokens": (
                    usage.get("total_tokens")
                    if isinstance(usage, dict)
                    else ""
                ),
                "usage_estimated_cost_usd": (
                    usage.get("estimated_cost_usd")
                    if isinstance(usage, dict)
                    else ""
                ),
            }
        )

    return output.getvalue()


@st.cache_data
def build_cached_export_data(
    conversation_snapshot: str,
    export_format: str,
) -> str | bytes:
    conversation_history = json.loads(conversation_snapshot)
    if export_format == "Markdown":
        return build_conversation_markdown(conversation_history)
    if export_format == "JSON":
        return build_conversation_json(conversation_history)
    if export_format == "CSV":
        return build_conversation_csv(conversation_history)
    if export_format == "PDF":
        return build_conversation_pdf(conversation_history)
    raise ValueError(f"Unsupported export format: {export_format}")


def get_export_artifact(
    conversation_history: list[dict[str, object]],
    export_format: str,
) -> dict[str, object]:
    conversation_snapshot = build_conversation_snapshot(conversation_history)
    if export_format == "Markdown":
        return {
            "data": build_cached_export_data(conversation_snapshot, export_format),
            "file_name": "conversation_export.md",
            "mime": "text/markdown",
        }
    if export_format == "JSON":
        return {
            "data": build_cached_export_data(conversation_snapshot, export_format),
            "file_name": "conversation_export.json",
            "mime": "application/json",
        }
    if export_format == "CSV":
        return {
            "data": build_cached_export_data(conversation_snapshot, export_format),
            "file_name": "conversation_export.csv",
            "mime": "text/csv",
        }
    if export_format == "PDF":
        return {
            "data": build_cached_export_data(conversation_snapshot, export_format),
            "file_name": "conversation_export.pdf",
            "mime": "application/pdf",
        }
    raise ValueError(f"Unsupported export format: {export_format}")
