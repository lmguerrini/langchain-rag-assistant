from __future__ import annotations

import csv
import json
import io
import re
import time

import streamlit as st
from fpdf import FPDF
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from build_index import KBRebuildResult, rebuild_knowledge_base
from src.chains import run_backend_query
from src.config import Settings, get_settings
from src.kb_status import KBStatusResult, get_kb_status
from src.logger import configure_logging, get_logger
from src.rate_limit import apply_rate_limit
from src.schemas import AnswerResult


PREVIEW_LENGTH = 80
EXPORT_FORMAT_OPTIONS = ("Markdown", "JSON", "CSV", "PDF")
KB_REBUILD_FEEDBACK_KEY = "kb_rebuild_feedback"
CHAT_MODEL_SESSION_KEY = "selected_chat_model"
PDF_TEXT_REPLACEMENTS = str.maketrans(
    {
        "\u00a0": " ",
        "\u200b": "",
        "\u2013": " ",
        "\u2014": " ",
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2022": "-",
        "\u2026": "...",
    }
)


class AppValidationError(ValueError):
    pass


LOGGER = get_logger(__name__)


@st.cache_resource
def _build_cached_vector_store(
    *,
    persist_directory: str,
    collection_name: str,
    embedding_model: str,
    api_key: str,
) -> Chroma:
    embeddings = OpenAIEmbeddings(
        api_key=api_key,
        model=embedding_model,
    )
    return Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=persist_directory,
    )


def build_vector_store_cache_inputs(settings: Settings) -> dict[str, str]:
    return {
        "persist_directory": str(settings.chroma_persist_dir),
        "collection_name": settings.chroma_collection_name,
        "embedding_model": settings.embedding_model,
        "api_key": settings.ensure_openai_api_key(),
    }


def get_vector_store(settings: Settings) -> Chroma:
    return _build_cached_vector_store(**build_vector_store_cache_inputs(settings))


@st.cache_resource
def _build_cached_chat_model(
    *,
    api_key: str,
    model_name: str,
    temperature: float,
) -> ChatOpenAI:
    return ChatOpenAI(
        api_key=api_key,
        model=model_name,
        temperature=temperature,
    )


def get_initial_chat_model_selection(
    settings: Settings,
    selected_model: object,
) -> str:
    if isinstance(selected_model, str) and selected_model in settings.supported_chat_models:
        return selected_model
    return settings.default_chat_model


def validate_selected_chat_model(
    selected_model: object,
    settings: Settings,
) -> str:
    if not isinstance(selected_model, str) or not selected_model.strip():
        raise AppValidationError("Select a supported chat model before sending a request.")

    try:
        return settings.ensure_supported_chat_model(selected_model)
    except ValueError as exc:
        raise AppValidationError(str(exc)) from exc


def build_chat_model_cache_inputs(
    settings: Settings,
    selected_model: str,
) -> dict[str, str | float]:
    return {
        "api_key": settings.ensure_openai_api_key(),
        "model_name": settings.ensure_supported_chat_model(selected_model),
        "temperature": 0,
    }


def get_chat_model(settings: Settings, selected_model: str) -> ChatOpenAI:
    return _build_cached_chat_model(
        **build_chat_model_cache_inputs(settings, selected_model)
    )


def render_latest_turn() -> None:
    conversation_history = st.session_state.get("conversation_history", [])
    if not conversation_history:
        return

    for turn in conversation_history:
        with st.chat_message("user"):
            st.write(turn["query"])

        with st.chat_message("assistant"):
            st.caption(f"Response type: {get_response_type_label(turn)}")
            st.caption(get_response_summary_line(turn))
            st.write(turn["answer"])
            with st.expander("How This Answer Was Generated"):
                st.write(get_response_generation_explanation(turn))
            if turn["tool_result"]:
                with st.expander("Tool Result"):
                    st.json(turn["tool_result"])
            if turn["sources"]:
                with st.expander("Sources"):
                    for source in turn["sources"]:
                        source_display = format_source_display(source)
                        if source_display["parse_failed"]:
                            st.write(source_display["raw_source"])
                            continue

                        st.markdown(f"**{source_display['title']}**")
                        if source_display["metadata_fragments"]:
                            st.caption(" | ".join(source_display["metadata_fragments"]))
                        if source_display["source_path"] is not None:
                            st.caption(f"Path: {source_display['source_path']}")
            st.caption(format_request_usage_label(turn))


def validate_query(raw_query: str, *, max_length: int) -> str:
    cleaned = raw_query.strip()
    if not cleaned:
        raise AppValidationError("Enter a question before sending a request.")
    if len(cleaned) > max_length:
        raise AppValidationError(
            f"Questions must be {max_length} characters or fewer."
        )
    return cleaned


def build_safe_log_metadata(query: str) -> dict[str, object]:
    preview = query[:PREVIEW_LENGTH].replace("\n", " ").strip()
    return {
        "query_length": len(query),
        "query_preview": preview,
    }


def get_user_facing_error_message(exc: Exception) -> str:
    message = str(exc)
    lowered_message = message.lower()
    if isinstance(exc, AppValidationError):
        return message
    if "OPENAI_API_KEY" in message:
        return "OpenAI is not configured yet. Add OPENAI_API_KEY and try again."
    if "Chroma vector store is unavailable" in message or "Chroma vector store is empty" in message:
        return "The local knowledge base is not ready. Build the Chroma index before asking questions."
    if "unsupported chat model" in lowered_message:
        return message
    if (
        "model_not_found" in lowered_message
        or "does not have access to the model" in lowered_message
        or "do not have access to the model" in lowered_message
        or ("model" in lowered_message and "does not exist" in lowered_message)
    ):
        return "The selected OpenAI model is unavailable for this API key. Choose a different model and try again."
    if "Connection error" in message:
        return "The AI backend could not be reached. Please try again in a moment."
    return "Something went wrong while processing your request. Please try again."


def build_turn_record(query: str, result: AnswerResult) -> dict[str, object]:
    return {
        "query": query,
        "answer": result.answer,
        "used_context": result.used_context,
        "sources": result.answer_sources,
        "tool_result": (
            result.tool_result.model_dump()
            if result.tool_result is not None
            else None
        ),
        "usage": result.usage.model_dump() if result.usage is not None else None,
    }


def get_response_type_label(turn: dict[str, object]) -> str:
    if turn["tool_result"]:
        return "Tool result"
    if turn["used_context"]:
        return "Grounded answer"
    return "No-context fallback"


def get_response_summary_line(turn: dict[str, object]) -> str:
    if turn["tool_result"]:
        return "Answered with a built-in tool."
    if turn["used_context"]:
        return "Used knowledge-base sources."
    return "No usable knowledge-base context found."


def get_response_generation_explanation(turn: dict[str, object]) -> str:
    if turn["tool_result"]:
        return (
            "A built-in tool handled this request directly, so the app did not "
            "generate a knowledge-base-grounded answer."
        )
    if turn["used_context"]:
        return (
            "The app used knowledge-base context to generate this answer. "
            "The sources below show what grounded the response."
        )
    return (
        "The app could not find usable knowledge-base context for this question, "
        "so it did not generate a grounded answer."
    )


def format_request_usage_label(turn: dict[str, object]) -> str:
    if turn["tool_result"] is not None or turn["used_context"] is False:
        return "No LLM usage"

    usage = turn.get("usage")
    if not isinstance(usage, dict):
        return "Usage unavailable"

    model_name = usage.get("model_name") or "unknown model"
    cost = usage.get("estimated_cost_usd")
    cost_text = (
        f"${cost:.6f}"
        if isinstance(cost, int | float)
        else "Cost unavailable"
    )
    return (
        f"LLM usage: {model_name} | "
        f"{usage['input_tokens']} in / {usage['output_tokens']} out / "
        f"{usage['total_tokens']} total | {cost_text}"
    )


def parse_source_string(source: str) -> dict[str, object] | None:
    segments = [segment.strip() for segment in source.split("|")]
    if not segments or not segments[0]:
        return None

    title = segments[0]
    metadata: dict[str, str] = {}
    for segment in segments[1:]:
        if not segment:
            continue
        if "=" not in segment:
            return None
        key, value = segment.split("=", 1)
        cleaned_key = key.strip()
        cleaned_value = value.strip()
        if not cleaned_key or not cleaned_value:
            return None
        metadata[cleaned_key] = cleaned_value

    return {
        "title": title,
        "metadata": metadata,
    }


def format_source_display(source: str) -> dict[str, object]:
    parsed = parse_source_string(source)
    if parsed is None:
        return {
            "title": "Source",
            "metadata_fragments": [],
            "source_path": None,
            "raw_source": source,
            "parse_failed": True,
        }

    metadata = parsed["metadata"]
    metadata_fragments = [
        _format_source_metadata_fragment(key, value)
        for key, value in metadata.items()
        if key != "source"
    ]
    return {
        "title": parsed["title"],
        "metadata_fragments": metadata_fragments,
        "source_path": metadata.get("source"),
        "raw_source": source,
        "parse_failed": False,
    }


def build_session_usage_totals(
    conversation_history: list[dict[str, object]],
) -> dict[str, int | float | None] | None:
    usage_entries = [
        turn["usage"]
        for turn in conversation_history
        if isinstance(turn.get("usage"), dict)
    ]
    if not usage_entries:
        return None

    estimated_costs = [entry.get("estimated_cost_usd") for entry in usage_entries]
    return {
        "request_count": len(usage_entries),
        "input_tokens": sum(entry["input_tokens"] for entry in usage_entries),
        "output_tokens": sum(entry["output_tokens"] for entry in usage_entries),
        "total_tokens": sum(entry["total_tokens"] for entry in usage_entries),
        "estimated_cost_usd": (
            round(sum(cost for cost in estimated_costs if isinstance(cost, int | float)), 6)
            if all(isinstance(cost, int | float) for cost in estimated_costs)
            else None
        ),
    }


def format_session_usage_label(conversation_history: list[dict[str, object]]) -> str:
    totals = build_session_usage_totals(conversation_history)
    if totals is None:
        return "No tracked LLM usage yet."

    cost = totals["estimated_cost_usd"]
    cost_text = (
        f"${cost:.6f}"
        if isinstance(cost, int | float)
        else "Cost unavailable"
    )
    return (
        f"{totals['request_count']} requests | "
        f"{totals['total_tokens']} total tokens | {cost_text}"
    )


def _format_source_metadata_fragment(key: str, value: str) -> str:
    labels = {
        "topic": "Topic",
        "library": "Library",
        "doc_type": "Type",
        "difficulty": "Difficulty",
        "error_family": "Error family",
        "chunk": "Chunk",
    }
    label = labels.get(key, key.replace("_", " ").title())
    return f"{label}: {value}"


def get_help_content() -> dict[str, list[str] | str]:
    return {
        "helps_with": (
            "LangChain-based RAG application design, Chroma retrieval decisions, "
            "Streamlit integration patterns, and the built-in domain tools."
        ),
        "out_of_scope": (
            "General programming help, unrelated topics, and broad questions outside "
            "LangChain, Chroma, Streamlit, and this app's domain."
        ),
        "example_questions": [
            "How should I persist and rebuild the Chroma index locally?",
            "How should I display retrieved sources in a Streamlit chat interface?",
            "Estimate OpenAI cost for model gpt-4.1-mini with 1000 input tokens, 500 output tokens, and 3 calls",
            "Alternative format: model=gpt-4.1-mini, input_tokens=1000, output_tokens=500, num_calls=3",
        ],
        "examples_intro": "You can ask natural questions or use structured inputs for tools.",
        "response_types": [
            "Grounded answer: generated from retrieved knowledge-base context.",
            "Tool result: returned by one structured rule-based tool.",
            "No-context fallback: shown when the system cannot find usable domain context.",
        ],
    }


def format_kb_status_label(status: KBStatusResult) -> str:
    labels = {
        "missing": "Missing",
        "up_to_date": "Up to date",
        "outdated": "Outdated",
    }
    return f"Status: {labels[status.state]}"


def should_show_kb_rebuild_trigger(status: KBStatusResult) -> bool:
    return status.state in {"missing", "outdated"}


def build_kb_rebuild_success_message(result: KBRebuildResult) -> str:
    return (
        "Knowledge base rebuilt successfully. "
        f"Indexed {result.indexed_chunk_count} chunks into '{result.collection_name}'."
    )


def build_kb_rebuild_error_message(exc: Exception) -> str:
    return f"Knowledge base rebuild failed: {exc}"


def run_kb_rebuild_action(
    *,
    settings: Settings,
    rebuild_fn=rebuild_knowledge_base,
    clear_vector_store_cache_fn=None,
) -> dict[str, object]:
    cache_clearer = clear_vector_store_cache_fn or _build_cached_vector_store.clear
    try:
        result = rebuild_fn(settings)
    except Exception as exc:
        return {
            "ok": False,
            "should_rerun": False,
            "feedback": {
                "kind": "error",
                "message": build_kb_rebuild_error_message(exc),
            },
        }

    cache_clearer()
    return {
        "ok": True,
        "should_rerun": True,
        "feedback": {
            "kind": "success",
            "message": build_kb_rebuild_success_message(result),
        },
    }


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
            for key, value in turn["tool_result"].items():
                lines.append(f"- {key}: {value}")
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


def build_conversation_pdf(conversation_history: list[dict[str, object]]) -> bytes:
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Helvetica", size=12)
    pdf.set_title(normalize_text_for_pdf("Conversation Export"))

    pdf.set_font("Helvetica", style="B", size=14)
    pdf.set_x(pdf.l_margin)
    pdf.cell(0, 10, normalize_text_for_pdf("Conversation Export"))
    pdf.ln(12)
    pdf.set_font("Helvetica", size=11)

    if not conversation_history:
        pdf.set_x(pdf.l_margin)
        pdf.multi_cell(0, 8, normalize_text_for_pdf("No conversation history available."))
    else:
        for index, turn in enumerate(conversation_history, start=1):
            pdf.set_font("Helvetica", style="B", size=12)
            pdf.set_x(pdf.l_margin)
            pdf.cell(0, 8, normalize_text_for_pdf(f"Turn {index}"))
            pdf.ln(8)
            pdf.set_font("Helvetica", size=11)
            pdf.set_x(pdf.l_margin)
            pdf.multi_cell(0, 7, normalize_text_for_pdf(f"User question: {turn['query']}"))
            pdf.set_x(pdf.l_margin)
            pdf.multi_cell(
                0,
                7,
                normalize_text_for_pdf(
                    f"Response type: {get_response_type_label(turn)}"
                ),
            )
            pdf.set_x(pdf.l_margin)
            pdf.multi_cell(
                0,
                7,
                normalize_text_for_pdf(
                    f"Assistant answer: {clean_markdown_text_for_pdf(turn['answer'])}"
                ),
            )

            if turn["sources"]:
                pdf.set_x(pdf.l_margin)
                pdf.multi_cell(0, 7, normalize_text_for_pdf("Sources:"))
                for source in turn["sources"]:
                    pdf.set_x(pdf.l_margin)
                    pdf.multi_cell(0, 7, normalize_text_for_pdf(f"- {source}"))

            if turn["tool_result"]:
                for line in build_pdf_detail_lines("Tool result", turn["tool_result"]):
                    pdf.set_x(pdf.l_margin)
                    pdf.multi_cell(0, 7, normalize_text_for_pdf(line))

            if turn["usage"]:
                for line in build_pdf_detail_lines("Usage", turn["usage"]):
                    pdf.set_x(pdf.l_margin)
                    pdf.multi_cell(0, 7, normalize_text_for_pdf(line))

            pdf.ln(3)

    rendered = pdf.output()
    if isinstance(rendered, bytearray):
        return bytes(rendered)
    if isinstance(rendered, bytes):
        return rendered
    return rendered.encode("latin-1")


def normalize_text_for_pdf(text: object) -> str:
    normalized = str(text).translate(PDF_TEXT_REPLACEMENTS)
    return normalized.encode("latin-1", errors="replace").decode("latin-1")


def clean_markdown_text_for_pdf(text: object) -> str:
    cleaned_lines: list[str] = []
    for raw_line in str(text).splitlines():
        line = re.sub(r"^\s{0,3}#{1,6}\s*", "", raw_line)
        line = line.replace("**", "")
        if re.fullmatch(r"\s*-{3,}\s*", line):
            continue
        cleaned_lines.append(line)
    return "\n".join(cleaned_lines).strip()


def build_pdf_detail_lines(title: str, data: dict[str, object]) -> list[str]:
    lines = [f"{title}:"]
    for key, value in data.items():
        if key == "estimated_cost_usd":
            label = "Estimated cost USD"
            value_text = (
                f"{value:.6f}"
                if isinstance(value, int | float)
                else "N/A"
            )
        else:
            label = key.replace("_", " ").capitalize()
            if isinstance(value, dict | list):
                value_text = json.dumps(value, ensure_ascii=False)
            elif value is None:
                value_text = "none"
            else:
                value_text = str(value)
        lines.append(f"- {label}: {value_text}")
    return lines


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


def render_help_section(
    settings: Settings,
    conversation_history: list[dict[str, object]],
    kb_status: KBStatusResult,
) -> None:
    help_content = get_help_content()
    with st.sidebar:
        feedback = st.session_state.pop(KB_REBUILD_FEEDBACK_KEY, None)
        if isinstance(feedback, dict) and isinstance(feedback.get("message"), str):
            if feedback.get("kind") == "success":
                st.success(feedback["message"])
            elif feedback.get("kind") == "error":
                st.error(feedback["message"])

        st.subheader("Knowledge Base")
        st.write(format_kb_status_label(kb_status))
        st.caption(kb_status.detail)
        if should_show_kb_rebuild_trigger(kb_status):
            if st.button("Rebuild knowledge base", use_container_width=True):
                with st.status("Rebuilding knowledge base...", expanded=True) as status:
                    status.write("Building the local Chroma index and writing the KB manifest")
                    rebuild_outcome = run_kb_rebuild_action(settings=settings)
                    if rebuild_outcome["ok"]:
                        status.write("Clearing the cached vector store")
                        status.update(label="Knowledge base rebuilt", state="complete")
                        st.session_state[KB_REBUILD_FEEDBACK_KEY] = rebuild_outcome[
                            "feedback"
                        ]
                        st.rerun()

                    status.update(label="Knowledge base rebuild failed", state="error")
                    st.error(rebuild_outcome["feedback"]["message"])

        if kb_status.rebuild_command is not None:
            st.caption(f"Rebuild manually: `{kb_status.rebuild_command}`")

        st.subheader("Chat Controls")
        current_chat_model = get_initial_chat_model_selection(
            settings,
            st.session_state.get(CHAT_MODEL_SESSION_KEY),
        )
        st.session_state[CHAT_MODEL_SESSION_KEY] = current_chat_model
        st.selectbox(
            "Chat model",
            options=settings.supported_chat_models,
            index=settings.supported_chat_models.index(current_chat_model),
            key=CHAT_MODEL_SESSION_KEY,
        )
        if st.button("Clear chat", use_container_width=True):
            st.session_state["conversation_history"] = []
            st.rerun()

        selected_export_format = st.selectbox(
            "Export format",
            EXPORT_FORMAT_OPTIONS,
            index=0,
        )
        export_artifact = get_export_artifact(
            conversation_history,
            selected_export_format,
        )
        st.download_button(
            "Download conversation export",
            data=export_artifact["data"],
            file_name=export_artifact["file_name"],
            mime=export_artifact["mime"],
            disabled=not conversation_history,
            use_container_width=True,
        )
        with st.expander("Help & Guide", expanded=False):
            st.write(help_content["helps_with"])
            st.caption("Out of scope")
            st.write(help_content["out_of_scope"])
            st.write(help_content["examples_intro"])
            st.caption("Example questions")
            for question in help_content["example_questions"]:
                st.write(f"- {question}")
            st.caption("How to read responses")
            for item in help_content["response_types"]:
                st.write(f"- {item}")

        st.caption("Session LLM usage (tracked)")
        st.caption(format_session_usage_label(conversation_history))


def main() -> None:
    settings = get_settings()
    configure_logging(settings.log_level)

    st.set_page_config(page_title="RAG Assistant", page_icon=":speech_balloon:")
    st.title("RAG Assistant")
    st.write(
        "Ask about LangChain-based RAG application development with Chroma and Streamlit."
    )

    if "conversation_history" not in st.session_state:
        st.session_state["conversation_history"] = []
    if "request_timestamps" not in st.session_state:
        st.session_state["request_timestamps"] = []
    if CHAT_MODEL_SESSION_KEY not in st.session_state:
        st.session_state[CHAT_MODEL_SESSION_KEY] = settings.default_chat_model

    kb_status = get_kb_status(settings)
    render_help_section(settings, st.session_state["conversation_history"], kb_status)

    render_latest_turn()

    question = st.chat_input("Ask a question about the knowledge base")
    if not question:
        return

    with st.status("Handling request...", expanded=True) as status:
        try:
            status.write("Validating request")
            validated_query = validate_query(
                question,
                max_length=settings.max_query_length,
            )
            safe_metadata = build_safe_log_metadata(validated_query)
            LOGGER.info("request_received %s", safe_metadata)
            status.write("Checking rate limit")
            rate_limit_result = apply_rate_limit(
                st.session_state["request_timestamps"],
                now=time.time(),
                max_requests=settings.rate_limit_request_count,
                window_seconds=settings.rate_limit_window_seconds,
            )
            st.session_state["request_timestamps"] = rate_limit_result.updated_timestamps
            if not rate_limit_result.allowed:
                LOGGER.warning("request_rate_limited %s", safe_metadata)
                status.update(label="Rate limit exceeded", state="error")
                st.warning(
                    "Too many requests in a short period. "
                    f"Wait about {rate_limit_result.retry_after_seconds} seconds and try again."
                )
                return

            status.write("Loading resources")
            selected_chat_model = validate_selected_chat_model(
                st.session_state.get(CHAT_MODEL_SESSION_KEY),
                settings,
            )
            vector_store = get_vector_store(settings)
            chat_model = get_chat_model(settings, selected_chat_model)
            status.write("Processing request")
            result = run_backend_query(
                query=validated_query,
                vector_store=vector_store,
                chat_model=chat_model,
            )
            status.update(label="Request completed", state="complete")
        except AppValidationError as exc:
            LOGGER.info(
                "request_rejected %s",
                build_safe_log_metadata(question.strip()),
            )
            status.update(label="Request rejected", state="error")
            st.warning(get_user_facing_error_message(exc))
            return
        except Exception as exc:
            LOGGER.exception("backend_error %s", safe_metadata if "safe_metadata" in locals() else {})
            status.update(label="Request failed", state="error")
            st.error(get_user_facing_error_message(exc))
            return

    path = "tool" if result.tool_result is not None else "rag"
    if result.used_context is False and result.tool_result is None:
        path = "fallback"
    LOGGER.info(
        "request_completed %s",
        {
            **safe_metadata,
            "path": path,
            "source_count": len(result.answer_sources),
        },
    )

    st.session_state["conversation_history"].append(
        build_turn_record(validated_query, result)
    )
    st.rerun()


if __name__ == "__main__":
    main()
