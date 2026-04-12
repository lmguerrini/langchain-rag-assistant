from __future__ import annotations

import json
import time

import streamlit as st
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from src.chains import run_backend_query
from src.config import get_settings
from src.kb_status import KBStatusResult, get_kb_status
from src.logger import configure_logging, get_logger
from src.rate_limit import apply_rate_limit
from src.schemas import AnswerResult


DEFAULT_CHAT_MODEL = "gpt-4.1-mini"
PREVIEW_LENGTH = 80


class AppValidationError(ValueError):
    pass


LOGGER = get_logger(__name__)


@st.cache_resource
def get_vector_store() -> Chroma:
    settings = get_settings()
    api_key = settings.ensure_openai_api_key()
    embeddings = OpenAIEmbeddings(
        api_key=api_key,
        model=settings.embedding_model,
    )
    return Chroma(
        collection_name=settings.chroma_collection_name,
        embedding_function=embeddings,
        persist_directory=str(settings.chroma_persist_dir),
    )


@st.cache_resource
def get_chat_model() -> ChatOpenAI:
    settings = get_settings()
    return ChatOpenAI(
        api_key=settings.ensure_openai_api_key(),
        model=DEFAULT_CHAT_MODEL,
        temperature=0,
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
            st.caption(format_request_usage_label(turn))
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
    if isinstance(exc, AppValidationError):
        return message
    if "OPENAI_API_KEY" in message:
        return "OpenAI is not configured yet. Add OPENAI_API_KEY and try again."
    if "Chroma vector store is unavailable" in message or "Chroma vector store is empty" in message:
        return "The local knowledge base is not ready. Build the Chroma index before asking questions."
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
            "Estimate OpenAI cost: model=gpt-4.1-mini, input_tokens=1000, output_tokens=500, num_calls=3",
        ],
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


def render_help_section(
    conversation_history: list[dict[str, object]],
    kb_status: KBStatusResult,
) -> None:
    help_content = get_help_content()
    with st.sidebar:
        st.subheader("Knowledge Base")
        st.write(format_kb_status_label(kb_status))
        st.caption(kb_status.detail)
        if kb_status.rebuild_command is not None:
            st.caption(f"Rebuild manually: `{kb_status.rebuild_command}`")

        st.subheader("Chat Controls")
        if st.button("Clear chat", use_container_width=True):
            st.session_state["conversation_history"] = []
            st.rerun()

        st.download_button(
            "Export conversation (.md)",
            data=build_conversation_markdown(conversation_history),
            file_name="conversation_export.md",
            mime="text/markdown",
            disabled=not conversation_history,
            use_container_width=True,
        )
        st.download_button(
            "Export conversation (.json)",
            data=build_conversation_json(conversation_history),
            file_name="conversation_export.json",
            mime="application/json",
            disabled=not conversation_history,
            use_container_width=True,
        )
        st.caption("Session LLM usage (tracked)")
        st.caption(format_session_usage_label(conversation_history))

        with st.expander("Help & Guide", expanded=False):
            st.write(help_content["helps_with"])
            st.caption("Out of scope")
            st.write(help_content["out_of_scope"])
            st.caption("Example questions")
            for question in help_content["example_questions"]:
                st.write(f"- {question}")
            st.caption("How to read responses")
            for item in help_content["response_types"]:
                st.write(f"- {item}")


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

    kb_status = get_kb_status(settings)
    render_help_section(st.session_state["conversation_history"], kb_status)

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
            vector_store = get_vector_store()
            chat_model = get_chat_model()
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
