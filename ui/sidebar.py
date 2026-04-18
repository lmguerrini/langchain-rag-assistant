from __future__ import annotations

import streamlit as st

from build_index import KBRebuildResult, rebuild_knowledge_base
from rendering.response_labels import format_session_usage_label
from rendering.export_renderer import EXPORT_FORMAT_OPTIONS, get_export_artifact
from services.chat_service import (
    clear_vector_store_cache,
    get_initial_chat_model_selection,
)
from src.config import Settings
from src.kb_status import KBStatusResult
from state.session_state import CHAT_MODEL_SESSION_KEY, KB_REBUILD_FEEDBACK_KEY


def get_help_content() -> dict[str, list[str] | str]:
    return {
        "helps_with": (
            "LangChain-based RAG application design, local knowledge-base guidance, "
            "official docs answers, Streamlit/Chroma implementation patterns, and "
            "built-in support tools."
        ),
        "out_of_scope": (
            "General programming help, unrelated topics, and broad questions outside "
            "LangChain, Chroma, Streamlit, and this app's domain."
        ),
        "example_questions": [
            "How should I persist and rebuild the Chroma index locally?",
            "According to the LangChain docs, how should I start a small RAG application?",
            "Estimate OpenAI cost for model gpt-4.1-mini with 1000 input tokens, 500 output tokens, and 3 calls",
            "Diagnose this Streamlit error: DuplicateWidgetID",
            "Recommend retrieval config for long technical documentation used for debugging questions",
            "What is the capital of France?",
        ],
        "review_actions": [
            "Open the Analytics tab to inspect response breakdowns, token usage, model usage, KB status, and recent diagnostics.",
            "Use Run evaluation snapshot in Analytics to show the current evaluation summary and per-case results.",
        ],
        "examples_intro": "Use these prompts to demonstrate the main response paths during review.",
        "response_types": [
            "Grounded answer: generated from retrieved knowledge-base context.",
            "Official docs answer: generated from official documentation evidence only.",
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
    cache_clearer = clear_vector_store_cache_fn or clear_vector_store_cache
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
            st.caption("Review actions")
            for item in help_content["review_actions"]:
                st.write(f"- {item}")
            st.caption("How to read responses")
            for item in help_content["response_types"]:
                st.write(f"- {item}")

        st.caption("Session LLM usage (tracked)")
        st.caption(format_session_usage_label(conversation_history))
