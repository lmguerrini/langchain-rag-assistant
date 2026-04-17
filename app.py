from __future__ import annotations

import time

import streamlit as st

from rendering.analytics_renderer import render_analytics_dashboard
from rendering.charts import (
    _format_cost_metric,
    build_horizontal_bar_chart,
    build_model_usage_chart,
    build_model_usage_chart_rows,
    build_response_behavior_chart,
    build_response_behavior_chart_rows,
)
from rendering.chat_renderer import (
    _clean_display_text,
    _format_source_metadata_fragment,
    _format_tool_field_lines,
    _format_tool_scalar_value,
    build_official_docs_display_data,
    build_session_usage_totals,
    build_tool_field_display_rows,
    build_tool_result_display_data,
    format_official_docs_library_label,
    format_official_docs_provider_label,
    format_request_usage_label,
    format_session_usage_label,
    format_source_display,
    format_tool_field_label,
    format_tool_name_label,
    get_response_generation_explanation,
    get_response_summary_line,
    get_response_type_label,
    parse_source_string,
    render_latest_turn,
)
from rendering.export_renderer import (
    EXPORT_FORMAT_OPTIONS,
    PDF_TEXT_REPLACEMENTS,
    _build_markdown_tool_field_lines,
    _build_text_tool_field_lines,
    build_cached_export_data,
    build_conversation_csv,
    build_conversation_json,
    build_conversation_markdown,
    build_conversation_pdf,
    build_conversation_snapshot,
    build_pdf_detail_lines,
    build_tool_result_markdown_lines,
    build_tool_result_text_lines,
    clean_markdown_text_for_pdf,
    get_export_artifact,
    normalize_text_for_pdf,
)
from services.chat_service import (
    PREVIEW_LENGTH,
    AppValidationError,
    _build_cached_chat_model,
    _build_cached_vector_store,
    build_chat_model_cache_inputs,
    build_safe_log_metadata,
    build_turn_record,
    build_vector_store_cache_inputs,
    get_chat_model,
    get_initial_chat_model_selection,
    get_user_facing_error_message,
    get_vector_store,
    run_non_streaming_query,
    run_tool_query,
    should_skip_resource_loading,
    should_stream_grounded_query,
    validate_query,
    validate_selected_chat_model,
)
from src.chains import (
    maybe_match_official_docs_query,
    run_backend_query,
    stream_answer_query,
)
from src.config import Settings, get_settings
from src.kb_status import KBStatusResult, get_kb_status
from src.logger import configure_logging, get_logger
from src.rate_limit import apply_rate_limit
from src.schemas import AnswerResult
from state.session_state import (
    ANALYTICS_EVAL_ERROR_KEY,
    ANALYTICS_EVAL_REPORT_KEY,
    CHAT_MODEL_SESSION_KEY,
    KB_REBUILD_FEEDBACK_KEY,
    initialize_session_state,
)
from ui.chat import (
    build_chat_input_visibility_script,
    render_chat_input_visibility_controller,
    render_streaming_grounded_answer,
)
from ui.sidebar import (
    build_kb_rebuild_error_message,
    build_kb_rebuild_success_message,
    format_kb_status_label,
    get_help_content,
    render_help_section,
    run_kb_rebuild_action,
    should_show_kb_rebuild_trigger,
)


LOGGER = get_logger(__name__)


def main() -> None:
    settings = get_settings()
    configure_logging(settings.log_level)

    st.set_page_config(page_title="LangChain RAG Assistant", page_icon=":speech_balloon:")
    st.title("LangChain RAG Assistant")
    st.write(
        "Ask about LangChain-based RAG application development with Chroma, Streamlit, official docs, and built-in support tools."
    )

    initialize_session_state(settings)

    kb_status = get_kb_status(settings)
    render_help_section(settings, st.session_state["conversation_history"], kb_status)

    chat_tab, analytics_tab = st.tabs(["Chat", "Analytics"])
    with chat_tab:
        render_latest_turn()
    with analytics_tab:
        render_analytics_dashboard(
            settings=settings,
            conversation_history=st.session_state["conversation_history"],
            kb_status=kb_status,
        )
    question = st.chat_input("Ask a question about the knowledge base")
    render_chat_input_visibility_controller()
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

            stream_grounded_answer = False
            if should_skip_resource_loading(validated_query):
                status.write("Handling request with a deterministic tool")
                result = run_tool_query(validated_query)
            else:
                status.write("Loading resources")
                selected_chat_model = validate_selected_chat_model(
                    st.session_state.get(CHAT_MODEL_SESSION_KEY),
                    settings,
                )
                vector_store = get_vector_store(settings)
                chat_model = get_chat_model(settings, selected_chat_model)
                if should_stream_grounded_query(validated_query):
                    status.write("Preparing grounded response")
                    stream_grounded_answer = True
                else:
                    status.write("Processing official docs request")
                    result = run_non_streaming_query(
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

    if stream_grounded_answer:
        try:
            result = render_streaming_grounded_answer(
                query=validated_query,
                vector_store=vector_store,
                chat_model=chat_model,
            )
        except Exception as exc:
            LOGGER.exception("backend_error %s", safe_metadata if "safe_metadata" in locals() else {})
            st.error(get_user_facing_error_message(exc))
            return

    path = "tool" if result.tool_result is not None else "rag"
    if result.official_docs_result is not None:
        path = "official_docs"
    elif result.used_context is False and result.tool_result is None:
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
