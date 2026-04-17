from __future__ import annotations

import streamlit as st

from rendering.response_labels import (
    build_session_usage_totals,
    format_request_usage_label,
    format_session_usage_label,
    get_response_generation_explanation,
    get_response_summary_line,
    get_response_type_label,
)
from rendering.structured_display import (
    _clean_display_text,
    _format_source_metadata_fragment,
    _format_tool_field_lines,
    _format_tool_scalar_value,
    build_official_docs_display_data,
    build_tool_field_display_rows,
    build_tool_result_display_data,
    format_official_docs_library_label,
    format_official_docs_provider_label,
    format_source_display,
    format_tool_field_label,
    format_tool_name_label,
    parse_source_string,
    render_tool_result_fields,
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
            tool_display = build_tool_result_display_data(turn.get("tool_result"))
            if tool_display is not None:
                with st.expander("Tool Result"):
                    st.markdown(f"**Tool:** {tool_display['tool_name']}")
                    if tool_display["raw_query"] is not None:
                        st.caption(f"Original query: {tool_display['raw_query']}")
                    if tool_display["input_fields"]:
                        st.markdown("**Input**")
                        render_tool_result_fields(tool_display["input_fields"])
                    if tool_display["output_fields"]:
                        st.markdown("**Result**")
                        render_tool_result_fields(tool_display["output_fields"])
                    if tool_display["error"] is not None:
                        st.error(tool_display["error"])
                    with st.expander("Raw tool payload"):
                        st.json(turn["tool_result"])
            official_docs_display = build_official_docs_display_data(
                turn.get("official_docs_result")
            )
            if official_docs_display is not None:
                with st.expander("Official Docs Result"):
                    if official_docs_display["library"] is not None:
                        st.caption(f"Library: {official_docs_display['library']}")
                    if not official_docs_display["documents"]:
                        st.write(
                            "No official documentation documents were returned for this request."
                        )
                    for index, document in enumerate(
                        official_docs_display["documents"],
                        start=1,
                    ):
                        st.markdown(f"**Document {index}: {document['title']}**")
                        metadata_fragments = []
                        if document["provider_label"] is not None:
                            metadata_fragments.append(document["provider_label"])
                        if metadata_fragments:
                            st.caption(" | ".join(metadata_fragments))
                        if document["url"] is not None:
                            st.caption(document["url"])
                        for snippet in document["snippets"]:
                            st.write(f"- {snippet}")
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
