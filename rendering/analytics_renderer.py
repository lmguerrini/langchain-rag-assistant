from __future__ import annotations

import streamlit as st

from rendering.charts import (
    _format_cost_metric,
    build_model_usage_chart,
    build_response_behavior_chart,
)
from services.chat_service import PREVIEW_LENGTH
from src.analytics import (
    build_evaluation_case_rows,
    build_evaluation_summary_metrics,
    build_grounded_source_summary,
    build_model_usage_breakdown,
    build_overview_metrics,
    build_recent_diagnostics_rows,
    build_response_type_breakdown,
    build_usage_totals,
)
from src.config import Settings
from src.evaluation import load_eval_cases, run_runtime_evaluation
from src.kb_status import KBStatusResult
from state.session_state import (
    ANALYTICS_EVAL_ERROR_KEY,
    ANALYTICS_EVAL_REPORT_KEY,
)
from ui.sidebar import format_kb_status_label, should_show_kb_rebuild_trigger


def render_analytics_dashboard(
    *,
    settings: Settings,
    conversation_history: list[dict[str, object]],
    kb_status: KBStatusResult,
    run_evaluation_fn=run_runtime_evaluation,
    load_eval_cases_fn=load_eval_cases,
) -> None:
    overview = build_overview_metrics(conversation_history, kb_status)
    response_breakdown = build_response_type_breakdown(conversation_history)
    grounded_summary = build_grounded_source_summary(conversation_history)
    usage_totals = build_usage_totals(conversation_history)
    model_usage_rows = build_model_usage_breakdown(conversation_history)
    recent_diagnostics_rows = build_recent_diagnostics_rows(
        conversation_history,
        limit=10,
        preview_length=PREVIEW_LENGTH,
    )

    st.subheader("Analytics Dashboard")
    st.caption(
        "This dashboard uses current session history, KB status, tracked usage data, "
        "and the optional evaluation snapshot."
    )

    st.markdown("**Overview**")
    overview_columns = st.columns(5)
    overview_columns[0].metric("Conversation turns", overview["total_conversation_turns"])
    overview_columns[1].metric("LLM-backed requests", overview["llm_backed_request_count"])
    overview_columns[2].metric("Total tokens", overview["total_tokens"])
    overview_columns[3].metric(
        "Estimated total cost",
        _format_cost_metric(overview["estimated_total_cost_usd"]),
    )
    with overview_columns[4]:
        st.caption("Knowledge base")
        st.write(format_kb_status_label(kb_status).removeprefix("Status: "))
        st.caption(kb_status.summary)
    st.caption(kb_status.detail)

    st.divider()
    st.markdown("**Response Behavior**")
    if any(row["count"] for row in response_breakdown):
        st.altair_chart(
            build_response_behavior_chart(response_breakdown),
            use_container_width=True,
        )
        response_table_column, response_detail_column = st.columns((2, 1))
        with response_table_column:
            st.dataframe(
                [
                    {
                        "Response type": row["response_type"],
                        "Count": row["count"],
                        "Share": f"{float(row['share']) * 100:.1f}%",
                    }
                    for row in response_breakdown
                ],
                hide_index=True,
                use_container_width=True,
            )
        with response_detail_column:
            st.metric("Grounded answers", grounded_summary["grounded_answer_count"])
            st.metric(
                "Grounded with sources",
                grounded_summary["grounded_answers_with_sources"],
            )
            st.metric(
                "Avg sources / grounded",
                f"{grounded_summary['average_sources_per_grounded']:.2f}",
            )
            st.metric(
                "Max sources / grounded",
                grounded_summary["max_sources_per_grounded"],
            )
    else:
        st.info("No response behavior data yet.")

    st.divider()
    st.markdown("**Token & Model Usage**")
    token_columns = st.columns(4)
    token_columns[0].metric("Tracked requests", usage_totals["request_count"])
    token_columns[1].metric("Input tokens", usage_totals["input_tokens"])
    token_columns[2].metric("Output tokens", usage_totals["output_tokens"])
    token_columns[3].metric("Total tokens", usage_totals["total_tokens"])
    if model_usage_rows:
        st.altair_chart(
            build_model_usage_chart(model_usage_rows),
            use_container_width=True,
        )
        st.dataframe(
            [
                {
                    "Model": row["model"],
                    "Requests": row["request_count"],
                    "Input tokens": row["input_tokens"],
                    "Output tokens": row["output_tokens"],
                    "Total tokens": row["total_tokens"],
                    "Estimated cost": _format_cost_metric(row["estimated_cost_usd"]),
                }
                for row in model_usage_rows
            ],
            hide_index=True,
            use_container_width=True,
        )
    else:
        st.info("No model usage data is available yet.")

    st.divider()
    st.markdown("**Knowledge Base**")
    kb_column, kb_detail_column = st.columns((1, 1))
    with kb_column:
        st.write(f"Status: {format_kb_status_label(kb_status).removeprefix('Status: ')}")
        st.caption(kb_status.summary)
        st.write(
            "Rebuild action available: "
            + ("Yes" if should_show_kb_rebuild_trigger(kb_status) else "No")
        )
    with kb_detail_column:
        st.caption(kb_status.detail)
        if kb_status.rebuild_command is not None:
            st.caption("Manual rebuild command")
            st.code(kb_status.rebuild_command, language="bash")

    st.markdown("**Evaluation**")
    with st.container():
        try:
            eval_cases = load_eval_cases_fn()
            evaluation_case_count = len(eval_cases)
            evaluation_dataset_error = None
        except Exception as exc:
            evaluation_case_count = 0
            evaluation_dataset_error = str(exc)

        top_row = st.columns(2)
        top_row[0].metric("Evaluation cases", evaluation_case_count)
        top_row[1].metric(
            "Snapshot available",
            "Yes" if st.session_state.get(ANALYTICS_EVAL_REPORT_KEY) else "No",
        )

        if evaluation_dataset_error is not None:
            st.error(f"Evaluation dataset unavailable: {evaluation_dataset_error}")

        if st.button("Run evaluation snapshot", key="run_evaluation_snapshot"):
            st.session_state[ANALYTICS_EVAL_ERROR_KEY] = None
            with st.status("Running evaluation snapshot...", expanded=True) as status:
                status.write(f"Loading evaluation cases from {settings.raw_data_dir.parent / 'eval' / 'eval_cases.json'}")
                try:
                    report = run_evaluation_fn()
                except Exception as exc:
                    st.session_state[ANALYTICS_EVAL_REPORT_KEY] = None
                    st.session_state[ANALYTICS_EVAL_ERROR_KEY] = f"Evaluation failed: {exc}"
                    status.update(label="Evaluation snapshot failed", state="error")
                else:
                    st.session_state[ANALYTICS_EVAL_REPORT_KEY] = report.model_dump()
                    st.session_state[ANALYTICS_EVAL_ERROR_KEY] = None
                    status.update(label="Evaluation snapshot completed", state="complete")

        evaluation_error = st.session_state.get(ANALYTICS_EVAL_ERROR_KEY)
        if isinstance(evaluation_error, str) and evaluation_error:
            st.error(evaluation_error)

        evaluation_report_payload = st.session_state.get(ANALYTICS_EVAL_REPORT_KEY)
        evaluation_summary_metrics = build_evaluation_summary_metrics(
            evaluation_report_payload
        )
        if evaluation_summary_metrics is not None:
            summary_columns = st.columns(3)
            summary_columns[0].metric(
                "Source recall",
                f"{evaluation_summary_metrics['average_source_recall']:.4f}",
            )
            summary_columns[1].metric(
                "Keyword recall",
                f"{evaluation_summary_metrics['average_keyword_recall']:.4f}",
            )
            summary_columns[2].metric(
                "Context match",
                f"{evaluation_summary_metrics['context_match_rate']:.4f}",
            )

            summary_columns = st.columns(3)
            summary_columns[0].metric(
                "No-context fallback rate",
                f"{evaluation_summary_metrics['no_context_fallback_rate']:.4f}",
            )
            summary_columns[1].metric(
                "Sources-present rate",
                f"{evaluation_summary_metrics['sources_present_rate_when_context_used']:.4f}",
            )
            summary_columns[2].metric(
                "Case count",
                evaluation_summary_metrics["case_count"],
            )

            evaluation_case_rows = build_evaluation_case_rows(
                evaluation_report_payload,
                limit=10,
            )
            if evaluation_case_rows:
                st.dataframe(
                    evaluation_case_rows,
                    hide_index=True,
                    use_container_width=True,
                )

    st.divider()
    st.markdown("**Recent Diagnostics**")
    if recent_diagnostics_rows:
        st.dataframe(
            recent_diagnostics_rows,
            hide_index=True,
            use_container_width=True,
        )
    else:
        st.info("No recent turn diagnostics are available yet.")
