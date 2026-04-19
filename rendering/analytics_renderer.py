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


EVALUATION_CASE_COLUMN_LABELS = {
    "question": "Question",
    "source_recall": "Source recall",
    "retrieved_chunks": "Retrieved chunks",
    "used_fallback": "Used fallback search",
    "context_match": "Context matched expectation",
    "keyword_recall": "Keyword recall",
}
RECENT_DIAGNOSTICS_COLUMN_LABELS = {
    "query_preview": "Query preview",
    "response_type": "Response type",
    "model": "Model",
    "source_count": "Source count",
    "total_tokens": "Total tokens",
    "estimated_cost_usd": "Estimated cost",
}


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
    overview_columns = st.columns((1, 1, 1, 1.7))
    overview_columns[0].metric("Conversation turns", overview["total_conversation_turns"])
    overview_columns[1].metric("LLM-backed requests", overview["llm_backed_request_count"])
    overview_columns[2].metric("Total tokens", overview["total_tokens"])
    overview_columns[3].metric(
        "Estimated total cost",
        _format_cost_metric(overview["estimated_total_cost_usd"]),
    )

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
                _format_percent_metric(
                    evaluation_summary_metrics["average_source_recall"]
                ),
            )
            summary_columns[1].metric(
                "Keyword recall",
                _format_percent_metric(
                    evaluation_summary_metrics["average_keyword_recall"]
                ),
            )
            summary_columns[2].metric(
                "Context match",
                _format_percent_metric(evaluation_summary_metrics["context_match_rate"]),
            )

            summary_columns = st.columns(3)
            summary_columns[0].metric(
                "No-context fallback rate",
                _format_percent_metric(
                    evaluation_summary_metrics["no_context_fallback_rate"]
                ),
            )
            summary_columns[1].metric(
                "Sources-present rate",
                _format_percent_metric(
                    evaluation_summary_metrics[
                        "sources_present_rate_when_context_used"
                    ]
                ),
            )
            summary_columns[2].metric(
                "Case count",
                evaluation_summary_metrics["case_count"],
            )
            interpretation = build_evaluation_interpretation(
                evaluation_summary_metrics
            )
            st.info(
                f"{interpretation['status']}: {interpretation['summary']}"
            )

            evaluation_case_rows = build_evaluation_case_rows(
                evaluation_report_payload,
                limit=10,
            )
            if evaluation_case_rows:
                st.dataframe(
                    format_evaluation_case_rows_for_display(evaluation_case_rows),
                    hide_index=True,
                    use_container_width=True,
                )

    st.divider()
    st.markdown("**Recent Request Diagnostics**")
    st.caption(
        "Latest requests first. Use this table to spot response type, source usage, "
        "token usage, and estimated cost for recent turns."
    )
    if recent_diagnostics_rows:
        st.dataframe(
            format_recent_diagnostics_rows_for_display(recent_diagnostics_rows),
            hide_index=True,
            use_container_width=True,
        )
    else:
        st.info("No recent turn diagnostics are available yet.")


def _format_percent_metric(value: object) -> str:
    if not isinstance(value, int | float):
        return "Unavailable"
    return f"{float(value) * 100:.1f}%"


def _format_optional_number(value: object) -> str:
    if isinstance(value, int | float):
        return str(value)
    return "No tracked usage"


def _format_optional_model(value: object) -> str:
    if isinstance(value, str) and value != "n/a":
        return value
    return "No LLM"


def _format_bool_label(value: object) -> str:
    return "Yes" if value is True else "No"


def format_evaluation_case_rows_for_display(
    rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    return [
        {
            EVALUATION_CASE_COLUMN_LABELS["question"]: row["question"],
            EVALUATION_CASE_COLUMN_LABELS["source_recall"]: _format_percent_metric(
                row["source_recall"]
            ),
            EVALUATION_CASE_COLUMN_LABELS["retrieved_chunks"]: row["retrieved_chunks"],
            EVALUATION_CASE_COLUMN_LABELS["used_fallback"]: _format_bool_label(
                row["used_fallback"]
            ),
            EVALUATION_CASE_COLUMN_LABELS["context_match"]: _format_bool_label(
                row["context_match"]
            ),
            EVALUATION_CASE_COLUMN_LABELS["keyword_recall"]: _format_percent_metric(
                row["keyword_recall"]
            ),
        }
        for row in rows
    ]


def format_recent_diagnostics_rows_for_display(
    rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    return [
        {
            RECENT_DIAGNOSTICS_COLUMN_LABELS["query_preview"]: row["query_preview"],
            RECENT_DIAGNOSTICS_COLUMN_LABELS["response_type"]: row["response_type"],
            RECENT_DIAGNOSTICS_COLUMN_LABELS["model"]: _format_optional_model(
                row["model"]
            ),
            RECENT_DIAGNOSTICS_COLUMN_LABELS["source_count"]: row["source_count"],
            RECENT_DIAGNOSTICS_COLUMN_LABELS["total_tokens"]: _format_optional_number(
                row["total_tokens"]
            ),
            RECENT_DIAGNOSTICS_COLUMN_LABELS["estimated_cost_usd"]: _format_cost_metric(
                row["estimated_cost_usd"]
            ),
        }
        for row in rows
    ]


def build_evaluation_interpretation(
    metrics: dict[str, int | float],
) -> dict[str, str]:
    source_recall = float(metrics.get("average_source_recall", 0.0))
    keyword_recall = float(metrics.get("average_keyword_recall", 0.0))
    context_match = float(metrics.get("context_match_rate", 0.0))
    case_count = int(metrics.get("case_count", 0))
    core_signals = {
        "source recall": source_recall,
        "keyword recall": keyword_recall,
        "context match": context_match,
    }
    strong_threshold = 0.8
    minimum_threshold = 0.6

    if case_count == 0:
        status = "Needs improvement"
        summary = "No evaluation cases were included in the latest snapshot."
    elif all(value == 1.0 for value in core_signals.values()):
        return {
            "status": "Good",
            "summary": (
                f"All evaluation metrics reached 100% across {case_count} cases."
            ),
        }
    elif all(value >= strong_threshold for value in core_signals.values()):
        status = "Good"
        summary = (
            "Core evaluation signals are strong; no tracked signal is below "
            f"the {_format_percent_metric(strong_threshold)} target."
        )
    elif all(value >= minimum_threshold for value in core_signals.values()):
        status = "Acceptable"
        weak_signals = [
            label
            for label, value in core_signals.items()
            if value < strong_threshold
        ]
        summary = (
            f"{_format_label_list(weak_signals)} "
            f"{_plural_verb(weak_signals)} below the "
            f"{_format_percent_metric(strong_threshold)} strong-result target, "
            "but all core signals remain in the acceptable range."
        )
    else:
        status = "Needs improvement"
        weak_signals = [
            label
            for label, value in core_signals.items()
            if value < minimum_threshold
        ]
        summary = (
            f"Prioritize {_format_label_list(weak_signals, capitalize=False)}; "
            f"{_plural_pronoun(weak_signals)} {_plural_verb(weak_signals)} below "
            f"the {_format_percent_metric(minimum_threshold)} minimum acceptable "
            "threshold."
        )

    return {
        "status": status,
        "summary": (
            f"{summary} Source recall {_format_percent_metric(source_recall)}, "
            f"keyword recall {_format_percent_metric(keyword_recall)}, "
            f"context match {_format_percent_metric(context_match)} across "
            f"{case_count} cases."
        ),
    }


def _format_label_list(labels: list[str], *, capitalize: bool = True) -> str:
    if not labels:
        return "No tracked signal"
    if len(labels) == 1:
        text = labels[0]
    else:
        text = ", ".join(labels[:-1]) + f" and {labels[-1]}"
    return text.capitalize() if capitalize else text


def _plural_verb(labels: list[str]) -> str:
    return "is" if len(labels) == 1 else "are"


def _plural_pronoun(labels: list[str]) -> str:
    return "it" if len(labels) == 1 else "they"
