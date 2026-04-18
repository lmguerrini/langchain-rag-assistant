from __future__ import annotations

from collections import Counter
from collections.abc import Mapping

from src.kb_status import KBStatusResult
from src.llm_response_utils import estimate_usage_cost_usd


RESPONSE_TYPE_ORDER = (
    "Grounded answer",
    "Official docs answer",
    "Tool result",
    "No-context fallback",
)
LLM_BACKED_RESPONSE_TYPES = {"Grounded answer", "Official docs answer"}


def get_turn_response_type(turn: Mapping[str, object]) -> str:
    if turn.get("tool_result") is not None:
        return "Tool result"
    if turn.get("official_docs_result") is not None:
        return "Official docs answer"
    if turn.get("used_context") is True:
        return "Grounded answer"
    return "No-context fallback"


def build_usage_totals(
    conversation_history: list[dict[str, object]],
) -> dict[str, int | float | None]:
    usage_entries = [
        usage
        for turn in conversation_history
        if isinstance((usage := turn.get("usage")), Mapping)
    ]
    if not usage_entries:
        return {
            "request_count": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "estimated_cost_usd": None,
        }

    estimated_costs = [estimate_usage_cost_usd(usage) for usage in usage_entries]
    return {
        "request_count": len(usage_entries),
        "input_tokens": sum(int(usage["input_tokens"]) for usage in usage_entries),
        "output_tokens": sum(int(usage["output_tokens"]) for usage in usage_entries),
        "total_tokens": sum(int(usage["total_tokens"]) for usage in usage_entries),
        "estimated_cost_usd": (
            round(sum(float(cost) for cost in estimated_costs), 6)
            if estimated_costs
            and all(isinstance(cost, int | float) for cost in estimated_costs)
            else None
        ),
    }


def build_overview_metrics(
    conversation_history: list[dict[str, object]],
    kb_status: KBStatusResult,
) -> dict[str, int | float | str | None]:
    usage_totals = build_usage_totals(conversation_history)
    llm_backed_request_count = sum(
        1
        for turn in conversation_history
        if get_turn_response_type(turn) in LLM_BACKED_RESPONSE_TYPES
    )
    return {
        "total_conversation_turns": len(conversation_history),
        "llm_backed_request_count": llm_backed_request_count,
        "usage_tracked_request_count": usage_totals["request_count"],
        "total_tokens": usage_totals["total_tokens"],
        "estimated_total_cost_usd": usage_totals["estimated_cost_usd"],
        "kb_state": kb_status.state,
        "kb_summary": kb_status.summary,
        "kb_detail": kb_status.detail,
    }


def build_response_type_breakdown(
    conversation_history: list[dict[str, object]],
) -> list[dict[str, int | float | str]]:
    counter = Counter(get_turn_response_type(turn) for turn in conversation_history)
    total_turns = len(conversation_history)
    return [
        {
            "response_type": response_type,
            "count": counter.get(response_type, 0),
            "share": round(counter.get(response_type, 0) / total_turns, 4)
            if total_turns
            else 0.0,
        }
        for response_type in RESPONSE_TYPE_ORDER
    ]


def build_grounded_source_summary(
    conversation_history: list[dict[str, object]],
) -> dict[str, int | float]:
    grounded_turns = [
        turn
        for turn in conversation_history
        if get_turn_response_type(turn) == "Grounded answer"
    ]
    source_counts = [len(_get_sources(turn)) for turn in grounded_turns]
    if not grounded_turns:
        return {
            "grounded_answer_count": 0,
            "grounded_answers_with_sources": 0,
            "average_sources_per_grounded": 0.0,
            "max_sources_per_grounded": 0,
        }

    return {
        "grounded_answer_count": len(grounded_turns),
        "grounded_answers_with_sources": sum(1 for count in source_counts if count > 0),
        "average_sources_per_grounded": round(sum(source_counts) / len(source_counts), 4),
        "max_sources_per_grounded": max(source_counts),
    }


def build_model_usage_breakdown(
    conversation_history: list[dict[str, object]],
) -> list[dict[str, int | float | str | None]]:
    model_totals: dict[str, dict[str, int | float | bool | None]] = {}

    for turn in conversation_history:
        usage = turn.get("usage")
        if not isinstance(usage, Mapping):
            continue

        model_name = str(usage.get("model_name") or "unknown model")
        if model_name not in model_totals:
            model_totals[model_name] = {
                "model": model_name,
                "request_count": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
                "estimated_cost_usd": 0.0,
                "has_complete_costs": True,
            }

        aggregate = model_totals[model_name]
        aggregate["request_count"] = int(aggregate["request_count"]) + 1
        aggregate["input_tokens"] = int(aggregate["input_tokens"]) + int(usage["input_tokens"])
        aggregate["output_tokens"] = int(aggregate["output_tokens"]) + int(usage["output_tokens"])
        aggregate["total_tokens"] = int(aggregate["total_tokens"]) + int(usage["total_tokens"])

        estimated_cost = estimate_usage_cost_usd(usage)
        if isinstance(estimated_cost, int | float) and aggregate["has_complete_costs"]:
            aggregate["estimated_cost_usd"] = (
                float(aggregate["estimated_cost_usd"]) + float(estimated_cost)
            )
        else:
            aggregate["has_complete_costs"] = False
            aggregate["estimated_cost_usd"] = None

    rows = []
    for aggregate in model_totals.values():
        estimated_cost = aggregate["estimated_cost_usd"]
        rows.append(
            {
                "model": str(aggregate["model"]),
                "request_count": int(aggregate["request_count"]),
                "input_tokens": int(aggregate["input_tokens"]),
                "output_tokens": int(aggregate["output_tokens"]),
                "total_tokens": int(aggregate["total_tokens"]),
                "estimated_cost_usd": (
                    round(float(estimated_cost), 6)
                    if isinstance(estimated_cost, int | float)
                    else None
                ),
            }
        )

    return sorted(rows, key=lambda row: (-int(row["request_count"]), str(row["model"])))


def build_recent_diagnostics_rows(
    conversation_history: list[dict[str, object]],
    *,
    limit: int = 10,
    preview_length: int = 80,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for turn in reversed(conversation_history):
        usage = turn.get("usage")
        sources = _get_sources(turn)
        rows.append(
            {
                "query_preview": _truncate_text(str(turn.get("query", "")), preview_length),
                "response_type": get_turn_response_type(turn),
                "model": (
                    usage.get("model_name")
                    if isinstance(usage, Mapping) and usage.get("model_name")
                    else "n/a"
                ),
                "source_count": len(sources),
                "total_tokens": (
                    int(usage["total_tokens"])
                    if isinstance(usage, Mapping) and isinstance(usage.get("total_tokens"), int)
                    else None
                ),
                "estimated_cost_usd": (
                    estimate_usage_cost_usd(usage)
                    if isinstance(usage, Mapping)
                    else None
                ),
            }
        )
        if len(rows) >= limit:
            break
    return rows


def build_evaluation_summary_metrics(
    report_payload: Mapping[str, object] | None,
) -> dict[str, int | float] | None:
    if not isinstance(report_payload, Mapping):
        return None

    summary = report_payload.get("summary")
    if not isinstance(summary, Mapping):
        return None

    return {
        "case_count": int(summary.get("case_count", 0)),
        "average_source_recall": float(summary.get("average_source_recall", 0.0)),
        "average_keyword_recall": float(summary.get("average_keyword_recall", 0.0)),
        "context_match_rate": float(summary.get("context_match_rate", 0.0)),
        "no_context_fallback_rate": float(summary.get("no_context_fallback_rate", 0.0)),
        "sources_present_rate_when_context_used": float(
            summary.get("sources_present_rate_when_context_used", 0.0)
        ),
    }


def build_evaluation_case_rows(
    report_payload: Mapping[str, object] | None,
    *,
    limit: int | None = None,
) -> list[dict[str, object]]:
    if not isinstance(report_payload, Mapping):
        return []

    raw_cases = report_payload.get("cases")
    if not isinstance(raw_cases, list):
        return []

    rows: list[dict[str, object]] = []
    for raw_case in raw_cases:
        if not isinstance(raw_case, Mapping):
            continue

        retrieval = raw_case.get("retrieval")
        answer = raw_case.get("answer")
        if not isinstance(retrieval, Mapping) or not isinstance(answer, Mapping):
            continue

        rows.append(
            {
                "question": raw_case.get("question", ""),
                "source_recall": float(retrieval.get("source_recall", 0.0)),
                "retrieved_chunks": int(retrieval.get("retrieved_chunk_count", 0)),
                "used_fallback": bool(retrieval.get("used_fallback", False)),
                "context_match": bool(answer.get("used_context_matches_expectation", False)),
                "keyword_recall": float(answer.get("keyword_recall", 0.0)),
            }
        )

    if limit is not None:
        return rows[:limit]
    return rows


def _get_sources(turn: Mapping[str, object]) -> list[object]:
    sources = turn.get("sources")
    if isinstance(sources, list):
        return sources
    return []


def _truncate_text(text: str, preview_length: int) -> str:
    cleaned = " ".join(text.split())
    if len(cleaned) <= preview_length:
        return cleaned

    truncated = cleaned[: max(0, preview_length - 3)].rstrip()
    if " " in truncated:
        truncated = truncated.rsplit(" ", 1)[0]
    return f"{truncated}..."
