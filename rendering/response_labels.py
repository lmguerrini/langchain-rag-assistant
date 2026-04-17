from __future__ import annotations


def get_response_type_label(turn: dict[str, object]) -> str:
    if turn.get("tool_result"):
        return "Tool result"
    if turn.get("official_docs_result"):
        return "Official docs answer"
    if turn["used_context"]:
        return "Grounded answer"
    return "No-context fallback"


def get_response_summary_line(turn: dict[str, object]) -> str:
    if turn.get("tool_result"):
        return "Answered with a built-in tool."
    if turn.get("official_docs_result"):
        return "Answered from official documentation evidence."
    if turn["used_context"]:
        return "Used knowledge-base sources."
    return "No usable knowledge-base context found."


def get_response_generation_explanation(turn: dict[str, object]) -> str:
    if turn.get("tool_result"):
        return (
            "A built-in tool handled this request directly, so the app did not "
            "generate a knowledge-base-grounded answer."
        )
    if turn.get("official_docs_result"):
        return (
            "The app looked up official documentation for the named library and "
            "generated this answer only from the retrieved official-docs evidence."
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
    if turn.get("tool_result") is not None:
        return "No LLM usage"

    usage = turn.get("usage")
    if not isinstance(usage, dict):
        if turn.get("official_docs_result") is not None or turn["used_context"] is True:
            return "Usage unavailable"
        return "No LLM usage"

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
