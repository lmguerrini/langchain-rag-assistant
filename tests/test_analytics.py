from src.analytics import (
    build_evaluation_case_rows,
    build_evaluation_summary_metrics,
    build_grounded_source_summary,
    build_model_usage_breakdown,
    build_overview_metrics,
    build_recent_diagnostics_rows,
    build_response_type_breakdown,
    build_usage_totals,
    get_turn_response_type,
)
from src.kb_status import KBStatusResult


def test_get_turn_response_type_distinguishes_supported_paths() -> None:
    assert get_turn_response_type(
        {
            "used_context": True,
            "tool_result": None,
            "official_docs_result": None,
        }
    ) == "Grounded answer"
    assert get_turn_response_type(
        {
            "used_context": False,
            "tool_result": {"tool_name": "estimate_openai_cost"},
            "official_docs_result": None,
        }
    ) == "Tool result"
    assert get_turn_response_type(
        {
            "used_context": False,
            "tool_result": None,
            "official_docs_result": {"library": "openai"},
        }
    ) == "Official docs answer"
    assert get_turn_response_type(
        {
            "used_context": False,
            "tool_result": None,
            "official_docs_result": None,
        }
    ) == "No-context fallback"


def test_build_response_type_breakdown_counts_all_response_types() -> None:
    history = [
        {"used_context": True, "tool_result": None, "official_docs_result": None},
        {"used_context": False, "tool_result": {"tool_name": "estimate_openai_cost"}, "official_docs_result": None},
        {"used_context": False, "tool_result": None, "official_docs_result": {"library": "openai"}},
        {"used_context": False, "tool_result": None, "official_docs_result": None},
    ]

    rows = build_response_type_breakdown(history)

    assert rows == [
        {"response_type": "Grounded answer", "count": 1, "share": 0.25},
        {"response_type": "Official docs answer", "count": 1, "share": 0.25},
        {"response_type": "Tool result", "count": 1, "share": 0.25},
        {"response_type": "No-context fallback", "count": 1, "share": 0.25},
    ]


def test_build_usage_totals_and_overview_metrics_aggregate_real_usage() -> None:
    history = [
        {
            "used_context": True,
            "tool_result": None,
            "official_docs_result": None,
            "usage": {
                "model_name": "gpt-4.1-mini",
                "input_tokens": 20,
                "output_tokens": 10,
                "total_tokens": 30,
                "estimated_cost_usd": 0.000024,
            },
        },
        {
            "used_context": False,
            "tool_result": None,
            "official_docs_result": {"library": "openai"},
            "usage": {
                "model_name": "gpt-4.1",
                "input_tokens": 100,
                "output_tokens": 50,
                "total_tokens": 150,
                "estimated_cost_usd": 0.0006,
            },
        },
        {
            "used_context": False,
            "tool_result": {"tool_name": "estimate_openai_cost"},
            "official_docs_result": None,
            "usage": None,
        },
    ]
    kb_status = KBStatusResult(
        state="up_to_date",
        summary="Knowledge base is up to date.",
        detail="The local Chroma index matches the current raw markdown snapshot.",
    )

    usage_totals = build_usage_totals(history)
    overview = build_overview_metrics(history, kb_status)

    assert usage_totals == {
        "request_count": 2,
        "input_tokens": 120,
        "output_tokens": 60,
        "total_tokens": 180,
        "estimated_cost_usd": 0.000624,
    }
    assert overview["total_conversation_turns"] == 3
    assert overview["llm_backed_request_count"] == 2
    assert overview["total_tokens"] == 180
    assert overview["estimated_total_cost_usd"] == 0.000624
    assert overview["kb_state"] == "up_to_date"


def test_build_usage_totals_infers_versioned_model_cost_when_missing() -> None:
    history = [
        {
            "used_context": True,
            "tool_result": None,
            "official_docs_result": None,
            "usage": {
                "model_name": "gpt-4.1-mini",
                "input_tokens": 20,
                "output_tokens": 10,
                "total_tokens": 30,
                "estimated_cost_usd": 0.000024,
            },
        },
        {
            "used_context": False,
            "tool_result": None,
            "official_docs_result": {"library": "openai"},
            "usage": {
                "model_name": "gpt-4.1-mini-2025-04-14",
                "input_tokens": 18,
                "output_tokens": 7,
                "total_tokens": 25,
                "estimated_cost_usd": None,
            },
        },
    ]
    kb_status = KBStatusResult(
        state="up_to_date",
        summary="Knowledge base is up to date.",
        detail="The local Chroma index matches the current raw markdown snapshot.",
    )

    usage_totals = build_usage_totals(history)
    overview = build_overview_metrics(history, kb_status)

    assert usage_totals == {
        "request_count": 2,
        "input_tokens": 38,
        "output_tokens": 17,
        "total_tokens": 55,
        "estimated_cost_usd": 0.000042,
    }
    assert overview["estimated_total_cost_usd"] == 0.000042


def test_build_usage_totals_keeps_unknown_model_cost_unavailable() -> None:
    history = [
        {
            "usage": {
                "model_name": "gpt-unknown-2025-04-14",
                "input_tokens": 18,
                "output_tokens": 7,
                "total_tokens": 25,
                "estimated_cost_usd": None,
            }
        }
    ]

    assert build_usage_totals(history)["estimated_cost_usd"] is None


def test_build_grounded_source_summary_counts_source_usage() -> None:
    history = [
        {
            "used_context": True,
            "tool_result": None,
            "official_docs_result": None,
            "sources": ["A", "B"],
        },
        {
            "used_context": True,
            "tool_result": None,
            "official_docs_result": None,
            "sources": ["C"],
        },
        {
            "used_context": False,
            "tool_result": None,
            "official_docs_result": None,
            "sources": [],
        },
    ]

    summary = build_grounded_source_summary(history)

    assert summary == {
        "grounded_answer_count": 2,
        "grounded_answers_with_sources": 2,
        "average_sources_per_grounded": 1.5,
        "max_sources_per_grounded": 2,
    }


def test_build_model_usage_breakdown_aggregates_by_model() -> None:
    history = [
        {
            "usage": {
                "model_name": "gpt-4.1-mini",
                "input_tokens": 20,
                "output_tokens": 10,
                "total_tokens": 30,
                "estimated_cost_usd": 0.000024,
            }
        },
        {
            "usage": {
                "model_name": "gpt-4.1-mini",
                "input_tokens": 30,
                "output_tokens": 20,
                "total_tokens": 50,
                "estimated_cost_usd": 0.000044,
            }
        },
        {
            "usage": {
                "model_name": "gpt-4.1",
                "input_tokens": 100,
                "output_tokens": 40,
                "total_tokens": 140,
                "estimated_cost_usd": 0.00052,
            }
        },
        {"usage": None},
    ]

    rows = build_model_usage_breakdown(history)

    assert rows == [
        {
            "model": "gpt-4.1-mini",
            "request_count": 2,
            "input_tokens": 50,
            "output_tokens": 30,
            "total_tokens": 80,
            "estimated_cost_usd": 0.000068,
        },
        {
            "model": "gpt-4.1",
            "request_count": 1,
            "input_tokens": 100,
            "output_tokens": 40,
            "total_tokens": 140,
            "estimated_cost_usd": 0.00052,
        },
    ]


def test_build_model_usage_breakdown_infers_versioned_cost_only_when_priced() -> None:
    history = [
        {
            "usage": {
                "model_name": "gpt-4.1-mini-2025-04-14",
                "input_tokens": 18,
                "output_tokens": 7,
                "total_tokens": 25,
                "estimated_cost_usd": None,
            }
        },
        {
            "usage": {
                "model_name": "gpt-unknown-2025-04-14",
                "input_tokens": 18,
                "output_tokens": 7,
                "total_tokens": 25,
                "estimated_cost_usd": None,
            }
        },
    ]

    rows = build_model_usage_breakdown(history)

    assert rows == [
        {
            "model": "gpt-4.1-mini-2025-04-14",
            "request_count": 1,
            "input_tokens": 18,
            "output_tokens": 7,
            "total_tokens": 25,
            "estimated_cost_usd": 0.000018,
        },
        {
            "model": "gpt-unknown-2025-04-14",
            "request_count": 1,
            "input_tokens": 18,
            "output_tokens": 7,
            "total_tokens": 25,
            "estimated_cost_usd": None,
        },
    ]


def test_build_recent_diagnostics_rows_shapes_latest_turns_first() -> None:
    history = [
        {
            "query": "How should I persist Chroma locally?",
            "used_context": True,
            "tool_result": None,
            "official_docs_result": None,
            "sources": ["A", "B"],
            "usage": {
                "model_name": "gpt-4.1-mini-2025-04-14",
                "input_tokens": 18,
                "output_tokens": 7,
                "total_tokens": 25,
                "estimated_cost_usd": None,
            },
        },
        {
            "query": "What is the capital of France and why does it not belong in this domain?",
            "used_context": False,
            "tool_result": None,
            "official_docs_result": None,
            "sources": [],
            "usage": None,
        },
    ]

    rows = build_recent_diagnostics_rows(history, limit=2, preview_length=35)

    assert rows == [
        {
            "query_preview": "What is the capital of France...",
            "response_type": "No-context fallback",
            "model": "n/a",
            "source_count": 0,
            "total_tokens": None,
            "estimated_cost_usd": None,
        },
        {
            "query_preview": "How should I persist Chroma...",
            "response_type": "Grounded answer",
            "model": "gpt-4.1-mini-2025-04-14",
            "source_count": 2,
            "total_tokens": 25,
            "estimated_cost_usd": 0.000018,
        },
    ]


def test_build_evaluation_summary_and_case_rows_shape_report_payload() -> None:
    report_payload = {
        "summary": {
            "case_count": 2,
            "average_source_recall": 0.5,
            "average_keyword_recall": 0.75,
            "context_match_rate": 1.0,
            "no_context_fallback_rate": 1.0,
            "sources_present_rate_when_context_used": 0.5,
        },
        "cases": [
            {
                "question": "How should I persist Chroma locally?",
                "retrieval": {
                    "source_recall": 1.0,
                    "retrieved_chunk_count": 2,
                    "used_fallback": False,
                },
                "answer": {
                    "used_context_matches_expectation": True,
                    "keyword_recall": 1.0,
                },
            },
            {
                "question": "What is the capital of France?",
                "retrieval": {
                    "source_recall": 1.0,
                    "retrieved_chunk_count": 0,
                    "used_fallback": False,
                },
                "answer": {
                    "used_context_matches_expectation": True,
                    "keyword_recall": 1.0,
                },
            },
        ],
    }

    summary = build_evaluation_summary_metrics(report_payload)
    rows = build_evaluation_case_rows(report_payload, limit=1)

    assert summary == {
        "case_count": 2,
        "average_source_recall": 0.5,
        "average_keyword_recall": 0.75,
        "context_match_rate": 1.0,
        "no_context_fallback_rate": 1.0,
        "sources_present_rate_when_context_used": 0.5,
    }
    assert rows == [
        {
            "question": "How should I persist Chroma locally?",
            "source_recall": 1.0,
            "retrieved_chunks": 2,
            "used_fallback": False,
            "context_match": True,
            "keyword_recall": 1.0,
        }
    ]
