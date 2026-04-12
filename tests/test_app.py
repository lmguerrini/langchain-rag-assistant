import csv
import io
import json

import pytest

from app import (
    AppValidationError,
    clean_markdown_text_for_pdf,
    build_conversation_csv,
    build_conversation_json,
    build_conversation_markdown,
    build_conversation_pdf,
    build_pdf_detail_lines,
    build_session_usage_totals,
    build_turn_record,
    format_kb_status_label,
    format_request_usage_label,
    format_source_display,
    format_session_usage_label,
    get_export_artifact,
    get_response_generation_explanation,
    get_help_content,
    get_response_summary_line,
    get_response_type_label,
    get_user_facing_error_message,
    normalize_text_for_pdf,
    parse_source_string,
    validate_query,
)
from src.kb_status import KBStatusResult
from src.schemas import AnswerResult, RequestUsage


def test_validate_query_trims_whitespace() -> None:
    assert validate_query("  hello world  ", max_length=20) == "hello world"


def test_validate_query_rejects_empty_input() -> None:
    with pytest.raises(AppValidationError, match="Enter a question"):
        validate_query("   ", max_length=20)


def test_validate_query_rejects_overly_long_input() -> None:
    with pytest.raises(AppValidationError, match="characters or fewer"):
        validate_query("x" * 21, max_length=20)


def test_get_user_facing_error_message_maps_known_errors() -> None:
    assert get_user_facing_error_message(ValueError("OPENAI_API_KEY missing")) == (
        "OpenAI is not configured yet. Add OPENAI_API_KEY and try again."
    )
    assert get_user_facing_error_message(
        ValueError("Chroma vector store is empty. Build the local index before retrieval.")
    ) == (
        "The local knowledge base is not ready. Build the Chroma index before asking questions."
    )
    assert get_user_facing_error_message(RuntimeError("Connection error.")) == (
        "The AI backend could not be reached. Please try again in a moment."
    )


def test_build_turn_record_keeps_only_ui_fields() -> None:
    result = AnswerResult(
        answer="Grounded answer",
        used_context=True,
        retrieval=None,
        answer_sources=["Source A"],
        tool_result=None,
        usage=RequestUsage(
            model_name="gpt-4.1-mini",
            input_tokens=20,
            output_tokens=10,
            total_tokens=30,
            estimated_cost_usd=0.000024,
        ),
    )

    turn = build_turn_record("How should I persist Chroma locally?", result)

    assert turn == {
        "query": "How should I persist Chroma locally?",
        "answer": "Grounded answer",
        "used_context": True,
        "sources": ["Source A"],
        "tool_result": None,
        "usage": {
            "model_name": "gpt-4.1-mini",
            "input_tokens": 20,
            "output_tokens": 10,
            "total_tokens": 30,
            "estimated_cost_usd": 0.000024,
        },
    }


def test_get_response_type_label_maps_turn_variants() -> None:
    assert get_response_type_label(
        {
            "query": "q",
            "answer": "a",
            "used_context": True,
            "sources": ["s"],
            "tool_result": None,
        }
    ) == "Grounded answer"
    assert get_response_type_label(
        {
            "query": "q",
            "answer": "a",
            "used_context": False,
            "sources": [],
            "tool_result": {"tool_name": "estimate_openai_cost"},
        }
    ) == "Tool result"
    assert get_response_type_label(
        {
            "query": "q",
            "answer": "a",
            "used_context": False,
            "sources": [],
            "tool_result": None,
        }
    ) == "No-context fallback"


def test_get_response_summary_line_maps_turn_variants() -> None:
    assert get_response_summary_line(
        {
            "query": "q",
            "answer": "a",
            "used_context": True,
            "sources": ["s"],
            "tool_result": None,
        }
    ) == "Used knowledge-base sources."
    assert get_response_summary_line(
        {
            "query": "q",
            "answer": "a",
            "used_context": False,
            "sources": [],
            "tool_result": {"tool_name": "estimate_openai_cost"},
        }
    ) == "Answered with a built-in tool."
    assert get_response_summary_line(
        {
            "query": "q",
            "answer": "a",
            "used_context": False,
            "sources": [],
            "tool_result": None,
        }
    ) == "No usable knowledge-base context found."


def test_get_response_generation_explanation_maps_turn_variants() -> None:
    assert get_response_generation_explanation(
        {
            "query": "q",
            "answer": "a",
            "used_context": True,
            "sources": ["s"],
            "tool_result": None,
        }
    ) == (
        "The app used knowledge-base context to generate this answer. "
        "The sources below show what grounded the response."
    )
    assert get_response_generation_explanation(
        {
            "query": "q",
            "answer": "a",
            "used_context": False,
            "sources": [],
            "tool_result": {"tool_name": "estimate_openai_cost"},
        }
    ) == (
        "A built-in tool handled this request directly, so the app did not "
        "generate a knowledge-base-grounded answer."
    )
    assert get_response_generation_explanation(
        {
            "query": "q",
            "answer": "a",
            "used_context": False,
            "sources": [],
            "tool_result": None,
        }
    ) == (
        "The app could not find usable knowledge-base context for this question, "
        "so it did not generate a grounded answer."
    )


def test_get_help_content_includes_practical_sections() -> None:
    content = get_help_content()

    assert "LangChain-based RAG application design" in content["helps_with"]
    assert "Out of scope" not in content["out_of_scope"]
    assert len(content["example_questions"]) >= 3
    assert any("Grounded answer" in item for item in content["response_types"])


def test_format_kb_status_label_uses_readable_state_text() -> None:
    status = KBStatusResult(
        state="up_to_date",
        summary="Knowledge base is up to date.",
        detail="The local Chroma index matches the current raw markdown snapshot.",
    )

    assert format_kb_status_label(status) == "Status: Up to date"


def test_build_conversation_markdown_handles_empty_history() -> None:
    markdown = build_conversation_markdown([])

    assert markdown == "# Conversation Export\n\n_No conversation history available._"


def test_build_conversation_markdown_formats_grounded_tool_and_fallback_turns() -> None:
    conversation_history = [
        {
            "query": "How should I persist Chroma locally?",
            "answer": "Persist the collection in a stable directory.",
            "used_context": True,
            "sources": ["Chroma Persistence and Reindexing Guide"],
            "tool_result": None,
        },
        {
            "query": "Estimate OpenAI cost",
            "answer": "Estimated total OpenAI cost: $0.002400 for model gpt-4.1-mini.",
            "used_context": False,
            "sources": [],
            "tool_result": {
                "tool_name": "estimate_openai_cost",
                "tool_error": None,
            },
        },
        {
            "query": "What is the capital of France?",
            "answer": "I could not find enough relevant context in the knowledge base to answer that safely.",
            "used_context": False,
            "sources": [],
            "tool_result": None,
        },
    ]

    markdown = build_conversation_markdown(conversation_history)

    assert "# Conversation Export" in markdown
    assert "**User question:** How should I persist Chroma locally?" in markdown
    assert "**Response type:** Grounded answer" in markdown
    assert "- Chroma Persistence and Reindexing Guide" in markdown
    assert "**Response type:** Tool result" in markdown
    assert "- tool_name: estimate_openai_cost" in markdown
    assert "**Response type:** No-context fallback" in markdown


def test_build_conversation_json_handles_empty_history() -> None:
    payload = json.loads(build_conversation_json([]))

    assert payload == {
        "export_format": "json",
        "turn_count": 0,
        "turns": [],
    }


def test_build_conversation_json_formats_grounded_answer_turn() -> None:
    payload = json.loads(
        build_conversation_json(
            [
                {
                    "query": "How should I persist Chroma locally?",
                    "answer": "Persist the collection in a stable directory.",
                    "used_context": True,
                    "sources": ["Chroma Persistence and Reindexing Guide"],
                    "tool_result": None,
                    "usage": {
                        "model_name": "gpt-4.1-mini",
                        "input_tokens": 20,
                        "output_tokens": 10,
                        "total_tokens": 30,
                        "estimated_cost_usd": 0.000024,
                    },
                }
            ]
        )
    )

    assert payload["export_format"] == "json"
    assert payload["turn_count"] == 1
    assert payload["turns"][0]["response_type"] == "Grounded answer"
    assert payload["turns"][0]["sources"] == ["Chroma Persistence and Reindexing Guide"]
    assert payload["turns"][0]["usage"] == {
        "model_name": "gpt-4.1-mini",
        "input_tokens": 20,
        "output_tokens": 10,
        "total_tokens": 30,
        "estimated_cost_usd": 0.000024,
    }


def test_build_conversation_json_formats_tool_result_turn() -> None:
    payload = json.loads(
        build_conversation_json(
            [
                {
                    "query": "Estimate OpenAI cost",
                    "answer": "Estimated total OpenAI cost: $0.002400 for model gpt-4.1-mini.",
                    "used_context": False,
                    "sources": [],
                    "tool_result": {
                        "tool_name": "estimate_openai_cost",
                        "tool_error": None,
                    },
                    "usage": None,
                }
            ]
        )
    )

    assert payload["turn_count"] == 1
    assert payload["turns"][0]["response_type"] == "Tool result"
    assert payload["turns"][0]["tool_result"] == {
        "tool_name": "estimate_openai_cost",
        "tool_error": None,
    }
    assert payload["turns"][0]["usage"] is None


def test_build_conversation_json_formats_no_context_fallback_turn() -> None:
    payload = json.loads(
        build_conversation_json(
            [
                {
                    "query": "What is the capital of France?",
                    "answer": (
                        "I could not find enough relevant context in the knowledge base "
                        "to answer that safely."
                    ),
                    "used_context": False,
                    "sources": [],
                    "tool_result": None,
                    "usage": None,
                }
            ]
        )
    )

    assert payload["turn_count"] == 1
    assert payload["turns"][0] == {
        "query": "What is the capital of France?",
        "answer": (
            "I could not find enough relevant context in the knowledge base "
            "to answer that safely."
        ),
        "response_type": "No-context fallback",
        "used_context": False,
        "sources": [],
        "tool_result": None,
        "usage": None,
    }


def test_build_conversation_csv_formats_grounded_answer_turn() -> None:
    csv_output = build_conversation_csv(
        [
            {
                "query": "How should I persist Chroma locally?",
                "answer": "Persist the collection in a stable directory.",
                "used_context": True,
                "sources": ["Chroma Persistence and Reindexing Guide"],
                "tool_result": None,
                "usage": {
                    "model_name": "gpt-4.1-mini",
                    "input_tokens": 20,
                    "output_tokens": 10,
                    "total_tokens": 30,
                    "estimated_cost_usd": 0.000024,
                },
            }
        ]
    )

    assert "turn_index,query,answer,response_type,used_context,sources" in csv_output
    assert "Grounded answer" in csv_output
    assert "Chroma Persistence and Reindexing Guide" in csv_output
    assert "gpt-4.1-mini" in csv_output


def test_build_conversation_csv_formats_tool_result_turn() -> None:
    csv_output = build_conversation_csv(
        [
            {
                "query": "Estimate OpenAI cost",
                "answer": "Estimated total OpenAI cost: $0.002400 for model gpt-4.1-mini.",
                "used_context": False,
                "sources": [],
                "tool_result": {
                    "tool_name": "estimate_openai_cost",
                    "tool_error": None,
                },
                "usage": None,
            }
        ]
    )

    row = next(csv.DictReader(io.StringIO(csv_output)))

    assert row["response_type"] == "Tool result"
    assert row["tool_name"] == "estimate_openai_cost"
    assert row["tool_result_json"] == (
        '{"tool_name":"estimate_openai_cost","tool_error":null}'
    )


def test_build_conversation_csv_formats_no_context_fallback_turn() -> None:
    csv_output = build_conversation_csv(
        [
            {
                "query": "What is the capital of France?",
                "answer": (
                    "I could not find enough relevant context in the knowledge base "
                    "to answer that safely."
                ),
                "used_context": False,
                "sources": [],
                "tool_result": None,
                "usage": None,
            }
        ]
    )

    row = next(csv.DictReader(io.StringIO(csv_output)))

    assert row["response_type"] == "No-context fallback"
    assert row["query"] == "What is the capital of France?"


def test_build_conversation_csv_preserves_sources_and_usage_fields() -> None:
    csv_output = build_conversation_csv(
        [
            {
                "query": "How should I show sources in Streamlit?",
                "answer": "Show source titles next to the answer.",
                "used_context": True,
                "sources": [
                    "Streamlit Chat Patterns",
                    "Chroma Persistence Guide",
                ],
                "tool_result": None,
                "usage": {
                    "model_name": "gpt-4.1-mini",
                    "input_tokens": 12,
                    "output_tokens": 5,
                    "total_tokens": 17,
                    "estimated_cost_usd": 0.000013,
                },
            }
        ]
    )

    row = next(csv.DictReader(io.StringIO(csv_output)))

    assert row["sources"] == "Streamlit Chat Patterns | Chroma Persistence Guide"
    assert row["usage_input_tokens"] == "12"
    assert row["usage_output_tokens"] == "5"
    assert row["usage_total_tokens"] == "17"
    assert row["usage_estimated_cost_usd"] in {"1.3e-05", "0.000013"}


def test_build_conversation_pdf_returns_non_empty_bytes_for_empty_history() -> None:
    pdf_output = build_conversation_pdf([])

    assert isinstance(pdf_output, bytes)
    assert len(pdf_output) > 0


def test_build_conversation_pdf_returns_non_empty_bytes_for_non_empty_history() -> None:
    pdf_output = build_conversation_pdf(
        [
            {
                "query": "How should I persist Chroma locally?",
                "answer": "Persist the collection in a stable directory.",
                "used_context": True,
                "sources": ["Chroma Persistence and Reindexing Guide"],
                "tool_result": None,
                "usage": {
                    "model_name": "gpt-4.1-mini",
                    "input_tokens": 20,
                    "output_tokens": 10,
                    "total_tokens": 30,
                    "estimated_cost_usd": 0.000024,
                },
            }
        ]
    )

    assert isinstance(pdf_output, bytes)
    assert len(pdf_output) > 0


def test_normalize_text_for_pdf_replaces_common_unicode_punctuation() -> None:
    normalized = normalize_text_for_pdf(
        'Smart “quotes”, apostrophes ’, joined—words, ranged–text, bullet •, ellipsis …, and\xa0space.'
    )

    assert normalized == (
        'Smart "quotes", apostrophes \', joined words, ranged text, bullet -, ellipsis ..., and space.'
    )


def test_clean_markdown_text_for_pdf_removes_basic_markdown_artifacts() -> None:
    cleaned = clean_markdown_text_for_pdf(
        "### Retrieval Tips\nUse **Chroma** carefully.\n---\n## Notes\nKeep it simple."
    )

    assert cleaned == "Retrieval Tips\nUse Chroma carefully.\nNotes\nKeep it simple."


def test_build_pdf_detail_lines_formats_readable_bullet_sections() -> None:
    lines = build_pdf_detail_lines(
        "Tool result",
        {
            "tool_name": "estimate_openai_cost",
            "tool_error": None,
        },
    )

    assert lines == [
        "Tool result:",
        "- Tool name: estimate_openai_cost",
        "- Tool error: none",
    ]


def test_build_pdf_detail_lines_formats_missing_estimated_cost_as_na() -> None:
    lines = build_pdf_detail_lines(
        "Usage",
        {
            "model_name": "gpt-4.1-mini",
            "estimated_cost_usd": None,
        },
    )

    assert lines == [
        "Usage:",
        "- Model name: gpt-4.1-mini",
        "- Estimated cost USD: N/A",
    ]


def test_build_pdf_detail_lines_formats_numeric_estimated_cost_stably() -> None:
    lines = build_pdf_detail_lines(
        "Usage",
        {
            "estimated_cost_usd": 0.000024,
        },
    )

    assert lines == [
        "Usage:",
        "- Estimated cost USD: 0.000024",
    ]


def test_build_conversation_pdf_handles_unicode_content_across_turn_types() -> None:
    pdf_output = build_conversation_pdf(
        [
            {
                "query": "How should I persist Chroma — locally?",
                "answer": 'Use a "stable" directory - do not rebuild on every run.',
                "used_context": True,
                "sources": ["Chroma Persistence Guide — intro"],
                "tool_result": None,
                "usage": {
                    "model_name": "gpt-4.1-mini — preview",
                    "input_tokens": 20,
                    "output_tokens": 10,
                    "total_tokens": 30,
                    "estimated_cost_usd": 0.000024,
                },
            },
            {
                "query": "Estimate OpenAI cost",
                "answer": "Answered with a built-in tool.",
                "used_context": False,
                "sources": [],
                "tool_result": {
                    "tool_name": "estimate_openai_cost",
                    "tool_error": "Temporary issue — try again…",
                },
                "usage": None,
            },
            {
                "query": "What is the capital of France?",
                "answer": "No usable knowledge-base context found — grounded answer skipped.",
                "used_context": False,
                "sources": [],
                "tool_result": None,
                "usage": None,
            },
        ]
    )

    assert isinstance(pdf_output, bytes)
    assert len(pdf_output) > 0
    assert pdf_output.startswith(b"%PDF")


def test_get_export_artifact_returns_expected_filename_and_mime_type_per_format() -> None:
    conversation_history = [
        {
            "query": "How should I persist Chroma locally?",
            "answer": "Persist the collection in a stable directory.",
            "used_context": True,
            "sources": ["Chroma Persistence and Reindexing Guide"],
            "tool_result": None,
            "usage": None,
        }
    ]

    markdown_artifact = get_export_artifact(conversation_history, "Markdown")
    json_artifact = get_export_artifact(conversation_history, "JSON")
    csv_artifact = get_export_artifact(conversation_history, "CSV")
    pdf_artifact = get_export_artifact(conversation_history, "PDF")

    assert markdown_artifact["file_name"] == "conversation_export.md"
    assert markdown_artifact["mime"] == "text/markdown"
    assert json_artifact["file_name"] == "conversation_export.json"
    assert json_artifact["mime"] == "application/json"
    assert csv_artifact["file_name"] == "conversation_export.csv"
    assert csv_artifact["mime"] == "text/csv"
    assert pdf_artifact["file_name"] == "conversation_export.pdf"
    assert pdf_artifact["mime"] == "application/pdf"
    assert isinstance(pdf_artifact["data"], bytes)


def test_parse_source_string_extracts_title_metadata_and_path() -> None:
    parsed = parse_source_string(
        "Streamlit Chat Patterns | topic=streamlit | library=streamlit | "
        "doc_type=example | difficulty=intermediate | "
        "source=data/raw/streamlit_chat_patterns.md | chunk=0 | error_family=ui"
    )

    assert parsed == {
        "title": "Streamlit Chat Patterns",
        "metadata": {
            "topic": "streamlit",
            "library": "streamlit",
            "doc_type": "example",
            "difficulty": "intermediate",
            "source": "data/raw/streamlit_chat_patterns.md",
            "chunk": "0",
            "error_family": "ui",
        },
    }


def test_format_source_display_returns_readable_fragments() -> None:
    source_display = format_source_display(
        "Chroma Persistence Guide | topic=chroma | library=chroma | "
        "doc_type=how_to | difficulty=intro | "
        "source=data/raw/chroma_persistence_guide.md | chunk=0 | "
        "error_family=persistence"
    )

    assert source_display == {
        "title": "Chroma Persistence Guide",
        "metadata_fragments": [
            "Topic: chroma",
            "Library: chroma",
            "Type: how_to",
            "Difficulty: intro",
            "Chunk: 0",
            "Error family: persistence",
        ],
        "source_path": "data/raw/chroma_persistence_guide.md",
        "raw_source": (
            "Chroma Persistence Guide | topic=chroma | library=chroma | "
            "doc_type=how_to | difficulty=intro | "
            "source=data/raw/chroma_persistence_guide.md | chunk=0 | "
            "error_family=persistence"
        ),
        "parse_failed": False,
    }


def test_format_source_display_falls_back_safely_for_malformed_source_string() -> None:
    source_display = format_source_display(
        "Streamlit Chat Patterns | topic=streamlit | malformed-fragment"
    )

    assert source_display == {
        "title": "Source",
        "metadata_fragments": [],
        "source_path": None,
        "raw_source": "Streamlit Chat Patterns | topic=streamlit | malformed-fragment",
        "parse_failed": True,
    }


def test_format_request_usage_label_handles_grounded_tool_and_fallback_turns() -> None:
    grounded_turn = {
        "query": "How should I persist Chroma locally?",
        "answer": "Persist the collection in a stable directory.",
        "used_context": True,
        "sources": ["Chroma Persistence and Reindexing Guide"],
        "tool_result": None,
        "usage": {
            "model_name": "gpt-4.1-mini",
            "input_tokens": 20,
            "output_tokens": 10,
            "total_tokens": 30,
            "estimated_cost_usd": 0.000024,
        },
    }
    tool_turn = {
        "query": "Estimate OpenAI cost",
        "answer": "Estimated total OpenAI cost: $0.002400 for model gpt-4.1-mini.",
        "used_context": False,
        "sources": [],
        "tool_result": {"tool_name": "estimate_openai_cost"},
        "usage": None,
    }
    unavailable_turn = {
        "query": "How should I show sources in Streamlit?",
        "answer": "Show source titles next to the answer.",
        "used_context": True,
        "sources": ["Streamlit Chat Patterns"],
        "tool_result": None,
        "usage": None,
    }
    fallback_turn = {
        "query": "What is the capital of France?",
        "answer": "I could not find enough relevant context in the knowledge base to answer that safely.",
        "used_context": False,
        "sources": [],
        "tool_result": None,
        "usage": None,
    }

    assert format_request_usage_label(grounded_turn) == (
        "LLM usage: gpt-4.1-mini | 20 in / 10 out / 30 total | $0.000024"
    )
    assert format_request_usage_label(tool_turn) == "No LLM usage"
    assert format_request_usage_label(unavailable_turn) == "Usage unavailable"
    assert format_request_usage_label(fallback_turn) == "No LLM usage"


def test_build_session_usage_totals_sums_usage_from_history() -> None:
    conversation_history = [
        {
            "query": "q1",
            "answer": "a1",
            "used_context": True,
            "sources": [],
            "tool_result": None,
            "usage": {
                "model_name": "gpt-4.1-mini",
                "input_tokens": 20,
                "output_tokens": 10,
                "total_tokens": 30,
                "estimated_cost_usd": 0.000024,
            },
        },
        {
            "query": "q2",
            "answer": "a2",
            "used_context": True,
            "sources": [],
            "tool_result": None,
            "usage": {
                "model_name": "gpt-4.1-mini",
                "input_tokens": 15,
                "output_tokens": 5,
                "total_tokens": 20,
                "estimated_cost_usd": 0.000014,
            },
        },
        {
            "query": "q3",
            "answer": "tool",
            "used_context": False,
            "sources": [],
            "tool_result": {"tool_name": "estimate_openai_cost"},
            "usage": None,
        },
    ]

    totals = build_session_usage_totals(conversation_history)

    assert totals == {
        "request_count": 2,
        "input_tokens": 35,
        "output_tokens": 15,
        "total_tokens": 50,
        "estimated_cost_usd": 0.000038,
    }
    assert format_session_usage_label(conversation_history) == (
        "2 requests | 50 total tokens | $0.000038"
    )


def test_format_session_usage_label_handles_empty_history() -> None:
    assert format_session_usage_label([]) == "No tracked LLM usage yet."
