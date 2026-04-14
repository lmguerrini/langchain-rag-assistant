import csv
import io
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from build_index import KBRebuildResult
from app import (
    AppValidationError,
    build_chat_model_cache_inputs,
    build_chat_input_visibility_script,
    build_conversation_csv,
    build_conversation_json,
    build_conversation_markdown,
    build_conversation_pdf,
    build_conversation_snapshot,
    build_model_usage_chart,
    build_official_docs_display_data,
    build_model_usage_chart_rows,
    build_pdf_detail_lines,
    build_response_behavior_chart,
    build_response_behavior_chart_rows,
    build_session_usage_totals,
    build_turn_record,
    build_vector_store_cache_inputs,
    build_kb_rebuild_error_message,
    build_kb_rebuild_success_message,
    clean_markdown_text_for_pdf,
    format_kb_status_label,
    format_request_usage_label,
    format_source_display,
    format_session_usage_label,
    get_export_artifact,
    get_response_generation_explanation,
    get_help_content,
    get_initial_chat_model_selection,
    get_response_summary_line,
    get_response_type_label,
    get_user_facing_error_message,
    normalize_text_for_pdf,
    parse_source_string,
    run_kb_rebuild_action,
    should_show_kb_rebuild_trigger,
    should_skip_resource_loading,
    validate_selected_chat_model,
    validate_query,
)
from src.config import SUPPORTED_CHAT_MODELS
from src.config import Settings
from src.kb_status import KBStatusResult
from src.schemas import AnswerResult, RequestUsage


class FakeSettings(SimpleNamespace):
    supported_chat_models = SUPPORTED_CHAT_MODELS
    default_chat_model = "gpt-4.1-mini"

    def ensure_openai_api_key(self) -> str:
        return self.openai_api_key

    def ensure_supported_chat_model(self, model_name: str) -> str:
        cleaned_model_name = model_name.strip()
        if cleaned_model_name not in self.supported_chat_models:
            raise ValueError(
                "Unsupported chat model selected. Choose one of: "
                + ", ".join(self.supported_chat_models)
            )
        return cleaned_model_name


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
    assert get_user_facing_error_message(
        RuntimeError("The model `gpt-4.1` does not exist or you do not have access to it.")
    ) == (
        "The selected OpenAI model is unavailable for this API key. Choose a different model and try again."
    )


def test_settings_default_chat_model_is_supported() -> None:
    settings = Settings(DEFAULT_CHAT_MODEL="gpt-4.1-mini")

    assert settings.default_chat_model == "gpt-4.1-mini"
    assert settings.default_chat_model in settings.supported_chat_models
    assert settings.supported_chat_models == SUPPORTED_CHAT_MODELS


def test_settings_reject_unsupported_default_chat_model() -> None:
    with pytest.raises(ValueError, match="Unsupported chat model selected"):
        Settings(DEFAULT_CHAT_MODEL="gpt-4.1-turbo")


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
        "official_docs_result": None,
        "usage": {
            "model_name": "gpt-4.1-mini",
            "input_tokens": 20,
            "output_tokens": 10,
            "total_tokens": 30,
            "estimated_cost_usd": 0.000024,
        },
    }


def test_should_skip_resource_loading_only_for_tool_queries() -> None:
    assert should_skip_resource_loading(
        "Estimate OpenAI cost for model gpt-4.1-mini with 1000 input tokens, 500 output tokens, and 3 calls"
    ) is True
    assert should_skip_resource_loading("Estimate OpenAI cost") is True
    assert should_skip_resource_loading(
        "How should I persist and rebuild the Chroma index locally?"
    ) is False


def test_build_official_docs_display_data_formats_readable_documents() -> None:
    display_data = build_official_docs_display_data(
        {
            "library": "langchain",
            "answer": "Start with a simple retrieval pipeline.",
            "lookup_result": {
                "library": "langchain",
                "documents": [
                    {
                        "title": "Build a RAG agent with LangChain",
                        "url": "https://docs.langchain.com/guides/rag",
                        "provider_mode": "official_mcp",
                        "snippets": [
                            {"text": "Start with a simple retrieval pipeline.", "rank": 1},
                            {"text": "Keep the first version narrow and testable.", "rank": 2},
                        ],
                    }
                ],
            },
        }
    )

    assert display_data == {
        "library": "LangChain",
        "documents": [
            {
                "title": "Build a RAG agent with LangChain",
                "url": "https://docs.langchain.com/guides/rag",
                "provider_label": "Provider: Official MCP",
                "snippets": [
                    "Start with a simple retrieval pipeline.",
                    "Keep the first version narrow and testable.",
                ],
            }
        ],
    }


def test_build_official_docs_display_data_handles_missing_or_empty_documents() -> None:
    assert build_official_docs_display_data(None) is None
    assert build_official_docs_display_data(
        {
            "library": "openai",
            "lookup_result": {
                "library": "openai",
                "documents": [],
            },
        }
    ) == {
        "library": "OpenAI",
        "documents": [],
    }


def test_build_response_behavior_chart_rows_keeps_chart_friendly_shape() -> None:
    assert build_response_behavior_chart_rows(
        [
            {"response_type": "Grounded answer", "count": 4, "share": 0.4},
            {"response_type": "Tool result", "count": 2, "share": 0.2},
        ]
    ) == [
        {"response_type": "Grounded answer", "count": 4},
        {"response_type": "Tool result", "count": 2},
    ]


def test_build_model_usage_chart_rows_keeps_chart_friendly_shape() -> None:
    assert build_model_usage_chart_rows(
        [
            {
                "model": "gpt-4.1-mini",
                "request_count": 2,
                "input_tokens": 40,
                "output_tokens": 20,
                "total_tokens": 60,
                "estimated_cost_usd": 0.000024,
            }
        ]
    ) == [
        {
            "model": "gpt-4.1-mini",
            "total_tokens": 60,
        }
    ]


def test_build_response_behavior_chart_is_horizontal() -> None:
    chart = build_response_behavior_chart(
        [
            {"response_type": "Grounded answer", "count": 4, "share": 0.4},
            {"response_type": "Tool result", "count": 2, "share": 0.2},
        ]
    )
    chart_dict = chart.to_dict(validate=False)

    assert chart_dict["mark"] == {"type": "bar"}
    assert chart_dict["encoding"]["x"]["field"] == "count"
    assert chart_dict["encoding"]["x"]["type"] == "quantitative"
    assert chart_dict["encoding"]["y"]["field"] == "response_type"
    assert chart_dict["encoding"]["y"]["type"] == "nominal"


def test_build_model_usage_chart_is_horizontal() -> None:
    chart = build_model_usage_chart(
        [
            {
                "model": "gpt-4.1-mini",
                "request_count": 2,
                "input_tokens": 40,
                "output_tokens": 20,
                "total_tokens": 60,
                "estimated_cost_usd": 0.000024,
            }
        ]
    )
    chart_dict = chart.to_dict(validate=False)

    assert chart_dict["mark"] == {"type": "bar"}
    assert chart_dict["encoding"]["x"]["field"] == "total_tokens"
    assert chart_dict["encoding"]["x"]["type"] == "quantitative"
    assert chart_dict["encoding"]["y"]["field"] == "model"
    assert chart_dict["encoding"]["y"]["type"] == "nominal"


def test_build_chat_input_visibility_script_targets_chat_tab_and_chat_input() -> None:
    script = build_chat_input_visibility_script()

    assert 'div[data-testid="stChatInput"]' in script
    assert "button[role=\"tab\"]" in script
    assert "Chat" in script
    assert "Analytics" in script
    assert "aria-selected" in script


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
            "tool_result": None,
            "official_docs_result": {
                "library": "langchain",
            },
        }
    ) == "Official docs answer"
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
            "tool_result": None,
            "official_docs_result": {
                "library": "langchain",
            },
        }
    ) == "Answered from official documentation evidence."
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
            "tool_result": None,
            "official_docs_result": {
                "library": "langchain",
            },
        }
    ) == (
        "The app looked up official documentation for the named library and "
        "generated this answer only from the retrieved official-docs evidence."
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
    assert (
        content["examples_intro"]
        == "You can ask natural questions or use structured inputs for tools."
    )
    assert content["example_questions"][-2:] == [
        "Estimate OpenAI cost for model gpt-4.1-mini with 1000 input tokens, 500 output tokens, and 3 calls",
        "Alternative format: model=gpt-4.1-mini, input_tokens=1000, output_tokens=500, num_calls=3",
    ]
    assert any("Grounded answer" in item for item in content["response_types"])
    assert any("Official docs answer" in item for item in content["response_types"])


def test_format_kb_status_label_uses_readable_state_text() -> None:
    status = KBStatusResult(
        state="up_to_date",
        summary="Knowledge base is up to date.",
        detail="The local Chroma index matches the current raw markdown snapshot.",
    )

    assert format_kb_status_label(status) == "Status: Up to date"


def test_should_show_kb_rebuild_trigger_only_for_missing_or_outdated() -> None:
    assert should_show_kb_rebuild_trigger(
        KBStatusResult(
            state="missing",
            summary="Knowledge base is missing.",
            detail="No usable local index was found.",
            rebuild_command="python build_index.py",
        )
    )
    assert should_show_kb_rebuild_trigger(
        KBStatusResult(
            state="outdated",
            summary="Knowledge base is outdated.",
            detail="The raw markdown snapshot changed.",
            rebuild_command="python build_index.py",
        )
    )
    assert not should_show_kb_rebuild_trigger(
        KBStatusResult(
            state="up_to_date",
            summary="Knowledge base is up to date.",
            detail="The local index matches the raw snapshot.",
            rebuild_command=None,
        )
    )


def test_build_kb_rebuild_success_message_is_short_and_readable() -> None:
    message = build_kb_rebuild_success_message(
        KBRebuildResult(
            indexed_chunk_count=7,
            collection_name="langchain_rag_knowledge_base",
            persist_directory=Path("data/chroma_db"),
            manifest_path=Path("data/chroma_db/kb_manifest.json"),
        )
    )

    assert message == (
        "Knowledge base rebuilt successfully. "
        "Indexed 7 chunks into 'langchain_rag_knowledge_base'."
    )


def test_build_kb_rebuild_error_message_is_clean() -> None:
    assert build_kb_rebuild_error_message(RuntimeError("temporary failure")) == (
        "Knowledge base rebuild failed: temporary failure"
    )


def test_run_kb_rebuild_action_clears_vector_store_cache_on_success() -> None:
    cleared: list[str] = []

    def rebuild_stub(settings: FakeSettings) -> KBRebuildResult:
        assert settings.openai_api_key == "key-123"
        return KBRebuildResult(
            indexed_chunk_count=4,
            collection_name="langchain_rag_knowledge_base",
            persist_directory=Path("data/chroma_db"),
            manifest_path=Path("data/chroma_db/kb_manifest.json"),
        )

    outcome = run_kb_rebuild_action(
        settings=FakeSettings(openai_api_key="key-123"),
        rebuild_fn=rebuild_stub,
        clear_vector_store_cache_fn=lambda: cleared.append("cleared"),
    )

    assert outcome == {
        "ok": True,
        "should_rerun": True,
        "feedback": {
            "kind": "success",
            "message": (
                "Knowledge base rebuilt successfully. "
                "Indexed 4 chunks into 'langchain_rag_knowledge_base'."
            ),
        },
    }
    assert cleared == ["cleared"]


def test_run_kb_rebuild_action_does_not_clear_vector_store_cache_on_failure() -> None:
    cleared: list[str] = []

    def rebuild_stub(settings: FakeSettings) -> KBRebuildResult:
        raise RuntimeError("index build failed")

    outcome = run_kb_rebuild_action(
        settings=FakeSettings(openai_api_key="key-123"),
        rebuild_fn=rebuild_stub,
        clear_vector_store_cache_fn=lambda: cleared.append("cleared"),
    )

    assert outcome == {
        "ok": False,
        "should_rerun": False,
        "feedback": {
            "kind": "error",
            "message": "Knowledge base rebuild failed: index build failed",
        },
    }
    assert cleared == []


def test_get_initial_chat_model_selection_uses_valid_state_or_default() -> None:
    settings = FakeSettings(openai_api_key="key-123")

    assert (
        get_initial_chat_model_selection(settings, "gpt-4.1")
        == "gpt-4.1"
    )
    assert (
        get_initial_chat_model_selection(settings, "not-a-real-model")
        == settings.default_chat_model
    )
    assert (
        get_initial_chat_model_selection(settings, None)
        == settings.default_chat_model
    )


def test_validate_selected_chat_model_rejects_unsupported_model() -> None:
    settings = FakeSettings(openai_api_key="key-123")

    with pytest.raises(AppValidationError, match="Unsupported chat model selected"):
        validate_selected_chat_model("not-a-real-model", settings)


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


def test_build_conversation_snapshot_is_stable_for_equivalent_turns() -> None:
    first_history = [
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
    second_history = [
        {
            "usage": {
                "total_tokens": 30,
                "estimated_cost_usd": 0.000024,
                "output_tokens": 10,
                "input_tokens": 20,
                "model_name": "gpt-4.1-mini",
            },
            "tool_result": None,
            "sources": ["Chroma Persistence and Reindexing Guide"],
            "used_context": True,
            "answer": "Persist the collection in a stable directory.",
            "query": "How should I persist Chroma locally?",
        }
    ]

    assert build_conversation_snapshot(first_history) == build_conversation_snapshot(
        second_history
    )


def test_build_conversation_snapshot_changes_when_conversation_changes() -> None:
    first_snapshot = build_conversation_snapshot(
        [
            {
                "query": "How should I persist Chroma locally?",
                "answer": "Persist the collection in a stable directory.",
                "used_context": True,
                "sources": ["Chroma Persistence and Reindexing Guide"],
                "tool_result": None,
                "usage": None,
            }
        ]
    )
    second_snapshot = build_conversation_snapshot(
        [
            {
                "query": "How should I persist Chroma locally?",
                "answer": "Use a durable persist directory.",
                "used_context": True,
                "sources": ["Chroma Persistence and Reindexing Guide"],
                "tool_result": None,
                "usage": None,
            }
        ]
    )

    assert first_snapshot != second_snapshot


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


def test_get_export_artifact_data_changes_when_conversation_changes() -> None:
    first_artifact = get_export_artifact(
        [
            {
                "query": "How should I persist Chroma locally?",
                "answer": "Persist the collection in a stable directory.",
                "used_context": True,
                "sources": ["Chroma Persistence Guide"],
                "tool_result": None,
                "usage": None,
            }
        ],
        "Markdown",
    )
    second_artifact = get_export_artifact(
        [
            {
                "query": "How should I persist Chroma locally?",
                "answer": "Use a durable persist directory instead.",
                "used_context": True,
                "sources": ["Chroma Persistence Guide"],
                "tool_result": None,
                "usage": None,
            }
        ],
        "Markdown",
    )

    assert first_artifact["data"] != second_artifact["data"]


def test_get_export_artifact_data_changes_when_format_changes() -> None:
    conversation_history = [
        {
            "query": "How should I persist Chroma locally?",
            "answer": "Persist the collection in a stable directory.",
            "used_context": True,
            "sources": ["Chroma Persistence Guide"],
            "tool_result": None,
            "usage": None,
        }
    ]

    markdown_artifact = get_export_artifact(conversation_history, "Markdown")
    json_artifact = get_export_artifact(conversation_history, "JSON")

    assert markdown_artifact["data"] != json_artifact["data"]
    assert markdown_artifact["file_name"] != json_artifact["file_name"]


def test_build_vector_store_cache_inputs_reflects_settings_values() -> None:
    settings = FakeSettings(
        chroma_persist_dir=Path("data/custom_db"),
        chroma_collection_name="custom_collection",
        embedding_model="text-embedding-3-large",
        openai_api_key="key-123",
    )

    assert build_vector_store_cache_inputs(settings) == {
        "persist_directory": "data/custom_db",
        "collection_name": "custom_collection",
        "embedding_model": "text-embedding-3-large",
        "api_key": "key-123",
    }


def test_build_chat_model_cache_inputs_reflects_settings_values() -> None:
    settings = FakeSettings(openai_api_key="key-456")

    assert build_chat_model_cache_inputs(settings, "gpt-4.1") == {
        "api_key": "key-456",
        "model_name": "gpt-4.1",
        "temperature": 0,
    }


def test_build_chat_model_cache_inputs_change_when_selected_model_changes() -> None:
    settings = FakeSettings(openai_api_key="key-456")

    first_inputs = build_chat_model_cache_inputs(settings, "gpt-4.1-mini")
    second_inputs = build_chat_model_cache_inputs(settings, "gpt-4o-mini")

    assert first_inputs != second_inputs
    assert first_inputs["model_name"] == "gpt-4.1-mini"
    assert second_inputs["model_name"] == "gpt-4o-mini"


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


def test_format_request_usage_label_handles_grounded_official_docs_tool_and_fallback_turns() -> None:
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
    official_docs_turn = {
        "query": "According to LangChain docs, how should I start a small RAG app?",
        "answer": "According to the official LangChain docs, start with a simple retrieval pipeline.",
        "used_context": False,
        "sources": [],
        "tool_result": None,
        "official_docs_result": {
            "library": "langchain",
        },
        "usage": {
            "model_name": "gpt-4.1-mini",
            "input_tokens": 18,
            "output_tokens": 7,
            "total_tokens": 25,
            "estimated_cost_usd": 0.000018,
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
    assert format_request_usage_label(official_docs_turn) == (
        "LLM usage: gpt-4.1-mini | 18 in / 7 out / 25 total | $0.000018"
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
