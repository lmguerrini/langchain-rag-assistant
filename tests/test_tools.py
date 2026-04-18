import re

import pytest
import src.tools as tools

from src.tools import (
    _extract_number,
    diagnose_stack_error,
    estimate_openai_cost,
    maybe_invoke_tool,
    recommend_retrieval_config,
)
from src.schemas import (
    DiagnoseStackErrorInput,
    EstimateOpenAICostInput,
    RecommendRetrievalConfigInput,
)


def test_estimate_openai_cost_calculates_supported_model_price() -> None:
    result = estimate_openai_cost(
        EstimateOpenAICostInput(
            model="gpt-4.1-mini",
            input_tokens=1000,
            output_tokens=500,
            num_calls=2,
        )
    )

    assert result.estimated_input_cost_usd == pytest.approx(0.0008)
    assert result.estimated_output_cost_usd == pytest.approx(0.0016)
    assert result.estimated_total_cost_usd == pytest.approx(0.0024)


def test_estimate_openai_cost_rejects_unsupported_model() -> None:
    with pytest.raises(ValueError, match="Unsupported model"):
        estimate_openai_cost(
            EstimateOpenAICostInput(
                model="unknown-model",
                input_tokens=1000,
                output_tokens=500,
                num_calls=1,
            )
        )


def test_diagnose_stack_error_returns_rule_based_diagnosis() -> None:
    result = diagnose_stack_error(
        DiagnoseStackErrorInput(
            library="openai",
            error_message="401 authentication error: incorrect api key provided",
        )
    )

    assert result.error_category == "api"
    assert any("OPENAI_API_KEY" in check for check in result.recommended_checks)


def test_diagnose_streamlit_missing_module_is_not_unknown() -> None:
    result = diagnose_stack_error(
        DiagnoseStackErrorInput(
            library="streamlit",
            error_message="No module named 'streamlit'",
        )
    )

    assert result.error_category == "imports"


def test_diagnose_streamlit_duplicate_widget_id_returns_specific_ui_guidance() -> None:
    result = diagnose_stack_error(
        DiagnoseStackErrorInput(
            library="streamlit",
            error_message="DuplicateWidgetID",
        )
    )

    assert result.error_category == "ui"
    assert any("unique key" in check for check in result.recommended_checks)
    assert any("same implicit or explicit key" in cause for cause in result.likely_causes)


def test_diagnose_invalid_api_key_is_not_unknown() -> None:
    result = diagnose_stack_error(
        DiagnoseStackErrorInput(
            library="openai",
            error_message="invalid_api_key",
        )
    )

    assert result.error_category == "api"


def test_recommend_retrieval_config_returns_rule_based_settings() -> None:
    result = recommend_retrieval_config(
        RecommendRetrievalConfigInput(
            content_type="troubleshooting",
            document_length="advanced",
            task_type="debugging",
        )
    )

    assert result.chunk_size == 600
    assert result.chunk_overlap == 100
    assert result.top_k == 6
    assert result.use_metadata_filters is True


def test_maybe_invoke_tool_matches_retrieval_config_request() -> None:
    result = maybe_invoke_tool(
        "Recommend retrieval config for troubleshooting advanced debugging "
        "retrieval settings"
    )

    assert result is not None
    assert result.tool_name == "recommend_retrieval_config"


def test_maybe_invoke_tool_matches_calculate_cost_phrasing() -> None:
    result = maybe_invoke_tool(
        "Estimate OpenAI cost for model gpt-4.1-mini with 1000 input tokens, "
        "500 output tokens, and 3 calls"
    )

    assert result is not None
    assert result.tool_name == "estimate_openai_cost"
    assert result.tool_error is None
    assert result.tool_input.num_calls == 3
    assert result.tool_output.estimated_total_cost_usd == pytest.approx(0.0036)


def test_extract_number_skips_empty_match_groups(monkeypatch) -> None:
    monkeypatch.setitem(
        tools.TOKEN_PATTERNS,
        "input_tokens",
        [
            re.compile(r"()"),
            re.compile(r"([\d,]+)\s*input tokens"),
        ],
    )

    assert _extract_number("1000 input tokens", "input_tokens") == 1000


def test_maybe_invoke_tool_returns_validation_error_for_missing_cost_parameters() -> None:
    result = maybe_invoke_tool("Estimate OpenAI cost")

    assert result is not None
    assert result.tool_name == "estimate_openai_cost"
    assert result.tool_output is None
    assert "supported model name" in result.tool_error


def test_maybe_invoke_tool_matches_short_markdown_retrieval_request() -> None:
    result = maybe_invoke_tool(
        "Recommend retrieval config for short markdown docs used for question answering"
    )

    assert result is not None
    assert result.tool_name == "recommend_retrieval_config"
    assert result.tool_error is None
    assert result.tool_input.content_type == "how_to"
    assert result.tool_input.document_length == "intro"
    assert result.tool_input.task_type == "question_answering"


def test_maybe_invoke_tool_matches_long_debugging_retrieval_request() -> None:
    result = maybe_invoke_tool(
        "Recommend retrieval config for long technical documentation used for debugging questions"
    )

    assert result is not None
    assert result.tool_name == "recommend_retrieval_config"
    assert result.tool_error is None
    assert result.tool_input.content_type == "how_to"
    assert result.tool_input.document_length == "advanced"
    assert result.tool_input.task_type == "debugging"


def test_maybe_invoke_tool_matches_code_heavy_troubleshooting_request() -> None:
    result = maybe_invoke_tool(
        "Recommend retrieval config for code-heavy content and troubleshooting"
    )

    assert result is not None
    assert result.tool_name == "recommend_retrieval_config"
    assert result.tool_error is None
    assert result.tool_input.content_type in {"example", "troubleshooting"}
    assert result.tool_input.document_length == "advanced"
    assert result.tool_input.task_type == "debugging"
