from types import SimpleNamespace

from src.llm_response_utils import (
    estimate_cost_usd,
    estimate_usage_cost_usd,
    extract_request_usage,
    resolve_priced_model_name,
)


def test_resolve_priced_model_name_handles_base_and_versioned_models() -> None:
    assert resolve_priced_model_name("gpt-4.1-mini") == "gpt-4.1-mini"
    assert (
        resolve_priced_model_name("gpt-4.1-mini-2025-04-14")
        == "gpt-4.1-mini"
    )


def test_resolve_priced_model_name_rejects_unknown_or_unsafe_suffixes() -> None:
    assert resolve_priced_model_name("gpt-unknown-2025-04-14") is None
    assert resolve_priced_model_name("gpt-4.1-mini-preview-2025-04-14") is None


def test_estimate_cost_usd_uses_base_pricing_for_versioned_model_name() -> None:
    assert (
        estimate_cost_usd(
            model_name="gpt-4.1-mini-2025-04-14",
            input_tokens=18,
            output_tokens=7,
        )
        == 0.000018
    )


def test_estimate_cost_usd_keeps_unknown_models_unpriced() -> None:
    assert (
        estimate_cost_usd(
            model_name="gpt-unknown-2025-04-14",
            input_tokens=18,
            output_tokens=7,
        )
        is None
    )


def test_extract_request_usage_keeps_original_model_name_and_estimates_cost() -> None:
    response = SimpleNamespace(
        content="Answer",
        response_metadata={"model_name": "gpt-4.1-mini-2025-04-14"},
        usage_metadata={
            "input_tokens": 18,
            "output_tokens": 7,
            "total_tokens": 25,
        },
    )

    usage = extract_request_usage(response, chat_model=SimpleNamespace())

    assert usage is not None
    assert usage.model_name == "gpt-4.1-mini-2025-04-14"
    assert usage.estimated_cost_usd == 0.000018


def test_estimate_usage_cost_usd_infers_missing_cost_only_when_priced() -> None:
    assert (
        estimate_usage_cost_usd(
            {
                "model_name": "gpt-4.1-mini-2025-04-14",
                "input_tokens": 18,
                "output_tokens": 7,
                "total_tokens": 25,
                "estimated_cost_usd": None,
            }
        )
        == 0.000018
    )
    assert (
        estimate_usage_cost_usd(
            {
                "model_name": "gpt-unknown-2025-04-14",
                "input_tokens": 18,
                "output_tokens": 7,
                "total_tokens": 25,
                "estimated_cost_usd": None,
            }
        )
        is None
    )
