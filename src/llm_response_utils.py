from __future__ import annotations

import re
from collections.abc import Mapping

from src.config import SUPPORTED_CHAT_MODELS
from src.schemas import RequestUsage


CHAT_MODEL_PRICING_PER_MILLION = {
    "gpt-4.1-mini": {"input": 0.40, "output": 1.60},
    "gpt-4.1": {"input": 2.00, "output": 8.00},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
}
MODEL_VERSION_SUFFIX_PATTERN = re.compile(r"\d{4}-\d{2}-\d{2}$")

UNPRICED_SUPPORTED_CHAT_MODELS = tuple(
    model_name
    for model_name in SUPPORTED_CHAT_MODELS
    if model_name not in CHAT_MODEL_PRICING_PER_MILLION
)

if UNPRICED_SUPPORTED_CHAT_MODELS:
    raise RuntimeError(
        "Missing chat-model pricing for supported models: "
        + ", ".join(UNPRICED_SUPPORTED_CHAT_MODELS)
    )


def extract_text(model_response) -> str:
    content = getattr(model_response, "content", model_response)
    if isinstance(content, str):
        return content.strip()
    return str(content).strip()


def extract_request_usage(
    model_response,
    *,
    chat_model: object,
) -> RequestUsage | None:
    response_metadata = getattr(model_response, "response_metadata", None)
    usage_metadata = getattr(model_response, "usage_metadata", None)
    usage_payload = normalize_usage_payload(usage_metadata)

    if usage_payload is None and isinstance(response_metadata, Mapping):
        usage_payload = normalize_usage_payload(response_metadata.get("token_usage"))

    if usage_payload is None:
        return None

    model_name = extract_model_name(model_response, chat_model=chat_model)
    estimated_cost_usd = estimate_cost_usd(
        model_name=model_name,
        input_tokens=usage_payload["input_tokens"],
        output_tokens=usage_payload["output_tokens"],
    )

    return RequestUsage(
        model_name=model_name,
        input_tokens=usage_payload["input_tokens"],
        output_tokens=usage_payload["output_tokens"],
        total_tokens=usage_payload["total_tokens"],
        estimated_cost_usd=estimated_cost_usd,
    )


def normalize_usage_payload(payload) -> dict[str, int] | None:
    if not isinstance(payload, Mapping):
        return None

    input_tokens = payload.get("input_tokens", payload.get("prompt_tokens"))
    output_tokens = payload.get("output_tokens", payload.get("completion_tokens"))
    total_tokens = payload.get("total_tokens")

    if not isinstance(input_tokens, int) or not isinstance(output_tokens, int):
        return None

    if not isinstance(total_tokens, int):
        total_tokens = input_tokens + output_tokens

    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
    }


def extract_model_name(model_response, *, chat_model: object) -> str | None:
    response_metadata = getattr(model_response, "response_metadata", None)
    if isinstance(response_metadata, Mapping):
        model_name = response_metadata.get("model_name")
        if isinstance(model_name, str) and model_name.strip():
            return model_name.strip()

    for attribute_name in ("model_name", "model"):
        model_name = getattr(chat_model, attribute_name, None)
        if isinstance(model_name, str) and model_name.strip():
            return model_name.strip()

    return None


def estimate_cost_usd(
    *,
    model_name: str | None,
    input_tokens: int,
    output_tokens: int,
) -> float | None:
    if model_name is None:
        return None

    priced_model_name = resolve_priced_model_name(model_name)
    if priced_model_name is None:
        return None

    pricing = CHAT_MODEL_PRICING_PER_MILLION[priced_model_name]
    input_cost = (input_tokens / 1_000_000) * pricing["input"]
    output_cost = (output_tokens / 1_000_000) * pricing["output"]
    return round(input_cost + output_cost, 6)


def estimate_usage_cost_usd(usage: Mapping[str, object]) -> float | None:
    estimated_cost = usage.get("estimated_cost_usd")
    if isinstance(estimated_cost, int | float):
        return round(float(estimated_cost), 6)

    input_tokens = usage.get("input_tokens")
    output_tokens = usage.get("output_tokens")
    model_name = usage.get("model_name")
    if (
        not isinstance(model_name, str)
        or not isinstance(input_tokens, int)
        or not isinstance(output_tokens, int)
    ):
        return None

    return estimate_cost_usd(
        model_name=model_name,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )


def resolve_priced_model_name(model_name: str) -> str | None:
    normalized_model_name = model_name.strip().lower()
    if normalized_model_name in CHAT_MODEL_PRICING_PER_MILLION:
        return normalized_model_name

    for base_model_name in sorted(CHAT_MODEL_PRICING_PER_MILLION, key=len, reverse=True):
        prefix = f"{base_model_name}-"
        if not normalized_model_name.startswith(prefix):
            continue
        version_suffix = normalized_model_name.removeprefix(prefix)
        if MODEL_VERSION_SUFFIX_PATTERN.fullmatch(version_suffix):
            return base_model_name

    return None
