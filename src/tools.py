from __future__ import annotations

import re

from src.schemas import (
    DiagnoseStackErrorInput,
    DiagnoseStackErrorOutput,
    EstimateOpenAICostInput,
    EstimateOpenAICostOutput,
    RecommendRetrievalConfigInput,
    RecommendRetrievalConfigOutput,
    ToolInvocationResult,
)


MODEL_PRICING_PER_MILLION = {
    "gpt-4.1-mini": {"input": 0.40, "output": 1.60},
    "gpt-4.1": {"input": 2.00, "output": 8.00},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "text-embedding-3-small": {"input": 0.02, "output": 0.0},
}

TOKEN_PATTERNS = {
    "input_tokens": [
        re.compile(r"(?:input_tokens|input tokens)\s*=?\s*([\d,]+)"),
        re.compile(r"([\d,]+)\s*input tokens"),
    ],
    "output_tokens": [
        re.compile(r"(?:output_tokens|output tokens)\s*=?\s*([\d,]+)"),
        re.compile(r"([\d,]+)\s*output tokens"),
    ],
    "num_calls": [
        re.compile(r"(?:num_calls|calls)\s*=?\s*([\d,]+)"),
        re.compile(r"for\s+([\d,]+)\s+calls"),
        re.compile(r"([\d,]+)\s+calls"),
    ],
}


def estimate_openai_cost(
    tool_input: EstimateOpenAICostInput,
) -> EstimateOpenAICostOutput:
    normalized_model = tool_input.model.strip().lower()
    pricing = MODEL_PRICING_PER_MILLION.get(normalized_model)
    if pricing is None:
        raise ValueError(
            f"Unsupported model for cost estimation: {tool_input.model}"
        )

    total_input_tokens = tool_input.input_tokens * tool_input.num_calls
    total_output_tokens = tool_input.output_tokens * tool_input.num_calls
    input_cost = round((total_input_tokens / 1_000_000) * pricing["input"], 6)
    output_cost = round((total_output_tokens / 1_000_000) * pricing["output"], 6)

    return EstimateOpenAICostOutput(
        model=normalized_model,
        input_tokens=tool_input.input_tokens,
        output_tokens=tool_input.output_tokens,
        num_calls=tool_input.num_calls,
        estimated_input_cost_usd=input_cost,
        estimated_output_cost_usd=output_cost,
        estimated_total_cost_usd=round(input_cost + output_cost, 6),
    )


def diagnose_stack_error(
    tool_input: DiagnoseStackErrorInput,
) -> DiagnoseStackErrorOutput:
    normalized = " ".join(
        part.strip().lower()
        for part in [tool_input.error_message, tool_input.code_context_summary or ""]
        if part.strip()
    )

    if tool_input.library == "langchain" and (
        "no module named" in normalized or "cannot import" in normalized
    ):
        return DiagnoseStackErrorOutput(
            library=tool_input.library,
            error_category="imports",
            likely_causes=[
                "The import path changed between LangChain package versions.",
                "The environment is missing the required LangChain package.",
            ],
            recommended_checks=[
                "Check the installed LangChain package versions.",
                "Confirm the import path matches the installed release.",
            ],
        )

    if tool_input.library == "chroma" and (
        "collection" in normalized or "persist" in normalized or "sqlite" in normalized
    ):
        return DiagnoseStackErrorOutput(
            library=tool_input.library,
            error_category="persistence",
            likely_causes=[
                "The Chroma persistence directory is missing or inconsistent.",
                "The collection name does not match the stored index.",
            ],
            recommended_checks=[
                "Verify CHROMA_PERSIST_DIR points to the built local database.",
                "Confirm the collection name matches the indexed collection.",
            ],
        )

    if tool_input.library == "streamlit" and (
        "duplicatewidgetid" in normalized
        or "duplicate widget" in normalized
    ):
        return DiagnoseStackErrorOutput(
            library=tool_input.library,
            error_category="ui",
            likely_causes=[
                "Multiple Streamlit widgets are being created with the same implicit or explicit key.",
                "A widget is rendered repeatedly in a loop or rerun path without a unique key.",
            ],
            recommended_checks=[
                "Give each repeated widget a stable unique key.",
                "Check loops, conditionals, and chat reruns for duplicate widget construction.",
            ],
        )

    if tool_input.library == "streamlit" and (
        "session state" in normalized
        or "no module named 'streamlit'" in normalized
        or 'no module named "streamlit"' in normalized
    ):
        return DiagnoseStackErrorOutput(
            library=tool_input.library,
            error_category=(
                "imports"
                if "no module named" in normalized
                else "ui"
            ),
            likely_causes=[
                "The active Python environment does not have Streamlit installed.",
                "The command is using a different virtual environment than the one where Streamlit is installed.",
            ],
            recommended_checks=[
                "Install Streamlit in the active environment and confirm the interpreter path.",
                "Run the same command with the project virtualenv to verify the module import.",
            ],
        )

    if tool_input.library == "openai" and (
        "api key" in normalized
        or "401" in normalized
        or "authentication" in normalized
        or "invalid_api_key" in normalized
    ):
        return DiagnoseStackErrorOutput(
            library=tool_input.library,
            error_category="api",
            likely_causes=[
                "The OpenAI API key is missing, invalid, or loaded from the wrong environment.",
                "The request is using credentials that do not have access to the requested model.",
            ],
            recommended_checks=[
                "Confirm OPENAI_API_KEY is set in the active environment.",
                "Verify the selected model is available for the current API key.",
            ],
        )

    return DiagnoseStackErrorOutput(
        library=tool_input.library,
        error_category="unknown",
        likely_causes=[
            "The error message does not match one of the supported rule-based diagnostics.",
        ],
        recommended_checks=[
            "Review the full stack trace and configuration values around the failing step.",
            "Reduce the failing example to the smallest reproducible case.",
        ],
    )


def recommend_retrieval_config(
    tool_input: RecommendRetrievalConfigInput,
) -> RecommendRetrievalConfigOutput:
    chunk_size = 800
    chunk_overlap = 120
    top_k = 3
    use_metadata_filters = False

    if tool_input.content_type == "troubleshooting":
        chunk_size = 500
        chunk_overlap = 80
        top_k = 4
        use_metadata_filters = True
    elif tool_input.content_type == "example":
        chunk_size = 700
        chunk_overlap = 100
        top_k = 3
        use_metadata_filters = True
    elif tool_input.content_type == "concept":
        chunk_size = 900
        chunk_overlap = 140
        top_k = 4

    if tool_input.document_length == "advanced":
        chunk_size += 100
        top_k += 1
    elif tool_input.document_length == "intro":
        chunk_size -= 100
        top_k = max(2, top_k - 1)

    if tool_input.task_type == "debugging":
        chunk_overlap += 20
        top_k += 1
        use_metadata_filters = True
    elif tool_input.task_type == "implementation":
        chunk_overlap += 10
        use_metadata_filters = True

    return RecommendRetrievalConfigOutput(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        top_k=top_k,
        use_metadata_filters=use_metadata_filters,
        rationale=(
            f"Configured for {tool_input.content_type} content, "
            f"{tool_input.document_length} document length, and "
            f"{tool_input.task_type} retrieval needs."
        ),
    )


def maybe_invoke_tool(query: str) -> ToolInvocationResult | None:
    normalized = query.lower()

    cost_result = _build_cost_tool_result(query, normalized)
    if cost_result is not None:
        return cost_result

    diagnosis_result = _build_diagnosis_tool_result(query, normalized)
    if diagnosis_result is not None:
        return diagnosis_result

    retrieval_result = _build_retrieval_tool_result(query, normalized)
    if retrieval_result is not None:
        return retrieval_result

    return None


def format_tool_answer(tool_result: ToolInvocationResult) -> str:
    if tool_result.tool_error is not None:
        return tool_result.tool_error

    if tool_result.tool_name == "estimate_openai_cost":
        output = tool_result.tool_output
        return (
            f"Estimated total OpenAI cost: ${output.estimated_total_cost_usd:.6f} "
            f"for model {output.model}."
        )

    if tool_result.tool_name == "diagnose_stack_error":
        output = tool_result.tool_output
        primary_cause = output.likely_causes[0] if output.likely_causes else "No cause identified."
        return (
            f"Diagnosed {output.library} issue as {output.error_category}: "
            f"{primary_cause}"
        )

    output = tool_result.tool_output
    return (
        "Recommended retrieval settings: "
        f"chunk_size={output.chunk_size}, chunk_overlap={output.chunk_overlap}, "
        f"top_k={output.top_k}."
    )


def _build_cost_tool_result(
    original_query: str,
    normalized_query: str,
) -> ToolInvocationResult | None:
    if not _is_cost_query(normalized_query):
        return None

    cost_input = _parse_cost_query(normalized_query)
    if cost_input is None:
        return ToolInvocationResult(
            tool_name="estimate_openai_cost",
            raw_query=original_query,
            tool_error=(
                "Estimate OpenAI cost needs a supported model name plus input_tokens "
                "and output_tokens. Optionally include num_calls."
            ),
        )

    return ToolInvocationResult(
        tool_name="estimate_openai_cost",
        raw_query=original_query,
        tool_input=cost_input,
        tool_output=estimate_openai_cost(cost_input),
    )


def _build_diagnosis_tool_result(
    original_query: str,
    normalized_query: str,
) -> ToolInvocationResult | None:
    diagnosis_input = _parse_error_query(original_query, normalized_query)
    if diagnosis_input is None:
        return None

    return ToolInvocationResult(
        tool_name="diagnose_stack_error",
        raw_query=original_query,
        tool_input=diagnosis_input,
        tool_output=diagnose_stack_error(diagnosis_input),
    )


def _build_retrieval_tool_result(
    original_query: str,
    normalized_query: str,
) -> ToolInvocationResult | None:
    if not _is_retrieval_config_query(normalized_query):
        return None

    retrieval_input = _parse_retrieval_config_query(normalized_query)
    if retrieval_input is None:
        return ToolInvocationResult(
            tool_name="recommend_retrieval_config",
            raw_query=original_query,
            tool_error=(
                "Recommend retrieval config needs enough detail to infer content type, "
                "document length, and task type."
            ),
        )

    return ToolInvocationResult(
        tool_name="recommend_retrieval_config",
        raw_query=original_query,
        tool_input=retrieval_input,
        tool_output=recommend_retrieval_config(retrieval_input),
    )


def _is_cost_query(normalized_query: str) -> bool:
    has_openai = "openai" in normalized_query
    has_cost_intent = any(
        phrase in normalized_query
        for phrase in ("cost", "price", "pricing", "estimate", "calculate")
    )
    has_cost_shape = (
        has_openai
        and (
            has_cost_intent
            or "input tokens" in normalized_query
            or "output tokens" in normalized_query
            or any(model in normalized_query for model in MODEL_PRICING_PER_MILLION)
        )
    )
    return has_cost_shape


def _parse_cost_query(normalized_query: str) -> EstimateOpenAICostInput | None:
    if not _is_cost_query(normalized_query):
        return None

    model_name = next(
        (model for model in MODEL_PRICING_PER_MILLION if model in normalized_query),
        None,
    )
    if model_name is None:
        return None

    input_tokens = _extract_number(normalized_query, "input_tokens")
    output_tokens = _extract_number(normalized_query, "output_tokens")
    if input_tokens is None or output_tokens is None:
        return None

    num_calls = _extract_number(normalized_query, "num_calls") or 1
    return EstimateOpenAICostInput(
        model=model_name,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        num_calls=num_calls,
    )


def _parse_error_query(
    original_query: str,
    normalized_query: str,
) -> DiagnoseStackErrorInput | None:
    if not any(keyword in normalized_query for keyword in ("error", "exception", "traceback")):
        return None

    library = next(
        (name for name in ("langchain", "chroma", "streamlit", "openai") if name in normalized_query),
        None,
    )
    if library is None:
        return None

    error_message = original_query
    if ":" in original_query:
        suffix = original_query.split(":", 1)[1].strip()
        if suffix:
            error_message = suffix

    return DiagnoseStackErrorInput(
        library=library,
        error_message=error_message,
    )


def _parse_retrieval_config_query(
    normalized_query: str,
) -> RecommendRetrievalConfigInput | None:
    if not _is_retrieval_config_query(normalized_query):
        return None

    content_type = _detect_content_type(normalized_query)
    document_length = _detect_document_length(normalized_query)
    task_type = _detect_task_type(normalized_query)
    if content_type is None or document_length is None or task_type is None:
        return None

    return RecommendRetrievalConfigInput(
        content_type=content_type,
        document_length=document_length,
        task_type=task_type,
    )


def _is_retrieval_config_query(normalized_query: str) -> bool:
    has_retrieval_intent = any(
        phrase in normalized_query
        for phrase in (
            "retrieval config",
            "retrieval settings",
            "retrieval setup",
            "chunk size",
            "top_k",
            "top k",
            "recommend retrieval",
            "recommend config",
        )
    )
    has_related_terms = "retrieval" in normalized_query or "chunk" in normalized_query
    return has_retrieval_intent or (
        "recommend" in normalized_query and has_related_terms
    )


def _extract_number(query: str, field_name: str) -> int | None:
    for pattern in TOKEN_PATTERNS[field_name]:
        match = pattern.search(query)
        if match is not None:
            captured_value = match.group(1).strip()
            if not captured_value:
                continue
            try:
                return int(captured_value.replace(",", ""))
            except ValueError:
                continue
    return None


def _detect_content_type(query: str) -> str | None:
    if "troubleshooting" in query:
        return "troubleshooting"
    if "code-heavy" in query or "code heavy" in query:
        return "example"
    if "example" in query:
        return "example"
    if "concept" in query:
        return "concept"
    if "technical documentation" in query or "documentation" in query:
        return "how_to"
    if "markdown docs" in query or "markdown documents" in query or "markdown" in query:
        return "how_to"
    if "how to" in query or "how_to" in query:
        return "how_to"
    return None


def _detect_document_length(query: str) -> str | None:
    if "intro" in query or "short" in query:
        return "intro"
    if "intermediate" in query or "medium" in query:
        return "intermediate"
    if "advanced" in query or "long" in query:
        return "advanced"
    if "code-heavy" in query or "code heavy" in query:
        return "advanced"
    return None


def _detect_task_type(query: str) -> str | None:
    if "debug" in query or "troubleshooting" in query:
        return "debugging"
    if "implementation" in query or "implement" in query:
        return "implementation"
    if "question" in query or "qa" in query or "answer" in query:
        return "question_answering"
    return None
