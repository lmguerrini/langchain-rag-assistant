from __future__ import annotations

from collections.abc import Callable

import streamlit as st
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from src.chains import (
    maybe_match_official_docs_query,
    run_backend_query,
    stream_answer_query,
)
from src.config import Settings
from src.schemas import AnswerResult
from src.tools import maybe_invoke_tool


PREVIEW_LENGTH = 80


class AppValidationError(ValueError):
    pass


@st.cache_resource
def _build_cached_vector_store(
    *,
    persist_directory: str,
    collection_name: str,
    embedding_model: str,
    api_key: str,
) -> Chroma:
    embeddings = OpenAIEmbeddings(
        api_key=api_key,
        model=embedding_model,
    )
    return Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=persist_directory,
    )


def build_vector_store_cache_inputs(settings: Settings) -> dict[str, str]:
    return {
        "persist_directory": str(settings.chroma_persist_dir),
        "collection_name": settings.chroma_collection_name,
        "embedding_model": settings.embedding_model,
        "api_key": settings.ensure_openai_api_key(),
    }


def get_vector_store(settings: Settings) -> Chroma:
    return _build_cached_vector_store(**build_vector_store_cache_inputs(settings))


def clear_vector_store_cache() -> None:
    _build_cached_vector_store.clear()


@st.cache_resource
def _build_cached_chat_model(
    *,
    api_key: str,
    model_name: str,
    temperature: float,
) -> ChatOpenAI:
    return ChatOpenAI(
        api_key=api_key,
        model=model_name,
        temperature=temperature,
        max_retries=3,
        request_timeout=30,
    )


def get_initial_chat_model_selection(
    settings: Settings,
    selected_model: object,
) -> str:
    if isinstance(selected_model, str) and selected_model in settings.supported_chat_models:
        return selected_model
    return settings.default_chat_model


def validate_selected_chat_model(
    selected_model: object,
    settings: Settings,
) -> str:
    if not isinstance(selected_model, str) or not selected_model.strip():
        raise AppValidationError("Select a supported chat model before sending a request.")

    try:
        return settings.ensure_supported_chat_model(selected_model)
    except ValueError as exc:
        raise AppValidationError(str(exc)) from exc


def build_chat_model_cache_inputs(
    settings: Settings,
    selected_model: str,
) -> dict[str, str | float]:
    return {
        "api_key": settings.ensure_openai_api_key(),
        "model_name": settings.ensure_supported_chat_model(selected_model),
        "temperature": 0,
    }


def get_chat_model(settings: Settings, selected_model: str) -> ChatOpenAI:
    return _build_cached_chat_model(
        **build_chat_model_cache_inputs(settings, selected_model)
    )


def validate_query(raw_query: str, *, max_length: int) -> str:
    cleaned = raw_query.strip()
    if not cleaned:
        raise AppValidationError("Enter a question before sending a request.")
    if len(cleaned) > max_length:
        raise AppValidationError(
            f"Questions must be {max_length} characters or fewer."
        )
    return cleaned


def build_safe_log_metadata(query: str) -> dict[str, object]:
    preview = query[:PREVIEW_LENGTH].replace("\n", " ").strip()
    return {
        "query_length": len(query),
        "query_preview": preview,
    }


def get_user_facing_error_message(exc: Exception) -> str:
    message = str(exc)
    lowered_message = message.lower()
    if isinstance(exc, AppValidationError):
        return message
    if "OPENAI_API_KEY" in message:
        return "OpenAI is not configured yet. Add OPENAI_API_KEY and try again."
    if "Chroma vector store is unavailable" in message or "Chroma vector store is empty" in message:
        return "The local knowledge base is not ready. Build the Chroma index before asking questions."
    if "unsupported chat model" in lowered_message:
        return message
    if (
        "model_not_found" in lowered_message
        or "does not have access to the model" in lowered_message
        or "do not have access to the model" in lowered_message
        or ("model" in lowered_message and "does not exist" in lowered_message)
    ):
        return "The selected OpenAI model is unavailable for this API key. Choose a different model and try again."
    if "Connection error" in message:
        return "The AI backend could not be reached. Please try again in a moment."
    return "Something went wrong while processing your request. Please try again."


def build_turn_record(query: str, result: AnswerResult) -> dict[str, object]:
    return {
        "query": query,
        "answer": result.answer,
        "used_context": result.used_context,
        "sources": result.answer_sources,
        "tool_result": (
            result.tool_result.model_dump()
            if result.tool_result is not None
            else None
        ),
        "official_docs_result": (
            result.official_docs_result.model_dump()
            if result.official_docs_result is not None
            else None
        ),
        "usage": result.usage.model_dump() if result.usage is not None else None,
    }


def should_skip_resource_loading(query: str) -> bool:
    return maybe_invoke_tool(query) is not None


def should_stream_grounded_query(query: str) -> bool:
    return maybe_match_official_docs_query(query) is None


def run_non_streaming_query(
    *,
    query: str,
    vector_store,
    chat_model,
) -> AnswerResult:
    return run_backend_query(
        query=query,
        vector_store=vector_store,
        chat_model=chat_model,
    )


def run_tool_query(query: str) -> AnswerResult:
    return run_backend_query(
        query=query,
        vector_store=None,
        chat_model=None,
    )


def run_streaming_grounded_query(
    *,
    query: str,
    vector_store,
    chat_model,
    on_token: Callable[[str], None],
) -> AnswerResult:
    return stream_answer_query(
        query=query,
        vector_store=vector_store,
        chat_model=chat_model,
        on_token=on_token,
    )
