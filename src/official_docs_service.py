from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol

from src.official_docs_sources import (
    OfficialDocsSourceAdapter,
    lookup_official_docs_documents,
)
from src.official_docs_summary import (
    OfficialDocsSummaryImplementation,
    summarize_official_docs_answer,
)
from src.schemas import (
    OfficialDocsAnswerResult,
    OfficialDocsLookupRequest,
    OfficialDocsLookupResult,
)

OFFICIAL_DOCS_LOOKUP_UNAVAILABLE_ANSWER = (
    "Official documentation lookup is not available for this library yet."
)


class OfficialDocsLookupImplementation(Protocol):
    def __call__(
        self,
        *,
        request: OfficialDocsLookupRequest,
        adapters: Mapping[str, OfficialDocsSourceAdapter] | None = None,
    ) -> OfficialDocsLookupResult:
        ...


def answer_official_docs_query(
    *,
    request: OfficialDocsLookupRequest,
    chat_model: object | None = None,
    adapters: Mapping[str, OfficialDocsSourceAdapter] | None = None,
    lookup_impl: OfficialDocsLookupImplementation = lookup_official_docs_documents,
    summary_impl: OfficialDocsSummaryImplementation | None = None,
) -> OfficialDocsAnswerResult:
    try:
        lookup_result = lookup_impl(
            request=request,
            adapters=adapters,
        )
    except Exception as exc:
        if _is_mcp_unavailable_error(exc):
            return OfficialDocsAnswerResult(
                library=request.library,
                answer=OFFICIAL_DOCS_LOOKUP_UNAVAILABLE_ANSWER,
                lookup_result=OfficialDocsLookupResult(
                    library=request.library,
                    documents=[],
                ),
                usage=None,
            )
        raise RuntimeError(f"Official docs lookup failed: {exc}") from exc

    try:
        return summarize_official_docs_answer(
            request=request,
            lookup_result=lookup_result,
            chat_model=chat_model,
            summary_impl=summary_impl,
        )
    except Exception as exc:
        raise RuntimeError(f"Official docs summary failed: {exc}") from exc


def _is_mcp_unavailable_error(exc: Exception) -> bool:
    if isinstance(exc, NotImplementedError):
        return True
    if isinstance(exc, RuntimeError):
        message = str(exc).lower()
        if "remote mcp not available" in message:
            return True
        if "official docs mcp request failed:" in message:
            return True
        transport_markers = (
            "certificate verify failed",
            "ssl",
            "tls",
            "timeout",
            "timed out",
            "connection",
            "urlopen error",
            "network is unreachable",
            "name resolution",
        )
        if any(marker in message for marker in transport_markers):
            return True
    return False
