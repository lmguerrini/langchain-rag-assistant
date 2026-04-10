from __future__ import annotations

from typing import Protocol

from src.retrieval import retrieve_chunks
from src.schemas import AnswerResult, RetrievalRequest, RetrievalResult


NO_CONTEXT_FALLBACK = (
    "I could not find enough relevant context in the knowledge base to answer "
    "that safely."
)


class ChatModelLike(Protocol):
    def invoke(self, prompt: str):
        ...


def answer_query(
    *,
    query: str,
    vector_store,
    chat_model: ChatModelLike,
    top_k: int = 3,
) -> AnswerResult:
    request = RetrievalRequest(query=query, top_k=top_k)
    retrieval_result = retrieve_chunks(
        vector_store=vector_store,
        request=request,
    )

    if not retrieval_result.chunks:
        return AnswerResult(
            answer=NO_CONTEXT_FALLBACK,
            used_context=False,
            retrieval=retrieval_result,
            answer_sources=[],
        )

    prompt = build_grounded_prompt(
        original_query=request.query,
        retrieval=retrieval_result,
    )
    model_response = chat_model.invoke(prompt)
    answer_text = _extract_text(model_response)

    return AnswerResult(
        answer=answer_text,
        used_context=True,
        retrieval=retrieval_result,
        answer_sources=retrieval_result.sources,
    )


def build_grounded_prompt(*, original_query: str, retrieval: RetrievalResult) -> str:
    context_blocks = []
    for index, chunk in enumerate(retrieval.chunks, start=1):
        context_blocks.append(f"[Chunk {index}]\n{chunk.content}")

    source_lines = "\n".join(f"- {source}" for source in retrieval.sources)
    context_text = "\n\n".join(context_blocks)

    return (
        "Answer the question using only the provided context.\n"
        "If the context is insufficient, say so plainly.\n\n"
        f"User query: {original_query}\n"
        f"Retrieval query: {retrieval.rewritten_query}\n\n"
        f"Context:\n{context_text}\n\n"
        f"Sources:\n{source_lines}"
    )


def _extract_text(model_response) -> str:
    content = getattr(model_response, "content", model_response)
    if isinstance(content, str):
        return content.strip()
    return str(content).strip()
