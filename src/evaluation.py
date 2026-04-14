from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Callable

from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from pydantic import BaseModel, ValidationError, field_validator

from src.chains import NO_CONTEXT_FALLBACK
from src.chains import run_backend_query
from src.config import get_settings
from src.schemas import AnswerResult


DEFAULT_EVAL_CASES_PATH = Path("data/eval/eval_cases.json")


class EvalCase(BaseModel):
    question: str
    expected_source_titles: list[str]
    expected_keywords: list[str]
    expect_context: bool

    @field_validator("question")
    @classmethod
    def validate_question(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("Evaluation case question must not be empty.")
        return cleaned


class RetrievalEvaluationResult(BaseModel):
    matched_source_titles: list[str]
    source_recall: float
    retrieved_chunk_count: int
    used_fallback: bool


class AnswerEvaluationResult(BaseModel):
    context_was_used: bool
    used_context_matches_expectation: bool
    keyword_recall: float
    correct_no_context_fallback: bool | None
    sources_present_when_context_used: bool | None


class EvaluationCaseResult(BaseModel):
    question: str
    retrieval: RetrievalEvaluationResult
    answer: AnswerEvaluationResult


class EvaluationSummary(BaseModel):
    case_count: int
    average_source_recall: float
    average_keyword_recall: float
    context_match_rate: float
    no_context_fallback_rate: float
    sources_present_rate_when_context_used: float


class EvaluationReport(BaseModel):
    cases: list[EvaluationCaseResult]
    summary: EvaluationSummary


def load_eval_cases(path: Path = DEFAULT_EVAL_CASES_PATH) -> list[EvalCase]:
    try:
        raw_data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"Evaluation dataset not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Evaluation dataset is not valid JSON: {path}") from exc

    if not isinstance(raw_data, list):
        raise ValueError("Evaluation dataset must be a JSON list of cases.")

    cases: list[EvalCase] = []
    for index, item in enumerate(raw_data):
        try:
            cases.append(EvalCase.model_validate(item))
        except ValidationError as exc:
            raise ValueError(
                f"Invalid evaluation case at index {index}: {exc}"
            ) from exc
    return cases


def evaluate_retrieval_quality(
    case: EvalCase,
    answer_result: AnswerResult,
) -> RetrievalEvaluationResult:
    retrieved_titles = _extract_retrieved_titles(answer_result)
    matched_titles = [
        title for title in case.expected_source_titles if title in retrieved_titles
    ]
    expected_count = len(case.expected_source_titles)
    source_recall = (
        len(matched_titles) / expected_count
        if expected_count > 0
        else 1.0
    )

    retrieval = answer_result.retrieval
    return RetrievalEvaluationResult(
        matched_source_titles=matched_titles,
        source_recall=round(source_recall, 4),
        retrieved_chunk_count=(len(retrieval.chunks) if retrieval is not None else 0),
        used_fallback=(retrieval.used_fallback if retrieval is not None else False),
    )


def evaluate_answer_quality(
    case: EvalCase,
    answer_result: AnswerResult,
) -> AnswerEvaluationResult:
    normalized_answer = answer_result.answer.lower()
    expected_keywords = [keyword.lower() for keyword in case.expected_keywords]
    matched_keywords = [
        keyword for keyword in expected_keywords if keyword in normalized_answer
    ]
    keyword_recall = (
        len(matched_keywords) / len(expected_keywords)
        if expected_keywords
        else 1.0
    )

    correct_no_context_fallback: bool | None = None
    if not case.expect_context:
        correct_no_context_fallback = (
            not answer_result.used_context
            and answer_result.answer == NO_CONTEXT_FALLBACK
            and not answer_result.answer_sources
        )

    sources_present_when_context_used: bool | None = None
    if answer_result.used_context:
        sources_present_when_context_used = (
        bool(answer_result.answer_sources)
        )

    return AnswerEvaluationResult(
        context_was_used=answer_result.used_context,
        used_context_matches_expectation=(
            answer_result.used_context == case.expect_context
        ),
        keyword_recall=round(keyword_recall, 4),
        correct_no_context_fallback=correct_no_context_fallback,
        sources_present_when_context_used=sources_present_when_context_used,
    )


def evaluate_case(
    case: EvalCase,
    answer_result: AnswerResult,
) -> EvaluationCaseResult:
    return EvaluationCaseResult(
        question=case.question,
        retrieval=evaluate_retrieval_quality(case, answer_result),
        answer=evaluate_answer_quality(case, answer_result),
    )


def summarize_results(results: list[EvaluationCaseResult]) -> EvaluationSummary:
    if not results:
        return EvaluationSummary(
            case_count=0,
            average_source_recall=0.0,
            average_keyword_recall=0.0,
            context_match_rate=0.0,
            no_context_fallback_rate=0.0,
            sources_present_rate_when_context_used=0.0,
        )

    context_used_results = [
        result
        for result in results
        if result.answer.sources_present_when_context_used is not None
    ]
    no_context_results = [
        result
        for result in results
        if result.answer.correct_no_context_fallback is not None
    ]

    return EvaluationSummary(
        case_count=len(results),
        average_source_recall=round(
            sum(result.retrieval.source_recall for result in results) / len(results),
            4,
        ),
        average_keyword_recall=round(
            sum(result.answer.keyword_recall for result in results) / len(results),
            4,
        ),
        context_match_rate=round(
            sum(
                1 for result in results
                if result.answer.used_context_matches_expectation
            ) / len(results),
            4,
        ),
        no_context_fallback_rate=round(
            (
                sum(
                    1 for result in no_context_results
                    if result.answer.correct_no_context_fallback
                ) / len(no_context_results)
            )
            if no_context_results
            else 0.0,
            4,
        ),
        sources_present_rate_when_context_used=round(
            (
                sum(
                    1 for result in context_used_results
                    if result.answer.sources_present_when_context_used
                ) / len(context_used_results)
            )
            if context_used_results
            else 0.0,
            4,
        ),
    )


def run_evaluation(
    *,
    answer_fn: Callable[[str], AnswerResult],
    cases: list[EvalCase] | None = None,
) -> EvaluationReport:
    resolved_cases = cases or load_eval_cases()
    results = [
        evaluate_case(case, answer_fn(case.question))
        for case in resolved_cases
    ]
    return EvaluationReport(
        cases=results,
        summary=summarize_results(results),
    )


def format_evaluation_report(report: EvaluationReport) -> str:
    lines = ["Evaluation Report", ""]

    for index, case_result in enumerate(report.cases, start=1):
        lines.extend(
            [
                f"Case {index}: {case_result.question}",
                (
                    "  Retrieval: "
                    f"source_recall={case_result.retrieval.source_recall:.4f}, "
                    f"matched_titles={case_result.retrieval.matched_source_titles}, "
                    f"retrieved_chunks={case_result.retrieval.retrieved_chunk_count}, "
                    f"used_fallback={case_result.retrieval.used_fallback}"
                ),
                (
                    "  Answer: "
                    f"context_match={case_result.answer.used_context_matches_expectation}, "
                    f"keyword_recall={case_result.answer.keyword_recall:.4f}, "
                    "correct_no_context_fallback="
                    f"{_format_optional_metric(case_result.answer.correct_no_context_fallback)}, "
                    "sources_present="
                    f"{_format_optional_metric(case_result.answer.sources_present_when_context_used)}"
                ),
                "",
            ]
        )

    summary = report.summary
    lines.extend(
        [
            "Summary",
            f"  case_count={summary.case_count}",
            f"  average_source_recall={summary.average_source_recall:.4f}",
            f"  average_keyword_recall={summary.average_keyword_recall:.4f}",
            f"  context_match_rate={summary.context_match_rate:.4f}",
            f"  no_context_fallback_rate={summary.no_context_fallback_rate:.4f}",
            (
                "  sources_present_rate_when_context_used="
                f"{summary.sources_present_rate_when_context_used:.4f}"
            ),
        ]
    )
    return "\n".join(lines)


def parse_cli_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the custom evaluation workflow for the local RAG system."
    )
    parser.add_argument(
        "--cases",
        type=Path,
        default=DEFAULT_EVAL_CASES_PATH,
        help=(
            "Path to the evaluation cases JSON file. "
            f"Defaults to {DEFAULT_EVAL_CASES_PATH}."
        ),
    )
    return parser.parse_args(argv if argv is not None else [])


def main(argv: list[str] | None = None) -> None:
    args = parse_cli_args(argv)
    try:
        report = run_runtime_evaluation(cases_path=args.cases)
    except Exception as exc:
        print(f"Evaluation failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    print(format_evaluation_report(report), flush=True)


def _extract_retrieved_titles(answer_result: AnswerResult) -> list[str]:
    retrieval = answer_result.retrieval
    if retrieval is None:
        return []

    titles: list[str] = []
    for chunk in retrieval.chunks:
        if chunk.metadata.title not in titles:
            titles.append(chunk.metadata.title)
    return titles


def _format_optional_metric(value: bool | None) -> str:
    if value is None:
        return "n/a"
    return str(value)


def run_runtime_evaluation(
    *,
    cases_path: Path = DEFAULT_EVAL_CASES_PATH,
) -> EvaluationReport:
    answer_fn = _build_runtime_answer_fn()
    cases = load_eval_cases(cases_path)
    return run_evaluation(answer_fn=answer_fn, cases=cases)


def _build_runtime_answer_fn() -> Callable[[str], AnswerResult]:
    settings = get_settings()
    api_key = settings.ensure_openai_api_key()
    embeddings = OpenAIEmbeddings(
        api_key=api_key,
        model=settings.embedding_model,
    )
    vector_store = Chroma(
        collection_name=settings.chroma_collection_name,
        embedding_function=embeddings,
        persist_directory=str(settings.chroma_persist_dir),
    )
    chat_model = ChatOpenAI(
        api_key=api_key,
        model="gpt-4.1-mini",
        temperature=0,
    )

    def answer_fn(question: str) -> AnswerResult:
        return run_backend_query(
            query=question,
            vector_store=vector_store,
            chat_model=chat_model,
        )

    return answer_fn


if __name__ == "__main__":
    main(sys.argv[1:])
