import json
from pathlib import Path

import pytest

from src.chains import NO_CONTEXT_FALLBACK
from src.evaluation import (
    AnswerEvaluationResult,
    DEFAULT_EVAL_CASES_PATH,
    EvalCase,
    EvaluationCaseResult,
    EvaluationReport,
    EvaluationSummary,
    RetrievalEvaluationResult,
    evaluate_answer_quality,
    evaluate_retrieval_quality,
    format_evaluation_report,
    load_eval_cases,
    parse_cli_args,
    main,
    run_evaluation,
    summarize_results,
)
from src.schemas import (
    AnswerResult,
    RetrievalFilters,
    RetrievalResult,
    RetrievedChunk,
)


def test_evaluate_retrieval_quality_calculates_source_recall() -> None:
    case = EvalCase(
        question="How should I persist the Chroma index?",
        expected_source_titles=[
            "Chroma Persistence and Reindexing Guide",
            "Streamlit Chat Patterns for RAG Interfaces",
        ],
        expected_keywords=["chroma", "persist"],
        expect_context=True,
    )
    answer_result = _make_answer_result(
        answer="Persist the Chroma collection locally.",
        used_context=True,
        source_titles=[
            "Chroma Persistence and Reindexing Guide",
            "LangChain Retrieval Patterns for Small RAG Apps",
        ],
        used_fallback=True,
    )

    metrics = evaluate_retrieval_quality(case, answer_result)

    assert metrics.matched_source_titles == ["Chroma Persistence and Reindexing Guide"]
    assert metrics.source_recall == 0.5
    assert metrics.retrieved_chunk_count == 2
    assert metrics.used_fallback is True


def test_evaluate_answer_quality_calculates_keyword_and_context_metrics() -> None:
    case = EvalCase(
        question="How should I display sources in Streamlit?",
        expected_source_titles=["Streamlit Chat Patterns for RAG Interfaces"],
        expected_keywords=["streamlit", "sources", "chat"],
        expect_context=True,
    )
    answer_result = _make_answer_result(
        answer="Use Streamlit chat elements and show sources beside the chat answer.",
        used_context=True,
        source_titles=["Streamlit Chat Patterns for RAG Interfaces"],
    )

    metrics = evaluate_answer_quality(case, answer_result)

    assert metrics.used_context_matches_expectation is True
    assert metrics.keyword_recall == 1.0
    assert metrics.correct_no_context_fallback is None
    assert metrics.sources_present_when_context_used is True


def test_evaluate_answer_quality_checks_no_context_fallback_behavior() -> None:
    case = EvalCase(
        question="How should I add rate limiting?",
        expected_source_titles=[],
        expected_keywords=[],
        expect_context=False,
    )
    answer_result = _make_answer_result(
        answer=NO_CONTEXT_FALLBACK,
        used_context=False,
        source_titles=[],
        include_retrieval=False,
    )

    metrics = evaluate_answer_quality(case, answer_result)

    assert metrics.used_context_matches_expectation is True
    assert metrics.keyword_recall == 1.0
    assert metrics.correct_no_context_fallback is True
    assert metrics.sources_present_when_context_used is None


def test_summarize_results_calculates_aggregate_metrics() -> None:
    results = [
        EvaluationCaseResult(
            question="case-1",
            retrieval=RetrievalEvaluationResult(
                matched_source_titles=["A"],
                source_recall=1.0,
                retrieved_chunk_count=1,
                used_fallback=False,
            ),
            answer=AnswerEvaluationResult(
                context_was_used=True,
                used_context_matches_expectation=True,
                keyword_recall=0.5,
                correct_no_context_fallback=None,
                sources_present_when_context_used=True,
            ),
        ),
        EvaluationCaseResult(
            question="case-2",
            retrieval=RetrievalEvaluationResult(
                matched_source_titles=[],
                source_recall=0.0,
                retrieved_chunk_count=0,
                used_fallback=True,
            ),
            answer=AnswerEvaluationResult(
                context_was_used=False,
                used_context_matches_expectation=False,
                keyword_recall=1.0,
                correct_no_context_fallback=False,
                sources_present_when_context_used=None,
            ),
        ),
    ]

    summary = summarize_results(results)

    assert summary.case_count == 2
    assert summary.average_source_recall == 0.5
    assert summary.average_keyword_recall == 0.75
    assert summary.context_match_rate == 0.5
    assert summary.no_context_fallback_rate == 0.0
    assert summary.sources_present_rate_when_context_used == 1.0


def test_summarize_results_uses_only_applicable_denominators() -> None:
    results = [
        EvaluationCaseResult(
            question="context-case",
            retrieval=RetrievalEvaluationResult(
                matched_source_titles=["A"],
                source_recall=1.0,
                retrieved_chunk_count=1,
                used_fallback=False,
            ),
            answer=AnswerEvaluationResult(
                context_was_used=True,
                used_context_matches_expectation=True,
                keyword_recall=1.0,
                correct_no_context_fallback=None,
                sources_present_when_context_used=False,
            ),
        ),
        EvaluationCaseResult(
            question="no-context-case",
            retrieval=RetrievalEvaluationResult(
                matched_source_titles=[],
                source_recall=1.0,
                retrieved_chunk_count=0,
                used_fallback=False,
            ),
            answer=AnswerEvaluationResult(
                context_was_used=False,
                used_context_matches_expectation=True,
                keyword_recall=1.0,
                correct_no_context_fallback=True,
                sources_present_when_context_used=None,
            ),
        ),
    ]

    summary = summarize_results(results)

    assert summary.no_context_fallback_rate == 1.0
    assert summary.sources_present_rate_when_context_used == 0.0


def test_run_evaluation_returns_per_case_results_and_summary() -> None:
    cases = [
        EvalCase(
            question="How should I persist Chroma locally?",
            expected_source_titles=["Chroma Persistence and Reindexing Guide"],
            expected_keywords=["chroma", "persist"],
            expect_context=True,
        ),
        EvalCase(
            question="How should I add rate limiting?",
            expected_source_titles=[],
            expected_keywords=[],
            expect_context=False,
        ),
    ]

    def fake_answer_fn(question: str) -> AnswerResult:
        if "rate limiting" in question.lower():
            return _make_answer_result(
                answer=NO_CONTEXT_FALLBACK,
                used_context=False,
                source_titles=[],
                include_retrieval=False,
            )
        return _make_answer_result(
            answer="Persist the Chroma collection locally.",
            used_context=True,
            source_titles=["Chroma Persistence and Reindexing Guide"],
        )

    report = run_evaluation(answer_fn=fake_answer_fn, cases=cases)

    assert len(report.cases) == 2
    assert report.summary.case_count == 2
    assert report.summary.context_match_rate == 1.0


def test_load_eval_cases_reads_six_cases_from_default_dataset() -> None:
    cases = load_eval_cases()

    assert len(cases) == 6
    assert cases[4].question == (
        "Why should metadata fields make filtered retrieval easier to implement?"
    )
    assert cases[5].question == "What is the capital of France?"


def test_load_eval_cases_fails_clearly_for_incomplete_case(tmp_path: Path) -> None:
    dataset_path = tmp_path / "bad_eval_cases.json"
    dataset_path.write_text(
        json.dumps(
            [
                {
                    "question": "Bad case",
                    "expected_source_titles": ["Some Title"],
                    "expect_context": True,
                }
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Invalid evaluation case at index 0"):
        load_eval_cases(dataset_path)


def test_format_evaluation_report_and_main_print_visible_output(monkeypatch, capsys) -> None:
    report = EvaluationReport(
        cases=[
            EvaluationCaseResult(
                question="How should I persist Chroma locally?",
                retrieval=RetrievalEvaluationResult(
                    matched_source_titles=["Chroma Persistence and Reindexing Guide"],
                    source_recall=1.0,
                    retrieved_chunk_count=1,
                    used_fallback=False,
                ),
                answer=AnswerEvaluationResult(
                    context_was_used=True,
                    used_context_matches_expectation=True,
                    keyword_recall=1.0,
                    correct_no_context_fallback=None,
                    sources_present_when_context_used=True,
                ),
            )
        ],
        summary=EvaluationSummary(
            case_count=1,
            average_source_recall=1.0,
            average_keyword_recall=1.0,
            context_match_rate=1.0,
            no_context_fallback_rate=1.0,
            sources_present_rate_when_context_used=1.0,
        ),
    )

    formatted = format_evaluation_report(report)
    assert "Evaluation Report" in formatted
    assert "Case 1: How should I persist Chroma locally?" in formatted
    assert "Summary" in formatted
    assert "correct_no_context_fallback=n/a" in formatted

    monkeypatch.setattr(
        "src.evaluation._build_runtime_answer_fn",
        lambda: (lambda question: _make_answer_result(
            answer="Persist the Chroma collection locally.",
            used_context=True,
            source_titles=["Chroma Persistence and Reindexing Guide"],
        )),
    )
    monkeypatch.setattr(
        "src.evaluation.load_eval_cases",
        lambda path=None: [
            EvalCase(
                question="How should I persist Chroma locally?",
                expected_source_titles=["Chroma Persistence and Reindexing Guide"],
                expected_keywords=["chroma", "persist"],
                expect_context=True,
            )
        ],
    )

    main()
    captured = capsys.readouterr()
    assert "Evaluation Report" in captured.out
    assert "Summary" in captured.out


def test_format_evaluation_report_shows_na_for_non_applicable_metrics() -> None:
    report = EvaluationReport(
        cases=[
            EvaluationCaseResult(
                question="What is the capital of France?",
                retrieval=RetrievalEvaluationResult(
                    matched_source_titles=[],
                    source_recall=1.0,
                    retrieved_chunk_count=0,
                    used_fallback=False,
                ),
                answer=AnswerEvaluationResult(
                    context_was_used=False,
                    used_context_matches_expectation=True,
                    keyword_recall=1.0,
                    correct_no_context_fallback=True,
                    sources_present_when_context_used=None,
                ),
            )
        ],
        summary=EvaluationSummary(
            case_count=1,
            average_source_recall=1.0,
            average_keyword_recall=1.0,
            context_match_rate=1.0,
            no_context_fallback_rate=1.0,
            sources_present_rate_when_context_used=0.0,
        ),
    )

    formatted = format_evaluation_report(report)

    assert "sources_present=n/a" in formatted


def test_main_prints_readable_error_when_runtime_evaluation_fails(monkeypatch, capsys) -> None:
    def fail_build_runtime_answer_fn():
        raise RuntimeError("Connection error.")

    monkeypatch.setattr(
        "src.evaluation._build_runtime_answer_fn",
        fail_build_runtime_answer_fn,
    )

    with pytest.raises(SystemExit, match="1"):
        main()

    captured = capsys.readouterr()
    assert "Evaluation failed: Connection error." in captured.err


def test_parse_cli_args_defaults_to_default_cases_path() -> None:
    args = parse_cli_args([])

    assert args.cases == DEFAULT_EVAL_CASES_PATH


def test_main_uses_custom_cases_path_from_cli(monkeypatch, tmp_path: Path, capsys) -> None:
    dataset_path = tmp_path / "custom_eval_cases.json"
    dataset_path.write_text(
        json.dumps(
            [
                {
                    "question": "How should I persist Chroma locally?",
                    "expected_source_titles": ["Chroma Persistence and Reindexing Guide"],
                    "expected_keywords": ["chroma", "persist"],
                    "expect_context": True,
                }
            ]
        ),
        encoding="utf-8",
    )

    loaded_paths: list[Path] = []

    def fake_load_eval_cases(path: Path = DEFAULT_EVAL_CASES_PATH):
        loaded_paths.append(path)
        return [
            EvalCase(
                question="How should I persist Chroma locally?",
                expected_source_titles=["Chroma Persistence and Reindexing Guide"],
                expected_keywords=["chroma", "persist"],
                expect_context=True,
            )
        ]

    monkeypatch.setattr("src.evaluation.load_eval_cases", fake_load_eval_cases)
    monkeypatch.setattr(
        "src.evaluation._build_runtime_answer_fn",
        lambda: (lambda question: _make_answer_result(
            answer="Persist the Chroma collection locally.",
            used_context=True,
            source_titles=["Chroma Persistence and Reindexing Guide"],
        )),
    )

    main(["--cases", str(dataset_path)])
    captured = capsys.readouterr()

    assert loaded_paths == [dataset_path]
    assert "Evaluation Report" in captured.out


def test_main_uses_default_cases_path_when_cli_flag_is_omitted(monkeypatch, capsys) -> None:
    loaded_paths: list[Path] = []

    def fake_load_eval_cases(path: Path = DEFAULT_EVAL_CASES_PATH):
        loaded_paths.append(path)
        return [
            EvalCase(
                question="How should I persist Chroma locally?",
                expected_source_titles=["Chroma Persistence and Reindexing Guide"],
                expected_keywords=["chroma", "persist"],
                expect_context=True,
            )
        ]

    monkeypatch.setattr("src.evaluation.load_eval_cases", fake_load_eval_cases)
    monkeypatch.setattr(
        "src.evaluation._build_runtime_answer_fn",
        lambda: (lambda question: _make_answer_result(
            answer="Persist the Chroma collection locally.",
            used_context=True,
            source_titles=["Chroma Persistence and Reindexing Guide"],
        )),
    )

    main([])
    captured = capsys.readouterr()

    assert loaded_paths == [DEFAULT_EVAL_CASES_PATH]
    assert "Summary" in captured.out


def test_main_keeps_readable_failure_behavior_with_custom_cases_arg(
    monkeypatch,
    tmp_path: Path,
    capsys,
) -> None:
    dataset_path = tmp_path / "custom_eval_cases.json"
    dataset_path.write_text("[]", encoding="utf-8")

    def fail_build_runtime_answer_fn():
        raise RuntimeError("Connection error.")

    monkeypatch.setattr(
        "src.evaluation._build_runtime_answer_fn",
        fail_build_runtime_answer_fn,
    )

    with pytest.raises(SystemExit, match="1"):
        main(["--cases", str(dataset_path)])

    captured = capsys.readouterr()
    assert "Evaluation failed: Connection error." in captured.err


def _make_answer_result(
    *,
    answer: str,
    used_context: bool,
    source_titles: list[str],
    used_fallback: bool = False,
    include_retrieval: bool = True,
) -> AnswerResult:
    chunks = [
        RetrievedChunk.model_validate(
            {
                "content": f"Context from {title}",
                "metadata": {
                    "doc_id": title.lower().replace(" ", "-"),
                    "source_path": f"data/raw/{index}.md",
                    "title": title,
                    "topic": "chroma" if "Chroma" in title else "streamlit",
                    "library": "chroma" if "Chroma" in title else "streamlit",
                    "doc_type": "how_to" if "Chroma" in title else "example",
                    "difficulty": "intro" if "Chroma" in title else "intermediate",
                    "error_family": "persistence" if "Chroma" in title else "ui",
                    "chunk_index": index,
                },
            }
        )
        for index, title in enumerate(source_titles)
    ]
    retrieval = None
    if include_retrieval:
        retrieval = RetrievalResult(
            rewritten_query="rewritten query",
            applied_filters=RetrievalFilters(),
            used_fallback=used_fallback,
            chunks=chunks,
            sources=[f"{title} | source=data/raw/{index}.md" for index, title in enumerate(source_titles)],
        )

    return AnswerResult(
        answer=answer,
        used_context=used_context,
        retrieval=retrieval,
        answer_sources=(retrieval.sources if retrieval is not None else []),
        tool_result=None,
    )
