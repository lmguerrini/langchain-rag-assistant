# Evaluation Workflow

This project uses a custom evaluation workflow for the current RAG system behavior. It does not use RAGAs.

## What It Covers

- retrieval-side source recall against expected source titles
- whether the answer used context when the case expected context
- answer-side keyword recall against expected keywords
- no-context fallback behavior for out-of-scope or unsupported questions
- whether grounded answers include sources

## What It Does Not Cover

- RAGAs metrics
- dashboard or analytics reporting
- automated regression triage
- broad UI or tool-path evaluation outside the current backend answer flow

## Prerequisites

- the project dependencies are installed
- `OPENAI_API_KEY` is set if you want to run the live backend evaluation path
- the local Chroma index has been built if the evaluation cases depend on the KB

## Cases

The default evaluation cases live in:

- `data/eval/eval_cases.json`

Each case includes:

- `question`
- `expected_source_titles`
- `expected_keywords`
- `expect_context`

## How To Run

Run the default evaluation dataset:

```bash
.venv/bin/python -m src.evaluation
```

Run evaluation with a custom cases file:

```bash
.venv/bin/python -m src.evaluation --cases path/to/custom_eval_cases.json
```

## Output

The CLI prints:

- one block per case with retrieval and answer metrics
- one aggregate summary at the end

Main metrics:

- `source_recall`: how many expected source titles were retrieved
- `retrieved_chunks`: how many chunks were returned for the case
- `used_fallback`: whether filtered retrieval fell back to plain similarity search
- `context_match`: whether `used_context` matched `expect_context`
- `keyword_recall`: how many expected keywords appeared in the answer
- `correct_no_context_fallback`: whether the app correctly returned the fallback behavior for no-context cases
- `sources_present_when_context_used`: whether grounded answers included sources

Non-applicable per-case metrics are shown as `n/a`.
