---
title: Official Docs Query Routing and Failures
topic: tool_calling
library: general
doc_type: troubleshooting
difficulty: advanced
---
# Official Docs Query Routing and Failures

## When This Matters

Use this document when a question should have gone to official documentation but did not, when a docs request unexpectedly used the local knowledge base, or when the official-docs path returned an unavailable message or a hard failure.

The current backend route order in `src/chains.py` is fixed:

1. built-in deterministic tools through `maybe_invoke_tool(...)`
2. official documentation route through `maybe_match_official_docs_query(...)`
3. local knowledge-base RAG through `answer_query(...)`

That order matters. Once a request matches an earlier route, later routes do not run.

## Recommended Pattern

Choose the path by intent, not by guesswork.

Use built-in tools when the question is really asking for a calculation, a rule-based diagnosis, or a retrieval-config recommendation. The current tools are:

- `estimate_openai_cost`
- `diagnose_stack_error`
- `recommend_retrieval_config`

Use the official-docs path when the user explicitly wants documentation or API-reference help for one supported library. `maybe_match_official_docs_query(...)` currently requires docs intent such as:

- `docs`
- `documentation`
- `api reference`

Supported explicit library matches are:

- `langchain`
- `openai`
- `streamlit`
- `chroma`

If the user clearly asks for docs but does not name a library, the route defaults to `openai`. If the user mentions more than one supported library in the same docs query, official-docs routing is skipped and the request falls through to normal KB behavior.

Use the local knowledge base when the question is about this project’s own implementation patterns, state keys, file names, rebuild behavior, or grounded debugging steps.

## Common Failure Modes

- A cost or diagnosis question never reaches official docs.
  That is expected. Tool routing runs first and wins when it matches cleanly.

- A docs-like query still goes to the local KB.
  This usually means there was no explicit docs intent term, or the query mentioned multiple supported libraries and the official-docs matcher intentionally refused the route.

- Official docs lookup returns `Official documentation lookup is not available for this library yet.`
  This is a controlled result from `answer_official_docs_query(...)`, not a local KB fallback. It is used when the lookup stage hits MCP-unavailable conditions such as transport failure, SSL issues, timeout, or explicit remote unavailability.

- Official docs lookup raises a hard error instead of returning the unavailable message.
  That usually means the transport worked, but the provider payload was malformed, empty, or logically unsupported. Those are treated as real lookup failures and are wrapped as `Official docs lookup failed: ...`.

- The query asked for Streamlit or Chroma docs, but the result does not come from a remote MCP endpoint.
  That is by design. `src/official_docs_sources.py` routes `langchain` and `openai` through MCP adapters, while `streamlit` and `chroma` use local fallback adapters.

- The app did not fall back to KB RAG after official-docs failure.
  That is also by design. Once the official-docs branch is selected, the backend does not silently switch to KB RAG.

## Implementation Notes

The official-docs path is handled by `answer_official_docs_query(...)` in `src/official_docs_service.py`. It has two stages:

1. lookup through `lookup_official_docs_documents(...)`
2. summary generation through `summarize_official_docs_answer(...)`

Failure handling is stage-aware:

- lookup failures become `Official docs lookup failed: ...`
- summary failures become `Official docs summary failed: ...`
- MCP-unavailable lookup failures become a controlled `OfficialDocsAnswerResult` with empty documents and `usage=None`

In the UI, these answers are not shown as normal grounded answers. The app stores `official_docs_result` in the turn record, labels the response type as `Official docs answer`, and renders a dedicated `Official Docs Result` expander.

Use official docs when you need library behavior or reference material. Use the local KB when you need project-specific guidance such as `conversation_history`, `python build_index.py`, `kb_manifest.json`, `selected_chat_model`, or `request_timestamps`.

## Retrieval Hints

- `Why did my docs question not route to official docs?`
- `What is the route order between tools official docs and local KB in src/chains.py?`
- `When does maybe_match_official_docs_query default to openai?`
- `Why does a multi-library docs question skip the official docs path?`
- `What failures return Official documentation lookup is not available for this library yet?`
- `When should I ask for official docs instead of local project guidance?`
- `Why does the backend not fall back to answer_query after official docs routing is selected?`
