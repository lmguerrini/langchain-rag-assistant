---
title: LangChain Retrieval Debugging Playbook
topic: langchain
library: langchain
doc_type: troubleshooting
difficulty: advanced
error_family: retrieval
---
# LangChain Retrieval Debugging Playbook

## When This Matters

Use this playbook when a grounded answer is missing, weak, or clearly based on the wrong chunks. In this project, the local retrieval path is simple and explicit: `run_backend_query(...)` falls through to `answer_query(...)`, which calls `retrieve_chunks(...)` from `src/retrieval.py`. If retrieval returns no usable chunks, the app returns `NO_CONTEXT_FALLBACK` instead of forcing an answer.

This is the right document when you are debugging:

- empty `sources` in a normal knowledge-base path
- a question that should match `data/raw/` content but falls back anyway
- over-constrained filtered retrieval
- weak chunk quality after the first similarity search result
- confusion about `top_k`, chunk size, overlap, or metadata filters

## Recommended Pattern

Start by checking the retrieval path in this order:

1. Validate the query shape.
   `RetrievalRequest` requires a non-empty query and `top_k` between `1` and `10`.
2. Inspect the rewritten query.
   `rewrite_query(...)` lowercases, removes noise tokens, and deduplicates terms. For example, tests show `How do I persist Chroma data in the local database?` becomes `do persist chroma data local database`.
3. Inspect inferred metadata filters.
   `infer_metadata_filters(...)` only uses strong signals for `topic`, `library`, `doc_type`, and `error_family`. Weak phrasing should produce no filters.
4. Check whether filtered retrieval ran.
   If `filters.as_chroma_filter()` is non-empty, `retrieve_chunks(...)` tries filtered `similarity_search(...)` first.
5. Check fallback behavior.
   If the filtered query returns no documents, the system retries without filters and marks `used_fallback=True`.
6. Check usability filtering after retrieval.
   Retrieved documents still have to pass `_filter_usable_documents(...)`. The code tokenizes the rewritten query and each document, then requires lexical overlap:
   at least `1` overlapping token for very short rewritten queries, otherwise at least `2`.

This last step is important. A similarity result can exist in Chroma and still be rejected as unusable context. That is how off-domain questions like `What is the capital of France?` correctly produce zero chunks and zero sources.

## Common Failure Modes

- The query is in-domain, but wording is too weak.
  If the query does not contain strong signals like `chroma`, `streamlit`, `error`, `persist`, or `build`, the inferred filters may stay empty. That is fine, but retrieval then depends entirely on semantic similarity and the later lexical-overlap gate.

- The filters are correct, but too narrow.
  A query like `Show a Streamlit persistence debugging example` can infer `topic=streamlit` and `error_family=persistence`. If no document matches both, the system falls back. That is expected behavior, not a bug.

- Chunks exist, but overlap filtering removes them.
  This usually happens when the rewritten query is too generic or the chunk text is too indirect. Check whether the source markdown contains the exact operational terms the user is likely to ask for.

- Chunk size is hiding specific troubleshooting details.
  Current defaults come from `Settings`: `CHUNK_SIZE=800` and `CHUNK_OVERLAP=120`. That is a reasonable starting point for compact internal docs, but smaller troubleshooting notes sometimes work better with tighter, more explicit sections.

- The source formatting looks fine, but the answer is still weak.
  `format_sources(...)` only describes what was retrieved. It does not mean the underlying chunk text was strong enough to support a detailed answer.

## Implementation Notes

The fastest concrete checks are:

- confirm the markdown file exists directly under `data/raw/`
- confirm the frontmatter uses valid `topic`, `library`, `doc_type`, `difficulty`, and optional `error_family`
- confirm the body contains exact project terms such as `conversation_history`, `python build_index.py`, `official docs`, `Chroma vector store`, or `selected_chat_model`
- rebuild the index after corpus changes so Chroma and `kb_manifest.json` reflect the new files

If you are changing corpus content rather than retrieval code, prefer improving document wording before changing retrieval rules. This project is intentionally using a fixed, reviewable retrieval flow. Small, high-signal markdown docs usually help more than adding retrieval complexity.

Useful code anchors:

- `src/retrieval.py`
- `src/chains.py`
- `src/knowledge_base.py`
- `tests/test_retrieval.py`

## Retrieval Hints

- `Why did retrieve_chunks return no usable chunks for an in-domain query?`
- `How does rewrite_query work in src/retrieval.py?`
- `When does LangChain retrieval use filtered similarity search and fallback search?`
- `Why did the app return NO_CONTEXT_FALLBACK even though Chroma returned documents?`
- `How should I debug weak grounded answers in this project?`
- `What top_k and chunking tradeoffs matter for small internal markdown docs?`
- `How do metadata filters topic library doc_type and error_family affect retrieval?`
