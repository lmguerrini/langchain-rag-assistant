---
title: Chroma Rebuild and Knowledge Base Troubleshooting
topic: chroma
library: chroma
doc_type: troubleshooting
difficulty: intermediate
error_family: persistence
---
# Chroma Rebuild and Knowledge Base Troubleshooting

## When This Matters

Use this document when the local knowledge base looks stale, the sidebar shows `Missing` or `Outdated`, new markdown files are not showing up in grounded answers, or the app reports that the Chroma vector store is unavailable or empty.

This project uses a persistent local Chroma index at `data/chroma_db/` by default, with raw markdown source files in `data/raw/`. The app does not rebuild the index automatically in the background. Freshness is tracked explicitly.

## Recommended Pattern

Think about the knowledge base as two artifacts that must stay aligned:

- the raw markdown corpus in `data/raw/`
- the built Chroma index plus `kb_manifest.json` in `data/chroma_db/`

The normal rebuild path is:

1. `rebuild_knowledge_base(settings)` in `build_index.py`
2. `build_index(settings=settings)` in `src/knowledge_base.py`
3. `write_kb_manifest(...)` in `src/kb_status.py`

The manifest records:

- `built_at`
- `collection_name`
- `indexed_chunk_count`
- `raw_file_count`
- `source_fingerprint`

`get_kb_status(settings)` compares the current raw snapshot to that manifest and returns one of three states:

- `missing`
- `up_to_date`
- `outdated`

If the sidebar shows `Missing` or `Outdated`, rebuild before debugging retrieval quality. Otherwise you may be inspecting stale data.

## Common Failure Modes

- The app says the local knowledge base is not ready.
  This usually comes from `retrieve_chunks(...)` calling `_validate_vector_store(...)`, which raises when Chroma cannot be read or contains no IDs.

- New markdown files exist, but answers still use old content.
  The index was not rebuilt after `data/raw/` changed. `get_kb_status(...)` detects this through raw file count or source fingerprint mismatch.

- The sidebar says `Outdated` even though the persist directory exists.
  A partial persist directory is not enough. The app also expects a valid manifest and a matching collection name.

- Duplicate or stale results appear after repeated builds.
  The current build path is intentionally conservative. `build_index(...)` deletes the existing collection before adding new chunks when `reset_collection=True`. If you bypass that path, duplicates become much more likely.

- The persist directory exists, but the wrong collection is loaded.
  `kb_manifest.json` stores the expected `collection_name`. If it does not match `settings.chroma_collection_name`, the KB is marked outdated.

## Implementation Notes

The markdown loader is intentionally strict:

- `data/raw/` must exist
- only `.md` files are supported
- every file must start with YAML-style frontmatter
- frontmatter must include valid `title`, `topic`, `library`, `doc_type`, and `difficulty`
- files with empty bodies are rejected

Chunking is configured through `Settings`:

- `CHUNK_SIZE=800`
- `CHUNK_OVERLAP=120`

During a successful in-app rebuild, the app clears only the cached vector store resource through `_build_cached_vector_store.clear()`. It does not clear:

- the cached chat model
- conversation history
- export cache
- rate-limit timestamps

That cache boundary is deliberate. Rebuilding the index should refresh retrieval data without wiping the rest of the session.

Manual rebuild is still supported through the same backend path. The sidebar displays `python build_index.py` as the rebuild command. The Streamlit button uses the same rebuild logic and then reruns the app.

## Retrieval Hints

- `Why does the sidebar say Knowledge base is outdated?`
- `How does kb_manifest.json decide whether the Chroma index is fresh?`
- `What does rebuild_knowledge_base do in build_index.py?`
- `Why is the Chroma vector store empty or unavailable in this app?`
- `How do I rebuild the local knowledge base after changing data/raw markdown files?`
- `What cache is cleared after an in-app knowledge base rebuild?`
- `How does this project avoid duplicate chunks when rebuilding Chroma?`
