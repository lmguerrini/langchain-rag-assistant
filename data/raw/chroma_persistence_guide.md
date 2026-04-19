---
title: Chroma Persistence and Reindexing Guide
topic: chroma
library: chroma
doc_type: how_to
difficulty: intermediate
error_family: persistence
---
# Chroma Persistence and Reindexing Guide

## When This Matters

Use this document when you need the intended persistence model for the local Chroma index rather than a failure checklist. This file explains where the index lives, how rebuilds are supposed to work, what the manifest means, and how to think about persistence separately from freshness.

This is the right reference when you are asking:

- where the project stores vector data
- what a normal rebuild does to an existing collection
- how `kb_manifest.json` fits into the build process
- what the app considers a valid rebuilt state
- why persistence alone is not enough to prove the knowledge base is current

If the main issue is a stale or broken build that already failed, the companion document `chroma_rebuild_and_kb_troubleshooting.md` is more direct.

## Recommended Pattern

Treat the local knowledge base as a persisted Chroma collection plus a build manifest.

The main persistence path is:

- raw docs in `data/raw/`
- Chroma index in `data/chroma_db/`
- manifest in `data/chroma_db/kb_manifest.json`

The normal rebuild sequence is explicit:

1. Load markdown files from `data/raw/`.
2. Parse YAML-style frontmatter into `DocumentMetadata`.
3. Split documents into chunks with `RecursiveCharacterTextSplitter`.
4. Recreate the Chroma collection.
5. Add all chunks with stable IDs such as `doc-id::chunk::0`.
6. Write `kb_manifest.json` with the current raw snapshot fingerprint.

In code, that path runs through:

- `build_index(settings=settings)` in `src/knowledge_base.py`
- `rebuild_knowledge_base(settings)` in `build_index.py`
- `write_kb_manifest(...)` in `src/kb_status.py`

This pattern is intentionally conservative. It favors a clean, deterministic local rebuild over incremental complexity.

## Common Failure Modes

- Assuming persistence means freshness.
  A persisted index can still be stale if `data/raw/` changed after the last build. The manifest exists specifically to catch that gap.

- Reusing an old collection without rebuilding.
  If new markdown files or updated frontmatter are not reflected in the collection, retrieval may still succeed but answer from outdated chunks.

- Treating any non-empty `data/chroma_db/` directory as valid.
  The app expects both index artifacts and a valid manifest. A partial directory is not considered up to date.

- Forgetting that collection identity matters.
  `kb_manifest.json` stores the expected `collection_name`. If the settings point at a different collection name, the persisted state is no longer considered current.

- Bypassing the normal reset behavior.
  `build_index(...)` uses `reset_collection=True` by default and attempts `delete_collection()` before re-adding chunks. That behavior reduces duplicate accumulation across rebuilds.

## Implementation Notes

The markdown corpus loader is strict because persistence quality starts before embeddings are created. `load_markdown_documents(...)` rejects:

- missing `data/raw/`
- non-markdown files in the raw directory
- incomplete frontmatter
- invalid metadata fields
- empty document bodies

That is useful for Chroma persistence because a clean index depends on predictable metadata. The chunk metadata later supports both retrieval and source display:

- `doc_id`
- `source_path`
- `title`
- `topic`
- `library`
- `doc_type`
- `difficulty`
- optional `error_family`
- `chunk_index`

## Metadata Fields and Filtered Retrieval

Metadata fields make filtered retrieval easier because they give Chroma stable
attributes to search before the model sees any context. In this project the raw
frontmatter is copied onto every chunk, so a retrieval request can narrow by
`topic`, `library`, `doc_type`, or `error_family` before ordinary similarity
ranking decides which chunks are most relevant.

That matters for persistence work because Chroma questions often share generic
words with unrelated RAG topics: `index`, `collection`, `metadata`, `retrieval`,
`filtered`, and `rebuild`. Accurate metadata keeps those terms attached to the
right operational source. A query about why metadata fields help filtered
retrieval should point back to the Chroma guide when the answer needs to explain
how frontmatter becomes chunk metadata and how that metadata supports precise
local retrieval.

Use these rules when adding or revising raw markdown:

1. Choose the narrowest truthful `topic` and `library`.
2. Set `doc_type` to the document's actual role, not just the question style.
3. Use `error_family` only when the document is meant to support that failure
   mode.
4. Include the operational terms users will ask for in the body, not only in
   frontmatter.

`write_kb_manifest(...)` records a summary of the successful build:

- `built_at`
- `collection_name`
- `indexed_chunk_count`
- `raw_file_count`
- `source_fingerprint`

A practical definition of a valid rebuilt state in this project is:

- the persist directory exists
- Chroma contains real document IDs from `vector_store.get()["ids"]`
- `kb_manifest.json` exists and parses cleanly
- the manifest collection name matches settings
- the manifest fingerprint matches the current `data/raw/` snapshot

That is why the app separates `persistence` from `freshness`. Persistence tells you the index survived. Freshness tells you the index still matches the corpus you intend to serve.

## Retrieval Hints

- `Where is the Chroma index stored in this project?`
- `How does build_index rebuild the local Chroma collection?`
- `What does kb_manifest.json store after a successful rebuild?`
- `What counts as a valid up-to-date knowledge base state for data/chroma_db?`
- `Why is persistence different from freshness in this Chroma setup?`
- `How does the project avoid duplicate chunks across repeated rebuilds?`
- `How do raw markdown metadata and chunk IDs affect Chroma persistence?`
