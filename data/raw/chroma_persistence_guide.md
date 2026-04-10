---
title: Chroma Persistence and Reindexing Guide
topic: chroma
library: chroma
doc_type: how_to
difficulty: intro
error_family: persistence
---
# Chroma Persistence and Reindexing Guide

Chroma should persist to a stable local directory so the knowledge base survives between runs.
In this project, `data/chroma_db` is the default persistence path.

Repeated indexing should be handled deliberately.
A safe first approach is to rebuild the collection before adding the new chunks so you do not accumulate duplicates from unchanged markdown files.

Metadata fields such as topic, library, and doc type make later filtered retrieval much easier to implement.
