---
title: Streamlit Chat Patterns for RAG Interfaces
topic: streamlit
library: streamlit
doc_type: example
difficulty: intermediate
error_family: ui
---
# Streamlit Chat Patterns for RAG Interfaces

Streamlit chat apps should keep the interaction simple: user message in, retrieved sources shown back, and tool outputs rendered only when they matter.

For AI application demos, progress indicators are useful during retrieval and generation because they make multi-step work visible to the user.

When building a RAG interface, show the source title and metadata next to the answer so the user can see what grounded the response.
