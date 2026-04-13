# Project Tools MCP Server

`project_tools_mcp_server.py` is a local stdin/stdout JSON-RPC handler for project tools. It supports only `tools/list` and `tools/call`.

Exposed tool:
- `search_internal_docs`

Manual `tools/list` example:
```bash
printf '%s\n' '{"jsonrpc":"2.0","id":"list-1","method":"tools/list"}' | python project_tools_mcp_server.py
```

Representative `tools/list` response:
```json
{"jsonrpc":"2.0","id":"list-1","result":{"tools":[{"name":"search_internal_docs","description":"Search local project markdown docs and return compact structured matches.","inputSchema":{"type":"object","properties":{"query":{"type":"string"}},"required":["query"]}}]}}
```

Manual `tools/call` example:
```bash
printf '%s\n' '{"jsonrpc":"2.0","id":"call-1","method":"tools/call","params":{"name":"search_internal_docs","arguments":{"query":"How do I persist Chroma collections?"}}}' | python project_tools_mcp_server.py
```

Representative `tools/call` response:
```json
{"jsonrpc":"2.0","id":"call-1","result":{"structuredContent":{"query":"How do I persist Chroma collections?","match_count":1,"matches":[{"title":"Chroma Persistence and Reindexing Guide","source_path":"data/raw/chroma_persistence_guide.md","doc_id":"chroma_persistence_guide","topic":"chroma","library":"chroma","doc_type":"how_to","excerpt":"Chroma should persist to a stable local directory so the knowledge base survives between runs."}]},"content":[{"type":"text","text":"Found 1 internal doc matches."}]}}
```
