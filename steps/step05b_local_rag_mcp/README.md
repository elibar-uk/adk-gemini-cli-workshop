# Step 05b: Local RAG + Airbnb MCP

This step replaces Vertex RAG with a local retrieval pipeline while keeping the Airbnb MCP tool.

## What this step does

- Uses local documents for retrieval-grounded destination advice.
- Keeps Airbnb MCP for accommodation search.
- Uses `now` for time/date context.
- Avoids Vertex RAG quotas entirely.

## Retrieval backend behavior

`local_rag.py` uses:

1. **Semantic retrieval**: `sentence-transformers` + `faiss-cpu` (default when model download/load works).
2. **Offline fallback**: TF-IDF lexical retrieval via `scikit-learn` (if semantic model cannot be loaded).

This means the step still works even in restricted/proxy environments.

## Run

From repo root:

```bash
uv run adk web steps
```

In ADK Web, select app:

- `step05b_local_rag_mcp`

## Optional environment variables

- `LOCAL_RAG_PATHS`: comma-separated files/folders to index.
  - Default: `WORKSHOP.md,docs`
  - Example: `LOCAL_RAG_PATHS="WORKSHOP.md,docs,notes/travel.md"`
- `LOCAL_RAG_EMBEDDING_MODEL`: sentence-transformers model name.
  - Default: `all-MiniLM-L6-v2`

Example:

```bash
export LOCAL_RAG_PATHS="WORKSHOP.md,docs"
export LOCAL_RAG_EMBEDDING_MODEL="all-MiniLM-L6-v2"
uv run adk web steps
```

## Test prompts

- "I'm going to Lisbon for 4 days in October. Recommend best neighborhoods and suggest 3 Airbnb options with tradeoffs."
- "Compare Alfama vs Baixa for a first-time couple trip and then suggest Airbnb options."
- "What should I know about transport and safety in Lisbon in October?"

## Notes

- This step does not require `RAG_CORPUS`.
- If retrieval returns weak context, the agent still continues with best-effort advice + Airbnb planning.
