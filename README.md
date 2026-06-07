# token-optimizer

Token-efficient RAG layer for agentic dev tasks using **Anthropic Claude + Voyage AI + pgvector**.

Supported task types: `jira` · `bug` · `analysis` · `coding` · `design` · `response`

## Architecture

```
optimize index  →  chunk files → embed (Voyage) → store in pgvector
optimize run    →  embed query → vector search → build context → call Claude
optimize eval   →  run golden.jsonl → report token savings & quality
```

Key components:
- `config.py` — budgets, model names, difficulty threshold, chunk sizes
- `token_optimizer/engine.py` — main pipeline (retrieve → route → build context → LLM)
- `token_optimizer/router.py` — picks fast/smart model based on retrieval confidence
- `token_optimizer/context_builder.py` — assembles token-budgeted prompt context
- `token_optimizer/retrieval/` — indexer (chunk+embed+store) and retriever (vector search)
- `token_optimizer/core/` — Anthropic LLM client, Voyage embeddings client, pgvector DB ops
- `token_optimizer/api.py` — FastAPI HTTP server (`/health`, `/run`)
- `eval/` — evaluation harness + `golden.jsonl` (6 task types)

## Quick start

```bash
# 1. Install
pip install -e .

# 2. Configure
cp .env.example .env
# edit .env: set ANTHROPIC_API_KEY and VOYAGE_API_KEY

# 3. Start Postgres+pgvector (Docker)
docker compose up -d db

# 4. Index a repo
optimize index --repo myrepo --root /path/to/code

# 5. Run a task
optimize run analysis --repo myrepo --input "What does the engine module do?"

# 6. HTTP API
uvicorn token_optimizer.api:app --port 8080
curl http://localhost:8080/health
curl -X POST http://localhost:8080/run \
  -H 'Content-Type: application/json' \
  -d '{"task_type":"response","repo":"myrepo","input":"What embedding model is used?"}'

# 7. Eval
optimize eval --repo myrepo
```

## Configuration (`config.py`)

| Setting | Default | Notes |
|---|---|---|
| `fast_model` | `claude-haiku-4-5-20251001` | Used when retrieval similarity ≥ threshold |
| `smart_model` | `claude-sonnet-4-6` | Used when similarity < threshold or complex task |
| `difficulty_threshold` | `0.65` | Cosine similarity below this → smart model |
| `candidate_chunks` | `20` | Chunks fetched from pgvector |
| `final_chunks` | `5` | Chunks kept for context |
| `chunk_lines` | `50` | Lines per chunk |
| `voyage_model` | `voyage-code-2` | Embedding model |

## Requirements

- Python 3.10+
- PostgreSQL 14+ with pgvector extension
- `ANTHROPIC_API_KEY` and `VOYAGE_API_KEY`
