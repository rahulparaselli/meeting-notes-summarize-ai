# Meeting Summariser

AI-powered meeting notes summariser built with **Gemma 4** (Google Gemini API) + **LangGraph** multi-agent system.

## Architecture

```
Audio / Text / Live stream
        │
        ▼
  Ingestion Pipeline
  ├── Whisper STT
  ├── Speaker diarization (pyannote)
  ├── Speaker-aware chunker (~300 tokens)
  └── Metadata tagging
        │
        ▼
  RAG Engine
  ├── text-embedding-004 (Google)
  ├── ChromaDB vector store
  ├── HyDE query expansion
  ├── MMR retrieval (diverse chunks)
  └── LLMLingua context compression
        │
        ▼
  LangGraph Agent Pipeline
  ├── classify_query      → detects intent
  ├── expand_and_retrieve → HyDE + MMR
  ├── compress_context    → token reduction
  └── [route] ──► summary_agent
                ──► action_items_agent
                ──► decisions_agent
                ──► qa_agent (cited answers)
        │
        ▼
  FastAPI + Redis cache
        │
        ▼
  Frontend (live agent trace UI)
```

## Folder structure

```
meeting-summariser/
├── src/
│   ├── core/
│   │   ├── config.py        # pydantic-settings
│   │   ├── models.py        # all shared types + AgentState
│   │   └── llm.py           # Gemini API client (text, structured, embed)
│   ├── ingestion/
│   │   ├── transcriber.py   # Whisper STT
│   │   ├── diarizer.py      # pyannote speaker diarization
│   │   ├── chunker.py       # speaker-aware token chunker
│   │   └── pipeline.py      # orchestrates full ingest flow
│   ├── rag/
│   │   ├── vector_store.py  # ChromaDB + MMR retrieval
│   │   ├── query_expander.py# HyDE + multi-query expansion
│   │   ├── compressor.py    # LLMLingua / extractive compression
│   │   └── cache.py         # Redis semantic cache
│   ├── agents/
│   │   ├── prompts.py       # all system + user prompts
│   │   ├── specialist_agents.py  # summary / action / decisions / qa nodes
│   │   └── graph.py         # LangGraph graph definition + run_pipeline()
│   └── api/
│       ├── main.py          # FastAPI app factory
│       └── routes/
│           ├── ingest.py    # POST /ingest/text, /ingest/audio
│           ├── query.py     # POST /query, /summarise
│           └── health.py    # GET /health
├── frontend/
│   └── index.html           # full UI with live LangGraph trace
├── tests/
│   ├── test_chunker.py
│   ├── test_graph_routing.py
│   └── test_compressor.py
├── scripts/
│   └── run_demo.py          # CLI for ingest + query
├── main.py                  # uvicorn entrypoint
├── pyproject.toml
└── .env.example
```

## Quick start

```bash
# 1. Install dependencies
pip install poetry
poetry install

# 2. Configure
cp .env.example .env
# Edit .env → add GEMINI_API_KEY

# 3. Start Redis (optional, for caching)
docker run -d -p 6379:6379 redis

# 4. Run API
python main.py

# 5. Open frontend
open frontend/index.html
```

## CLI usage

```bash
# Ingest a transcript file
python scripts/run_demo.py ingest \
  --text transcript.txt \
  --title "Q4 Planning" \
  --attendees "Alice, Bob, Carol"

# Query a meeting (use ID printed by ingest)
python scripts/run_demo.py query \
  --meeting-id abc123def456 \
  --query "What were the action items?"
```

## API endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/v1/ingest/text` | Ingest transcript text |
| POST | `/api/v1/ingest/audio` | Upload + transcribe audio |
| POST | `/api/v1/query` | Run LangGraph pipeline |
| POST | `/api/v1/summarise?meeting_id=X` | Full summary shortcut |
| GET  | `/health` | Health check |

## Key techniques

| Technique | Where | Benefit |
|-----------|-------|---------|
| HyDE expansion | `rag/query_expander.py` | Better embedding recall |
| MMR retrieval | `rag/vector_store.py` | Diverse, non-redundant chunks |
| LLMLingua compression | `rag/compressor.py` | 30–60% token reduction |
| `response_schema` | `core/llm.py` | Zero-parse-failure structured output |
| Speaker-aware chunking | `ingestion/chunker.py` | Context-preserving splits |
| LangGraph routing | `agents/graph.py` | Typed agent orchestration |
| Redis cache | `rag/cache.py` | Skip LLM on repeated queries |
| Semantic cache | `rag/cache.py` | Near-duplicate query dedup |

## Running tests

```bash
poetry run pytest
```
