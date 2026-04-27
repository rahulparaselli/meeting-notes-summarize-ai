# Meeting Notes Summariser -- AI Agent

An AI-powered meeting analysis system built with LangGraph agents, RAG (Retrieval-Augmented Generation), and Google Gemini. Ingest meeting transcripts through a web UI, then chat with an intelligent agent that can summarise discussions, extract action items, identify decisions, and answer specific questions.

---

## Table of Contents

- [Features](#features)
- [Architecture Overview](#architecture-overview)
- [How It Works](#how-it-works)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Configuration](#configuration)
- [Debugging and Tracing](#debugging-and-tracing)
- [Testing](#testing)
- [Tech Stack](#tech-stack)

---

## Features

- **Multi-format Ingestion** -- Paste text, upload `.txt`/`.md` files, or upload audio files (MP3, WAV, M4A)
- **Agentic RAG Pipeline** -- LangGraph-powered agent classifies queries and routes to specialist agents
- **Meeting Summaries** -- TL;DR and key discussion points from the transcript
- **Action Item Extraction** -- Tasks with owners and deadlines in structured format
- **Decision Tracking** -- Confirmed decisions with context and participants
- **Question Answering** -- Ask any question with answers citing specific speakers and transcript sections
- **Agent Trace Visibility** -- See every step the agent takes through the pipeline
- **LangSmith Integration** -- Full observability with trace-level detail for every LLM call
- **Smart Caching** -- In-memory LRU cache with optional Redis for instant repeat queries
- **Custom Web UI** -- ChatGPT-style interface with persistent chat history per meeting

---

## Architecture Overview

<!-- Architecture diagram placeholder -->

The system follows a four-stage pipeline for every user query:

```
User Query --> Classify --> Retrieve --> Compress --> Specialist Agent --> Response
```

1. **Classify** -- Determines the intent (summary, action items, decisions, or Q&A)
2. **Retrieve** -- Expands the query using HyDE and searches the vector store with MMR
3. **Compress** -- Reduces retrieved context to keep only relevant content
4. **Specialist Agent** -- Generates the final answer using the appropriate agent

<!-- Detailed flow diagram placeholder -->

---

## How It Works

### Ingestion Pipeline

When a meeting transcript is submitted:

1. The transcript is parsed into speaker-separated segments
2. Segments are split into approximately 300-token chunks with 40-token overlap, preserving speaker boundaries
3. Each chunk is embedded using Gemini Embedding and batched in groups of 100
4. Chunks and embeddings are stored in ChromaDB with metadata (speaker, timestamps, meeting ID)

### Query Classification

The orchestrator uses Gemini to classify incoming queries:

| Query Type     | Example                                 | Routed To           |
|----------------|----------------------------------------|---------------------|
| `summary`      | "Summarise this meeting"               | Summary Agent       |
| `action_items` | "What are the action items?"           | Action Items Agent  |
| `decisions`    | "What decisions were made?"            | Decisions Agent     |
| `qa`           | "What did Alice say about the budget?" | Q&A Agent           |
| `general`      | Any other question                     | Q&A Agent (default) |

### RAG Retrieval

**HyDE (Hypothetical Document Embeddings)** -- Instead of embedding the raw query, the system first generates a hypothetical transcript excerpt that would answer the question. This produces embeddings closer to actual document embeddings, improving retrieval quality.

**MMR (Maximal Marginal Relevance)** -- After initial retrieval, results are re-ranked to maximise both relevance to the query and diversity among results. This prevents retrieving multiple chunks that say the same thing.

### Context Compression

Retrieved chunks are compressed before being sent to the specialist agent:

- **LLMLingua** (if available) -- Neural compression that preserves meaning while reducing token count
- **Extractive fallback** -- Keyword-scored sentence selection that keeps relevant sentences plus surrounding context
- Contexts under 2,000 characters skip compression to avoid information loss

### Specialist Agents

Each agent uses Gemini with task-specific prompts and token-limited output:

- **Summary Agent** -- Returns a TL;DR and a list of key discussion points
- **Action Items Agent** -- Returns structured data with task, owner, and deadline fields
- **Decisions Agent** -- Returns structured data with description, context, and participants
- **Q&A Agent** -- Returns a cited answer with source chunk references

### Caching

- **In-memory LRU** (256 entries) -- Always active, no setup required
- **Redis** (optional) -- If Redis is available, responses are cached with a configurable TTL
- Cache key is a SHA256 hash of the meeting ID and query, so identical queries return instantly

---

## Project Structure

```
meeting-notes-summarize-ai/
|-- server.py                       # FastAPI backend (REST API + static file serving)
|-- main.py                         # Graph re-export for LangGraph Studio
|-- langgraph.json                  # LangGraph Studio configuration
|-- .env                            # API keys and settings
|-- requirements.txt                # Python dependencies
|
|-- static/
|   |-- index.html                  # Web UI entry point
|   |-- style.css                   # UI styles
|   +-- app.js                      # Frontend logic (chat, ingestion, history)
|
|-- src/
|   |-- agents/
|   |   |-- graph.py                # LangGraph pipeline definition
|   |   |-- prompts.py              # All LLM prompts with format instructions
|   |   +-- specialist_agents.py    # Summary, Action Items, Decisions, Q&A agents
|   |
|   |-- core/
|   |   |-- config.py               # Settings loaded from .env via Pydantic
|   |   |-- llm.py                  # Gemini API wrapper (text, structured, embeddings)
|   |   +-- models.py               # Data models (Chunk, AgentState, etc.)
|   |
|   |-- ingestion/
|   |   |-- pipeline.py             # Ingest text/audio, parse speakers, chunk, embed, store
|   |   |-- chunker.py              # Token-based chunking with speaker-boundary awareness
|   |   |-- transcriber.py          # OpenAI Whisper audio-to-text
|   |   +-- diarizer.py             # Pyannote speaker diarization
|   |
|   +-- rag/
|       |-- vector_store.py         # ChromaDB wrapper with MMR search
|       |-- query_expander.py       # HyDE query expansion
|       |-- compressor.py           # LLMLingua and extractive context compression
|       +-- cache.py                # In-memory LRU + optional Redis cache
|
|-- data/chroma/                    # Vector store data (auto-created)
|-- scripts/
|   +-- test_pipeline.py            # End-to-end test script
+-- tests/                          # Unit tests
```

---

## Getting Started

### Prerequisites

- Python 3.12 or higher
- A Google AI Studio API key (free at https://aistudio.google.com/apikey)

### Installation

```bash
git clone https://github.com/your-username/meeting-notes-summarize-ai.git
cd meeting-notes-summarize-ai
pip install -r requirements.txt
```

### Configuration

Copy and edit the environment file:

```
GEMINI_API_KEY=your_gemini_api_key_here
```

See the [Configuration](#configuration) section below for all available settings.

### Running the Application

```bash
uvicorn server:app --reload
```

Open http://localhost:8000 in your browser.

### Using the Application

1. **Ingest a Meeting** -- Click "New Meeting", paste or upload a transcript, add a title, and submit
2. **Ask Questions** -- Use the quick action buttons (Summary, Action Items, Decisions) or type any question
3. **View Traces** -- Expand the "Agent Thinking" section on any response to see the full pipeline trace
4. **Switch Meetings** -- Use the sidebar to navigate between previously ingested meetings

---

## Configuration

All settings are loaded from the `.env` file.

### Required

| Variable         | Description                  |
|------------------|------------------------------|
| `GEMINI_API_KEY` | Google AI Studio API key     |

### LLM Settings

| Variable            | Default                  | Description                |
|---------------------|--------------------------|----------------------------|
| `GEMMA_MODEL`       | `gemma-4-26b-a4b-it`    | LLM model for generation  |
| `EMBEDDING_MODEL`   | `gemini-embedding-001`  | Model for vector embeddings|
| `GEMMA_TEMPERATURE` | `0.2`                   | Generation temperature     |
| `GEMMA_MAX_TOKENS`  | `4096`                  | Max output tokens (global) |

### RAG Settings

| Variable           | Default          | Description                          |
|--------------------|------------------|--------------------------------------|
| `CHUNK_SIZE`       | `300`            | Tokens per chunk during ingestion    |
| `CHUNK_OVERLAP`    | `40`             | Overlap tokens between chunks        |
| `TOP_K_RETRIEVAL`  | `6`              | Number of chunks to retrieve         |
| `RERANK_TOP_K`     | `3`              | Chunks to keep after re-ranking      |

### Storage

| Variable             | Default              | Description                |
|----------------------|----------------------|----------------------------|
| `CHROMA_PERSIST_DIR` | `./data/chroma`      | ChromaDB storage path      |
| `CHROMA_COLLECTION`  | `meetings`           | ChromaDB collection name   |
| `REDIS_URL`          | `redis://localhost:6379` | Redis URL (optional)   |
| `CACHE_TTL_SECONDS`  | `3600`               | Cache time-to-live         |

### Observability

| Variable              | Default               | Description                |
|-----------------------|-----------------------|----------------------------|
| `LANGSMITH_API_KEY`   | --                    | LangSmith API key          |
| `LANGCHAIN_TRACING_V2`| `true`               | Enable LangChain tracing   |
| `LANGCHAIN_PROJECT`   | `meeting-summariser`  | LangSmith project name     |

### Audio (Optional)

| Variable        | Default | Description              |
|-----------------|---------|--------------------------|
| `WHISPER_MODEL` | `base`  | Whisper model size       |

---

## Debugging and Tracing

### LangGraph Studio

For visual debugging and step-by-step agent inspection:

```bash
pip install "langgraph-cli[inmem]"
python -m langgraph_cli dev
```

This starts a dev server at http://127.0.0.1:2024 with a Studio UI where you can visualise the agent graph, inspect state at each node, and view LLM inputs and outputs.

### LangSmith

When `LANGSMITH_API_KEY` is configured, every pipeline run is traced to your LangSmith dashboard at https://smith.langchain.com. Traces include LLM inputs and outputs, token usage, latency per node, and error details.

---

## Testing

Run the end-to-end pipeline test:

```bash
python scripts/test_pipeline.py
```

This ingests a sample transcript and runs queries to verify speaker parsing, chunking, embedding, vector storage, HyDE expansion, MMR retrieval, context compression, query classification, and specialist agent output.

---

## Tech Stack

| Component              | Technology                                    |
|------------------------|-----------------------------------------------|
| LLM                    | Google Gemini (gemma-4-26b-a4b-it)            |
| Embeddings             | Google Gemini (gemini-embedding-001)          |
| Agent Framework        | LangGraph                                     |
| Vector Store           | ChromaDB                                      |
| Query Expansion        | HyDE (Hypothetical Document Embeddings)       |
| Retrieval Strategy     | MMR (Maximal Marginal Relevance)              |
| Context Compression    | LLMLingua / Extractive fallback               |
| Backend                | FastAPI + Uvicorn                              |
| Frontend               | Vanilla HTML, CSS, JavaScript                  |
| Audio Transcription    | OpenAI Whisper                                |
| Speaker Diarization    | Pyannote Audio                                |
| Caching                | In-memory LRU + Redis (optional)              |
| Observability          | LangSmith                                     |
| Data Models            | Pydantic v2                                   |

---

## License

This project is for educational and personal use.
