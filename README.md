# Meeting Notes Summariser — AI Agent

An AI-powered system that ingests meeting transcripts and lets you chat with them. Upload a transcript or audio file, then ask questions — get summaries, action items, decisions, or answers about specific topics and speakers.

Built with **LangGraph**, **Google Gemini**, and **RAG** (Retrieval-Augmented Generation).

---

## Architecture

Every user query flows through a four-stage pipeline:

1. **Classify** — Identifies query intent (summary, action items, decisions, or Q&A)
2. **Retrieve** — Expands the query using HyDE, then searches the vector store with MMR
3. **Compress** — Removes irrelevant content from retrieved chunks
4. **Specialist Agent** — Generates a structured response using the right agent

The pipeline is built as a **LangGraph StateGraph** — each stage is a node, and every step is logged to a trace visible in the UI.

---

## Features

- **Multi-format ingestion** — Paste text, upload `.txt`/`.md` files, or upload audio (MP3, WAV, M4A)
- **Audio transcription** — Whisper converts speech to text; Pyannote identifies speakers
- **Speaker-aware chunking** — Chunks never mix text from different speakers
- **HyDE query expansion** — Generates a hypothetical answer before searching, improving retrieval accuracy
- **MMR retrieval** — Returns relevant *and* diverse results from ChromaDB
- **Context compression** — LLMLingua or extractive fallback strips filler before generation
- **Smart caching** — In-memory LRU + optional Redis; identical queries return instantly
- **Agent trace visibility** — See every pipeline step in the UI
- **LangSmith integration** — Full observability with per-node tracing
- **ChatGPT-style web UI** — Chat history, meeting sidebar, quick-action buttons

---

## Components

### Specialist Agents

| Agent | What It Returns |
|---|---|
| Summary Agent | TL;DR + key discussion points |
| Action Items Agent | Tasks with owner and deadline |
| Decisions Agent | Decisions with context and participants |
| Q&A Agent | Cited answer with source references |

The orchestrator classifies each query and routes it to the right agent. Unrecognised queries default to Q&A.

### RAG Pipeline

| Component | Role |
|---|---|
| HyDE | Generates a fake transcript excerpt to improve embedding similarity |
| MMR Search | Retrieves chunks that are both relevant and diverse |
| Compressor | Reduces context size while keeping important content |
| ChromaDB | Stores embeddings with metadata, supports per-meeting filtering |

### Ingestion

| Input | Process |
|---|---|
| Text / File | Parse speakers → chunk (~300 tokens) → embed → store |
| Audio | Whisper transcribe → diarize speakers → chunk → embed → store |

---

## How It Works

### Ingestion

The transcript is parsed into speaker segments. Each segment is split into ~300-token chunks with overlap, preserving speaker boundaries. Chunks are embedded with Gemini and stored in ChromaDB with speaker, timestamp, and meeting ID metadata.

### Query

The system checks the cache first. On a miss, it classifies the query, expands it with HyDE, retrieves relevant chunks via MMR, compresses the context, and routes to the appropriate specialist agent. The response is cached and returned with a full pipeline trace.

### Routing

| Query Type | Example | Agent |
|---|---|---|
| `summary` | "Summarise this meeting" | Summary |
| `action_items` | "What are the action items?" | Action Items |
| `decisions` | "What decisions were made?" | Decisions |
| `qa` / `general` | "What did Alice say about the budget?" | Q&A |

---

## Tech Stack

| Layer | Technology |
|---|---|
| LLM | Google Gemini |
| Embeddings | Gemini Embedding |
| Agent Framework | LangGraph |
| Vector Store | ChromaDB |
| Backend | FastAPI |
| Frontend | HTML, CSS, JavaScript |
| Transcription | OpenAI Whisper |
| Diarization | Pyannote Audio |
| Caching | LRU + Redis |
| Observability | LangSmith |
| Validation | Pydantic v2 |

---

## License

This project is for educational and personal use.
