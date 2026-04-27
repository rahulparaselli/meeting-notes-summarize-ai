# Meeting Notes Summariser — AI Agent

An intelligent meeting analysis system that transforms raw transcripts into structured, actionable insights. Built on a multi-agent RAG architecture powered by LangGraph and Google Gemini, the system ingests meeting transcripts — text or audio — and enables natural-language conversations about their content. Ask for a summary, extract action items, surface decisions, or drill into what a specific participant said, all through a conversational web interface.

---

## Architecture

The system is designed around a **four-stage agentic pipeline** that separates concerns cleanly: understanding intent, retrieving evidence, reducing noise, and generating a precise answer.

```
┌──────────┐     ┌──────────────────┐     ┌──────────────┐     ┌──────────────────┐     ┌──────────┐
│  User    │────▶│  1. Classify     │────▶│  2. Retrieve │────▶│  3. Compress     │────▶│ 4. Agent │
│  Query   │     │  (Intent)        │     │  (HyDE+MMR)  │     │  (LLMLingua)     │     │ (Answer) │
└──────────┘     └──────────────────┘     └──────────────┘     └──────────────────┘     └──────────┘
```

**Why this design?**

- **Classify first** — Routing to task-specific agents avoids a one-size-fits-all prompt. A summary request needs a fundamentally different instruction set than a question about a specific speaker.
- **Retrieve with HyDE** — Raw user queries produce weak embeddings. Generating a hypothetical transcript excerpt first yields embeddings that sit closer to actual document vectors, significantly improving recall.
- **Compress before generation** — Retrieved chunks contain filler. Compressing them keeps the specialist agent focused on relevant content and reduces token waste.
- **Specialist agents** — Each agent has its own system prompt, output schema, and token budget. This produces consistently structured output and prevents the LLM from drifting.

The entire pipeline is orchestrated as a **LangGraph state graph**, where each stage is a node and data flows through a typed `GraphState`. This makes the flow debuggable, traceable, and compatible with LangGraph Studio for visual inspection.

---

## Features & Components

### Ingestion Engine

| Capability | Description | Why It's Included |
|---|---|---|
| **Text ingestion** | Paste transcripts or upload `.txt`/`.md` files directly | Lowest-friction path for users who already have transcripts |
| **Audio ingestion** | Upload MP3, WAV, M4A, WebM, OGG, or FLAC files | Eliminates the need for external transcription tools |
| **Whisper transcription** | Automatic speech-to-text via OpenAI Whisper | Best open-source transcription model, runs locally |
| **Speaker diarization** | Pyannote-based speaker identification from audio | Attributing statements to speakers is critical for action items and Q&A |
| **Speaker-aware chunking** | Token-based splitting (~300 tokens, 40-token overlap) that respects speaker boundaries | Prevents a single chunk from blending two speakers, which would confuse downstream retrieval |
| **Batch embedding** | Gemini Embedding in batches of 100 | Keeps ingestion fast without hitting rate limits |

### RAG Pipeline

| Component | What It Does | Why It Matters |
|---|---|---|
| **HyDE query expansion** | Generates a hypothetical transcript excerpt before embedding the query | Bridges the semantic gap between short questions and long transcript passages — embeddings of a fake answer are closer to real document embeddings than embeddings of a question |
| **MMR retrieval** | Maximal Marginal Relevance search over ChromaDB | Balances relevance and diversity — prevents retrieving five chunks that all say the same thing |
| **Context compression** | LLMLingua neural compression with extractive fallback | Strips filler from retrieved chunks so the specialist agent receives only signal. Small contexts (<2,000 chars) skip compression to avoid information loss |
| **ChromaDB vector store** | Persistent, local vector database with metadata filtering | Enables per-meeting scoped search using meeting IDs, so queries never leak across meetings |

### Specialist Agents

Each agent is a focused LangGraph node with a task-specific system prompt, output schema, and capped token budget.

| Agent | Output | Use Case |
|---|---|---|
| **Summary Agent** | TL;DR + bullet-point key discussion topics | "What happened in this meeting?" |
| **Action Items Agent** | Structured list with task, owner, and deadline fields | "What needs to be done?" |
| **Decisions Agent** | Structured list with description, context, and participants | "What was decided?" |
| **Q&A Agent** | Cited answer with source chunk references | "What did Alice say about the budget?" |

An **orchestrator classifier** powered by Gemini determines which agent handles each query. If classification fails, it defaults to Q&A — the most general-purpose agent.

### Caching Layer

| Layer | Behaviour | Purpose |
|---|---|---|
| **In-memory LRU** (256 entries) | Always active, zero configuration | Instant responses for repeated queries within a session |
| **Redis** (optional) | Persistent cache with configurable TTL | Survives restarts, shared across instances |

Cache keys are SHA-256 hashes of `(meeting_id, query)`, so identical questions to the same meeting return instantly without touching the LLM.

### Web Interface

A custom ChatGPT-style web UI built with vanilla HTML, CSS, and JavaScript.

- **Persistent chat history** — Conversations are stored per meeting and survive page reloads
- **Meeting sidebar** — Switch between previously ingested meetings without losing context
- **Agent trace visibility** — Expand the "Agent Thinking" panel on any response to see every pipeline step: classification reasoning, retrieval counts, compression ratios, and agent output
- **Quick actions** — One-click buttons for Summary, Action Items, and Decisions

### Observability

- **LangSmith integration** — When configured, every pipeline run is traced with full LLM input/output, token usage, and per-node latency
- **LangGraph Studio** — Visual graph debugger for stepping through agent states node by node
- **Structured logging** — Every pipeline stage logs its inputs, outputs, and timing

---

## How It Works

### 1. Ingestion Flow

```
Transcript (text or audio)
  │
  ├─ [Audio only] Whisper transcription → raw text
  ├─ [Audio only] Pyannote diarization → speaker labels
  │
  ▼
Speaker-separated segments
  │
  ▼
Token-based chunking (~300 tokens, 40-token overlap)
  │  • Preserves speaker boundaries
  │  • Each chunk tagged with speaker, timestamps, meeting ID
  │
  ▼
Gemini Embedding (batched, 100 per request)
  │
  ▼
ChromaDB (persistent vector store)
```

The transcript is first parsed into speaker-separated segments using pattern matching (`"Alice: ..."` format). If no speaker labels are detected, the text is split into sentence groups. Segments are then chunked with token-level precision, ensuring no chunk crosses a speaker boundary. Each chunk is embedded and stored with rich metadata that enables scoped retrieval later.

### 2. Query Flow

```
User asks: "What action items came out of this meeting?"
  │
  ▼
┌─────────────────────────────────────────────────────────────────┐
│  CLASSIFY                                                       │
│  Gemini classifies the query → "action_items"                   │
│  Reasoning: "User is asking for tasks and assignments"          │
└──────────────────────────────────┬──────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────┐
│  EXPAND & RETRIEVE                                              │
│  HyDE generates a hypothetical transcript excerpt:              │
│    "Alice: Let's assign the API migration to Bob, due Friday."  │
│  This hypothetical is embedded and used for MMR search          │
│  Result: 6 relevant chunks retrieved from ChromaDB              │
└──────────────────────────────────┬──────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────┐
│  COMPRESS                                                       │
│  LLMLingua reduces retrieved context by ~20%                    │
│  Extractive fallback: keyword-scored sentence selection          │
│  Small contexts (<2K chars) pass through unchanged              │
└──────────────────────────────────┬──────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────┐
│  SPECIALIST AGENT                                               │
│  Action Items Agent processes compressed context                │
│  Output: structured JSON with task, owner, deadline fields      │
│  Token budget: 1024 (capped to prevent runaway generation)      │
└──────────────────────────────────┬──────────────────────────────┘
                                   │
                                   ▼
Response rendered in chat UI with agent trace expandable
```

Every step records its activity to the `agent_scratchpad` — a running log visible in the UI under "Agent Thinking". This makes the system transparent: users can see *why* the agent gave a particular answer.

### 3. Caching Flow

Before the pipeline runs, the cache is checked using a SHA-256 hash of the meeting ID and query. On a hit, the cached response is returned immediately. On a miss, the full pipeline executes and the result is stored in both the in-memory LRU and (optionally) Redis.

---

## Tech Stack

| Layer | Technology |
|---|---|
| **LLM** | Google Gemini (gemma-4-26b-a4b-it) |
| **Embeddings** | Google Gemini (gemini-embedding-001) |
| **Agent Orchestration** | LangGraph (StateGraph) |
| **Vector Store** | ChromaDB |
| **Query Expansion** | HyDE (Hypothetical Document Embeddings) |
| **Retrieval** | MMR (Maximal Marginal Relevance) |
| **Context Compression** | LLMLingua / extractive fallback |
| **Backend** | FastAPI + Uvicorn |
| **Frontend** | Vanilla HTML, CSS, JavaScript |
| **Audio Transcription** | OpenAI Whisper |
| **Speaker Diarization** | Pyannote Audio |
| **Caching** | In-memory LRU + Redis (optional) |
| **Observability** | LangSmith + LangGraph Studio |
| **Data Validation** | Pydantic v2 |

---

## Summary

This project demonstrates a production-grade approach to meeting analysis by combining retrieval-augmented generation with a multi-agent architecture. Rather than dumping an entire transcript into an LLM prompt, it intelligently retrieves only relevant context, compresses it, and routes it to purpose-built agents — each with their own prompt engineering and structured output schemas. The result is fast, accurate, and transparent: every answer comes with a full trace of how the system arrived at it.

---

## License

This project is for educational and personal use.
