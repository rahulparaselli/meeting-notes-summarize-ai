# Meeting Notes Summariser — AI Agent

An intelligent meeting analysis system that transforms raw transcripts into structured, actionable insights. Built on a multi-agent RAG architecture powered by LangGraph and Google Gemini, the system ingests meeting transcripts — text or audio — and enables natural-language conversations about their content. Ask for a summary, extract action items, surface decisions, or drill into what a specific participant said, all through a conversational web interface.

---

## Architecture

The system is built around a **four-stage agentic pipeline** where each stage has a single responsibility. A user query enters the pipeline and flows sequentially through classification, retrieval, compression, and specialist generation before the final response is returned.

**Stage 1 — Classify.** The orchestrator sends the user's query to Gemini, which classifies it into one of five intent categories: `summary`, `action_items`, `decisions`, `qa`, or `general`. This classification determines which specialist agent will handle the final generation. Routing to task-specific agents avoids a one-size-fits-all prompt — a summary request needs a fundamentally different instruction set than a question about a specific speaker.

**Stage 2 — Retrieve.** Rather than embedding the raw query directly, the system first generates a hypothetical transcript excerpt that would answer the question (HyDE — Hypothetical Document Embeddings). This produces embeddings that sit closer to actual document vectors, significantly improving recall. The hypothetical is then used for MMR (Maximal Marginal Relevance) search over ChromaDB, which balances relevance and diversity to prevent retrieving multiple chunks that say the same thing.

**Stage 3 — Compress.** Retrieved chunks often contain filler. Before passing them to the specialist agent, the system compresses the context using LLMLingua (neural compression) when available, or falls back to an extractive method that scores sentences by keyword relevance and keeps only the highest-scoring ones plus surrounding context. Contexts under 2,000 characters skip compression entirely to avoid information loss.

**Stage 4 — Specialist Agent.** The compressed context is routed to the appropriate agent based on the classification from Stage 1. Each agent has its own system prompt, structured output schema, and a capped token budget of 1,024 tokens to prevent runaway generation. The agent produces a structured response that is formatted and returned to the user.

The entire pipeline is orchestrated as a **LangGraph StateGraph**, where each stage is a node and data flows through a typed state object. Every node writes to an `agent_scratchpad` — a running trace log that is surfaced in the UI, making the system fully transparent.

---

## Features

### Multi-Format Ingestion

The system accepts meeting data in multiple formats. Users can paste transcript text directly, upload `.txt` or `.md` files, or upload audio files (MP3, WAV, M4A, WebM, OGG, FLAC). Audio files are transcribed locally using OpenAI Whisper and optionally processed with Pyannote for speaker diarization, so each statement is attributed to a specific participant.

### Speaker-Aware Chunking

Transcripts are parsed into speaker-separated segments using pattern matching. Each segment is then split into approximately 300-token chunks with 40-token overlap. The chunker respects speaker boundaries — no single chunk will blend text from two different speakers. This is critical because mixed-speaker chunks would confuse retrieval when a user asks what a specific person said.

### HyDE Query Expansion

Short user queries like "what were the action items?" produce weak embeddings that don't match well against long transcript passages. HyDE bridges this gap by first generating a hypothetical transcript excerpt that would answer the question, then embedding that hypothetical instead. The result is embeddings that are semantically closer to actual document content, yielding substantially better retrieval accuracy.

### MMR Retrieval

Standard similarity search often returns several chunks that are nearly identical in content. MMR (Maximal Marginal Relevance) re-ranks results to maximise both relevance to the query and diversity among the returned chunks. This ensures the specialist agent receives a broad view of the relevant discussion rather than multiple copies of the same point.

### Context Compression

Retrieved chunks are compressed before being sent to the specialist agent. When LLMLingua is available, it applies neural compression that preserves meaning while reducing token count by approximately 20%. When LLMLingua is not installed, the system falls back to an extractive method that extracts keywords from the query, scores each sentence by keyword overlap, and keeps only the relevant sentences plus the first sentence of each chunk for context. Small contexts under 2,000 characters bypass compression entirely.

### Specialist Agents

Four purpose-built agents handle different query types, each with its own system prompt and output schema:

- **Summary Agent** — Produces a TL;DR paragraph and a bulleted list of key discussion points from the transcript.
- **Action Items Agent** — Extracts structured action items with task description, assigned owner, and deadline fields. Uses Gemini's structured output mode with a Pydantic schema, falling back to text generation with JSON parsing if structured mode fails.
- **Decisions Agent** — Identifies confirmed decisions with description, supporting context, and participant information. Uses the same structured output approach as the Action Items Agent.
- **Q&A Agent** — Generates a free-text answer to any question, with source citations referencing specific speakers and chunk locations from the transcript.

An orchestrator classifier determines which agent handles each query. If classification fails for any reason, the system defaults to the Q&A Agent as the most general-purpose option.

### Smart Caching

Every pipeline response is cached using a SHA-256 hash of the meeting ID and query as the cache key. An in-memory LRU cache (256 entries) is always active with zero configuration. When Redis is available, responses are additionally stored with a configurable TTL for persistence across restarts. Identical queries return instantly without touching the LLM or vector store.

### Custom Web Interface

The frontend is a ChatGPT-style web UI built with vanilla HTML, CSS, and JavaScript, served directly by the FastAPI backend. It includes a meeting sidebar for switching between previously ingested meetings, persistent chat history that survives page reloads, quick-action buttons for Summary, Action Items, and Decisions, and an expandable "Agent Thinking" panel on every response that shows the full pipeline trace — classification reasoning, retrieval counts, compression ratios, and agent output.

### Observability

When a LangSmith API key is configured, every pipeline run is automatically traced to the LangSmith dashboard with full LLM input/output, token usage, and per-node latency. The project also supports LangGraph Studio for visual graph debugging, allowing step-by-step inspection of agent state at each node.

---

## How It Works

### Ingestion Flow

When a meeting transcript is submitted, the system first parses the text into speaker-separated segments by detecting patterns like `"Alice: Hello everyone"`. If no speaker labels are found, the text is split into sentence groups attributed to an unknown speaker. These segments are then chunked into approximately 300-token pieces with 40-token overlap, with the chunker ensuring no chunk crosses a speaker boundary. Each chunk is embedded using the Gemini Embedding model in batches of 100 and stored in ChromaDB along with metadata including the speaker name, timestamps, chunk index, and meeting ID.

For audio files, the system first transcribes the audio using OpenAI Whisper, optionally runs Pyannote speaker diarization to identify who is speaking when, and then follows the same chunking and embedding process as text ingestion.

### Query Flow

When a user sends a query, the system first checks the cache. On a cache hit, the stored response is returned immediately. On a miss, the query enters the four-stage pipeline.

The orchestrator classifies the query intent using Gemini with structured output, producing both a category and a reasoning explanation. The query is then expanded using HyDE — Gemini generates a short hypothetical transcript excerpt that would answer the question, and this hypothetical is combined with the original query. The combined text is embedded and used for MMR search over ChromaDB, scoped to the specific meeting ID. If no chunks are found, the system retries with a direct similarity search, and if still empty, searches across all meetings as a last resort.

The retrieved chunks are compressed to remove filler content, then routed to the appropriate specialist agent based on the earlier classification. The agent generates a structured response within its token budget, and the result is cached, saved to chat history, and returned to the user along with the full pipeline trace.

### Query Classification

The orchestrator maps queries to agents using the following routing logic:

| Query Type | Example | Handled By |
|---|---|---|
| `summary` | "Summarise this meeting" | Summary Agent |
| `action_items` | "What are the action items?" | Action Items Agent |
| `decisions` | "What decisions were made?" | Decisions Agent |
| `qa` | "What did Alice say about the budget?" | Q&A Agent |
| `general` | Any unclassified query | Q&A Agent (default) |

---

## Tech Stack

| Layer | Technology |
|---|---|
| LLM | Google Gemini (gemma-4-26b-a4b-it) |
| Embeddings | Google Gemini (gemini-embedding-001) |
| Agent Orchestration | LangGraph (StateGraph) |
| Vector Store | ChromaDB |
| Query Expansion | HyDE (Hypothetical Document Embeddings) |
| Retrieval Strategy | MMR (Maximal Marginal Relevance) |
| Context Compression | LLMLingua / extractive fallback |
| Backend | FastAPI + Uvicorn |
| Frontend | Vanilla HTML, CSS, JavaScript |
| Audio Transcription | OpenAI Whisper |
| Speaker Diarization | Pyannote Audio |
| Caching | In-memory LRU + Redis (optional) |
| Observability | LangSmith + LangGraph Studio |
| Data Validation | Pydantic v2 |

---

## Summary

This project demonstrates a production-grade approach to meeting analysis by combining retrieval-augmented generation with a multi-agent architecture. Rather than dumping an entire transcript into an LLM prompt, it intelligently retrieves only relevant context, compresses it to remove noise, and routes it to purpose-built agents — each with their own prompt engineering and structured output schemas. The four-stage pipeline design ensures each concern is handled independently, making the system modular, debuggable, and easy to extend. Every answer comes with a full trace of how the system arrived at it, providing complete transparency into the AI's reasoning process.

---

## License

This project is for educational and personal use.
