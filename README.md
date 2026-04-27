# 🎙️ Meeting Notes Summariser — AI Agent

An **AI-powered meeting analysis system** that uses **LangGraph agents**, **RAG (Retrieval-Augmented Generation)**, and **Google Gemini** to answer questions about your meetings. Ingest transcripts (text or audio), then chat with an intelligent agent that can summarise, extract action items, identify decisions, and answer specific questions.

---

## ✨ Key Features

| Feature | Description |
|---------|-------------|
| 📝 **Multi-format Ingestion** | Paste text, upload `.txt`/`.md` files, or upload audio files (MP3, WAV, M4A) |
| 🤖 **Agentic RAG Pipeline** | LangGraph-powered agent that classifies queries and routes to specialist agents |
| 📋 **Meeting Summaries** | TL;DR + key discussion points from the transcript |
| ✅ **Action Item Extraction** | Tasks with owners and deadlines in structured format |
| 🎯 **Decision Tracking** | Confirmed decisions with context and participants |
| 💬 **Q&A** | Ask any question — answers cite specific speakers and transcript sections |
| 🔍 **Agent Trace Visibility** | See every step the agent takes (classify → retrieve → compress → answer) |
| 📊 **LangSmith Integration** | Full observability — trace every LLM call, token usage, and latency |
| 🗄️ **Smart Caching** | In-memory LRU + optional Redis for instant repeat queries |

---

## 🏗️ Architecture

```
User Query
    │
    ▼
┌────────────────┐
│ classify_query  │  ← Gemini classifies: summary / action_items / decisions / qa / general
└───────┬────────┘
        ▼
┌────────────────────┐
│ expand_and_retrieve │  ← HyDE generates hypothetical transcript, then MMR search in ChromaDB
└───────┬────────────┘
        ▼
┌──────────────────┐
│ compress_context  │  ← LLMLingua or extractive compression to reduce tokens
└───────┬──────────┘
        ▼
┌──────────────────────┐
│ Specialist Agent      │  ← One of: summary_agent / action_items_agent / decisions_agent / qa_agent
└───────┬──────────────┘
        ▼
  Formatted Response
  (Markdown + Agent Trace)
```

---

## 🔧 How It Works

### 1. Ingestion Pipeline

When you upload or paste a meeting transcript:

1. **Speaker Parsing** — The transcript is parsed into speaker-separated segments (e.g., `Alice: ...`, `Bob: ...`)
2. **Chunking** — Segments are split into ~300-token chunks with 40-token overlap, preserving speaker boundaries
3. **Embedding** — Each chunk is embedded using `gemini-embedding-001` (batched in groups of 100)
4. **Storage** — Chunks + embeddings are stored in ChromaDB with metadata (speaker, timestamps, meeting ID)

### 2. Query Classification

When you ask a question, the orchestrator agent uses Gemini to classify it:

| Query Type | Trigger Examples | Agent |
|------------|-----------------|-------|
| `summary` | "Summarise this meeting", "Give me an overview" | Summary Agent |
| `action_items` | "What are the action items?", "Who needs to do what?" | Action Items Agent |
| `decisions` | "What decisions were made?" | Decisions Agent |
| `qa` | "What did Alice say about the budget?" | Q&A Agent |
| `general` | Any other question | Q&A Agent (fallback) |

### 3. RAG Retrieval (HyDE + MMR)

**HyDE (Hypothetical Document Embeddings):**
Instead of searching with the raw query, the system first asks Gemini to generate a *hypothetical transcript excerpt* that would answer the question. This hypothetical text is embedded and used for search — producing much better retrieval results than raw query embedding.

**MMR (Maximal Marginal Relevance):**
After initial retrieval, MMR re-ranks results to maximise both **relevance** (similar to query) and **diversity** (different from each other). This prevents getting 6 chunks that all say the same thing.

### 4. Context Compression

Retrieved chunks are compressed before being sent to the specialist agent:

- **LLMLingua** (if available): Neural compression that preserves meaning while reducing tokens by 20%
- **Extractive fallback**: Keyword-scored sentence selection — keeps relevant sentences + context

Contexts under 2000 chars skip compression entirely to avoid losing information.

### 5. Specialist Agents

Each specialist agent uses Gemini with carefully engineered prompts:

- **Summary Agent** → Returns `TL;DR` + `Key Points` (bullet list)
- **Action Items Agent** → Returns structured JSON: `[{task, owner, deadline}]`
- **Decisions Agent** → Returns structured JSON: `[{description, context, participants}]`
- **Q&A Agent** → Returns a cited answer with source chunk references

### 6. Caching

- **In-memory LRU cache** (256 entries) — always active, no setup needed
- **Redis** (optional) — if Redis is running on `localhost:6379`, responses are also cached there with TTL
- Cache key = `SHA256(meeting_id + query)` — identical queries return instantly

---

## 📁 Project Structure

```
meeting-notes-summarize-ai/
├── app.py                          ← Streamlit UI (main entry point)
├── main.py                         ← Graph re-export for LangGraph Studio
├── langgraph.json                  ← LangGraph Studio configuration
├── .env                            ← API keys and settings
├── requirements.txt                ← Python dependencies
│
├── src/
│   ├── agents/
│   │   ├── graph.py                ← LangGraph pipeline (classify → retrieve → compress → agent)
│   │   ├── prompts.py              ← All LLM prompts with format instructions
│   │   └── specialist_agents.py    ← Summary, Action Items, Decisions, Q&A agents
│   │
│   ├── core/
│   │   ├── config.py               ← Settings loaded from .env
│   │   ├── llm.py                  ← Gemini API calls (text, structured, embeddings)
│   │   └── models.py               ← Pydantic data models (Chunk, AgentState, etc.)
│   │
│   ├── ingestion/
│   │   ├── pipeline.py             ← Ingest text/audio → parse speakers → chunk → embed → store
│   │   ├── chunker.py              ← Token-based chunking with speaker-boundary awareness
│   │   ├── transcriber.py          ← OpenAI Whisper audio → text
│   │   └── diarizer.py             ← Pyannote speaker diarization
│   │
│   └── rag/
│       ├── vector_store.py         ← ChromaDB with MMR search
│       ├── query_expander.py       ← HyDE query expansion
│       ├── compressor.py           ← LLMLingua / extractive context compression
│       └── cache.py                ← In-memory LRU + optional Redis cache
│
├── data/chroma/                    ← Vector store data (auto-created)
├── scripts/
│   └── test_pipeline.py            ← End-to-end test script
└── tests/                          ← Unit tests
```

---

## 🚀 Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/your-username/meeting-notes-summarize-ai.git
cd meeting-notes-summarize-ai
pip install -r requirements.txt
```

### 2. Configure

Create a `.env` file (or edit the existing one):

```env
# Required — get your free key from https://aistudio.google.com/apikey
GEMINI_API_KEY=your_gemini_api_key_here

# LLM settings
GEMMA_MODEL=gemma-4-26b-a4b-it
EMBEDDING_MODEL=gemini-embedding-001
GEMMA_TEMPERATURE=0.2
GEMMA_MAX_TOKENS=4096

# RAG settings
CHUNK_SIZE=300
CHUNK_OVERLAP=40
TOP_K_RETRIEVAL=6
RERANK_TOP_K=3

# Vector store
CHROMA_PERSIST_DIR=./data/chroma
CHROMA_COLLECTION=meetings

# Cache (optional — works without Redis)
REDIS_URL=redis://localhost:6379
CACHE_TTL_SECONDS=3600

# LangSmith tracing (optional — get free key from https://smith.langchain.com)
LANGSMITH_API_KEY=your_langsmith_api_key_here
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=meeting-summariser

# Whisper (for audio ingestion)
WHISPER_MODEL=base
```

### 3. Run the Streamlit App

```bash
streamlit run app.py
```

Open **http://localhost:8501** in your browser.

### 4. Use the App

1. **Sidebar → Ingest Meeting**
   - Paste transcript text, upload a `.txt` file, or upload an audio file
   - Add a title and attendees (optional)
   - Click **⚡ Ingest**

2. **Main Area → Chat**
   - Use quick action buttons: **Summarise**, **Action Items**, **Decisions**, **Ownership**
   - Or type any question in the chat input
   - View the **Agent Trace** expander to see every pipeline step

---

## 🔍 LangGraph Studio (Optional)

For debugging and step-by-step agent visualisation:

```bash
python -m langgraph_cli dev
```

This starts the LangGraph dev server at `http://127.0.0.1:2024` and opens LangGraph Studio in your browser where you can:
- Visualise the agent graph
- Inspect state at each node
- View LLM inputs/outputs
- Debug retrieval and compression

> **Note:** LangGraph Studio requires data to already be ingested via the Streamlit app.

---

## 📊 LangSmith Tracing (Optional)

When `LANGSMITH_API_KEY` is configured:
- Every pipeline run is traced to your [LangSmith dashboard](https://smith.langchain.com)
- View: LLM inputs/outputs, token usage, latency per node, error traces
- The Streamlit sidebar shows a green "LangSmith tracing: ON" indicator

---

## 🧪 Testing

Run the end-to-end pipeline test:

```bash
python scripts/test_pipeline.py
```

This ingests a sample transcript and runs summary + action items queries, verifying:
- ✅ Speaker parsing and chunking
- ✅ Embedding and vector storage
- ✅ HyDE query expansion
- ✅ MMR retrieval
- ✅ Context compression
- ✅ Query classification
- ✅ Specialist agent output formatting

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| **LLM** | Google Gemini (`gemma-4-26b-a4b-it`) |
| **Embeddings** | Google Gemini (`gemini-embedding-001`) |
| **Agent Framework** | LangGraph |
| **Vector Store** | ChromaDB |
| **Query Expansion** | HyDE (Hypothetical Document Embeddings) |
| **Retrieval** | MMR (Maximal Marginal Relevance) |
| **Context Compression** | LLMLingua / Extractive fallback |
| **UI** | Streamlit |
| **Audio Transcription** | OpenAI Whisper |
| **Speaker Diarization** | Pyannote Audio |
| **Caching** | In-memory LRU + Redis (optional) |
| **Observability** | LangSmith |
| **Data Models** | Pydantic v2 |

---

## 📋 Environment Variables Reference

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `GEMINI_API_KEY` | ✅ | — | Google AI Studio API key |
| `GEMMA_MODEL` | ❌ | `gemma-4-26b-a4b-it` | LLM model for generation |
| `EMBEDDING_MODEL` | ❌ | `gemini-embedding-001` | Model for vector embeddings |
| `GEMMA_TEMPERATURE` | ❌ | `0.2` | LLM temperature (0-1) |
| `GEMMA_MAX_TOKENS` | ❌ | `4096` | Max output tokens |
| `CHUNK_SIZE` | ❌ | `300` | Tokens per chunk |
| `CHUNK_OVERLAP` | ❌ | `40` | Overlap tokens between chunks |
| `TOP_K_RETRIEVAL` | ❌ | `6` | Number of chunks to retrieve |
| `CHROMA_PERSIST_DIR` | ❌ | `./data/chroma` | ChromaDB storage path |
| `REDIS_URL` | ❌ | `redis://localhost:6379` | Redis URL (optional) |
| `LANGSMITH_API_KEY` | ❌ | — | LangSmith API key (optional) |
| `LANGCHAIN_PROJECT` | ❌ | `meeting-summariser` | LangSmith project name |
| `WHISPER_MODEL` | ❌ | `base` | Whisper model size |
