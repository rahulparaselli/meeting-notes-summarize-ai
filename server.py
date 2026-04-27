"""
Meeting Summariser — FastAPI Backend

Serves the frontend UI and provides REST API endpoints for
meeting ingestion, chat, and history management.

Run:
    uvicorn server:app --reload
"""
import asyncio
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

load_dotenv()

# ── LangSmith tracing (auto-enabled if key is set) ──────────────────────────

if os.getenv("LANGSMITH_API_KEY") and os.getenv("LANGSMITH_API_KEY") != "your_langsmith_api_key_here":
    os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
    os.environ.setdefault("LANGCHAIN_PROJECT", "meeting-summariser")

# ── Logging ──────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("server")

# ── Imports from existing pipeline ───────────────────────────────────────────

from src.agents.graph import run_pipeline
from src.ingestion.pipeline import ingest_text
from src.rag.vector_store import VectorStore

# ── Data paths ───────────────────────────────────────────────────────────────

DATA_DIR = Path("./data")
MEETINGS_FILE = DATA_DIR / "meetings.json"
HISTORY_DIR = DATA_DIR / "history"

DATA_DIR.mkdir(exist_ok=True)
HISTORY_DIR.mkdir(parents=True, exist_ok=True)

# ── Persistence helpers ──────────────────────────────────────────────────────


def _load_meetings() -> dict[str, Any]:
    if MEETINGS_FILE.exists():
        try:
            return json.loads(MEETINGS_FILE.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def _save_meetings(meetings: dict[str, Any]) -> None:
    MEETINGS_FILE.write_text(
        json.dumps(meetings, indent=2, default=str), encoding="utf-8"
    )


def _history_path(meeting_id: str) -> Path:
    return HISTORY_DIR / f"{meeting_id}.json"


def _load_history(meeting_id: str) -> list[dict[str, Any]]:
    p = _history_path(meeting_id)
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return []
    return []


def _save_history(meeting_id: str, messages: list[dict[str, Any]]) -> None:
    _history_path(meeting_id).write_text(
        json.dumps(messages, indent=2, default=str), encoding="utf-8"
    )


# ── FastAPI app ──────────────────────────────────────────────────────────────

app = FastAPI(title="Meeting Summariser AI", version="2.0.0")

# Serve static files
app.mount("/static", StaticFiles(directory="static"), name="static")


# ── Routes: UI ───────────────────────────────────────────────────────────────


@app.get("/")
async def serve_ui():
    """Serve the main HTML page."""
    return FileResponse("static/index.html")


# ── Routes: Meetings ─────────────────────────────────────────────────────────


@app.get("/api/meetings")
async def list_meetings():
    """List all meetings."""
    meetings = _load_meetings()
    return [{"id": mid, **mdata} for mid, mdata in meetings.items()]


@app.get("/api/meetings/{meeting_id}")
async def get_meeting(meeting_id: str):
    """Get a single meeting's metadata."""
    meetings = _load_meetings()
    if meeting_id not in meetings:
        raise HTTPException(status_code=404, detail="Meeting not found")
    return {"id": meeting_id, **meetings[meeting_id]}


@app.delete("/api/meetings/{meeting_id}")
async def delete_meeting(meeting_id: str):
    """Delete a meeting and its history."""
    meetings = _load_meetings()
    if meeting_id in meetings:
        del meetings[meeting_id]
        _save_meetings(meetings)

    # Delete history file
    hp = _history_path(meeting_id)
    if hp.exists():
        hp.unlink()

    return {"ok": True}


# ── Routes: Ingest ───────────────────────────────────────────────────────────


class IngestRequest(BaseModel):
    title: str
    text: str
    attendees: list[str] = []


@app.post("/api/meetings/ingest")
async def ingest_meeting(req: IngestRequest):
    """Ingest transcript text into the vector store."""
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Transcript text is empty")

    logger.info(f"Ingesting meeting: '{req.title}' ({len(req.text)} chars)")

    meta = await ingest_text(
        text=req.text,
        title=req.title,
        attendees=req.attendees,
    )

    # Save meeting metadata
    meetings = _load_meetings()
    meetings[meta.id] = {
        "title": meta.title,
        "date": datetime.now().isoformat(),
        "attendees": meta.attendees,
    }
    _save_meetings(meetings)

    logger.info(f"Ingested meeting: id={meta.id}")

    return {
        "id": meta.id,
        "title": meta.title,
        "date": meetings[meta.id]["date"],
        "attendees": meta.attendees,
    }


@app.post("/api/meetings/ingest-file")
async def ingest_meeting_file(
    file: UploadFile = File(...),
    title: str = Form(""),
    attendees: str = Form(""),
):
    """Ingest a transcript file (.txt/.md/.csv)."""
    content = await file.read()
    text = content.decode("utf-8")

    if not text.strip():
        raise HTTPException(status_code=400, detail="File is empty")

    meeting_title = title or file.filename.rsplit(".", 1)[0]
    attendee_list = [a.strip() for a in attendees.split(",") if a.strip()] if attendees else []

    logger.info(f"Ingesting file '{file.filename}' as '{meeting_title}'")

    meta = await ingest_text(
        text=text,
        title=meeting_title,
        attendees=attendee_list,
    )

    # Save meeting metadata
    meetings = _load_meetings()
    meetings[meta.id] = {
        "title": meta.title,
        "date": datetime.now().isoformat(),
        "attendees": meta.attendees,
    }
    _save_meetings(meetings)

    return {
        "id": meta.id,
        "title": meta.title,
        "date": meetings[meta.id]["date"],
        "attendees": meta.attendees,
    }


@app.post("/api/meetings/ingest-audio")
async def ingest_meeting_audio(
    file: UploadFile = File(...),
    title: str = Form(""),
    attendees: str = Form(""),
):
    """Ingest an audio file (mp3, wav, m4a, webm, ogg) — transcribes via Whisper then ingests."""
    from src.ingestion.pipeline import ingest_audio

    allowed_extensions = {".mp3", ".wav", ".m4a", ".webm", ".ogg", ".flac", ".mp4"}
    filename = file.filename or "audio.wav"
    ext = Path(filename).suffix.lower()

    if ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported audio format '{ext}'. Supported: {', '.join(sorted(allowed_extensions))}",
        )

    # Save uploaded file to a temp location
    temp_dir = DATA_DIR / "temp"
    temp_dir.mkdir(exist_ok=True)
    temp_path = temp_dir / f"upload_{filename}"

    try:
        content = await file.read()
        if not content:
            raise HTTPException(status_code=400, detail="Audio file is empty")

        temp_path.write_bytes(content)
        logger.info(f"Saved audio to {temp_path} ({len(content)} bytes)")

        meeting_title = title or filename.rsplit(".", 1)[0]
        attendee_list = (
            [a.strip() for a in attendees.split(",") if a.strip()]
            if attendees
            else []
        )

        logger.info(f"Transcribing audio '{filename}' as '{meeting_title}'")

        # Run the full audio pipeline: transcribe → chunk → embed → store
        meta = await ingest_audio(
            audio_path=temp_path,
            title=meeting_title,
            attendees=attendee_list,
        )

        # Save meeting metadata
        meetings = _load_meetings()
        meetings[meta.id] = {
            "title": meta.title,
            "date": datetime.now().isoformat(),
            "attendees": meta.attendees,
            "duration_seconds": meta.duration_seconds,
            "source": "audio",
        }
        _save_meetings(meetings)

        logger.info(f"Audio ingested: id={meta.id}, duration={meta.duration_seconds}s")

        return {
            "id": meta.id,
            "title": meta.title,
            "date": meetings[meta.id]["date"],
            "attendees": meta.attendees,
            "duration_seconds": meta.duration_seconds,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Audio ingestion failed: {e}")
        raise HTTPException(status_code=500, detail=f"Audio processing failed: {e}")
    finally:
        # Cleanup temp file
        if temp_path.exists():
            try:
                temp_path.unlink()
            except Exception:
                pass


# ── Routes: Chat History ────────────────────────────────────────────────────


@app.get("/api/meetings/{meeting_id}/history")
async def get_history(meeting_id: str):
    """Get chat history for a meeting."""
    messages = _load_history(meeting_id)
    return {"messages": messages}


# ── Routes: Chat ─────────────────────────────────────────────────────────────


class ChatRequest(BaseModel):
    meeting_id: str
    query: str


@app.post("/api/chat")
async def chat(req: ChatRequest):
    """Send a query to the agent pipeline and return the result."""
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query is empty")

    meetings = _load_meetings()
    if req.meeting_id not in meetings:
        raise HTTPException(status_code=404, detail="Meeting not found")

    logger.info(f"Chat: meeting={req.meeting_id[:8]}, query='{req.query[:60]}'")

    # Run the agent pipeline (handles caching internally)
    result = await run_pipeline(req.meeting_id, req.query)

    output = result.get("output", {})
    query_type = result.get("query_type", "general")
    thinking = result.get("thinking", [])

    # Convert query_type enum to string if needed
    qt = query_type.value if hasattr(query_type, "value") else str(query_type)

    # Format the response
    content = _format_response(qt, output)

    # Save to history
    history = _load_history(req.meeting_id)
    history.append({
        "role": "user",
        "content": req.query,
        "timestamp": datetime.now().isoformat(),
    })
    history.append({
        "role": "assistant",
        "content": content,
        "traces": thinking,
        "query_type": qt,
        "timestamp": datetime.now().isoformat(),
    })
    _save_history(req.meeting_id, history)

    return {
        "content": content,
        "query_type": qt,
        "traces": thinking,
    }


# ── Response formatter ───────────────────────────────────────────────────────


def _format_response(query_type: str, output: dict[str, Any]) -> str:
    """Format pipeline output into readable markdown text."""
    parts: list[str] = []

    if query_type == "summary":
        tldr = output.get("tldr", "")
        key_points = output.get("key_points", [])
        if tldr:
            parts.append(f"## 📋 TL;DR\n{tldr}")
        if key_points:
            parts.append("\n## Key Points")
            for kp in key_points:
                parts.append(f"- {kp}")

    elif query_type == "action_items":
        items = output.get("action_items", [])
        if items:
            parts.append(f"## ✅ Action Items ({len(items)})")
            for item in items:
                task = item.get("task", "")
                owner = item.get("owner")
                deadline = item.get("deadline")
                line = f"- **{task}**"
                if owner:
                    line += f"  →  {owner}"
                if deadline:
                    line += f"  📅 {deadline}"
                parts.append(line)
        else:
            parts.append("No action items found in this meeting.")

    elif query_type == "decisions":
        decs = output.get("decisions", [])
        if decs:
            parts.append(f"## 🎯 Decisions ({len(decs)})")
            for d in decs:
                desc = d.get("description", "")
                ctx = d.get("context", "")
                parts.append(f"- **{desc}**")
                if ctx:
                    parts.append(f"  > {ctx}")
        else:
            parts.append("No decisions found in this meeting.")

    else:
        # QA / general
        answer = output.get("answer", output.get("tldr", ""))
        if not answer:
            answer = str(output) if output else "I couldn't find an answer to that question."
        parts.append(answer)

        sources = output.get("sources", [])
        if sources:
            parts.append("\n---\n**📚 Sources:**")
            for s in sources:
                speaker = s.get("speaker", "Unknown")
                preview = s.get("text_preview", "")[:100]
                parts.append(f"- **[{speaker}]** {preview}")

    return "\n".join(parts) if parts else "No output generated."


# ── Run directly ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
