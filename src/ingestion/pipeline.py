"""Ingestion pipeline — transcript text or audio → chunked + embedded → vector store."""
import re
import uuid
from datetime import datetime
from pathlib import Path

from src.core.models import MeetingMetadata
from src.ingestion.chunker import chunk_segments
from src.ingestion.diarizer import assign_speaker, diarize
from src.ingestion.transcriber import file_hash, transcribe
from src.rag.vector_store import VectorStore


async def ingest_audio(
    audio_path: Path,
    title: str,
    attendees: list[str] | None = None,
    hf_token: str = "",
    store: VectorStore | None = None,
) -> MeetingMetadata:
    """Full pipeline: audio -> chunks -> vector store."""
    meeting_id = file_hash(audio_path)

    # transcribe
    segments = transcribe(audio_path)

    # diarize (optional — degrades gracefully)
    speakers = diarize(audio_path, hf_token) if hf_token else []
    for seg in segments:
        seg["speaker"] = assign_speaker(seg["start"], seg["end"], speakers) or "Unknown"

    # chunk
    chunks = chunk_segments(segments, meeting_id)

    # embed + store
    vs = store or VectorStore()
    await vs.add_chunks(chunks)

    duration = segments[-1]["end"] if segments else None
    meta = MeetingMetadata(
        id=meeting_id,
        title=title,
        date=datetime.now(),
        attendees=attendees or [],
        duration_seconds=duration,
        source_file=str(audio_path),
    )
    return meta


def _parse_transcript_to_segments(text: str) -> list[dict]:
    """Parse transcript text into speaker-separated segments.

    Handles formats like:
      - "Alice: Hello everyone"
      - "Bob: Thanks"
      - Plain text without speaker labels
    """
    segments = []
    # Try to detect speaker pattern: "Name:" at start of line
    speaker_pattern = re.compile(r'^([A-Za-z][A-Za-z0-9_ ]{0,30}):\s*(.+)', re.MULTILINE)

    lines = text.strip().splitlines()
    current_speaker = None
    current_text = []
    line_idx = 0

    for line in lines:
        line = line.strip()
        if not line:
            continue

        match = speaker_pattern.match(line)
        if match:
            # Save previous segment
            if current_text:
                segments.append({
                    "text": " ".join(current_text),
                    "start": float(max(0, line_idx - len(current_text))),
                    "end": float(line_idx),
                    "speaker": current_speaker or "Unknown",
                })

            current_speaker = match.group(1).strip()
            current_text = [match.group(2).strip()]
        else:
            current_text.append(line)

        line_idx += 1

    # Flush last segment
    if current_text:
        segments.append({
            "text": " ".join(current_text),
            "start": float(max(0, line_idx - len(current_text))),
            "end": float(line_idx),
            "speaker": current_speaker or "Unknown",
        })

    # Fallback: if no speaker patterns found, split into ~3 sentence chunks
    if not segments:
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        chunk_size = 3
        for i in range(0, len(sentences), chunk_size):
            chunk = " ".join(sentences[i:i + chunk_size])
            if chunk.strip():
                segments.append({
                    "text": chunk,
                    "start": float(i),
                    "end": float(min(i + chunk_size, len(sentences))),
                    "speaker": "Unknown",
                })

    return segments


async def ingest_text(
    text: str,
    title: str,
    meeting_id: str | None = None,
    attendees: list[str] | None = None,
    store: VectorStore | None = None,
) -> MeetingMetadata:
    """Ingest plain transcript text directly.

    Parses speaker turns from the text, chunks them, embeds, and stores.
    """
    mid = meeting_id or str(uuid.uuid4())[:16]

    # Parse into speaker segments instead of stuffing everything into 1 segment
    segments = _parse_transcript_to_segments(text)

    if not segments:
        # Ultimate fallback
        segments = [{"text": text, "start": 0.0, "end": 0.0, "speaker": "Unknown"}]

    chunks = chunk_segments(segments, mid)

    vs = store or VectorStore()
    await vs.add_chunks(chunks)

    return MeetingMetadata(
        id=mid,
        title=title,
        date=datetime.now(),
        attendees=attendees or [],
        source_file=None,
    )
