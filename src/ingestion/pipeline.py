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
    """Full pipeline: audio → chunks → vector store."""
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


async def ingest_text(
    text: str,
    title: str,
    meeting_id: str | None = None,
    attendees: list[str] | None = None,
    store: VectorStore | None = None,
) -> MeetingMetadata:
    """Ingest plain transcript text directly."""
    mid = meeting_id or str(uuid.uuid4())[:16]
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
