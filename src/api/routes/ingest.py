import tempfile
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

from src.ingestion.pipeline import ingest_audio, ingest_text

router = APIRouter()


class IngestTextRequest(BaseModel):
    text: str
    title: str
    attendees: list[str] = []
    meeting_id: str | None = None


class IngestResponse(BaseModel):
    meeting_id: str
    title: str
    chunks_ingested: bool = True


@router.post("/ingest/text", response_model=IngestResponse)
async def ingest_text_endpoint(body: IngestTextRequest) -> IngestResponse:
    meta = await ingest_text(
        text=body.text,
        title=body.title,
        meeting_id=body.meeting_id,
        attendees=body.attendees,
    )
    return IngestResponse(meeting_id=meta.id, title=meta.title)


@router.post("/ingest/audio", response_model=IngestResponse)
async def ingest_audio_endpoint(
    file: UploadFile = File(...),
    title: str = Form(...),
    attendees: str = Form(""),
    hf_token: str = Form(""),
) -> IngestResponse:
    if file.content_type not in ("audio/mpeg", "audio/wav", "audio/x-wav", "audio/mp4"):
        raise HTTPException(status_code=415, detail="Unsupported audio format")

    with tempfile.NamedTemporaryFile(suffix=Path(file.filename or "audio.mp3").suffix, delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = Path(tmp.name)

    try:
        attendee_list = [a.strip() for a in attendees.split(",") if a.strip()]
        meta = await ingest_audio(
            audio_path=tmp_path,
            title=title,
            attendees=attendee_list,
            hf_token=hf_token,
        )
    finally:
        tmp_path.unlink(missing_ok=True)

    return IngestResponse(meeting_id=meta.id, title=meta.title)
