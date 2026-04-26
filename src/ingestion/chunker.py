import re
import uuid

from src.core.config import get_settings
from src.core.models import Chunk

_settings = get_settings()


def _count_tokens(text: str) -> int:
    # approx 1 token ≈ 0.75 words; avoids network calls from tiktoken
    return int(len(text.split()) * 1.33)


def _split_sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


def chunk_segments(
    segments: list[dict],
    meeting_id: str,
    chunk_size: int | None = None,
    overlap: int | None = None,
) -> list[Chunk]:
    size = chunk_size or _settings.chunk_size
    ovlp = overlap or _settings.chunk_overlap

    chunks: list[Chunk] = []
    current_sentences: list[str] = []
    current_tokens = 0
    current_start: float | None = None
    current_end: float | None = None
    current_speaker: str | None = None
    chunk_idx = 0

    def flush() -> None:
        nonlocal current_sentences, current_tokens, current_start, current_end, chunk_idx
        if not current_sentences:
            return
        text = " ".join(current_sentences)
        chunks.append(
            Chunk(
                id=str(uuid.uuid4()),
                text=text,
                speaker=current_speaker,
                start_time=current_start,
                end_time=current_end,
                meeting_id=meeting_id,
                chunk_index=chunk_idx,
            )
        )
        chunk_idx += 1
        # keep overlap sentences
        overlap_sents = current_sentences[-2:]
        overlap_tok = _count_tokens(" ".join(overlap_sents))
        current_sentences[:] = overlap_sents if overlap_tok <= ovlp else []
        current_tokens = _count_tokens(" ".join(current_sentences))
        current_start = None

    for seg in segments:
        speaker = seg.get("speaker")
        if current_speaker and speaker and speaker != current_speaker:
            flush()
            current_speaker = speaker

        if current_speaker is None:
            current_speaker = speaker

        for sentence in _split_sentences(seg["text"]):
            tok = _count_tokens(sentence)
            if current_tokens + tok > size and current_sentences:
                flush()
            current_sentences.append(sentence)
            current_tokens += tok
            if current_start is None:
                current_start = seg.get("start")
            current_end = seg.get("end")

    flush()
    return chunks
