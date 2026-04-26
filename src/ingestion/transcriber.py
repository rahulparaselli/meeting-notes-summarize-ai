import hashlib
from pathlib import Path

import whisper  # type: ignore[import-untyped]

from src.core.config import get_settings

_settings = get_settings()
_model: whisper.Whisper | None = None


def _get_model() -> whisper.Whisper:
    global _model
    if _model is None:
        _model = whisper.load_model(_settings.whisper_model)
    return _model


def transcribe(audio_path: Path) -> list[dict]:
    """Return list of {text, start, end} segments."""
    model = _get_model()
    result = model.transcribe(str(audio_path), word_timestamps=False, verbose=False)
    return [
        {"text": seg["text"].strip(), "start": seg["start"], "end": seg["end"]}
        for seg in result["segments"]
    ]


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8192), b""):
            h.update(block)
    return h.hexdigest()[:16]
