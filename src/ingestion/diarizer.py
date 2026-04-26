from pathlib import Path
from typing import Any

from src.core.models import Speaker


def diarize(audio_path: Path, hf_token: str) -> list[Speaker]:
    """Run pyannote diarization. Returns speaker segments."""
    try:
        from pyannote.audio import Pipeline  # type: ignore[import-untyped]
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=hf_token,
        )
        diarization = pipeline(str(audio_path))

        speakers: dict[str, Speaker] = {}
        for turn, _, speaker_label in diarization.itertracks(yield_label=True):
            if speaker_label not in speakers:
                speakers[speaker_label] = Speaker(name=speaker_label)
            speakers[speaker_label].segments.append((turn.start, turn.end))

        return list(speakers.values())

    except ImportError:
        # pyannote not available — return empty (graceful degradation)
        return []


def assign_speaker(
    start: float, end: float, speakers: list[Speaker]
) -> str | None:
    """Match a timestamp range to the most overlapping speaker."""
    best_speaker: str | None = None
    best_overlap = 0.0

    for speaker in speakers:
        for seg_start, seg_end in speaker.segments:
            overlap = max(0.0, min(end, seg_end) - max(start, seg_start))
            if overlap > best_overlap:
                best_overlap = overlap
                best_speaker = speaker.name

    return best_speaker
