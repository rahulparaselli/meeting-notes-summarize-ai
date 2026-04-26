from src.core.models import Chunk
from src.rag.compressor import _extractive_compress


def _make_chunk(text: str, speaker: str = "Alice", start: float = 0.0) -> Chunk:
    return Chunk(
        id="c1", text=text, speaker=speaker,
        start_time=start, end_time=start + 5,
        meeting_id="m1", chunk_index=0,
    )


def test_extractive_compress_keeps_relevant_sentences():
    chunk = _make_chunk(
        "The budget was approved. We discussed the timeline. Action items were assigned."
    )
    result = _extractive_compress([chunk], query="budget approval")
    assert "budget" in result.lower()


def test_extractive_compress_fallback_keeps_first_sentences():
    chunk = _make_chunk("Completely unrelated text here. Nothing matches query at all.")
    result = _extractive_compress([chunk], query="xyzabc123")
    assert len(result) > 0


def test_extractive_compress_includes_speaker():
    chunk = _make_chunk("We agreed on the new design.", speaker="Bob", start=30.0)
    result = _extractive_compress([chunk], query="design")
    assert "[Bob]" in result
    assert "(30s)" in result
