import pytest
from unittest.mock import AsyncMock, patch

from src.ingestion.chunker import chunk_segments


def test_chunk_segments_basic():
    segments = [
        {"text": "Let us discuss the budget. We need to allocate funds.", "start": 0.0, "end": 10.0, "speaker": "Alice"},
        {"text": "I agree. The Q4 budget should be finalised this week.", "start": 10.0, "end": 20.0, "speaker": "Bob"},
    ]
    chunks = chunk_segments(segments, meeting_id="test-001", chunk_size=50)
    assert len(chunks) >= 1
    for c in chunks:
        assert c.meeting_id == "test-001"
        assert c.text


def test_chunk_speaker_boundary():
    """Speaker change should trigger a new chunk."""
    segments = [
        {"text": "Hello everyone. " * 5, "start": 0.0, "end": 10.0, "speaker": "Alice"},
        {"text": "Thanks Alice. " * 5, "start": 10.0, "end": 20.0, "speaker": "Bob"},
    ]
    chunks = chunk_segments(segments, meeting_id="test-002", chunk_size=500)
    speakers = [c.speaker for c in chunks]
    # both speakers should appear
    assert "Alice" in speakers
    assert "Bob" in speakers


def test_chunk_preserves_metadata():
    segments = [{"text": "Test sentence.", "start": 5.0, "end": 8.0, "speaker": "Carol"}]
    chunks = chunk_segments(segments, meeting_id="meta-test")
    assert chunks[0].start_time == 5.0
    assert chunks[0].speaker == "Carol"
