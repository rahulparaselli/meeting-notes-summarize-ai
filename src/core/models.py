from datetime import datetime
from enum import Enum
from typing import Any
from pydantic import BaseModel, Field


class QueryType(str, Enum):
    SUMMARY = "summary"
    ACTION_ITEMS = "action_items"
    QA = "qa"
    DECISIONS = "decisions"
    GENERAL = "general"


class Speaker(BaseModel):
    name: str
    segments: list[tuple[float, float]] = Field(default_factory=list)  # (start, end)


class Chunk(BaseModel):
    id: str
    text: str
    speaker: str | None = None
    start_time: float | None = None
    end_time: float | None = None
    meeting_id: str
    chunk_index: int
    metadata: dict[str, Any] = Field(default_factory=dict)


class ActionItem(BaseModel):
    task: str
    owner: str | None = None
    deadline: str | None = None
    source_chunk_id: str | None = None


class Decision(BaseModel):
    description: str
    context: str
    participants: list[str] = Field(default_factory=list)
    source_chunk_id: str | None = None


class MeetingMetadata(BaseModel):
    id: str
    title: str
    date: datetime
    attendees: list[str] = Field(default_factory=list)
    duration_seconds: float | None = None
    source_file: str | None = None


class CitedAnswer(BaseModel):
    answer: str
    sources: list[dict[str, Any]] = Field(default_factory=list)  # chunk refs
    confidence: float = 1.0


class MeetingSummary(BaseModel):
    tldr: str
    key_points: list[str]
    action_items: list[ActionItem]
    decisions: list[Decision]
    attendees: list[str]
    duration_summary: str | None = None


# LangGraph state
class AgentState(BaseModel):
    meeting_id: str
    query: str
    query_type: QueryType = QueryType.GENERAL
    retrieved_chunks: list[Chunk] = Field(default_factory=list)
    compressed_context: str = ""
    agent_scratchpad: list[dict[str, str]] = Field(default_factory=list)  # thinking log
    final_output: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None
    steps_taken: list[str] = Field(default_factory=list)
