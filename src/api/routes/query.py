from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from src.agents.graph import run_pipeline

router = APIRouter()


class QueryRequest(BaseModel):
    meeting_id: str
    query: str


class QueryResponse(BaseModel):
    meeting_id: str
    query: str
    query_type: str
    output: dict
    steps: list[str]
    thinking: list[dict]  # agent scratchpad — powers the UI trace


@router.post("/query", response_model=QueryResponse)
async def query_meeting(body: QueryRequest) -> QueryResponse:
    if not body.query.strip():
        raise HTTPException(status_code=422, detail="Query cannot be empty")

    result = await run_pipeline(
        meeting_id=body.meeting_id,
        query=body.query,
    )

    return QueryResponse(
        meeting_id=body.meeting_id,
        query=body.query,
        query_type=result.get("query_type", "general"),
        output=result.get("output", {}),
        steps=result.get("steps", []),
        thinking=result.get("thinking", []),
    )


@router.post("/summarise", response_model=QueryResponse)
async def summarise_meeting(meeting_id: str) -> QueryResponse:
    """Convenience endpoint — always returns full summary."""
    result = await run_pipeline(
        meeting_id=meeting_id,
        query="Give me a complete summary of this meeting.",
    )
    return QueryResponse(
        meeting_id=meeting_id,
        query="Full summary",
        query_type="summary",
        output=result.get("output", {}),
        steps=result.get("steps", []),
        thinking=result.get("thinking", []),
    )
