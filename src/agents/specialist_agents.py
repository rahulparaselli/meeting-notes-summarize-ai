"""Each function is a LangGraph node. Receives AgentState, returns partial state update."""
from typing import Any

from src.agents.prompts import (
    ACTION_ITEMS_PROMPT,
    ACTION_ITEMS_SYSTEM,
    DECISIONS_PROMPT,
    DECISIONS_SYSTEM,
    QA_PROMPT,
    QA_SYSTEM,
    SUMMARY_PROMPT,
    SUMMARY_SYSTEM,
)
from src.core.llm import generate_text
from src.core.models import (
    ActionItem,
    AgentState,
    CitedAnswer,
    Decision,
    MeetingSummary,
    QueryType,
)


def _log(state: AgentState, step: str, detail: str = "") -> None:
    state.steps_taken.append(step)
    state.agent_scratchpad.append({"step": step, "detail": detail})


async def summary_agent(state: AgentState) -> dict[str, Any]:
    _log(state, "summary_agent", "Generating meeting TL;DR and key points")

    ctx = state.compressed_context or "\n\n".join(c.text for c in state.retrieved_chunks)
    raw = await generate_text(
        prompt=SUMMARY_PROMPT.format(context=ctx),
        system=SUMMARY_SYSTEM,
    )

    output = MeetingSummary(
        tldr=_extract_tldr(raw),
        key_points=_extract_bullets(raw),
        action_items=[],
        decisions=[],
        attendees=[],
    )
    return {"final_output": output.model_dump(), "steps_taken": state.steps_taken,
            "agent_scratchpad": state.agent_scratchpad}


async def action_items_agent(state: AgentState) -> dict[str, Any]:
    _log(state, "action_items_agent", "Extracting tasks, owners, and deadlines")

    ctx = state.compressed_context or "\n\n".join(c.text for c in state.retrieved_chunks)

    from src.core.llm import generate_structured
    from pydantic import BaseModel

    class _Items(BaseModel):
        items: list[ActionItem]

    result = await generate_structured(
        prompt=ACTION_ITEMS_PROMPT.format(context=ctx),
        schema=_Items,
        system=ACTION_ITEMS_SYSTEM,
    )
    return {"final_output": {"action_items": [i.model_dump() for i in result.items]},
            "steps_taken": state.steps_taken, "agent_scratchpad": state.agent_scratchpad}


async def decisions_agent(state: AgentState) -> dict[str, Any]:
    _log(state, "decisions_agent", "Identifying confirmed decisions from discussion")

    ctx = state.compressed_context or "\n\n".join(c.text for c in state.retrieved_chunks)

    from src.core.llm import generate_structured
    from pydantic import BaseModel

    class _Decisions(BaseModel):
        decisions: list[Decision]

    result = await generate_structured(
        prompt=DECISIONS_PROMPT.format(context=ctx),
        schema=_Decisions,
        system=DECISIONS_SYSTEM,
    )
    return {"final_output": {"decisions": [d.model_dump() for d in result.decisions]},
            "steps_taken": state.steps_taken, "agent_scratchpad": state.agent_scratchpad}


async def qa_agent(state: AgentState) -> dict[str, Any]:
    _log(state, "qa_agent", f"Answering: '{state.query[:60]}...'")

    ctx = state.compressed_context or "\n\n".join(c.text for c in state.retrieved_chunks)
    raw = await generate_text(
        prompt=QA_PROMPT.format(context=ctx, query=state.query),
        system=QA_SYSTEM,
    )

    sources = [
        {
            "chunk_id": c.id,
            "speaker": c.speaker,
            "start_time": c.start_time,
            "text_preview": c.text[:120],
        }
        for c in state.retrieved_chunks[:3]
    ]
    cited = CitedAnswer(answer=raw, sources=sources)
    return {"final_output": cited.model_dump(),
            "steps_taken": state.steps_taken, "agent_scratchpad": state.agent_scratchpad}


# ── helpers ──────────────────────────────────────────────────────────────────

def _extract_tldr(text: str) -> str:
    for line in text.splitlines():
        if line.strip() and not line.startswith("-"):
            return line.strip()
    return text[:200]


def _extract_bullets(text: str) -> list[str]:
    bullets = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith(("-", "•", "*", "·")):
            bullets.append(stripped.lstrip("-•*· ").strip())
    return bullets or [text[:200]]
