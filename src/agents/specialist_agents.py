"""Each function is a LangGraph node. Receives AgentState, returns partial state update."""
import json
import logging
import re
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

logger = logging.getLogger(__name__)

# ── Max-token budgets per specialist (avoid default 4096 overkill) ────────────
_SUMMARY_MAX_TOKENS = 1024
_ACTION_ITEMS_MAX_TOKENS = 1024
_DECISIONS_MAX_TOKENS = 1024
_QA_MAX_TOKENS = 1024


def _log(state: AgentState, step: str, detail: str = "") -> None:
    state.steps_taken.append(step)
    state.agent_scratchpad.append({"step": step, "detail": detail})


# ── Summary Agent ─────────────────────────────────────────────────────────────

async def summary_agent(state: AgentState) -> dict[str, Any]:
    _log(state, "summary_agent", "Generating meeting TL;DR and key points")

    ctx = state.compressed_context or "\n\n".join(c.text for c in state.retrieved_chunks)
    raw = await generate_text(
        prompt=SUMMARY_PROMPT.format(context=ctx),
        system=SUMMARY_SYSTEM,
        max_tokens=_SUMMARY_MAX_TOKENS,
    )

    tldr = _extract_tldr(raw)
    key_points = _extract_bullets(raw)

    output = MeetingSummary(
        tldr=tldr,
        key_points=key_points,
        action_items=[],
        decisions=[],
        attendees=[],
    )

    _log(state, "summary_done", f"TL;DR: {tldr[:80]}... | {len(key_points)} key points")

    return {
        "final_output": output.model_dump(),
        "steps_taken": state.steps_taken,
        "agent_scratchpad": state.agent_scratchpad,
    }


# ── Action Items Agent ────────────────────────────────────────────────────────

async def action_items_agent(state: AgentState) -> dict[str, Any]:
    _log(state, "action_items_agent", "Extracting tasks, owners, and deadlines")

    ctx = state.compressed_context or "\n\n".join(c.text for c in state.retrieved_chunks)

    from src.core.llm import generate_structured
    from pydantic import BaseModel

    class _Items(BaseModel):
        items: list[ActionItem]

    try:
        result = await generate_structured(
            prompt=ACTION_ITEMS_PROMPT.format(context=ctx),
            schema=_Items,
            system=ACTION_ITEMS_SYSTEM,
            max_tokens=_ACTION_ITEMS_MAX_TOKENS,
        )
        items = [i.model_dump() for i in result.items]
    except Exception as e:
        logger.warning(f"Structured generation failed for action_items: {e} — parsing as text")
        # Fallback: try text generation and parse manually
        raw = await generate_text(
            prompt=ACTION_ITEMS_PROMPT.format(context=ctx),
            system=ACTION_ITEMS_SYSTEM,
            max_tokens=_ACTION_ITEMS_MAX_TOKENS,
        )
        items = _parse_json_items(raw, "items")

    _log(state, "action_items_done", f"Found {len(items)} action items")

    return {
        "final_output": {"action_items": items},
        "steps_taken": state.steps_taken,
        "agent_scratchpad": state.agent_scratchpad,
    }


# ── Decisions Agent ───────────────────────────────────────────────────────────

async def decisions_agent(state: AgentState) -> dict[str, Any]:
    _log(state, "decisions_agent", "Identifying confirmed decisions from discussion")

    ctx = state.compressed_context or "\n\n".join(c.text for c in state.retrieved_chunks)

    from src.core.llm import generate_structured
    from pydantic import BaseModel

    class _Decisions(BaseModel):
        decisions: list[Decision]

    try:
        result = await generate_structured(
            prompt=DECISIONS_PROMPT.format(context=ctx),
            schema=_Decisions,
            system=DECISIONS_SYSTEM,
            max_tokens=_DECISIONS_MAX_TOKENS,
        )
        decs = [d.model_dump() for d in result.decisions]
    except Exception as e:
        logger.warning(f"Structured generation failed for decisions: {e} — parsing as text")
        # Fallback: try text generation and parse manually
        raw = await generate_text(
            prompt=DECISIONS_PROMPT.format(context=ctx),
            system=DECISIONS_SYSTEM,
            max_tokens=_DECISIONS_MAX_TOKENS,
        )
        decs = _parse_json_items(raw, "decisions")

    _log(state, "decisions_done", f"Found {len(decs)} decisions")

    return {
        "final_output": {"decisions": decs},
        "steps_taken": state.steps_taken,
        "agent_scratchpad": state.agent_scratchpad,
    }


# ── Q&A Agent ─────────────────────────────────────────────────────────────────

async def qa_agent(state: AgentState) -> dict[str, Any]:
    _log(state, "qa_agent", f"Answering: '{state.query[:60]}'")

    ctx = state.compressed_context or "\n\n".join(c.text for c in state.retrieved_chunks)
    raw = await generate_text(
        prompt=QA_PROMPT.format(context=ctx, query=state.query),
        system=QA_SYSTEM,
        max_tokens=_QA_MAX_TOKENS,
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

    _log(state, "qa_done", f"Answer length: {len(raw)} chars, {len(sources)} sources")

    return {
        "final_output": cited.model_dump(),
        "steps_taken": state.steps_taken,
        "agent_scratchpad": state.agent_scratchpad,
    }


# ── Helpers ──────────────────────────────────────────────────────────────────

def _extract_tldr(text: str) -> str:
    """Extract TL;DR from formatted output."""
    # Look for "TL;DR:" prefix
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.upper().startswith("TL;DR"):
            return re.sub(r"^TL;DR\s*:?\s*", "", stripped, flags=re.IGNORECASE).strip()

    # Fallback: first non-empty, non-bullet line
    for line in text.splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith(("-", "*", "•", "#", "KEY")):
            return stripped
    return text[:300]


def _extract_bullets(text: str) -> list[str]:
    """Extract bullet points from formatted output."""
    bullets = []
    in_key_points = False
    for line in text.splitlines():
        stripped = line.strip()

        # Start capturing after KEY POINTS header
        if "KEY POINT" in stripped.upper() or "KEY DISCUSSION" in stripped.upper():
            in_key_points = True
            continue

        if stripped.startswith(("-", "•", "*", "·")):
            bullet = stripped.lstrip("-•*· ").strip()
            if bullet:
                bullets.append(bullet)
                in_key_points = True  # we found bullets

        # Numbered items like "1. something"
        elif re.match(r"^\d+[.)]\s+", stripped):
            bullet = re.sub(r"^\d+[.)]\s+", "", stripped).strip()
            if bullet:
                bullets.append(bullet)

    return bullets if bullets else [text[:300]]


def _parse_json_items(raw: str, key: str) -> list[dict]:
    """Try to parse JSON from raw LLM text output as fallback."""
    try:
        # Try direct JSON parse
        data = json.loads(raw)
        if isinstance(data, dict) and key in data:
            return data[key]
        if isinstance(data, list):
            return data
    except json.JSONDecodeError:
        pass

    # Try to find JSON block in the text
    json_match = re.search(r"\{[\s\S]*\}", raw)
    if json_match:
        try:
            data = json.loads(json_match.group())
            if isinstance(data, dict) and key in data:
                return data[key]
        except json.JSONDecodeError:
            pass

    # Try to find JSON array
    arr_match = re.search(r"\[[\s\S]*\]", raw)
    if arr_match:
        try:
            return json.loads(arr_match.group())
        except json.JSONDecodeError:
            pass

    return []
