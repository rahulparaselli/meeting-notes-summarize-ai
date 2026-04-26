"""
LangGraph agent pipeline:

  START
    │
    ▼
  classify_query          ← orchestrator decides query type
    │
    ▼
  expand_and_retrieve     ← HyDE + MMR retrieval
    │
    ▼
  compress_context        ← LLMLingua / extractive compression
    │
    ▼
  route_to_agent          ← conditional branch
   ├─ summary    → summary_agent
   ├─ action_items → action_items_agent
   ├─ decisions  → decisions_agent
   └─ qa / general → qa_agent
    │
    ▼
  END
"""
from typing import Any, Literal

from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel

from src.agents.prompts import ORCHESTRATOR_SYSTEM, QUERY_CLASSIFIER_PROMPT
from src.agents.specialist_agents import (
    action_items_agent,
    decisions_agent,
    qa_agent,
    summary_agent,
)
from src.core.llm import generate_structured
from src.core.models import AgentState, QueryType
from src.rag.compressor import compress_chunks
from src.rag.query_expander import expand_query_hyde
from src.rag.vector_store import VectorStore


# ── node: classify ────────────────────────────────────────────────────────────

class _Classification(BaseModel):
    query_type: str
    reasoning: str


async def classify_query(state: dict[str, Any]) -> dict[str, Any]:
    s = AgentState(**state)
    s.steps_taken.append("classify_query")
    s.agent_scratchpad.append({"step": "classify_query", "detail": f"Classifying: '{s.query[:80]}'"})

    result = await generate_structured(
        prompt=QUERY_CLASSIFIER_PROMPT.format(query=s.query),
        schema=_Classification,
        system=ORCHESTRATOR_SYSTEM,
    )

    try:
        qtype = QueryType(result.query_type)
    except ValueError:
        qtype = QueryType.GENERAL

    s.query_type = qtype
    s.agent_scratchpad.append({
        "step": "classify_result",
        "detail": f"Type={qtype.value} | {result.reasoning}"
    })
    return s.model_dump()


# ── node: expand + retrieve ───────────────────────────────────────────────────

async def expand_and_retrieve(state: dict[str, Any]) -> dict[str, Any]:
    s = AgentState(**state)
    s.steps_taken.append("expand_and_retrieve")
    s.agent_scratchpad.append({"step": "expand_and_retrieve", "detail": "HyDE expansion + MMR retrieval"})

    vs = VectorStore()
    expanded_query = await expand_query_hyde(s.query)

    chunks = await vs.mmr_search(expanded_query, meeting_id=s.meeting_id)
    s.retrieved_chunks = chunks

    s.agent_scratchpad.append({
        "step": "retrieval_result",
        "detail": f"Retrieved {len(chunks)} diverse chunks via MMR"
    })
    return s.model_dump()


# ── node: compress ────────────────────────────────────────────────────────────

async def compress_context(state: dict[str, Any]) -> dict[str, Any]:
    s = AgentState(**state)
    s.steps_taken.append("compress_context")
    s.agent_scratchpad.append({"step": "compress_context", "detail": "Reducing context tokens"})

    if s.retrieved_chunks:
        s.compressed_context = compress_chunks(s.retrieved_chunks, s.query)

    s.agent_scratchpad.append({
        "step": "compression_done",
        "detail": f"Context: {len(s.compressed_context)} chars"
    })
    return s.model_dump()


# ── node wrappers (LangGraph needs dict I/O) ──────────────────────────────────

async def run_summary_agent(state: dict[str, Any]) -> dict[str, Any]:
    s = AgentState(**state)
    updates = await summary_agent(s)
    return {**state, **updates}


async def run_action_items_agent(state: dict[str, Any]) -> dict[str, Any]:
    s = AgentState(**state)
    updates = await action_items_agent(s)
    return {**state, **updates}


async def run_decisions_agent(state: dict[str, Any]) -> dict[str, Any]:
    s = AgentState(**state)
    updates = await decisions_agent(s)
    return {**state, **updates}


async def run_qa_agent(state: dict[str, Any]) -> dict[str, Any]:
    s = AgentState(**state)
    updates = await qa_agent(s)
    return {**state, **updates}


# ── conditional router ────────────────────────────────────────────────────────

def route_to_agent(
    state: dict[str, Any],
) -> Literal["run_summary_agent", "run_action_items_agent", "run_decisions_agent", "run_qa_agent"]:
    qtype = state.get("query_type", QueryType.GENERAL)
    routes = {
        QueryType.SUMMARY: "run_summary_agent",
        QueryType.ACTION_ITEMS: "run_action_items_agent",
        QueryType.DECISIONS: "run_decisions_agent",
        QueryType.QA: "run_qa_agent",
        QueryType.GENERAL: "run_qa_agent",
    }
    return routes.get(qtype, "run_qa_agent")  # type: ignore[return-value]


# ── build graph ───────────────────────────────────────────────────────────────

def build_graph() -> Any:
    g = StateGraph(dict)

    g.add_node("classify_query", classify_query)
    g.add_node("expand_and_retrieve", expand_and_retrieve)
    g.add_node("compress_context", compress_context)
    g.add_node("run_summary_agent", run_summary_agent)
    g.add_node("run_action_items_agent", run_action_items_agent)
    g.add_node("run_decisions_agent", run_decisions_agent)
    g.add_node("run_qa_agent", run_qa_agent)

    g.add_edge(START, "classify_query")
    g.add_edge("classify_query", "expand_and_retrieve")
    g.add_edge("expand_and_retrieve", "compress_context")
    g.add_conditional_edges(
        "compress_context",
        route_to_agent,
        {
            "run_summary_agent": "run_summary_agent",
            "run_action_items_agent": "run_action_items_agent",
            "run_decisions_agent": "run_decisions_agent",
            "run_qa_agent": "run_qa_agent",
        },
    )
    g.add_edge("run_summary_agent", END)
    g.add_edge("run_action_items_agent", END)
    g.add_edge("run_decisions_agent", END)
    g.add_edge("run_qa_agent", END)

    return g.compile()


# singleton compiled graph
graph = build_graph()


async def run_pipeline(meeting_id: str, query: str) -> dict[str, Any]:
    """Entry point for running the full agent pipeline."""
    from src.rag.cache import get_cached, set_cached

    cached = get_cached(meeting_id, query)
    if cached:
        return cached

    initial_state: dict[str, Any] = AgentState(
        meeting_id=meeting_id,
        query=query,
    ).model_dump()

    final_state = await graph.ainvoke(initial_state)
    result: dict[str, Any] = {
        "output": final_state.get("final_output", {}),
        "query_type": final_state.get("query_type"),
        "steps": final_state.get("steps_taken", []),
        "thinking": final_state.get("agent_scratchpad", []),
    }

    set_cached(meeting_id, query, result)
    return result
