"""
LangGraph agent pipeline for meeting analysis.

  classify_query → expand_and_retrieve → compress_context → specialist_agent

Called directly from the Streamlit app via run_pipeline / run_pipeline_stream.
"""
import logging
import operator
from typing import Annotated, Any, Literal, TypedDict

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
from src.core.models import AgentState, Chunk, QueryType
from src.rag.compressor import compress_chunks
from src.rag.query_expander import expand_query_hyde
from src.rag.vector_store import VectorStore

logger = logging.getLogger(__name__)


# ── Graph state with proper reducers ─────────────────────────────────────────

class GraphState(TypedDict, total=False):
    meeting_id: str
    query: str
    query_type: str
    retrieved_chunks: list[dict[str, Any]]
    compressed_context: str
    agent_scratchpad: Annotated[list[dict[str, str]], operator.add]
    final_output: dict[str, Any]
    steps_taken: Annotated[list[str], operator.add]
    error: str


# ── node: classify ────────────────────────────────────────────────────────────

class _Classification(BaseModel):
    query_type: str
    reasoning: str


async def classify_query(state: GraphState) -> dict[str, Any]:
    query = state.get("query", "")

    try:
        result = await generate_structured(
            prompt=QUERY_CLASSIFIER_PROMPT.format(query=query),
            schema=_Classification,
            system=ORCHESTRATOR_SYSTEM,
        )
        try:
            qtype = QueryType(result.query_type)
        except ValueError:
            qtype = QueryType.GENERAL

        return {
            "query_type": qtype.value,
            "steps_taken": ["classify_query"],
            "agent_scratchpad": [
                {"step": "classify_query", "detail": f"Classifying: '{query[:80]}'"},
                {"step": "classify_result", "detail": f"Type={qtype.value} | {result.reasoning}"},
            ],
        }
    except Exception as e:
        logger.error(f"Classification failed: {e}")
        return {
            "query_type": "general",
            "steps_taken": ["classify_query"],
            "agent_scratchpad": [
                {"step": "classify_query", "detail": f"Classification failed: {e} — defaulting to QA"},
            ],
        }


# ── node: expand + retrieve ───────────────────────────────────────────────────

async def expand_and_retrieve(state: GraphState) -> dict[str, Any]:
    query = state.get("query", "")
    meeting_id = state.get("meeting_id", "")

    try:
        vs = VectorStore()

        # HyDE expansion (falls back to raw query on failure)
        expanded_query = await expand_query_hyde(query)
        logger.info(f"Expanded query length: {len(expanded_query)} chars")

        # MMR retrieval
        chunks = await vs.mmr_search(expanded_query, meeting_id=meeting_id)
        logger.info(f"Retrieved {len(chunks)} chunks for meeting {meeting_id}")

        # If no chunks found with MMR, try direct search
        if not chunks:
            logger.warning("MMR returned 0 chunks — trying direct search")
            chunks = await vs.search(query, meeting_id=meeting_id)

        # If still no chunks, try without meeting_id filter
        if not chunks:
            logger.warning(f"No chunks for meeting_id={meeting_id} — searching all meetings")
            chunks = await vs.search(query, meeting_id="")

        # Serialize chunks to dicts for state storage
        chunk_dicts = [
            {
                "id": c.id,
                "text": c.text,
                "speaker": c.speaker,
                "start_time": c.start_time,
                "end_time": c.end_time,
                "meeting_id": c.meeting_id,
                "chunk_index": c.chunk_index,
            }
            for c in chunks
        ]

        detail = f"Retrieved {len(chunks)} chunks via MMR"
        if not chunks:
            detail = "WARNING: No chunks found — vector store may be empty or meeting_id doesn't match"

        return {
            "retrieved_chunks": chunk_dicts,
            "steps_taken": ["expand_and_retrieve"],
            "agent_scratchpad": [
                {"step": "hyde_expansion", "detail": f"Expanded query: {len(expanded_query)} chars"},
                {"step": "retrieval_result", "detail": detail},
            ],
        }
    except Exception as e:
        logger.error(f"Retrieval failed: {e}")
        return {
            "retrieved_chunks": [],
            "steps_taken": ["expand_and_retrieve"],
            "agent_scratchpad": [
                {"step": "retrieval_error", "detail": f"Retrieval failed: {e}"},
            ],
        }


# ── node: compress ────────────────────────────────────────────────────────────

async def compress_context(state: GraphState) -> dict[str, Any]:
    query = state.get("query", "")
    chunk_dicts = state.get("retrieved_chunks", [])

    try:
        compressed = ""
        if chunk_dicts:
            chunk_objs = [Chunk(**cd) for cd in chunk_dicts]
            compressed = compress_chunks(chunk_objs, query)
        else:
            logger.warning("No chunks to compress — context will be empty")

        return {
            "compressed_context": compressed,
            "steps_taken": ["compress_context"],
            "agent_scratchpad": [
                {"step": "compress_context", "detail": f"Input: {len(chunk_dicts)} chunks"},
                {"step": "compression_done", "detail": f"Output: {len(compressed)} chars"},
            ],
        }
    except Exception as e:
        logger.error(f"Compression failed: {e}")
        # Fallback: concatenate raw chunks
        fallback = "\n\n".join(cd.get("text", "") for cd in chunk_dicts)
        return {
            "compressed_context": fallback,
            "steps_taken": ["compress_context"],
            "agent_scratchpad": [
                {"step": "compression_error", "detail": f"Compression failed: {e} — using raw text"},
            ],
        }


# ── specialist agent wrappers ────────────────────────────────────────────────

def _build_agent_state(state: GraphState) -> AgentState:
    """Convert graph state to internal AgentState for specialist agents."""
    chunk_dicts = state.get("retrieved_chunks", [])
    chunks = [Chunk(**cd) for cd in chunk_dicts] if chunk_dicts else []

    return AgentState(
        meeting_id=state.get("meeting_id", ""),
        query=state.get("query", ""),
        query_type=QueryType(state.get("query_type", "general")),
        retrieved_chunks=chunks,
        compressed_context=state.get("compressed_context", ""),
        agent_scratchpad=[],
        steps_taken=[],
    )


async def _run_agent_safe(state: GraphState, agent_fn, agent_name: str) -> dict[str, Any]:
    """Run a specialist agent with error handling."""
    try:
        agent_state = _build_agent_state(state)
        updates = await agent_fn(agent_state)
        return {
            "final_output": updates.get("final_output", {}),
            "steps_taken": updates.get("steps_taken", []),
            "agent_scratchpad": updates.get("agent_scratchpad", []),
        }
    except Exception as e:
        logger.error(f"{agent_name} failed: {e}")
        return {
            "final_output": {"answer": f"Agent error: {e}", "error": True},
            "steps_taken": [agent_name],
            "agent_scratchpad": [
                {"step": f"{agent_name}_error", "detail": str(e)},
            ],
        }


async def run_summary_agent(state: GraphState) -> dict[str, Any]:
    return await _run_agent_safe(state, summary_agent, "summary_agent")


async def run_action_items_agent(state: GraphState) -> dict[str, Any]:
    return await _run_agent_safe(state, action_items_agent, "action_items_agent")


async def run_decisions_agent(state: GraphState) -> dict[str, Any]:
    return await _run_agent_safe(state, decisions_agent, "decisions_agent")


async def run_qa_agent(state: GraphState) -> dict[str, Any]:
    return await _run_agent_safe(state, qa_agent, "qa_agent")


# ── conditional router ────────────────────────────────────────────────────────

def route_to_agent(
    state: GraphState,
) -> Literal["run_summary_agent", "run_action_items_agent", "run_decisions_agent", "run_qa_agent"]:
    qtype = state.get("query_type", "general")
    routes = {
        "summary": "run_summary_agent",
        "action_items": "run_action_items_agent",
        "decisions": "run_decisions_agent",
        "qa": "run_qa_agent",
        "general": "run_qa_agent",
    }
    return routes.get(qtype, "run_qa_agent")  # type: ignore[return-value]


# ── build graph ───────────────────────────────────────────────────────────────

def build_graph() -> Any:
    g = StateGraph(GraphState)

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
    """Entry point — run the full agent pipeline and return result."""
    from src.rag.cache import get_cached, set_cached

    cached = get_cached(meeting_id, query)
    if cached:
        logger.info(f"Cache HIT for meeting={meeting_id[:8]}, query='{query[:40]}'")
        return cached

    logger.info(f"Running pipeline: meeting={meeting_id[:8]}, query='{query[:40]}'")

    initial_state: GraphState = {
        "meeting_id": meeting_id,
        "query": query,
        "query_type": "general",
        "retrieved_chunks": [],
        "compressed_context": "",
        "agent_scratchpad": [],
        "final_output": {},
        "steps_taken": [],
    }

    final_state = await graph.ainvoke(initial_state)
    result: dict[str, Any] = {
        "output": final_state.get("final_output", {}),
        "query_type": final_state.get("query_type"),
        "steps": final_state.get("steps_taken", []),
        "thinking": final_state.get("agent_scratchpad", []),
    }

    set_cached(meeting_id, query, result)
    logger.info(f"Pipeline complete: {len(result.get('steps', []))} steps")
    return result


async def run_pipeline_stream(meeting_id: str, query: str):
    """Stream each LangGraph node as it completes."""
    initial_state: GraphState = {
        "meeting_id": meeting_id,
        "query": query,
        "query_type": "general",
        "retrieved_chunks": [],
        "compressed_context": "",
        "agent_scratchpad": [],
        "final_output": {},
        "steps_taken": [],
    }

    final_state = dict(initial_state)
    async for event in graph.astream(initial_state, stream_mode="updates"):
        for node_name, node_output in event.items():
            final_state.update(node_output)
            scratchpad = node_output.get("agent_scratchpad", [])
            yield node_name, scratchpad, node_output

    yield "__done__", [], final_state
