"""
Meeting Summariser — AI Agent

Run with:
    uvicorn server:app --reload
    (then open http://localhost:8000)

For LangSmith tracing, set LANGSMITH_API_KEY in .env
"""
from src.agents.graph import graph  # noqa: F401

__all__ = ["graph"]
