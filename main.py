"""
Meeting Summariser — AI Agent

Run with:
    streamlit run app.py

For LangSmith tracing, set LANGSMITH_API_KEY in .env
"""
from src.agents.graph import graph  # noqa: F401

__all__ = ["graph"]
