import pytest
from src.agents.graph import route_to_agent
from src.core.models import QueryType


def test_route_summary():
    state = {"query_type": QueryType.SUMMARY}
    assert route_to_agent(state) == "run_summary_agent"


def test_route_action_items():
    state = {"query_type": QueryType.ACTION_ITEMS}
    assert route_to_agent(state) == "run_action_items_agent"


def test_route_decisions():
    state = {"query_type": QueryType.DECISIONS}
    assert route_to_agent(state) == "run_decisions_agent"


def test_route_qa():
    state = {"query_type": QueryType.QA}
    assert route_to_agent(state) == "run_qa_agent"


def test_route_general_falls_back_to_qa():
    state = {"query_type": QueryType.GENERAL}
    assert route_to_agent(state) == "run_qa_agent"


def test_route_missing_type_falls_back():
    state = {}
    assert route_to_agent(state) == "run_qa_agent"
