from __future__ import annotations

from langgraph.graph import StateGraph, END

from app.nodes.cleaning import cleaning_node
from app.nodes.discovery import discovery_node
from app.nodes.insights import insights_node
from app.nodes.planner import planner_node
from app.nodes.python_analyst import python_analyst_node
from app.nodes.report_writer import report_writer_node
from app.nodes.reviewer import reviewer_node
from app.nodes.sql_agent import sql_agent_node
from app.nodes.visualization import visualization_node
from app.state import AgentState


def build_graph():
    """Build and compile the LangGraph agent graph.

    Returns:
        A compiled LangGraph state machine wiring all agent nodes in sequence.
    """
    graph = StateGraph(AgentState)

    graph.add_node("discovery", discovery_node)
    graph.add_node("planner", planner_node)
    graph.add_node("cleaning", cleaning_node)
    graph.add_node("sql_agent", sql_agent_node)
    graph.add_node("python_analyst", python_analyst_node)
    graph.add_node("visualization", visualization_node)
    graph.add_node("insights", insights_node)
    graph.add_node("report_writer", report_writer_node)
    graph.add_node("reviewer", reviewer_node)

    graph.set_entry_point("discovery")
    graph.add_edge("discovery", "planner")
    graph.add_edge("planner", "cleaning")
    graph.add_edge("cleaning", "sql_agent")
    graph.add_edge("sql_agent", "python_analyst")
    graph.add_edge("python_analyst", "visualization")
    graph.add_edge("visualization", "insights")
    graph.add_edge("insights", "report_writer")
    graph.add_edge("report_writer", "reviewer")
    graph.add_edge("reviewer", END)

    return graph.compile()


# Compiled once, reused across requests.
agent_graph = build_graph()
