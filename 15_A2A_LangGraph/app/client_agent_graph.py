"""Simple client LangGraph that uses the A2A tool to call the server app.

Usage:
    uv run python app/client_agent_graph.py "Your question"
"""
from __future__ import annotations

import os
import sys
from typing import Annotated, Dict, List, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

from app.a2a_tool import call_general_agent


class ClientState(TypedDict):
    messages: Annotated[List[AnyMessage], add_messages]


def _build_model_with_a2a_tool():
    # Ensure .env is loaded for OPENAI_API_KEY
    load_dotenv()
    # Prefer a stable default model for the client to avoid invalid env overrides
    model_name = os.getenv("CLIENT_LLM_NAME", "gpt-4o-mini")
    model = ChatOpenAI(
        model=model_name,
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        openai_api_base=os.getenv("TOOL_LLM_URL", "https://api.openai.com/v1"),
        temperature=0,
    )
    return model.bind_tools([call_general_agent])


CLIENT_SYSTEM_PROMPT = (
    "You are a thin client that must always use the call_general_agent tool to answer the user's question. "
    "Do not answer directly. Call the tool with the user's query and return the tool's result."
)


def _agent_node(state: Dict, model):
    messages = [("system", CLIENT_SYSTEM_PROMPT)] + state["messages"]
    return {"messages": [model.invoke(messages)]}


def _router(state: Dict):
    last = state["messages"][-1]
    return "action" if getattr(last, "tool_calls", None) else "end"


def build_client_graph():
    model = _build_model_with_a2a_tool()
    tool_node = ToolNode([call_general_agent])

    def agent(state: ClientState):
        return _agent_node(state, model)

    graph = StateGraph(ClientState)
    graph.add_node("agent", agent)
    graph.add_node("action", tool_node)
    graph.set_entry_point("agent")
    graph.add_conditional_edges("agent", _router, {"action": "action", "end": END})
    # Terminal is implicit via conditional edge mapping
    graph.add_edge("action", "agent")
    return graph.compile()


def main():
    if len(sys.argv) < 2:
        print("Usage: uv run python app/client_agent_graph.py \"Your question\"")
        sys.exit(1)

    query = sys.argv[1]
    graph = build_client_graph()
    inputs = {"messages": [("user", query)]}

    # Stream for visibility, then print final model response
    last_values: Dict | None = None
    for values in graph.stream(inputs, stream_mode="values"):
        last_values = values

    if last_values is None:
        result = graph.invoke(inputs)
        last_message = result["messages"][-1]
    else:
        last_message = last_values["messages"][-1]
    print(getattr(last_message, "content", ""))


if __name__ == "__main__":
    main()


