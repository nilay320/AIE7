"""Toolbelt assembly for agents.

Collects third-party tools and local tools (like RAG) into a single list that
graphs can bind to their language models.
"""
from __future__ import annotations

from typing import List

from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.tools.arxiv.tool import ArxivQueryRun
from langchain_core.tools import tool
import json
from app.rag import retrieve_information
from app.mcp_runtime import get_mcp_server


def get_tool_belt() -> List:
    """Return the list of tools available to agents (Tavily, Arxiv, RAG, MCP)."""
    tavily_tool = TavilySearchResults(max_results=5)
    return [tavily_tool, ArxivQueryRun(), retrieve_information, mcp_time, mcp_echo, mcp_call]


@tool
def mcp_call(name: str, arguments: dict | None = None) -> str:
    """
    Call a local MCP stdio server tool.

    - name: tool name, e.g., "echo" or "time"
    - arguments: dict of arguments for the tool
    """
    arguments = arguments or {}
    try:
        server = get_mcp_server()
        resp = server.call(name, arguments)
        if "result" in resp:
            return json.dumps(resp["result"], ensure_ascii=False)
        return json.dumps(resp)
    except Exception as e:
        return f"MCP_ERROR: {e}"


@tool
def mcp_echo(text: str) -> str:
    """Echo text using the MCP server."""
    try:
        resp = get_mcp_server().call("echo", {"text": text})
        if "result" in resp:
            return json.dumps(resp["result"], ensure_ascii=False)
        return json.dumps(resp)
    except Exception as e:
        return f"MCP_ERROR: {e}"


@tool
def mcp_time() -> str:
    """Get current UTC time from the MCP server."""
    try:
        resp = get_mcp_server().call("time", {})
        if "result" in resp:
            return json.dumps(resp["result"], ensure_ascii=False)
        return json.dumps(resp)
    except Exception as e:
        return f"MCP_ERROR: {e}"


