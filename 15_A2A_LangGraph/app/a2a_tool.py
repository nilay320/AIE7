"""A2A client tool for LangGraph agents.

This module exposes a LangChain Tool that calls the running A2A server
for the General Purpose Agent and returns the result text.
"""
from __future__ import annotations

import os
import asyncio
from typing import Annotated
from uuid import uuid4

import httpx
from a2a.client import A2ACardResolver, A2AClient
from a2a.types import MessageSendParams, SendMessageRequest
from langchain_core.tools import tool
from dotenv import load_dotenv


async def _send_a2a_message(query: str, base_url: str) -> str:
    """Send a single-turn message to the A2A server and return response text.

    Attempts to extract the final result text from artifacts first, then from
    the result message parts. Falls back to returning the JSON payload.
    """
    # Increase timeout to allow LLM/tool latencies on the server side
    async with httpx.AsyncClient(timeout=httpx.Timeout(120.0)) as httpx_client:
        resolver = A2ACardResolver(httpx_client=httpx_client, base_url=base_url)
        agent_card = await resolver.get_agent_card()

        client = A2AClient(httpx_client=httpx_client, agent_card=agent_card)

        send_message_payload: dict[str, object] = {
            "message": {
                "role": "user",
                "parts": [{"kind": "text", "text": query}],
                "message_id": uuid4().hex,
            }
        }
        request = SendMessageRequest(
            id=str(uuid4()), params=MessageSendParams(**send_message_payload)
        )

        response = await client.send_message(request)

        # Try artifacts → message parts → raw JSON
        try:
            result = response.root.result  # type: ignore[attr-defined]
            artifacts = getattr(result, "artifacts", None)
            if artifacts:
                for artifact in artifacts:
                    parts = getattr(artifact, "parts", None)
                    if parts:
                        first_part = parts[0]
                        root = getattr(first_part, "root", None)
                        if root and hasattr(root, "text"):
                            return str(root.text)

            message = getattr(result, "message", None)
            if message:
                parts = getattr(message, "parts", None)
                if parts:
                    first_part = parts[0]
                    root = getattr(first_part, "root", None)
                    if root and hasattr(root, "text"):
                        return str(root.text)
        except Exception:
            pass

        try:
            return response.model_dump_json(exclude_none=True)
        except Exception:
            return str(response)


@tool
def call_general_agent(
    query: Annotated[str, "query for the General Purpose Agent via A2A"]
) -> str:
    """Call the running A2A server (General Agent) and return its result text.

    The A2A base URL can be configured with the environment variable
    `A2A_BASE_URL` (default: http://localhost:10000).
    """
    # Load environment variables
    load_dotenv()
    base_url = os.environ.get("A2A_BASE_URL", "http://localhost:10000")

    # Safely run async from sync contexts (avoid nested event loop hangs)
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(_send_a2a_message(query, base_url))
    else:
        import threading

        result_holder: dict[str, str] = {}
        error_holder: dict[str, BaseException] = {}

        def _runner():
            try:
                result_holder["result"] = asyncio.run(_send_a2a_message(query, base_url))
            except BaseException as e:  # propagate later
                error_holder["error"] = e

        t = threading.Thread(target=_runner)
        t.start()
        t.join()
        if "error" in error_holder:
            raise error_holder["error"]
        return result_holder.get("result", "")


