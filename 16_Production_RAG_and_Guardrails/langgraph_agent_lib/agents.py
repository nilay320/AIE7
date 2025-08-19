"""LangGraph agent integration with production features."""

from typing import Dict, Any, List, Optional
import os

from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage, SystemMessage
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.tools.arxiv.tool import ArxivQueryRun
from langchain_core.tools import tool
from typing_extensions import TypedDict, Annotated
from langgraph.graph.message import add_messages
from textwrap import dedent

from .models import get_openai_model
from .rag import ProductionRAGChain


class AgentState(TypedDict):
    """State schema for agent graphs."""
    messages: Annotated[List[BaseMessage], add_messages]


def create_rag_tool(rag_chain: ProductionRAGChain):
    """Create a RAG tool from a ProductionRAGChain."""
    
    @tool
    def retrieve_information(query: str) -> str:
        """Use Retrieval Augmented Generation to retrieve information from the student loan documents."""
        try:
            result = rag_chain.invoke(query)
            return result.content if hasattr(result, 'content') else str(result)
        except Exception as e:
            return f"Error retrieving information: {str(e)}"
    
    return retrieve_information


def get_default_tools(rag_chain: Optional[ProductionRAGChain] = None) -> List:
    """Get default tools for the agent.
    
    Args:
        rag_chain: Optional RAG chain to include as a tool
        
    Returns:
        List of tools
    """
    tools = []
    
    # Add Tavily search if API key is available
    if os.getenv("TAVILY_API_KEY"):
        tools.append(TavilySearchResults(max_results=5))
    
    # Add Arxiv tool
    tools.append(ArxivQueryRun())
    
    # Add RAG tool if provided
    if rag_chain:
        tools.append(create_rag_tool(rag_chain))
    
    return tools


def create_langgraph_agent(
    model_name: str = "gpt-4",
    temperature: float = 0.1,
    tools: Optional[List] = None,
    rag_chain: Optional[ProductionRAGChain] = None
):
    """Create a simple LangGraph agent.
    
    Args:
        model_name: OpenAI model name
        temperature: Model temperature
        tools: List of tools to bind to the model
        rag_chain: Optional RAG chain to include as a tool
        
    Returns:
        Compiled LangGraph agent
    """
    if tools is None:
        tools = get_default_tools(rag_chain)
    
    # Get model and bind tools
    model = get_openai_model(model_name=model_name, temperature=temperature)
    model_with_tools = model.bind_tools(tools)
    
    def call_model(state: AgentState) -> Dict[str, Any]:
        """Invoke the model with messages."""
        messages = state["messages"]
        response = model_with_tools.invoke(messages)
        return {"messages": [response]}
    
    def should_continue(state: AgentState):
        """Route to tools if the last message has tool calls."""
        last_message = state["messages"][-1]
        if getattr(last_message, "tool_calls", None):
            return "action"
        return END
    
    # Build graph
    graph = StateGraph(AgentState)
    tool_node = ToolNode(tools)
    
    graph.add_node("agent", call_model)
    graph.add_node("action", tool_node)
    graph.set_entry_point("agent")
    graph.add_conditional_edges("agent", should_continue, {"action": "action", END: END})
    graph.add_edge("action", "agent")
    
    return graph.compile()


# --- Guardrails-enabled Simple Agent ----------------------------------------------------------

# Import Guardrails lazily/optionally so the library still works if deps are missing
try:
    from guardrails import Guard  # type: ignore
    from guardrails.hub import (  # type: ignore
        RestrictToTopic,
        DetectJailbreak,
        GuardrailsPII,
        ProfanityFree,
        LlmRagEvaluator,
        HallucinationPrompt,
    )
    _GUARDS_AVAILABLE = True
except Exception:
    _GUARDS_AVAILABLE = False


def _build_guard_suites() -> Dict[str, Any]:
    """Create guard instances, skipping missing ones. Returns a dict of guards.

    The function degrades gracefully: if any import/config fails, that guard is omitted.
    """
    if not _GUARDS_AVAILABLE:
        return {}

    suites: Dict[str, Any] = {}
    # Topic restriction
    try:
        suites["topic"] = Guard().use(
            RestrictToTopic(
                valid_topics=[
                    "student loans",
                    "financial aid",
                    "education financing",
                    "loan repayment",
                ],
                invalid_topics=["crypto", "gambling", "politics", "investment advice"],
                disable_classifier=True,
                disable_llm=False,
                on_fail="exception",
            )
        )
    except Exception:
        pass

    # Jailbreak
    try:
        suites["jailbreak"] = Guard().use(DetectJailbreak())
    except Exception:
        pass

    # PII (auto-fix/redact)
    try:
        suites["pii"] = Guard().use(
            GuardrailsPII(
                entities=["CREDIT_CARD", "SSN", "PHONE_NUMBER", "EMAIL_ADDRESS"],
                on_fail="fix",
            )
        )
    except Exception:
        pass

    # Profanity (block)
    try:
        suites["profanity"] = Guard().use(
            ProfanityFree(threshold=0.8, validation_method="sentence", on_fail="exception")
        )
    except Exception:
        pass

    # Factuality (soft)
    try:
        suites["factuality"] = Guard().use(
            LlmRagEvaluator(
                eval_llm_prompt_generator=HallucinationPrompt(prompt_name="hallucination_judge_llm"),
                llm_evaluator_fail_response="hallucinated",
                llm_evaluator_pass_response="factual",
                llm_callable="gpt-4.1-mini",
                on_fail="exception",
                on="prompt",
            )
        )
    except Exception:
        pass

    return suites


def _last_human_text(messages: List[BaseMessage]) -> str:
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            return getattr(message, "content", "")
    return ""


def _replace_last_human(messages: List[BaseMessage], new_text: str) -> None:
    for i in range(len(messages) - 1, -1, -1):
        if isinstance(messages[i], HumanMessage):
            messages[i] = HumanMessage(content=new_text)
            return


def create_langgraph_agent_with_guardrails(
    model_name: str = "gpt-4",
    temperature: float = 0.1,
    tools: Optional[List] = None,
    rag_chain: Optional[ProductionRAGChain] = None,
):
    """Create a Simple LangGraph agent with input/output Guardrails nodes.

    This is an additive variant that leaves existing agents untouched. Guards are
    optional; if unavailable, the graph behaves like the simple agent.
    """
    if tools is None:
        tools = get_default_tools(rag_chain)

    # Model with tools
    model = get_openai_model(model_name=model_name, temperature=temperature)
    model_with_tools = model.bind_tools(tools)

    guard_suites = _build_guard_suites()

    def input_guard_node(state: AgentState) -> Dict[str, Any]:
        # No-op if no guards
        if not guard_suites:
            return {}

        user_text = _last_human_text(state["messages"])

        # Hard blockers: topic + jailbreak
        for name in ("topic", "jailbreak"):
            guard = guard_suites.get(name)
            if not guard:
                continue
            try:
                result = guard.validate(user_text)
            except Exception:
                continue
            if not getattr(result, "validation_passed", True):
                # Provide user-friendly guidance
                return {
                    "messages": [
                        SystemMessage(
                            content=f"Blocked by {name} policy. Please rephrase your request to be on-topic and compliant."
                        )
                    ]
                }

        # PII: redact and replace last human message if changed
        pii = guard_suites.get("pii")
        if pii:
            try:
                p = pii.validate(user_text)
                redacted = getattr(p, "validated_output", user_text)
                if redacted and redacted != user_text:
                    _replace_last_human(state["messages"], redacted)
            except Exception:
                pass

        return {}

    def call_model(state: AgentState) -> Dict[str, Any]:
        messages = state["messages"]
        response = model_with_tools.invoke(messages)
        return {"messages": [response]}

    def route_after_agent(state: AgentState):
        last_message = state["messages"][-1]
        if getattr(last_message, "tool_calls", None):
            return "action"
        return "output_guard"

    def output_guard_node(state: AgentState) -> Dict[str, Any]:
        if not guard_suites:
            return {}

        text = getattr(state["messages"][-1], "content", "")

        # Profanity -> request rewrite
        prof = guard_suites.get("profanity")
        if prof:
            try:
                r = prof.validate(text)
                if not getattr(r, "validation_passed", True):
                    return {
                        "messages": [
                            SystemMessage(
                                content=(
                                    "Your last reply contained inappropriate language. Rewrite it professionally "
                                    "and ensure it adheres to content policy."
                                )
                            )
                        ]
                    }
            except Exception:
                pass

        # Optional factuality -> request revision
        fact = guard_suites.get("factuality")
        if fact:
            try:
                r = fact.validate(text)
                if not getattr(r, "validation_passed", True):
                    return {
                        "messages": [
                            SystemMessage(
                                content=(
                                    "Your last reply may not align with the provided context. Please revise to be "
                                    "factual and cite context where appropriate."
                                )
                            )
                        ]
                    }
            except Exception:
                pass

        return {}

    def output_guard_decision(state: AgentState):
        # If a SystemMessage was added to request a repair, retry once
        last = state["messages"][-1]
        if isinstance(last, SystemMessage):
            return "retry"
        return END

    def should_continue(state: AgentState):
        last_message = state["messages"][-1]
        if getattr(last_message, "tool_calls", None):
            return "action"
        return END

    # Build graph
    graph = StateGraph(AgentState)
    tool_node = ToolNode(tools)

    graph.add_node("input_guard", input_guard_node)
    graph.add_node("agent", call_model)
    graph.add_node("action", tool_node)
    graph.add_node("output_guard", output_guard_node)

    graph.set_entry_point("input_guard")
    graph.add_edge("input_guard", "agent")
    graph.add_conditional_edges("agent", route_after_agent, {"action": "action", "output_guard": "output_guard"})
    graph.add_edge("action", "agent")
    graph.add_conditional_edges("output_guard", output_guard_decision, {"retry": "agent", END: END})

    return graph.compile()

def create_langgraph_agent_with_helpfulness(
    model_name: str = "gpt-4",
    temperature: float = 0.1,
    tools: Optional[List] = None,
    rag_chain: Optional[ProductionRAGChain] = None
):
    """Create a LangGraph agent with helpfulness.
    
    Args:
        model_name: OpenAI model name
        temperature: Model temperature
        tools: List of tools to bind to the model
        rag_chain: Optional RAG chain to include as a tool
        
    Returns:
        Compiled LangGraph agent
    """
    if tools is None:
        tools = get_default_tools(rag_chain)
    
    # Get model and bind tools
    model = get_openai_model(model_name=model_name, temperature=temperature)
    model_with_tools = model.bind_tools(tools)
    
    def call_model(state: AgentState) -> Dict[str, Any]:
        """Invoke the model with messages."""
        messages = state["messages"]
        response = model_with_tools.invoke(messages)
        return {"messages": [response]}
    
    def route_to_action_or_helpfulness(state: AgentState):
        """Decide whether to execute tools or run the helpfulness evaluator."""
        last_message = state["messages"][-1]
        if getattr(last_message, "tool_calls", None):
            return "action"
        return "helpfulness"

    def helpfulness_node(state: AgentState) -> Dict[str, Any]:
        """Evaluate helpfulness of the latest response relative to the initial query."""
        # If we've exceeded loop limit, short-circuit with END decision marker
        if len(state["messages"]) > 10:
            return {"messages": [AIMessage(content="HELPFULNESS:END")]}    

        initial_query = state["messages"][0]
        final_response = state["messages"][-1]

        prompt_template = dedent("""
            Given an initial query and a final response, determine if the final response is extremely helpful or not. Please indicate helpfulness with a 'Y' and unhelpfulness as an 'N'.

            Initial Query:
            {initial_query}

            Final Response:
            {final_response}
        """).strip()

        helpfulness_prompt_template = PromptTemplate.from_template(prompt_template)
        helpfulness_check_model = get_openai_model(model_name="gpt-4.1-mini", temperature=temperature)
        helpfulness_chain = (
            helpfulness_prompt_template | helpfulness_check_model | StrOutputParser()
        )

        helpfulness_response = helpfulness_chain.invoke(
            {
                "initial_query": initial_query.content,
                "final_response": final_response.content,
            }
        )

        decision = "Y" if "Y" in helpfulness_response else "N"
        return {"messages": [AIMessage(content=f"HELPFULNESS:{decision}")]}

    def helpfulness_decision(state: AgentState):
        """Terminate on 'HELPFULNESS:Y' or loop otherwise; guard against infinite loops."""
        # Check loop-limit marker
        if any(getattr(m, "content", "") == "HELPFULNESS:END" for m in state["messages"][-1:]):
            return END

        last = state["messages"][-1]
        text = getattr(last, "content", "")
        if "HELPFULNESS:Y" in text:
            return "end"
        return "continue"

    graph = StateGraph(AgentState)
    tool_node = ToolNode(tools)
    graph.add_node("agent", call_model)
    graph.add_node("action", tool_node)
    graph.add_node("helpfulness", helpfulness_node)
    graph.set_entry_point("agent")
    graph.add_conditional_edges(
        "agent",
        route_to_action_or_helpfulness,
        {"action": "action", "helpfulness": "helpfulness"},
    )
    graph.add_conditional_edges(
        "helpfulness",
        helpfulness_decision,
        {"continue": "agent", "end": END, END: END},
    )
    graph.add_edge("action", "agent")    
    return graph.compile()
