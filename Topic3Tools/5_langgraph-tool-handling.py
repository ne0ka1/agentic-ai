"""
Tool Calling with LangGraph
Single long-running conversation with checkpointing and recovery.
"""

import os
import math
from typing import Annotated, TypedDict, List, Dict, Any

import numexpr as ne
from datetime import datetime
from zoneinfo import ZoneInfo, available_timezones

from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage, BaseMessage

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite import SqliteSaver


api_key = os.getenv("OPENAI_API_KEY")


# ============================================
# PART 1: Define Your Tools
# ============================================

@tool
def get_weather(location: str) -> str:
    """Get the current weather for a given location"""
    # Simulated weather data
    weather_data = {
        "San Francisco": "Sunny, 72°F",
        "New York": "Cloudy, 55°F",
        "London": "Rainy, 48°F",
        "Tokyo": "Clear, 65°F"
    }
    return weather_data.get(location, f"Weather data not available for {location}")

@tool
def calculator(expression: str) -> str:
    """
    Evaluate arithmetic and geometric expressions
    Supported:
    - Arithmetic + - * / **
    - Triangle: sin, cos, tan
    - Constants: pi, e
    Examples: "sin(pi/2) + 3"
    """
    local_dict = {
        "pi": math.pi,
        "e": math.e,
        "sin": math.sin,
        "cos": math.cos,
        "tan": math.tan,
        "sqrt": math.sqrt
    }
    value = float(ne.evaluate(expression, local_dict=local_dict))
    return f"{value:.6f}"

@tool
def count_letter(text: str, letter: str) -> str:
    """
    Count the number of occurrences of letter in a text
    """
    count = text.count(letter)
    return f"{count:d}"

@tool
def timezone_duration(time1: str, zone1: str, time2: str, zone2: str, fmt: str = "%Y-%m-%d %H:%M") -> str:
    """
    Compute the duration starting from time1 zone1 to time2 zone2, and return signed minutes.
    """
    if zone1 not in available_timezones() or zone2 not in available_timezones():
        raise ValueError("Invalid timezone")

    d1 = datetime.strptime(time1, fmt).replace(tzinfo=ZoneInfo(zone1))
    d2 = datetime.strptime(time2, fmt).replace(tzinfo=ZoneInfo(zone2))

    minutes = (d2 - d1).total_seconds() / 60
    return f"{minutes:.1f} minutes"


# ============================================
# PART 2: Define Agent State and Graph
# ============================================
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]


tools = [get_weather, calculator, count_letter, timezone_duration]
tool_map = {tool.name: tool for tool in tools}

llm = ChatOpenAI(model="gpt-4o-mini")
llm_with_tools = llm.bind_tools(tools)


def agent_node(state: AgentState) -> AgentState:
    """Call the LLM (with tools bound) and append its response."""
    response = llm_with_tools.invoke(state["messages"])
    print("Assistant (model):", response.content or "[tool call]")
    return {"messages": [response]}


def tool_node(state: AgentState) -> AgentState:
    """Execute any requested tools and append ToolMessages."""
    messages = state["messages"]
    last = messages[-1]

    tool_messages: List[ToolMessage] = []

    if getattr(last, "tool_calls", None):
        for tool_call in last.tool_calls:
            name = tool_call["name"]
            args = tool_call["args"]
            print(f"  Tool: {name}")
            print(f"  Args: {args}")

            if name in tool_map:
                result = tool_map[name].invoke(args)
            else:
                result = f"Error: Unknown function {name}"

            print(f"  Result: {result}")

            tool_messages.append(
                ToolMessage(
                    content=result,
                    tool_call_id=tool_call["id"],
                )
            )

    return {"messages": tool_messages}


def should_continue(state: AgentState) -> str:
    """Route based on whether the LLM requested tools."""
    messages = state["messages"]
    last = messages[-1]
    if getattr(last, "tool_calls", None):
        return "tools"
    return "end"


def build_app(checkpointer: Any):
    """Build and compile the LangGraph app."""
    graph = StateGraph(AgentState)

    graph.add_node("agent", agent_node)
    graph.add_node("tools", tool_node)

    graph.add_edge(START, "agent")
    graph.add_conditional_edges(
        "agent",
        should_continue,
        {
            "tools": "tools",
            "end": END,
        },
    )
    graph.add_edge("tools", "agent")

    return graph.compile(checkpointer=checkpointer)


# ============================================
# PART 3: Convenience helpers
# ============================================
def run_turn(
    app, thread_config: Dict[str, Any], user_text: str, first_turn: bool = False
):
    """
    Run a single user turn against a persistent conversation.
    If first_turn is True, a SystemMessage is injected to set behavior.
    """
    messages: List[BaseMessage] = []
    if first_turn:
        messages.append(
            SystemMessage(
                content="You are a helpful travel-planning assistant. "
                "Use the provided tools when helpful, and keep track of prior turns."
            )
        )
    messages.append(HumanMessage(content=user_text))

    print(f"\nUser: {user_text}\n")

    final_state = None
    for event in app.stream({"messages": messages}, thread_config):
        final_state = list(event.values())[-1]

    if final_state is None:
        return

    last_message = final_state["messages"][-1]
    if isinstance(last_message, BaseMessage):
        print("Assistant:", last_message.content)


def demo_conversation(app):
    """
    Demonstrate:
    - Tool use
    - Conversation context across turns
    - Recovery by reusing the same thread_id
    """
    thread_config = {"configurable": {"thread_id": "itinerary-thread"}}

    # First turn: uses timezone and weather tools.
    run_turn(
        app,
        thread_config,
        "Today's January 29, 2026. I'll take the plane at 17:00 in New York "
        "and land in Tokyo at 21:30 tomorrow. How long is the flight in minutes?",
        first_turn=True,
    )

    # Second turn: relies on context from the first answer.
    run_turn(
        app,
        thread_config,
        "Based on that duration, is it less than 15 hours? "
        "If so, get the weather for New York and Tokyo and tell me which is warmer.",
    )

    # Simulated recovery: imagine the program crashed here.
    print("\n--- Simulating crash and recovery (restarting program) ---\n")

    # On "restart" we reuse the same thread_id and the conversation continues.
    run_turn(
        app,
        thread_config,
        "We just recovered from a crash. What were we planning, in one sentence?",
    )


# ============================================
# PART 4: Entry point
# ============================================
if __name__ == "__main__":
    db_path = "topic3_tools_conversation.sqlite"

    with SqliteSaver.from_conn_string(db_path) as checkpointer:
        app = build_app(checkpointer)
        demo_conversation(app)