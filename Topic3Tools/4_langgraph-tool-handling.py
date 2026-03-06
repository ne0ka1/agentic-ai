"""
Tool Calling with LangChain
Shows how LangChain abstracts tool calling.
"""

import os
import math
import numexpr as ne
from datetime import datetime
from zoneinfo import ZoneInfo, available_timezones
from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

api_key=os.getenv("OPENAI_API_KEY")

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
# PART 2: Create LLM with Tools
# ============================================

# Tools

tools = [get_weather, calculator, count_letter, timezone_duration]
tool_map = {tool.name: tool for tool in tools}

# Create LLM
llm = ChatOpenAI(model="gpt-4o-mini")

# Bind tools to LLM
llm_with_tools = llm.bind_tools(tools)


# ============================================
# PART 3: The Agent Loop
# ============================================

def run_agent(user_query: str):
    """
    Simple agent that can use tools.
    Shows the manual loop that LangGraph automates.
    """
    
    # Start conversation with user query
    messages = [
        SystemMessage(content="You are a helpful assistant. Use the provided tools when needed."),
        HumanMessage(content=user_query)
    ]
    
    print(f"User: {user_query}\n")
    
    # Agent loop - can iterate up to 5 times
    for iteration in range(5):
        print(f"--- Iteration {iteration + 1} ---")
        
        # Call the LLM
        response = llm_with_tools.invoke(messages)
        
        # Check if the LLM wants to call a tool
        if response.tool_calls:
            print(f"LLM wants to call {len(response.tool_calls)} tool(s)")
            
            # Add the assistant's response to messages
            messages.append(response)
            
            # Execute each tool call
            for tool_call in response.tool_calls:
                function_name = tool_call["name"]
                function_args = tool_call["args"]
                
                print(f"  Tool: {function_name}")
                print(f"  Args: {function_args}")
                
                # Execute the tool
                if function_name in tool_map:
                    result = tool_map[function_name].invoke(function_args)
                else:
                    result = f"Error: Unknown function {function_name}"
                
                print(f"  Result: {result}")
                
                # Add the tool result back to the conversation
                messages.append(ToolMessage(
                    content=result,
                    tool_call_id=tool_call["id"]
                ))
            
            print()
            # Loop continues - LLM will see the tool results
            
        else:
            # No tool calls - LLM provided a final answer
            print(f"Assistant: {response.content}\n")
            return response.content
    
    return "Max iterations reached"


# ============================================
# PART 4: Test It
# ============================================

if __name__ == "__main__":
    # Test query that requires tool use
    # print("="*60)
    # print("TEST 1: Query requiring tool")
    # print("="*60)
    # run_agent("What's the weather like in San Francisco?")
    #
    # print("\n" + "="*60)
    # print("TEST 2: Query not requiring tool")
    # print("="*60)
    # run_agent("Say hello!")
    #
    # print("\n" + "="*60)
    # print("TEST 3: Multiple tool calls")
    # print("="*60)
    # run_agent("What's the weather in New York and London?")
    #
    # print("=" * 60)
    # print("TEST 1: Query requiring count_letter")
    # print("=" * 60)
    # run_agent("How many s are in Mississippi riverboats?")
    #
    # print("=" * 60)
    # print("TEST 2: Query requiring count_letter")
    # print("=" * 60)
    # run_agent("Are there more i's than s's in Mississippi riverboats?")
    #
    # print("=" * 60)
    # print("TEST 3: Query requiring count_letter and calculator")
    # print("=" * 60)
    # run_agent("What is the sin of the difference between the number of i's and the number of s's in Mississippi riverboats?")

    print("=" * 60)
    print("TEST 4: Query requiring timezone duration")
    print("=" * 60)
    run_agent("Today's January 29, 2026. I'll take the plane at 5pm tonight in New York and land in Tokyo 9:30pm tomorrow, how long does the flight take?")

    print("=" * 60)
    print("TEST 5: Query using all tools")
    print("=" * 60)
    run_agent("Today's January 29, 2026. First get the weather for New York, then count how many times the letter 'o' appears in the weather description. " +
              "I'll take the plane at 5pm tonight in New York and land in Tokyo 9:30pm tomorrow, how long does the flight take? " +
              "If that is less than 15 hours, I'll take the flight, so please me the weather for Tokyo. " +
              "If I'll be in Tokyo tomorrow, convert the temperature in Tokyo to Celsius.")