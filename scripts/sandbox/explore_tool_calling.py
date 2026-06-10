"""Tool Calling Exploration Script: Inspect OpenAI's native function-calling mechanism.

Note: MLP4CS does NOT use OpenAI tool calling. Our pipeline extracts slots via structured JSON output and calls find_entity()/book_entity() in Python directly.
This script explores the alternative approach where the LLM decides when and how to invoke tools, for reference and comparison with our hardcoded pipeline flow.
"""
import os
import json
from typing import Any

from src.utils import print_separator
from openai import OpenAI

from dotenv import load_dotenv
load_dotenv()


# Tool definitions for OpenAI function calling
TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "search_hotels",
            "description": "Search for hotels matching price range and area constraints",
            "parameters": {
                "type": "object",
                "properties": {
                    "pricerange": {
                        "type": "string",
                        "enum": ["cheap", "moderate", "expensive"],
                        "description": "Price category"
                    },
                    "area": {
                        "type": "string",
                        "enum": ["centre", "north", "south", "east", "west"],
                        "description": "Location area"
                    }
                },
                "required": ["pricerange", "area"]
            }
        }
    }
]


def create_dummy_scenario() -> dict[str, Any]:
    """
    Create a hardcoded hotel search scenario mimicking MultiWOZ structure.

    Purpose: Simulate a user query with slot constraints
    Returns: Dictionary with user query and expected slots
    """
    dummy_scenario = {
        "user_query": "I need a cheap hotel in the centre",
        "expected_slots": {
            "domain": "hotel",
            "pricerange": "cheap",
            "area": "centre"
        },
        "database": [
            {"name": "el shaddai", "pricerange": "cheap", "area": "centre", "type": "guesthouse", "stars": "2", "phone": "01onal"},
            {"name": "alexander bed and breakfast", "pricerange": "cheap", "area": "centre", "type": "guesthouse", "stars": "4", "phone": "01223525725"},
            {"name": "university arms hotel", "pricerange": "expensive", "area": "centre", "type": "hotel", "stars": "4", "phone": "01223351241"}
        ]
    }

    print_separator("DUMMY SCENARIO")
    print(json.dumps(dummy_scenario, indent=2))

    return dummy_scenario


def search_hotels(pricerange: str, area: str, database: list[dict[str, Any]], verbose: bool = False) -> list[dict[str, Any]]:
    """
    Search hotel database by pricerange and area constraints.

    Purpose: Simulate a tool/API that the LLM can call (analogous to find_entity in src/db.py)
    Args:
        pricerange: Price category (cheap, moderate, expensive)
        area: Location area (center, north, south, east, west)
        database: List of hotel records to search
        verbose: If True, print execution details (default: False)
    Returns: List of matching hotels
    """
    search_results = []
    for hotel in database:
        if hotel.get("pricerange") == pricerange and hotel.get("area") == area:
            search_results.append(hotel)

    if verbose:
        print_separator("SEARCH TOOL EXECUTION")
        print(f"Query: pricerange='{pricerange}', area='{area}'")
        print(f"\nFound {len(search_results)} matching hotels:")
        for hotel in search_results:
            print(f"  - {hotel['name']} (type: {hotel['type']}, stars: {hotel['stars']})")

    return search_results


def call_llm(system_prompt: str = "", user_query: str = "", model_name: str = "gpt-4o-mini", tools: list[dict[str, Any]] | None = None, temperature: float = 0, messages: list[dict[str, Any]] | None = None) -> Any:
    """
    Helper function to call OpenAI LLM with consistent settings.

    Purpose: Centralize LLM API calls to avoid code duplication.
             For simplicity, we use only gpt-4o-mini (our goal is to understand
             tool calling mechanics, not compare model quality).

    Args:
        system_prompt: System instructions (ignored if messages provided)
        user_query: User's input (ignored if messages provided)
        model_name: Which OpenAI model to use (default: gpt-4o-mini)
        tools: Optional tool definitions for function calling
        temperature: Sampling temperature (0 = deterministic)
        messages: Optional pre-built conversation history (overrides system_prompt/user_query)
    Returns: OpenAI ChatCompletion response object
    """
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    if messages is None:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_query}
        ]

    request_params = {
        "model": model_name,
        "messages": messages,
        "temperature": temperature
    }

    if tools is not None:
        request_params["tools"] = tools

    response = client.chat.completions.create(**request_params)

    return response


def run_standard_llm(user_query: str, model_name: str = "gpt-4o-mini") -> dict[str, Any]:
    """
    Raw LLM call without any tool definitions (zero-shot baseline).
    Purpose: Model answers from training data only, no tool access.

    Args:
        user_query: The user's question/request
        model_name: Which LLM to use
    Returns: Dictionary with model response and metadata
    """
    system_prompt = """You are a helpful hotel search assistant.
Help the user find hotels based on their preferences."""

    print_separator(f"STANDARD LLM - NO TOOLS ({model_name})")
    print(f"User Query: {user_query}")

    response = call_llm(
        system_prompt=system_prompt,
        user_query=user_query,
        model_name=model_name
    )

    answer = response.choices[0].message.content

    print(f"\nLLM Response:\n{answer}")

    result = {
        "agent_type": "standard_llm",
        "model": model_name,
        "user_query": user_query,
        "tools_available": [],
        "tools_used": [],
        "response": answer,
        "tokens_used": response.usage.total_tokens
    }

    return result


def run_simple_tool_calling(user_query: str, database: list[dict[str, Any]], model_name: str = "gpt-4o-mini") -> dict[str, Any]:
    """
    LLM with tool calling: single request-response cycle with tool execution.
    Purpose: Model calls tools once, then generates final answer with results.

    In MLP4CS, this is done differently: the pipeline calls find_entity() directly
    in Python after DST extracts slots. Here, the LLM decides when to call the tool.

    Args:
        user_query: The user's question/request
        database: Hotel database to search
        model_name: Which LLM to use
    Returns: Dictionary with model response and actual tool usage
    """
    system_prompt = """You are a helpful hotel search assistant.
You have access to a search_hotels tool if needed to find hotels matching specific criteria."""

    print_separator(f"SIMPLE TOOL CALLING ({model_name})")
    print(f"User Query: {user_query}")

    # Step 1: Initial LLM call (LLM decides whether to use a tool)
    response = call_llm(
        system_prompt=system_prompt,
        user_query=user_query,
        model_name=model_name,
        tools=TOOL_DEFINITIONS
    )

    message = response.choices[0].message
    tools_used = []
    tool_results = []

    print_separator("STEP 1: LLM DECISION (First LLM Call)")

    # Step 2: Check if model wants to call a tool
    if message.tool_calls:
        print("Decision: Use tool(s) to gather information")
        print(f"\nModel requested {len(message.tool_calls)} tool call(s):")

        for tool_call in message.tool_calls:
            tool_name = tool_call.function.name
            tool_args = json.loads(tool_call.function.arguments)

            print(f"  - {tool_name}({tool_args})")
            tools_used.append(tool_name)

            # Step 3: Execute the tool
            if tool_name == "search_hotels":
                results = search_hotels(
                    pricerange=tool_args["pricerange"],
                    area=tool_args["area"],
                    database=database,
                    verbose=True
                )
                tool_results.append(results)

        # Step 4: Feed tool results back to LLM
        print_separator("STEP 2: TOOL RESULTS → LLM (Second LLM Call)")
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_query},
            message,
            {
                "role": "tool",
                "tool_call_id": message.tool_calls[0].id,
                "content": json.dumps(tool_results[0])
            }
        ]

        # Step 5: Get final answer from model
        final_response = call_llm(
            model_name=model_name,
            messages=messages
        )

        final_answer = final_response.choices[0].message.content
        total_tokens = response.usage.total_tokens + final_response.usage.total_tokens

        print(f"Final Answer:\n{final_answer}")

    else:
        print("Decision: Answer directly without tools")
        final_answer = message.content
        total_tokens = response.usage.total_tokens

    print_separator("FINAL ANSWER")
    print(final_answer)

    result = {
        "agent_type": "simple_tool_calling",
        "model": model_name,
        "user_query": user_query,
        "tools_available": ["search_hotels"],
        "tools_used": tools_used,
        "tool_results": tool_results,
        "response": final_answer,
        "tokens_used": total_tokens
    }

    return result


def run_react_tool_calling(user_query: str, database: list[dict[str, Any]], model_name: str = "gpt-4o-mini", max_iterations: int = 5) -> dict[str, Any]:
    """
    ReAct-style iterative Thought-Action-Observation loop with tool calling.
    Purpose: LLM can reason, act, observe results, and repeat until task complete.

    Args:
        user_query: The user's question/request
        database: Hotel database to search
        model_name: Which LLM to use
        max_iterations: Maximum reasoning loops to prevent infinite cycles
    Returns: Dictionary with complete reasoning trace and final response
    """
    system_prompt = """You are a helpful hotel search assistant using ReAct reasoning.
You have access to a search_hotels tool if needed.

Follow this pattern for each step:
Thought: [Reason about what you need to do]
Action: [Either use a tool or provide Final Answer]

Available actions:
- search_hotels(pricerange, area) - search for hotels
- Final Answer: [your response to the user]

Continue reasoning until you can provide a Final Answer."""

    print_separator(f"REACT TOOL CALLING - ITERATIVE REASONING ({model_name})")
    print(f"User Query: {user_query}")
    print(f"Max Iterations: {max_iterations}")

    conversation_history = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_query}
    ]

    tools_used = []
    tool_results = []
    reasoning_trace = []

    for iteration in range(1, max_iterations + 1):
        print_separator(f"ITERATION {iteration}")

        # Step 1 (THINK): LLM thinks and decides action
        response = call_llm(
            model_name=model_name,
            messages=conversation_history,
            tools=TOOL_DEFINITIONS
        )

        message = response.choices[0].message

        # Step 2 (ACT): Execute tool, if decided
        if message.tool_calls:
            print("Action: Use tool")

            iteration_tools = []
            iteration_observations = []

            conversation_history.append(message)

            for tool_call in message.tool_calls:
                tool_name = tool_call.function.name
                tool_args = json.loads(tool_call.function.arguments)

                print(f"  → {tool_name}({tool_args})")
                tools_used.append(tool_name)

                if tool_name == "search_hotels":
                    react_results = search_hotels(
                        pricerange=tool_args["pricerange"],
                        area=tool_args["area"],
                        database=database
                    )
                    tool_results.append(react_results)

                    print(f"Observation: Found {len(react_results)} hotels")

                    # Step 3 (OBSERVE): Add tool result to conversation
                    conversation_history.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": json.dumps(react_results)
                    })

                    iteration_tools.append({
                        "tool": tool_name,
                        "args": tool_args
                    })
                    iteration_observations.append(react_results)

            reasoning_trace.append({
                "iteration": iteration,
                "action": "tool_calls",
                "tools": iteration_tools,
                "observations": iteration_observations
            })
        else:
            print("Action: Provide Final Answer")
            final_answer = message.content
            print(f"\nFinal Answer:\n{final_answer}")

            reasoning_trace.append({
                "iteration": iteration,
                "action": "final_answer",
                "response": final_answer
            })

            break
    else:
        final_answer = "Maximum iterations reached without completing task."
        print(f"\n{final_answer}")

    print_separator("REACT SUMMARY")
    print(f"Total Iterations: {len(reasoning_trace)}")
    print(f"LLM Calls Made: {len(reasoning_trace)}")
    print(f"Tools Executed: {len(tools_used)} calls")
    print(f"Tool Breakdown: {', '.join(tools_used) if tools_used else 'none'}")

    result = {
        "agent_type": "react_tool_calling",
        "model": model_name,
        "user_query": user_query,
        "tools_available": ["search_hotels"],
        "tools_used": tools_used,
        "tool_results": tool_results,
        "reasoning_trace": reasoning_trace,
        "iterations_used": len(reasoning_trace),
        "response": final_answer,
    }

    return result


def explore_all() -> None:
    """Explore and compare three tool-calling approaches."""

    scenario = create_dummy_scenario()

    print_separator("TOOL CALLING EXPLORATION")
    print("\nComparing 3 approaches: No Tools, Simple Tool Calling, ReAct Tool Calling")

    # 1. Standard LLM (no tools)
    run_standard_llm(user_query=scenario["user_query"])

    # 2. Simple tool calling (relevant query, should use tool)
    run_simple_tool_calling(user_query=scenario["user_query"], database=scenario["database"])

    # 3. Simple tool calling (irrelevant query, should NOT use tool)
    run_simple_tool_calling(user_query="What is your cancellation policy?", database=scenario["database"])

    # 4. ReAct tool calling (simple query)
    run_react_tool_calling(user_query=scenario["user_query"], database=scenario["database"])

    # 5. ReAct tool calling (multi-step query)
    run_react_tool_calling(
        user_query="I want a cheap hotel in the centre, but also show me expensive hotels in the same area for comparison.",
        database=scenario["database"]
    )

    print_separator("END OF TOOL CALLING EXPLORATION")


# Run with: python -m scripts.sandbox.explore_tool_calling
if __name__ == "__main__":
    explore_all()