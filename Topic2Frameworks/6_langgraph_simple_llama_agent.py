# langgraph_multi_agent.py

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_huggingface import HuggingFacePipeline, ChatHuggingFace
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from typing import TypedDict, Annotated
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, BaseMessage

def get_device():
    """Detect and return the best available compute device."""
    if torch.cuda.is_available():
        print("Using CUDA (NVIDIA GPU) for inference")
        return "cuda"
    elif torch.backends.mps.is_available():
        print("Using MPS (Apple Silicon) for inference")
        return "mps"
    else:
        print("Using CPU for inference")
        return "cpu"

# =============================================================================
# STATE DEFINITION
# =============================================================================
class AgentState(TypedDict):
    user_input: str
    should_exit: bool
    verbose: bool
    active_model: str  # Tracks who is currently talking: "llama" or "qwen"
    llm_response: str
    messages: Annotated[list[BaseMessage], add_messages]

def vprint(state, *args, **kwargs):
    """Helper function for printing state in verbose mode."""
    if state.get("verbose", True):
        print(*args, **kwargs)

def create_models():
    """
    Load both Llama and Qwen models into memory.
    """
    device = get_device()
    models = {}

    def load_single_model(model_id):
        print(f"Loading {model_id}...")
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            dtype=torch.float16 if device != "cpu" else torch.float32,
            device_map=device if device == "cuda" else None,
        )
        if device == "mps":
            model = model.to(device)

        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=256,
            do_sample=True,
            temperature=0.7,
            top_p=0.95,
            pad_token_id=tokenizer.eos_token_id,
            return_full_text=False,
        )
        base_llm = HuggingFacePipeline(pipeline=pipe)
        return ChatHuggingFace(llm=base_llm)

    # Load models (ensure you have access to these specific repos or update the IDs)
    models["llama"] = load_single_model("meta-llama/Llama-3.2-1B-Instruct")
    models["qwen"] = load_single_model("Qwen/Qwen2.5-1.5B-Instruct")

    print("Both models loaded successfully!")
    return models

def create_graph(models: dict):
    """Create the LangGraph state graph with natural model switching."""

    def get_user_input(state: AgentState) -> dict:
        current_model = state.get("active_model", "llama")
        
        print("\n" + "=" * 50)
        print(
            f"Active: {current_model.capitalize()} | "
            "Prefix with 'Hey Qwen' or 'Hey Llama' to switch"
        )
        print("=" * 50)

        print("\n> ", end="")
        user_input = input()
        user_lower = user_input.lower()

        if user_lower in ['quit', 'exit', 'q']:
            print("Goodbye!")
            return {"user_input": user_input, "should_exit": True}

        if user_lower == 'verbose':
            return {"user_input": "", "verbose": True}

        if user_lower == 'quiet':
            return {"user_input": "", "verbose": False}

        if user_lower == "switch":
            print("\n[System] Use 'Hey Qwen' or 'Hey Llama' to switch models.")
            return {"user_input": "", "active_model": current_model}

        # Natural language switching logic
        active_model = current_model
        if user_lower.startswith("hey qwen"):
            active_model = "qwen"
            print("\n[System] Switched active model to: Qwen")
        elif user_lower.startswith("hey llama"):
            active_model = "llama"
            print("\n[System] Switched active model to: Llama")

        return {
            "user_input": user_input,
            "should_exit": False,
            "active_model": active_model,
            "messages": [HumanMessage(content=user_input)],
        }

    def call_llm(state: AgentState) -> dict:
        active_model = state.get("active_model", "llama")
        other_model = "qwen" if active_model == "llama" else "llama"
        chat_model = models[active_model]
        
        # 1. Dynamically build the System Message with explicit participant roles
        system_content = (
            f"You are {active_model.capitalize()}, an AI assistant in a three-party conversation.\n"
            f"Participants:\n"
            f"- Human: The person typing into the console.\n"
            f"- {active_model.capitalize()}: You, the active AI assistant.\n"
            f"- {other_model.capitalize()}: The other AI assistant.\n\n"
            f"Messages from the Human or {other_model.capitalize()} will be prefixed with their name. "
            f"Respond directly and concisely as {active_model.capitalize()}. "
            f"Do not start your reply with any name or label (e.g. do not write '{active_model.capitalize()}:'); just give your answer."
        )
        
        formatted_messages = [SystemMessage(content=system_content)]
        
        # 2. Iterate through history and cast the inactive model as a "Human" user
        for msg in state.get("messages", []):
            if isinstance(msg, HumanMessage):
                formatted_messages.append(HumanMessage(content=f"Human: {msg.content}"))
                
            elif isinstance(msg, AIMessage):
                if msg.name == active_model:
                    # It's from the currently active model. Keep it as an AI message.
                    formatted_messages.append(AIMessage(content=msg.content))
                else:
                    # It's from the OTHER AI. Treat it as a user message with a name prefix!
                    sender = msg.name.capitalize() if msg.name else "Other AI"
                    formatted_messages.append(HumanMessage(content=f"{sender}: {msg.content}"))

        vprint(state, f"\nProcessing your input with {active_model.capitalize()}...")

        response_msg = chat_model.invoke(formatted_messages)
        content = response_msg.content or ""

        # Strip leading "Qwen:" / "Llama:" etc. — model may echo the other agent's
        # name from history when it's supposed to reply as the current agent.
        for prefix in ("Qwen:", "Llama:", "Assistant:"):
            if content.strip().startswith(prefix):
                content = content.strip()[len(prefix):].lstrip()
                break

        # 3. Return the response, tagging it with the generating model's name
        return {
            "llm_response": content,
            "messages": [AIMessage(content=content, name=active_model)],
        }

    def print_response(state: AgentState) -> dict:
        active = state.get("active_model", "llama").capitalize()
        vprint(state, "\n" + "-" * 50)
        vprint(state, f"{active} Response:")
        vprint(state, "-" * 50)
        vprint(state, state["llm_response"])
        return {}

    def route_after_input(state: AgentState) -> str:
        if state.get("should_exit", False):
            return END
        if state["user_input"] == "":
            return "get_user_input"
        return "call_llm"

    graph_builder = StateGraph(AgentState)

    graph_builder.add_node("get_user_input", get_user_input)
    graph_builder.add_node("call_llm", call_llm)
    graph_builder.add_node("print_response", print_response)

    graph_builder.add_edge(START, "get_user_input")
    graph_builder.add_conditional_edges(
        "get_user_input",
        route_after_input,
        {
            "call_llm": "call_llm",
            "get_user_input": "get_user_input",
            END: END
        }
    )
    graph_builder.add_edge("call_llm", "print_response")
    graph_builder.add_edge("print_response", "get_user_input")

    return graph_builder.compile()

def main():
    print("=" * 50)
    print("Multi-Agent LangGraph (Llama & Qwen)")
    print("=" * 50)
    print()

    models = create_models()

    print("\nCreating LangGraph...")
    graph = create_graph(models)
    print("Graph created successfully!")

    initial_state: AgentState = {
        "messages": [],
        "user_input": "",
        "should_exit": False,
        "verbose": True,
        "active_model": "llama", 
        "llm_response": "",
    }

    graph.invoke(initial_state)

if __name__ == "__main__":
    main()