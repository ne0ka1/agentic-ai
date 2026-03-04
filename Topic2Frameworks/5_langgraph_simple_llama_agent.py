# langgraph_simple_agent.py

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
    """
    State object that flows through the LangGraph nodes.
    
    Using `add_messages` for the 'messages' key tells LangGraph to append 
    new messages to the existing list rather than overwriting it, natively 
    building our chat history.
    """
    user_input: str
    should_exit: bool
    verbose: bool
    llm_response: str
    messages: Annotated[list[BaseMessage], add_messages]

def vprint(state, *args, **kwargs):
    """Helper function for printing state in verbose mode."""
    if state.get("verbose", True):
        print(*args, **kwargs)

def create_llm():
    """
    Create and configure the LLM using HuggingFace's transformers library.
    Wraps it in ChatHuggingFace to natively support LangChain's Message API.
    """
    device = get_device()
    model_id = "meta-llama/Llama-3.2-1B-Instruct"

    print(f"Loading model: {model_id}")
    
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
        return_full_text=False, # Essential: prevents echoing the prompt back
    )

    base_llm = HuggingFacePipeline(pipeline=pipe)
    # Wrap in ChatHuggingFace to natively handle System, Human, and AI roles
    chat_model = ChatHuggingFace(llm=base_llm)

    print("Model loaded successfully!")
    return chat_model

def create_graph(chat_model):
    """Create the LangGraph state graph."""

    def get_user_input(state: AgentState) -> dict:
        print("\n" + "=" * 50)
        print("Enter your text (or 'quit' to exit):")
        print("=" * 50)

        print("\n> ", end="")
        user_input = input()

        if user_input.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            return {"user_input": user_input, "should_exit": True}

        if user_input.lower() == 'verbose':
            print("Verbose mode ON.")
            return {"user_input": "", "verbose": True}

        if user_input.lower() == 'quiet':
            print("Verbose mode OFF.")
            return {"user_input": "", "verbose": False}

        # Return the new HumanMessage. The `add_messages` reducer appends this automatically.
        return {
            "user_input": user_input,
            "should_exit": False,
            "messages": [HumanMessage(content=user_input)],
        }

    def call_llm(state: AgentState) -> dict:
        # Grab the full history of messages built up by LangGraph
        chat_history = state.get("messages", [])
        
        vprint(state, "\nProcessing your input...")

        # Invoke the ChatHuggingFace model with the raw Message objects directly
        response_msg = chat_model.invoke(chat_history)

        # Return the AIMessage so `add_messages` can append it to the context
        return {
            "llm_response": response_msg.content,
            "messages": [response_msg],
        }

    def print_response(state: AgentState) -> dict:
        vprint(state, "\n" + "-" * 50)
        vprint(state, "LLM Response:")
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

def save_graph_image(graph, filename="lg_graph.png"):
    try:
        png_data = graph.get_graph(xray=True).draw_mermaid_png()
        with open(filename, "wb") as f:
            f.write(png_data)
        print(f"Graph image saved to {filename}")
    except Exception as e:
        print(f"Could not save graph image: {e}")

def main():
    print("=" * 50)
    print("LangGraph Agent with Memory (Llama-3.2-1B-Instruct)")
    print("=" * 50)
    print()

    chat_model = create_llm()

    print("\nCreating LangGraph...")
    graph = create_graph(chat_model)
    print("Graph created successfully!")

    print("\nSaving graph visualization...")
    save_graph_image(graph)

    # Initialize state with the required SystemMessage
    initial_state: AgentState = {
        "messages": [
            SystemMessage(
                content=(
                    "You are Llama, a helpful and concise AI assistant. "
                    "Use the provided chat history to answer the user's questions."
                )
            )
        ],
        "user_input": "",
        "should_exit": False,
        "verbose": True,
        "llm_response": "",
    }

    graph.invoke(initial_state)

if __name__ == "__main__":
    main()
