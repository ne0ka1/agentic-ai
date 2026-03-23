# Multi-turn image chat agent (LangGraph + Ollama LLaVA).
#
# Conversation history lives in `messages`; the image
# path is separate state so checkpoints stay small (pixels are re-read each turn).
#
# Dependencies: pip install langgraph langchain-core langchain-ollama
#               pip install langgraph-checkpoint-sqlite   # optional, for disk checkpoints
#
# Ollama: `ollama pull llava` (or another vision tag) and keep `ollama serve` running.

from __future__ import annotations

import argparse
import base64
import mimetypes
import os
from contextlib import nullcontext
from typing import Annotated, TypedDict

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    RemoveMessage,
    SystemMessage,
)
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import REMOVE_ALL_MESSAGES, add_messages

try:
    from langchain_ollama import ChatOllama as OllamaChat
except ImportError as exc:
    raise ImportError(
        "This script requires langchain-ollama (replaces deprecated "
        "langchain_community.chat_models.ChatOllama). "
        "Install with: pip install langchain-ollama"
    ) from exc

# -----------------------------------------------------------------------------
# State
# -----------------------------------------------------------------------------


class VLMChatState(TypedDict):
    """Graph state: transcript + which image to ground every user turn on."""

    messages: Annotated[list[BaseMessage], add_messages]
    image_path: str
    user_input: str
    should_exit: bool
    skip_llm: bool
    assistant_text: str
    verbose: bool


SYSTEM_PROMPT = (
    "You are a helpful assistant. The user is discussing a single image they loaded. "
    "Answer from what you see; if something is unclear, say so. Be concise unless "
    "they ask for detail."
)


def vprint(state: VLMChatState, *args, **kwargs) -> None:
    if state.get("verbose", True):
        print(*args, **kwargs)


def image_path_to_data_url(path: str) -> str:
    mime, _ = mimetypes.guess_type(path)
    mime = mime or "image/jpeg"
    with open(path, "rb") as f:
        b64 = base64.standard_b64encode(f.read()).decode("ascii")
    return f"data:{mime};base64,{b64}"


def build_multimodal_messages(
    state: VLMChatState,
    data_url: str,
    max_messages: int,
) -> list[BaseMessage]:
    """Map stored text history → vision messages (image re-attached on every user turn)."""
    raw = state.get("messages") or []
    if max_messages > 0 and len(raw) > max_messages:
        raw = raw[-max_messages:]

    out: list[BaseMessage] = [SystemMessage(content=SYSTEM_PROMPT)]
    for msg in raw:
        if isinstance(msg, HumanMessage):
            text = msg.content if isinstance(msg.content, str) else str(msg.content)
            out.append(
                HumanMessage(
                    content=[
                        {"type": "text", "text": text},
                        {
                            "type": "image_url",
                            "image_url": {"url": data_url},
                        },
                    ]
                )
            )
        elif isinstance(msg, AIMessage):
            out.append(msg)
    return out


# -----------------------------------------------------------------------------
# Graph
# -----------------------------------------------------------------------------


def create_graph(llm, max_context_messages: int, checkpointer):
    def get_user_input(state: VLMChatState) -> dict:
        ip = state.get("image_path") or ""
        short = os.path.basename(ip) if ip else "(none)"

        print("\n" + "=" * 56)
        print(f" Image: {short}  |  model: {getattr(llm, 'model', '?')}")
        print(" Commands: image <path>  |  reset  |  quiet / verbose  |  quit / exit / q")
        print("=" * 56)
        print("\n> ", end="", flush=True)
        line = input()
        low = line.strip().lower()

        if low == "":
            return {"user_input": "", "skip_llm": True}

        if low in ("quit", "exit", "q"):
            print("Goodbye.")
            return {
                "user_input": line,
                "should_exit": True,
                "skip_llm": True,
            }

        if low == "help":
            print(
                "Load an image:  image /path/to/file.jpg\n"
                "Then ask questions about it.  reset clears chat + image.\n"
            )
            return {"user_input": "", "skip_llm": True}

        if low == "verbose":
            return {"user_input": "", "verbose": True, "skip_llm": True}
        if low == "quiet":
            return {"user_input": "", "verbose": False, "skip_llm": True}

        if low == "reset":
            return {
                "user_input": "",
                "image_path": "",
                "messages": [RemoveMessage(id=REMOVE_ALL_MESSAGES)],
                "skip_llm": True,
            }

        if low.startswith("image "):
            path = line[6:].strip().strip('"').strip("'")
            if not path:
                print("Usage: image <path-to-file>")
                return {"user_input": "", "skip_llm": True}
            path = os.path.expanduser(path)
            if not os.path.isfile(path):
                print(f"Not a file: {path}")
                return {"user_input": "", "skip_llm": True}
            print(f"[Loaded image: {path}]")
            return {
                "user_input": "",
                "image_path": os.path.abspath(path),
                "skip_llm": True,
            }

        if not (state.get("image_path") or "").strip():
            print("Load an image first, e.g.  image ./photo.jpg")
            return {"user_input": "", "skip_llm": True}

        return {
            "user_input": line,
            "should_exit": False,
            "skip_llm": False,
            "messages": [HumanMessage(content=line)],
        }

    def call_vlm(state: VLMChatState) -> dict:
        path = state.get("image_path") or ""
        data_url = image_path_to_data_url(path)
        vprint(state, "\nThinking…")
        msgs = build_multimodal_messages(state, data_url, max_context_messages)
        reply = llm.invoke(msgs)
        text = (reply.content or "").strip()
        return {
            "assistant_text": text,
            "messages": [AIMessage(content=text)],
        }

    def print_reply(state: VLMChatState) -> dict:
        vprint(state, "\n" + "-" * 56)
        vprint(state, "Assistant")
        vprint(state, "-" * 56)
        print(state.get("assistant_text", ""))
        return {"skip_llm": True}

    def route_after_input(state: VLMChatState) -> str:
        if state.get("should_exit"):
            return END
        if state.get("skip_llm"):
            return "get_user_input"
        return "call_vlm"

    g = StateGraph(VLMChatState)
    g.add_node("get_user_input", get_user_input)
    g.add_node("call_vlm", call_vlm)
    g.add_node("print_reply", print_reply)

    g.add_edge(START, "get_user_input")
    g.add_conditional_edges(
        "get_user_input",
        route_after_input,
        {
            "get_user_input": "get_user_input",
            "call_vlm": "call_vlm",
            END: END,
        },
    )
    g.add_edge("call_vlm", "print_reply")
    g.add_edge("print_reply", "get_user_input")

    return g.compile(checkpointer=checkpointer)


def make_checkpointer(db_path: str):
    try:
        from langgraph.checkpoint.sqlite import SqliteSaver

        return SqliteSaver.from_conn_string(db_path), f"sqlite:{db_path}"
    except ImportError:
        return nullcontext(MemorySaver()), "memory (install langgraph-checkpoint-sqlite for disk)"


def main() -> None:
    parser = argparse.ArgumentParser(description="LangGraph + Ollama LLaVA image chat")
    parser.add_argument(
        "--model",
        default=os.environ.get("OLLAMA_VLM_MODEL", "llava"),
        help="Ollama vision model name (default: llava or OLLAMA_VLM_MODEL)",
    )
    parser.add_argument(
        "--base-url",
        default=os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434"),
        help="Ollama HTTP API base URL",
    )
    parser.add_argument(
        "--thread-id",
        default="vlm-main",
        help="Checkpoint thread id (separate conversations per id)",
    )
    parser.add_argument(
        "--max-msgs",
        type=int,
        default=24,
        help="Max prior messages sent to the model (sliding window)",
    )
    args = parser.parse_args()

    llm = OllamaChat(model=args.model, base_url=args.base_url, temperature=0.4)

    here = os.path.dirname(os.path.abspath(__file__))
    db_path = os.path.join(here, "vlm_checkpoints.sqlite")
    cm, desc = make_checkpointer(db_path)

    print("LangGraph VLM chat — checkpoint:", desc)

    with cm as checkpointer:
        graph = create_graph(llm, args.max_msgs, checkpointer)
        config = {"configurable": {"thread_id": args.thread_id}}

        initial: VLMChatState = {
            "messages": [],
            "image_path": "",
            "user_input": "",
            "should_exit": False,
            "skip_llm": True,
            "assistant_text": "",
            "verbose": True,
        }

        snap = graph.get_state(config)
        has_cp = bool(
            snap.values
            and (
                (snap.values.get("messages") or [])
                or snap.next
            )
        )
        if has_cp:
            print(f"[Resuming thread {args.thread_id!r}]\n")
            graph.invoke(None, config)
        else:
            graph.invoke(initial, config)


if __name__ == "__main__":
    main()
