import json
import requests
import os

ASTA_MCP_URL = "https://asta-tools.allen.ai/mcp/v1"

def list_tools():
    # JSON-RPC 2.0 tools/list request
    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/list",
        "params": {},
    }

    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json, text/event-stream",
        "x-api-key": os.environ["ASTA_API_KEY"]
    }

    # Stream the SSE response so we can see raw events
    response = requests.post(
        ASTA_MCP_URL,
        headers=headers,
        json=payload,
        stream=True,
    )
    response.raise_for_status()

    raw_events = []
    tools = None

    for line in response.iter_lines(decode_unicode=True):
        if not line:
            continue
        # Show the raw SSE line
        raw_events.append(line)

        # Parse JSON only from data: lines
        if not line.startswith("data:"):
            continue

        data_str = line[len("data:") :].strip()
        if not data_str:
            continue

        try:
            msg = json.loads(data_str)
        except json.JSONDecodeError:
            # Keep the raw line for debugging, but skip parsing errors
            continue

        # Look for a JSON-RPC style response with result.tools
        if isinstance(msg, dict):
            result = msg.get("result")
            if isinstance(result, dict) and "tools" in result:
                tools = result["tools"]
                break

    if tools is None:
        raise RuntimeError(
            "Did not find tools in SSE stream. Raw events:\n"
            + "\n".join(raw_events[:50])
        )

    return tools, raw_events

def print_tools(tools):
    for tool in tools:
        name = tool.get("name", "<no name>")
        description = tool.get("description", "").strip().replace("\n", " ")
        print(f"Tool: {name}")
        if description:
            # Take only the first sentence (up to the first period).
            description = description.split(".")[0].strip()
            print(f"  Description: {description}")
        else:
            print("  Description: <none>")

        # Parameters: assuming JSON Schema-like object in tool["inputSchema"]
        input_schema = tool.get("inputSchema") or tool.get("parameters") or {}
        props = (input_schema.get("properties") or {}) if isinstance(input_schema, dict) else {}
        required = list(input_schema.get("required") or [])

        def format_param(name: str, schema: dict) -> str:
            ptype = schema.get("type")
            if isinstance(ptype, list):
                ptype = "|".join(ptype)
            if not ptype:
                ptype = "any"
            return f"{name} ({ptype})"

        # Required params line
        if not required:
            print("  Required: <none>")
        else:
            required_parts = []
            for param_name in required:
                schema = props.get(param_name, {})
                required_parts.append(format_param(param_name, schema))
            print("  Required: " + ", ".join(required_parts))

        # Optional params line
        optional_names = [name for name in props.keys() if name not in required]
        if not optional_names:
            print("  Optional: <none>")
        else:
            optional_parts = []
            for param_name in optional_names:
                schema = props.get(param_name, {})
                optional_parts.append(format_param(param_name, schema))
            print("  Optional: " + ", ".join(optional_parts))

        print()

def main():
    try:
        tools, _ = list_tools()
        print_tools(tools)
    except Exception as e:
        print(f"Error while listing tools: {e}")

if __name__ == "__main__":
    main()