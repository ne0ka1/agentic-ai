import json
import os
import requests

ASTA_MCP_URL = "https://asta-tools.allen.ai/mcp/v1"

def search_papers_and_print_top_5() -> None:
    """Call Asta's search_papers tool and print top 5 results."""
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json, text/event-stream",
        "x-api-key": os.environ["ASTA_API_KEY"],
    }

    payload = {
        "jsonrpc": "2.0",
        "id": 2,
        "method": "tools/call",
        "params": {
            "name": "snippet_search",
            "arguments": {
                "query": "large language model agents",
                "fields": "paper",
                "limit": 5,
            },
        },
    }

    # Stream SSE response and extract the JSON-RPC result from data: lines
    response = requests.post(
        ASTA_MCP_URL,
        headers=headers,
        json=payload,
        stream=True,
    )
    response.raise_for_status()

    result = None
    raw_sse_lines = []
    for line in response.iter_lines(decode_unicode=True):
        if not line:
            continue
        raw_sse_lines.append(line)
        if not line.startswith("data:"):
            continue

        data_str = line[len("data:") :].strip()
        if not data_str:
            continue

        try:
            msg = json.loads(data_str)
        except json.JSONDecodeError:
            continue

        if isinstance(msg, dict) and "result" in msg:
            result = msg
            break

    if result is None:
        print("Did not find tools/call result in SSE stream. Raw SSE lines:")
        for l in raw_sse_lines:
            print(l)
        raise RuntimeError("Did not find tools/call result in SSE stream")

    content_part = result["result"]["content"][0]
    # Debug: show raw content part before parsing
    # print("Raw content part from tools/call:")
    #print(json.dumps(content_part, indent=2)[:10000])

    # Normalize tool response into a list of paper dicts.
    papers = []

    content_type = content_part.get("type")
    if content_type == "json" and "json" in content_part:
        payload = content_part["json"]
    else:
        text = content_part.get("text", "").strip()
        if not text:
            print("No JSON content returned from search_papers.")
            return

        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            print("Could not decode JSON from tool response. Raw text:")
            print(text)
            return

    if isinstance(payload, list):
        papers = payload
    elif isinstance(payload, dict):
        # Try common field names that might hold the list.
        for key in ("papers", "results", "data", "items"):
            if isinstance(payload.get(key), list):
                papers = payload[key]
                break
        if not papers:
            # If it's a single paper dict, wrap it.
            papers = [payload]

    print(f"Parsed {len(papers)} papers from tool response.")

    # Print top 5 results as numbered list with title and year.
    for idx, paper in enumerate(papers[:5], start=1):
        # Some snippet_search results may nest the paper info.
        if "title" not in paper and isinstance(paper.get("paper"), dict):
            paper = paper["paper"]

        title = (paper.get("title") or "<no title>")
        print(f"{idx}. {title}")


if __name__ == "__main__":
    search_papers_and_print_top_5()