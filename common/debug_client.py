"""Raw JSON-RPC/SSE debug client for any A2A agent in this repo.

Step 2 of the SPEC: a standalone script (not the chair) that talks to an
agent's Agent Card + `message/stream` (or `message/send`, for
non-streaming agents) directly over HTTP, and prints a readable, timed
trace of the events as they arrive.

Usage:
    python -m common.debug_client --url http://localhost:9001 --text "A2A adversarial robustness"

    # Agents whose skill takes a DataPart (critic, synthesis):
    python -m common.debug_client --url http://localhost:9003 --data '{"draft": "...", "sources": [...]}'
    python -m common.debug_client --url http://localhost:9002 --data-file payload.json

    # Continuing a paused (input-required) task with a follow-up reply:
    python -m common.debug_client --url http://localhost:9002 --text "prioritize source 0" \\
        --task-id <id> --context-id <id>

    # Disable preview truncation to inspect a full drafted answer, verdict
    # list, etc.:
    python -m common.debug_client --url http://localhost:9002 --data-file payload.json --full
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
import uuid
from pathlib import Path
from typing import Any

import httpx

TERMINAL_STATES = {"completed", "failed", "canceled"}
PREVIEW_MAX_LEN = 300
_NESTED_VALUE_MAX_LEN = 60

# Set from --full in main(); disables preview truncation everywhere below.
FULL_OUTPUT = False


def elapsed(t0: float) -> str:
    return f"+{time.monotonic() - t0:6.3f}s"


def extract_message_text(message: dict[str, Any] | None) -> str:
    if not message:
        return ""
    texts = [
        part.get("text", "")
        for part in message.get("parts", [])
        if part.get("kind") == "text"
    ]
    return " ".join(t for t in texts if t)


def format_value(value: Any, max_len: int = _NESTED_VALUE_MAX_LEN) -> str:
    """Renders a (possibly nested) DataPart value as a compact summary.

    Deliberately not `json.dumps` — this is a human-readable preview, not a
    raw dump. Dicts/lists render structurally; scalars are truncated.
    """
    if isinstance(value, dict):
        return "{" + ", ".join(f"{k}={format_value(v)}" for k, v in value.items()) + "}"
    if isinstance(value, list):
        return "[" + "; ".join(format_value(v) for v in value) + "]"
    text = str(value)
    if not FULL_OUTPUT and len(text) > max_len:
        text = text[: max_len - 1] + "…"
    return text


def preview_part(part: dict[str, Any]) -> str:
    kind = part.get("kind")
    if kind == "text":
        content = part.get("text", "")
    elif kind == "data":
        data = part.get("data", {})
        content = (
            ", ".join(f"{k}={format_value(v)}" for k, v in data.items())
            if isinstance(data, dict)
            else str(data)
        )
    elif kind == "file":
        file_info = part.get("file", {})
        content = file_info.get("name") or file_info.get("uri") or "<file>"
    else:
        content = str(part)

    content = " ".join(content.split())
    if not FULL_OUTPUT and len(content) > PREVIEW_MAX_LEN:
        content = content[: PREVIEW_MAX_LEN - 1] + "…"
    return content


def print_status(
    t0: float,
    state: str,
    message: dict[str, Any] | None,
    ids_suffix: str = "",
) -> None:
    text = extract_message_text(message)
    text_suffix = f" — {text}" if text else ""
    print(f"[{elapsed(t0)}] [STATUS] {state}{text_suffix}{ids_suffix}")


def print_artifact(t0: float, event: dict[str, Any]) -> None:
    artifact = event.get("artifact", {})
    append = event.get("append", False)
    last_chunk = event.get("lastChunk", False)
    previews = [preview_part(p) for p in artifact.get("parts", [])]
    preview = "; ".join(previews)
    print(
        f"[{elapsed(t0)}] [ARTIFACT] append={append} lastChunk={last_chunk} "
        f"{preview}"
    )


def handle_envelope(data: dict[str, Any], t0: float) -> bool:
    """Prints one JSON-RPC response/event. Returns True if the stream is done."""
    if "error" in data:
        print(f"[{elapsed(t0)}] [ERROR] {data['error']}")
        return True

    result = data.get("result", data)
    kind = result.get("kind")

    if kind in ("task", "status-update"):
        status = result["status"] if kind == "task" else result.get("status", {})
        state = status.get("state", "unknown")
        ids_suffix = ""
        if kind == "task":
            ids_suffix = f" (task={result.get('id')}, context={result.get('contextId')})"
        print_status(t0, state, status.get("message"), ids_suffix)
        if state == "input-required":
            return True
        return bool(result.get("final")) or state in TERMINAL_STATES

    if kind == "artifact-update":
        print_artifact(t0, result)
        return False

    if kind == "message":
        text = extract_message_text(result)
        print(f"[{elapsed(t0)}] [MESSAGE] {text}")
        return True

    dumped = json.dumps(result)
    if not FULL_OUTPUT:
        dumped = dumped[:PREVIEW_MAX_LEN]
    print(f"[{elapsed(t0)}] [OTHER] {dumped}")
    return False


async def fetch_agent_card(client: httpx.AsyncClient, base_url: str) -> dict[str, Any]:
    resp = await client.get(f"{base_url.rstrip('/')}/.well-known/agent-card.json")
    resp.raise_for_status()
    return resp.json()


def print_agent_card_summary(card: dict[str, Any]) -> None:
    skill_ids = [s.get("id", "?") for s in card.get("skills", [])]
    streaming = card.get("capabilities", {}).get("streaming", False)
    print(f"Agent: {card.get('name', '?')}")
    print(f"Skills: {', '.join(skill_ids) or '(none)'}")
    print(f"Streaming: {streaming}")
    print()


async def run_streaming(
    client: httpx.AsyncClient, rpc_url: str, message: dict[str, Any], t0: float
) -> None:
    payload = {
        "jsonrpc": "2.0",
        "id": "1",
        "method": "message/stream",
        "params": {"message": message},
    }
    async with client.stream(
        "POST", rpc_url, json=payload, headers={"Accept": "text/event-stream"}
    ) as resp:
        resp.raise_for_status()
        async for line in resp.aiter_lines():
            if not line.startswith("data:"):
                continue
            data = json.loads(line[len("data:") :].strip())
            if handle_envelope(data, t0):
                return


async def run_send(
    client: httpx.AsyncClient, rpc_url: str, message: dict[str, Any], t0: float
) -> None:
    payload = {
        "jsonrpc": "2.0",
        "id": "1",
        "method": "message/send",
        "params": {"message": message},
    }
    resp = await client.post(rpc_url, json=payload)
    resp.raise_for_status()
    handle_envelope(resp.json(), t0)


def build_message(args: argparse.Namespace) -> dict[str, Any]:
    if args.data is not None:
        try:
            data = json.loads(args.data)
        except json.JSONDecodeError as exc:
            raise SystemExit(f"--data is not valid JSON: {exc}") from exc
        parts = [{"kind": "data", "data": data}]
    elif args.data_file is not None:
        data = json.loads(Path(args.data_file).read_text())
        parts = [{"kind": "data", "data": data}]
    else:
        parts = [{"kind": "text", "text": args.text}]

    message: dict[str, Any] = {
        "role": "user",
        "messageId": str(uuid.uuid4()),
        "parts": parts,
    }
    if args.task_id:
        message["taskId"] = args.task_id
    if args.context_id:
        message["contextId"] = args.context_id
    return message


async def async_main(base_url: str, message: dict[str, Any]) -> None:
    async with httpx.AsyncClient(timeout=httpx.Timeout(10.0, read=60.0)) as client:
        card = await fetch_agent_card(client, base_url)
        print_agent_card_summary(card)

        rpc_url = card.get("url") or base_url
        t0 = time.monotonic()
        streaming = bool(card.get("capabilities", {}).get("streaming"))
        if streaming:
            await run_streaming(client, rpc_url, message, t0)
        else:
            await run_send(client, rpc_url, message, t0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True, help="Agent base URL, e.g. http://localhost:9001")
    payload_group = parser.add_mutually_exclusive_group(required=True)
    payload_group.add_argument("--text", help="Send a TextPart message")
    payload_group.add_argument("--data", help="Send a DataPart message (inline JSON string)")
    payload_group.add_argument("--data-file", help="Send a DataPart message (JSON loaded from this file)")
    parser.add_argument(
        "--task-id",
        help="Continue an existing task (e.g. replying to an input-required prompt)",
    )
    parser.add_argument("--context-id", help="Context ID to attach to the message")
    parser.add_argument(
        "--full",
        action="store_true",
        help="Disable preview truncation — print full Part/artifact content",
    )
    args = parser.parse_args()

    global FULL_OUTPUT
    FULL_OUTPUT = args.full

    message = build_message(args)
    try:
        asyncio.run(async_main(args.url, message))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
