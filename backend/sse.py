# backend/app/sse.py
"""
Server-Sent Events (SSE) primitives.

Exports
-------
- sse_event(event: str, data: dict | str) -> bytes
    Formats a single SSE event frame (UTF-8 encoded bytes).

- keepalive_comment() -> bytes
    Produces a heartbeat comment frame like ':ping\\n\\n'.

- stream_lines(source, keepalive: float = 20.0) -> AsyncIterator[bytes]
    Tiny adapter: consume an async iterator of events and yield SSE byte frames.
    Accepts items shaped as:
      * (event: str, data: dict|str) tuple
      * {"event": str, "data": any} dict
      * {"type": str, "payload": any} dict
      * raw bytes (already SSE framed)
      * str (sent as 'message' event)
      * any other object (JSON-encoded as 'message')
"""
from __future__ import annotations

import asyncio
from typing import Any, AsyncIterator, Dict, Tuple, Union

import orjson

SSELike = Union[
    Tuple[str, Union[dict, str]],
    Dict[str, Any],
    bytes,
    str,
    Any,
]


# ---------------------------------------------------------------------------
# Encoding helpers
# ---------------------------------------------------------------------------


def _json_dumps(obj: Any) -> str:
    """Serialize any Python object to a compact JSON string (UTF-8)."""
    return orjson.dumps(obj, option=orjson.OPT_SERIALIZE_NUMPY).decode("utf-8")


def _sse_data_lines(data: Union[dict, str]) -> list[str]:
    """Convert payload to list of 'data: ...' lines per SSE spec."""
    if isinstance(data, str):
        payload = data
    else:
        payload = _json_dumps(data)
    # Each physical line must be prefixed with 'data: '
    # Normalize CRLF -> LF to be safe.
    lines = payload.replace("\r\n", "\n").split("\n") or [""]
    return [f"data: {ln}" for ln in lines]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def sse_event(event: str, data: dict | str) -> bytes:
    """
    Build a single SSE frame.

    Parameters
    ----------
    event : str
        Event name (e.g., 'token', 'trace', 'final').
    data : dict | str
        Payload; dict is JSON-encoded, str is sent verbatim.

    Returns
    -------
    bytes
        UTF-8 encoded SSE frame terminated by a blank line.
    """
    # Event field must be a single line
    ev = (event or "message").splitlines()[0].strip() or "message"
    parts = [f"event: {ev}"]
    parts.extend(_sse_data_lines(data))
    # End of event
    frame = "\n".join(parts) + "\n\n"
    return frame.encode("utf-8")


def keepalive_comment(comment: str = "ping") -> bytes:
    """
    Produce a heartbeat comment frame. Browsers ignore comment lines but they keep
    the connection warm on proxies/load balancers.

    Example: b":ping\\n\\n"
    """
    return f":{comment}\n\n".encode("utf-8")


async def stream_lines(source: AsyncIterator[SSELike], keepalive: float = 20.0) -> AsyncIterator[bytes]:
    """
    Adapt an async iterator of heterogeneous items into SSE-framed byte chunks.

    This helper also emits periodic heartbeat comments if no data is produced
    within `keepalive` seconds, without cancelling the underlying iterator.

    Parameters
    ----------
    source : AsyncIterator[SSELike]
        Items can be:
          - (event, data) tuple
          - {"event": ..., "data": ...}
          - {"type": ..., "payload": ...}
          - bytes (already SSE framed)
          - str (sent as 'message')
          - any other object (JSON-encoded as 'message')
    keepalive : float
        Seconds between heartbeats when idle.

    Yields
    ------
    AsyncIterator[bytes]
        Properly framed SSE chunks ready to write to the HTTP response.
    """
    it = source.__aiter__()
    # Prime the first fetch task
    next_task = asyncio.create_task(it.__anext__())

    while True:
        done, _pending = await asyncio.wait({next_task}, timeout=keepalive)
        if not done:
            # No item arrived in time — send a heartbeat, keep waiting.
            yield keepalive_comment()
            continue

        try:
            item = next_task.result()
        except StopAsyncIteration:
            # Source is exhausted.
            break

        # Schedule the next item fetch immediately (keeps the pipeline warm).
        next_task = asyncio.create_task(it.__anext__())

        # Fast path: bytes already framed
        if isinstance(item, (bytes, bytearray)):
            yield bytes(item)
            continue

        # Tuple form: (event, data)
        if isinstance(item, tuple) and len(item) == 2:
            ev, payload = item
            yield sse_event(str(ev), payload if isinstance(payload, (dict, str)) else _json_dumps(payload))
            continue

        # Dict forms
        if isinstance(item, dict):
            if "event" in item and "data" in item:
                yield sse_event(str(item["event"]), item["data"])
                continue
            if "type" in item and "payload" in item:
                yield sse_event(str(item["type"]), item["payload"])
                continue
            # Fallback: treat as message with JSON body excluding typical event keys
            ev = str(item.get("event") or item.get("type") or "message")
            data = item.get("data")
            if data is None and "payload" in item:
                data = item["payload"]
            if data is None:
                data = {k: v for k, v in item.items() if k not in {"event", "type"}}
            yield sse_event(ev, data if isinstance(data, (dict, str)) else _json_dumps(data))
            continue

        # String form
        if isinstance(item, str):
            yield sse_event("message", item)
            continue

        # Generic object fallback
        yield sse_event("message", _json_dumps(item))


__all__ = ["sse_event", "keepalive_comment", "stream_lines"]
