# backend/app/llm.py
"""
Thin wrappers for OpenAI embeddings + chat; JSON enforcement + streaming bridge.

Exports
-------
- embed_texts(texts, model, api_key) -> list[list[float]]
    Async embeddings with batching. Returns L2-normalized vectors.

- answer_stream_json(question, selection, context_blocks, model, api_key)
    Async generator that streams tokens from OpenAI (JSON mode), buffers the
    raw text, and on completion parses to JSON. If parsing fails, performs a
    one-shot "repair" call to coerce valid JSON. Yields dicts:
        {"type": "trace", "payload": {...}}
        {"type": "token", "payload": {"delta": str}}
        {"type": "final", "payload": {"answer": str, "citations": [...], "alternatives": [...]?}}

Design notes
------------
- The prompt strictly instructs the model to answer *only* from CONTEXT and to
  cite using the provided ALLOWED_CITATIONS table. We keep the context compact.
- Determinism: pass a fixed `seed` and `temperature=0`.
- Token budget: context is truncated to a soft budget using `tiktoken`.
"""
from __future__ import annotations

import asyncio
import math
from typing import Any, AsyncIterator, Dict, Iterable, List, Optional, Tuple

import orjson

# OpenAI (async client)
try:
    from openai import AsyncOpenAI  # type: ignore
except Exception:  # pragma: no cover - helpful error if import mismatch
    AsyncOpenAI = None  # type: ignore

# Tokenizer (optional; fallback to char/4 heuristic)
try:
    import tiktoken  # type: ignore
except Exception:  # pragma: no cover
    tiktoken = None  # type: ignore

# ---------------------------
# Embeddings
# ---------------------------


def _l2_normalize(vec: List[float]) -> List[float]:
    s = sum(x * x for x in vec)
    if s <= 0:
        return [0.0] * len(vec)
    inv = 1.0 / math.sqrt(s)
    return [x * inv for x in vec]


async def embed_texts(texts: List[str], model: str, api_key: str) -> List[List[float]]:
    """
    Compute embeddings for a list of texts with batching and return
    L2-normalized vectors. Order is preserved.

    Parameters
    ----------
    texts : list[str]
    model : str
    api_key : str

    Returns
    -------
    list[list[float]]
    """
    if AsyncOpenAI is None:
        raise RuntimeError("openai>=1.0 is required")

    client = AsyncOpenAI(api_key=api_key)

    BATCH = 128
    out: List[List[float]] = []
    for i in range(0, len(texts), BATCH):
        batch = texts[i : i + BATCH]
        resp = await client.embeddings.create(model=model, input=batch)
        # Ensure ordering
        for row in resp.data:
            vec = list(row.embedding)
            out.append(_l2_normalize(vec))
    return out


# ---------------------------
# Chat / JSON streaming
# ---------------------------


def _get_encoder():
    if tiktoken is None:
        return None
    # Prefer o-series encoding when available; fallback to cl100k_base
    try:
        return tiktoken.get_encoding("o200k_base")  # for GPT-4o family
    except Exception:
        return tiktoken.get_encoding("cl100k_base")


def _count_tokens(text: str) -> int:
    enc = _get_encoder()
    if enc is None:
        # ~4 chars per token rough rule
        return max(1, (len(text) + 3) // 4)
    return len(enc.encode(text))


def _truncate_context_items(
    items: List[Dict[str, Any]],
    token_budget: int,
    per_snippet_chars: int = 900,
) -> List[Dict[str, Any]]:
    """
    Keep as many context items as fit under token_budget tokens (approx).
    Snippets are clipped to per_snippet_chars to avoid long tails.
    """
    total = 0
    kept: List[Dict[str, Any]] = []
    for it in items:
        snippet = it.get("text", "")[:per_snippet_chars]
        rough = _count_tokens(snippet) + 16  # include metadata overhead
        if total + rough > token_budget and kept:
            break
        it = dict(it)
        it["text"] = snippet
        kept.append(it)
        total += rough
    return kept


def _format_context_and_citations(
    blocks: List[Dict[str, Any]],
    selection: Optional[str],
    token_budget: int = 6000,
) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Build a compact CONTEXT string and the ALLOWED_CITATIONS table.

    Returns (context_text, allowed_citations)
    """
    # Deduplicate by blockId (blocks may already be deduped, guard anyway)
    seen: set[str] = set()
    uniq: List[Dict[str, Any]] = []
    for b in blocks:
        bid = str(b["blockId"])
        if bid in seen:
            continue
        seen.add(bid)
        uniq.append(b)

    uniq = _truncate_context_items(uniq, token_budget)

    allowed = [
        {
            "blockId": str(b["blockId"]),
            "page": int(b.get("page", 0)),
            # anchorId may be empty here; downstream can map block -> derived anchor
            "anchorId": str(b.get("anchorId") or ""),
        }
        for b in uniq
    ]

    # Plain, compact representation to reduce tokens.
    lines = []
    if selection:
        lines.append(f"SELECTION: {selection.strip()}")
    lines.append("CONTEXT BLOCKS:")
    for i, b in enumerate(uniq, 1):
        sec = " / ".join(b.get("sectionPath") or [])
        src = b.get("source", "local")
        score = f"{float(b.get('score', 0.0)):.3f}"
        # Keep bbox/page as hints; client won't see this text.
        lines.append(
            f"[{i}] id={b['blockId']} page={b.get('page',0)} src={src} score={score} sec='{sec}' :: {b.get('text','')}"
        )
    ctx_text = "\n".join(lines)
    return ctx_text, allowed


_SYSTEM_INSTRUCTIONS = (
    "You are a faithful, terse scientific assistant. Answer strictly from CONTEXT.\n"
    "If uncertain, say so and present top alternatives with locations.\n"
    "For every non-trivial claim, attach citations as {blockId, page, anchorId}.\n"
    "Never invent definitions. Keep symbols verbatim.\n"
    "Output ONLY a single JSON object with schema:\n"
    '{ "answer": string, "citations": [ { "blockId": string, "page": number, "anchorId": string } ], '
    '"alternatives": [ { "title": string, "explanation": string, "citations": [ { "blockId": string, "page": number, "anchorId": string } ] } ]? }\n'
    "Citation rules:\n"
    " - Cite ONLY blocks listed in ALLOWED_CITATIONS (exact blockId and page).\n"
    " - Use the anchorId given there. If it is empty, still include it as an empty string (runtime will map it).\n"
    " - Prefer 1–3 citations that directly support each key claim.\n"
    "Do not include extra keys. No markdown, no prose outside JSON."
)


async def _chat_stream_json(
    *,
    model: str,
    api_key: str,
    messages: List[Dict[str, Any]],
    seed: int = 7,
) -> AsyncIterator[Dict[str, Any]]:
    """
    Internal helper to stream JSON-mode chat completion, yielding token deltas and final text.
    """
    if AsyncOpenAI is None:
        raise RuntimeError("openai>=1.0 is required")

    client = AsyncOpenAI(api_key=api_key)

    # Stream response
    stream = await client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0,
        stream=True,
        response_format={"type": "json_object"},
        seed=seed,
    )

    # Let caller know the LLM stage started
    yield {"type": "trace", "payload": {"stage": "llm", "meta": {"model": model}}}

    buf_parts: List[str] = []
    async for event in stream:
        try:
            delta = event.choices[0].delta.content  # type: ignore[attr-defined]
        except Exception:
            delta = None
        if delta:
            buf_parts.append(delta)
            yield {"type": "token", "payload": {"delta": delta}}

    # Final (non-stream) response object (contains usage etc.), not used here.
    try:
        await stream.get_final_response()
    except Exception:
        # Not fatal; some SDK versions require context manager for finalization.
        pass

    final_text = "".join(buf_parts).strip()
    yield {"type": "trace", "payload": {"stage": "llm", "meta": {"event": "done"}}}
    yield {"type": "final_text", "payload": final_text}  # internal marker for outer orchestrator


def _safe_json_loads(text: str) -> Optional[dict]:
    try:
        return orjson.loads(text)
    except Exception:
        # Try to extract a JSON object substring (first '{' ... last '}')
        try:
            first = text.find("{")
            last = text.rfind("}")
            if first != -1 and last != -1 and last > first:
                return orjson.loads(text[first : last + 1])
        except Exception:
            return None
    return None


async def _repair_json(
    *, model: str, api_key: str, broken_text: str, seed: int = 7
) -> Optional[dict]:
    """
    One-shot attempt to coerce valid JSON using the model in JSON mode.
    """
    if AsyncOpenAI is None:
        return None
    client = AsyncOpenAI(api_key=api_key)
    sys = (
        "You fix malformed JSON emitted by another model. Output only a valid JSON object. "
        "Preserve keys and structure if possible: "
        'answer (string), citations (array of {blockId,page,anchorId}), alternatives (optional array).'
    )
    msgs = [
        {"role": "system", "content": sys},
        {"role": "user", "content": broken_text},
    ]
    try:
        resp = await client.chat.completions.create(
            model=model,
            messages=msgs,
            temperature=0,
            response_format={"type": "json_object"},
            seed=seed,
        )
        text = (resp.choices[0].message.content or "").strip()  # type: ignore
        return _safe_json_loads(text)
    except Exception:
        return None


# ---------------------------
# Public API: answer_stream_json
# ---------------------------


async def answer_stream_json(
    question: str,
    selection: Optional[str],
    context_blocks: List[Dict[str, Any]],
    model: str,
    api_key: str,
) -> AsyncIterator[Dict[str, Any]]:
    """
    Stream a JSON answer grounded in CONTEXT.

    Yields
    ------
    AsyncIterator[dict]
        {"type":"token","payload":{"delta":str}}
        {"type":"trace","payload":{...}}
        {"type":"final","payload":{"answer":..., "citations":[...], "alternatives":[...]?}}
    """
    # 1) Build compact CONTEXT and ALLOWED_CITATIONS under a token budget
    ctx_text, allowed_citations = _format_context_and_citations(
        blocks=context_blocks, selection=selection, token_budget=6000
    )

    # 2) Compose messages
    user_prompt = (
        f"QUESTION:\n{question.strip()}\n\n"
        "You must answer strictly from the CONTEXT below and cite using ALLOWED_CITATIONS.\n\n"
        f"{ctx_text}\n\nALLOWED_CITATIONS:\n{orjson.dumps(allowed_citations).decode('utf-8')}\n"
    )

    messages = [
        {"role": "system", "content": _SYSTEM_INSTRUCTIONS},
        {"role": "user", "content": user_prompt},
    ]

    # 3) Stream model output (JSON mode)
    #     We intercept the internal "final_text" marker to parse/repair.
    async for evt in _chat_stream_json(model=model, api_key=api_key, messages=messages, seed=7):
        if evt.get("type") != "final_text":
            yield evt  # token or trace
            continue

        raw_text: str = evt["payload"]

        # 4) Parse JSON; if fails, run a one-shot repair
        parsed = _safe_json_loads(raw_text)
        if parsed is None:
            yield {"type": "trace", "payload": {"stage": "llm", "meta": {"repair": True}}}
            parsed = await _repair_json(model=model, api_key=api_key, broken_text=raw_text, seed=7)

        if parsed is None:
            # As a last resort, return a minimal, safe object.
            parsed = {
                "answer": "Sorry — I could not produce a valid JSON answer this time.",
                "citations": [],
            }

        # 5) Post-process citations: coerce to allowed entries and fill pages/anchorIds
        #    - ensure blockId exists in allowed list
        allow_map = {c["blockId"]: c for c in allowed_citations}
        fixed_cites: List[Dict[str, Any]] = []
        for c in (parsed.get("citations") or []):
            try:
                bid = str(c.get("blockId"))
                if not bid or bid not in allow_map:
                    continue
                page = int(c.get("page", allow_map[bid]["page"]))
                anchor_id = str(c.get("anchorId") or allow_map[bid].get("anchorId") or "")
                fixed_cites.append({"blockId": bid, "page": page, "anchorId": anchor_id})
            except Exception:
                continue
        parsed["citations"] = fixed_cites

        # Also normalize alternatives if present
        if "alternatives" in parsed and isinstance(parsed["alternatives"], list):
            alt_fixed = []
            for alt in parsed["alternatives"]:
                title = str(alt.get("title") or "").strip() or "Alternative"
                explanation = str(alt.get("explanation") or "").strip()
                cites = []
                for c in (alt.get("citations") or []):
                    try:
                        bid = str(c.get("blockId"))
                        if not bid or bid not in allow_map:
                            continue
                        page = int(c.get("page", allow_map[bid]["page"]))
                        anchor_id = str(c.get("anchorId") or allow_map[bid].get("anchorId") or "")
                        cites.append({"blockId": bid, "page": page, "anchorId": anchor_id})
                    except Exception:
                        continue
                alt_fixed.append({"title": title, "explanation": explanation, "citations": cites})
            parsed["alternatives"] = alt_fixed

        yield {"type": "final", "payload": parsed}
