# backend/app/retrieval.py
"""
Retrieval engine: local cluster + vector + lexical + RRF + light rerank + context pack.

Overview
--------
Given a user's selection anchor, we:
 1) Resolve the most likely block for that anchor (bbox/char overlap).
 2) Build a *local cluster* around that block on the same page (orderIdx ±k).
 3) Run global semantic search via Atlas `$vectorSearch`.
 4) Run lexical BM25 via Atlas `$search` with highlight enabled.
 5) Fuse results with Reciprocal Rank Fusion (RRF) and lightly rerank.
 6) Assemble a compact context pack with deduped blocks (local pinned first).

Return shape:
  { anchor, selectionText, blocks: [
      { blockId, page, sectionPath, bboxNorm, text, score, source: "local"|"vector"|"lexical" }
    ] }

Notes
-----
- All coordinates are normalized [0..1].
- Multi-tenant: All queries are filtered by docId; caller should ensure orgId
  and access via the auth layer (this module assumes docId is already vetted).
- `$vectorSearch` and `$search` index names are provided by `db.py`.
"""
from __future__ import annotations

import math
import re
from typing import Any, AsyncIterator, Dict, Iterable, List, Optional, Sequence, Tuple

from motor.motor_asyncio import AsyncIOMotorDatabase

from .db import blocks_search_index_name, blocks_vector_index_name
from .schemas import Bbox

# ---------------------------------------------------------------------------
# Small math/text helpers
# ---------------------------------------------------------------------------

DEFN_CUE_RE = re.compile(r"\b(where|let|denote|means)\b", flags=re.IGNORECASE)
DEFINE_INTENT_RE = re.compile(r"\b(define|what\s+is|meaning\s+of|denote|stands\s+for)\b", re.IGNORECASE)
TOKEN_RE = re.compile(r"(?<![A-Za-z0-9_])([A-Za-z](?:_[A-Za-z0-9]+)?)")


def _iou(a: Bbox, b: Bbox) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    aw = max(0.0, ax1 - ax0)
    ah = max(0.0, ay1 - ay0)
    bw = max(0.0, bx1 - bx0)
    bh = max(0.0, by1 - by0)
    union = aw * ah + bw * bh - inter
    return (inter / union) if union > 0 else 0.0


def _centroid(b: Bbox) -> Tuple[float, float]:
    x0, y0, x1, y1 = b
    return (0.5 * (x0 + x1), 0.5 * (y0 + y1))


def _char_overlap(anchor: Tuple[int, int], ranges: List[Dict[str, int]]) -> float:
    """Compute fraction of the anchor [start,end) overlapping any of the block's charRanges."""
    a0, a1 = anchor
    if a1 <= a0:
        return 0.0
    total = 0
    for r in ranges or []:
        r0, r1 = int(r.get("start", 0)), int(r.get("end", 0))
        inter = max(0, min(a1, r1) - max(a0, r0))
        total += inter
    return float(total) / float(a1 - a0)


def _shorten(text: str, max_len: int = 1200) -> str:
    """Trim long blocks (we'll keep the first ~1200 chars to stay token-lean)."""
    return text if len(text) <= max_len else text[: max_len - 1] + "…"


def _as_ctx_block(doc: dict, *, score: float, source: str) -> dict:
    """Normalize a Mongo block doc into the context block shape."""
    return {
        "blockId": str(doc["_id"]),
        "page": int(doc.get("page", 0)),
        "sectionPath": list(doc.get("sectionPath", [])),
        "bboxNorm": tuple(doc.get("bboxNorm", (0.0, 0.0, 1.0, 1.0))),
        "text": _shorten(doc.get("text", "")),
        "score": float(score),
        "source": source,
        # internal hints (not part of spec, but useful internally)
        "_orderIdx": int(doc.get("orderIdx", 0)),
        "_type": doc.get("type", "paragraph"),
    }


# ---------------------------------------------------------------------------
# Anchor resolution
# ---------------------------------------------------------------------------


async def block_for_anchor(db: AsyncIOMotorDatabase, anchor: dict) -> Optional[dict]:
    """
    Resolve the block that best matches the given anchor on a page using:
      1) explicit anchor.blockId if present
      2) bbox IoU + charRange overlap
      3) fallback: nearest centroid distance on the page
    """
    doc_id = anchor["docId"]
    page = int(anchor["page"])
    anchor_bbox: Bbox = tuple(anchor.get("bboxNorm", (0.0, 0.0, 0.0, 0.0)))  # type: ignore
    a_char = (int(anchor.get("charStart", 0)), int(anchor.get("charEnd", 0)))

    # 1) Explicit link
    if anchor.get("blockId"):
        blk = await db["blocks"].find_one(
            {"_id": anchor["blockId"], "docId": doc_id, "page": page},
            {"_id": 1, "page": 1, "orderIdx": 1, "bboxNorm": 1, "text": 1, "sectionPath": 1, "type": 1, "charRanges": 1},
        )
        if blk:
            return blk

    # 2) Score by IoU + char overlap
    cursor = db["blocks"].find(
        {"docId": doc_id, "page": page},
        {"_id": 1, "page": 1, "orderIdx": 1, "bboxNorm": 1, "text": 1, "sectionPath": 1, "type": 1, "charRanges": 1},
        sort=[("orderIdx", 1)],
    )
    best = None
    best_score = -1.0
    a_cx, a_cy = _centroid(anchor_bbox)
    for blk in await cursor.to_list(length=1000):
        bb = tuple(blk.get("bboxNorm", (0.0, 0.0, 0.0, 0.0)))  # type: ignore
        iou = _iou(anchor_bbox, bb)
        cover = 0.0
        if a_char[1] > a_char[0]:
            cover = _char_overlap(a_char, blk.get("charRanges") or [])
        score = 0.7 * iou + 0.3 * cover
        if score > best_score:
            best, best_score = blk, score

    if best and best_score > 0:
        return best

    # 3) Nearest centroid if nothing overlapped
    cursor = db["blocks"].find(
        {"docId": doc_id, "page": page},
        {"_id": 1, "page": 1, "orderIdx": 1, "bboxNorm": 1, "text": 1, "sectionPath": 1, "type": 1},
        sort=[("orderIdx", 1)],
    )
    best, best_d = None, float("inf")
    for blk in await cursor.to_list(length=1000):
        bb = tuple(blk.get("bboxNorm", (0.0, 0.0, 0.0, 0.0)))  # type: ignore
        cx, cy = _centroid(bb)
        d = math.hypot(cx - a_cx, cy - a_cy)
        if d < best_d:
            best, best_d = blk, d
    return best


# ---------------------------------------------------------------------------
# Local cluster
# ---------------------------------------------------------------------------


async def local_cluster(
    db: AsyncIOMotorDatabase, doc_id: str, page: int, order_idx: int, k: int = 3
) -> List[dict]:
    """
    Return neighboring blocks on the same page within orderIdx ±k.
    We slightly prefer definitional text near equations.
    """
    lo, hi = max(0, order_idx - k), order_idx + k
    cursor = db["blocks"].find(
        {"docId": doc_id, "page": page, "orderIdx": {"$gte": lo, "$lte": hi}},
        {
            "_id": 1,
            "page": 1,
            "orderIdx": 1,
            "bboxNorm": 1,
            "text": 1,
            "sectionPath": 1,
            "type": 1,
        },
        sort=[("orderIdx", 1)],
    )

    results: List[dict] = []
    for b in await cursor.to_list(length=100):
        delta = abs(int(b.get("orderIdx", 0)) - order_idx)
        base = 1.0 / (1.0 + float(delta))
        text = b.get("text", "")
        boost = 0.0
        if DEFN_CUE_RE.search(text):
            boost += 0.25
        if b.get("type") == "equation":
            boost += 0.1
        score = base + boost
        results.append(_as_ctx_block(b, score=score, source="local"))

    # Keep deterministic order primarily by orderIdx, then score just in case.
    results.sort(key=lambda d: (d["_orderIdx"], -d["score"]))
    return results


# ---------------------------------------------------------------------------
# Global search (vector & lexical)
# ---------------------------------------------------------------------------


async def vector_search(
    db: AsyncIOMotorDatabase,
    doc_id: str,
    query_vec: List[float],
    filters: Dict[str, Any],
    k: int = 48,
    ef: Optional[int] = None,
) -> List[dict]:
    """
    Atlas `$vectorSearch` over blocks.embedding.vector.

    Parameters
    ----------
    filters : dict
        Additional MQL filters merged with {'docId': doc_id}.
        Example: {'type': {'$in': ['paragraph', 'equation', 'caption']}}
    """
    filt = {"docId": doc_id}
    if filters:
        filt.update(filters)

    index_name = blocks_vector_index_name()
    pipeline = [
        {
            "$vectorSearch": {
                "index": index_name,
                "path": "embedding.vector",
                "queryVector": query_vec,
                "numCandidates": int(ef or max(400, k * 8)),
                "limit": int(k),
                "filter": filt,
            }
        },
        {
            "$project": {
                "_id": 1,
                "docId": 1,
                "page": 1,
                "orderIdx": 1,
                "bboxNorm": 1,
                "text": 1,
                "sectionPath": 1,
                "type": 1,
                "score": {"$meta": "vectorSearchScore"},
            }
        },
    ]
    rows = [r async for r in db["blocks"].aggregate(pipeline)]
    return [_as_ctx_block(r, score=float(r.get("score", 0.0)), source="vector") for r in rows]


async def lexical_search(
    db: AsyncIOMotorDatabase,
    doc_id: str,
    text: str,
    selected: Optional[str],
    k: int = 48,
) -> List[dict]:
    """
    Atlas Search BM25 ($search) over blocks.text with highlight enabled.
    Uses a compound 'should' query to mix the question and the selected text.
    """
    idx = blocks_search_index_name()

    should: List[dict] = []
    if text and text.strip():
        should.append({"text": {"query": text.strip(), "path": "text"}})
    if selected and selected.strip():
        should.append({"text": {"query": selected.strip(), "path": "text"}})
    if not should:
        # Degenerate fallback to avoid $search failure on empty 'should'
        should.append({"text": {"query": " ", "path": "text"}})

    pipeline = [
        {
            "$search": {
                "index": idx,
                "compound": {
                    "should": should,
                    "minimumShouldMatch": 1,
                },
                "highlight": {"path": "text"},
                "returnStoredSource": True,
            }
        },
        {"$match": {"docId": doc_id}},
        {"$match": {"type": {"$in": ["paragraph", "equation", "caption"]}}},
        {
            "$project": {
                "_id": 1,
                "docId": 1,
                "page": 1,
                "orderIdx": 1,
                "bboxNorm": 1,
                "text": 1,
                "sectionPath": 1,
                "type": 1,
                "score": {"$meta": "searchScore"},
                "highlights": {"$meta": "searchHighlights"},
            }
        },
        {"$limit": int(k)},
    ]
    rows = [r async for r in db["blocks"].aggregate(pipeline)]
    return [_as_ctx_block(r, score=float(r.get("score", 0.0)), source="lexical") for r in rows]


# ---------------------------------------------------------------------------
# Fusion & rerank
# ---------------------------------------------------------------------------


def rrf_fuse(vec: List[dict], lex: List[dict], k: int = 60, top: int = 16) -> List[dict]:
    """
    Reciprocal Rank Fusion:
        score = Σ 1 / (k + rank_i)
    Dedup by blockId. Keep the doc payload from the better-ranked list.
    """
    # Rank maps (1-based)
    v_rank = {d["blockId"]: i + 1 for i, d in enumerate(vec)}
    l_rank = {d["blockId"]: i + 1 for i, d in enumerate(lex)}

    merged: Dict[str, dict] = {}
    for d in vec:
        merged[d["blockId"]] = dict(d)  # copy
    for d in lex:
        if d["blockId"] not in merged:
            merged[d["blockId"]] = dict(d)
        else:
            # Prefer the one with the better individual rank
            if l_rank.get(d["blockId"], 10**9) < v_rank.get(d["blockId"], 10**9):
                merged[d["blockId"]] = dict(d)

    for bid, d in merged.items():
        score = 0.0
        if bid in v_rank:
            score += 1.0 / (k + v_rank[bid])
        if bid in l_rank:
            score += 1.0 / (k + l_rank[bid])
        d["score"] = float(d.get("score", 0.0)) + score
        # Normalize source to one of the allowed labels
        if bid in v_rank and bid in l_rank:
            # choose the modality with the *better* rank
            d["source"] = "vector" if v_rank[bid] <= l_rank[bid] else "lexical"
        elif bid in v_rank:
            d["source"] = "vector"
        else:
            d["source"] = "lexical"

    ranked = sorted(merged.values(), key=lambda x: x.get("score", 0.0), reverse=True)
    return ranked[:top]


async def rerank(db: AsyncIOMotorDatabase, docs: List[dict], question: str, selection: Optional[str]) -> List[dict]:
    """
    Lightweight heuristic rerank (no external calls by default).

    Heuristics:
      - Boost definitional cues: 'where|let|denote|means'
      - If user intent looks like a definition ('what is', 'meaning of', 'define'),
        boost blocks with definitional cues even more.
      - Small bonus for equations (often followed by explanations).
      - If the selected token(s) appears in text, boost.
    """
    define_intent = bool(DEFINE_INTENT_RE.search(question or ""))
    sel_tokens = set(t.group(1) for t in TOKEN_RE.finditer(selection or ""))

    reranked: List[dict] = []
    for d in docs:
        boost = 0.0
        text = d.get("text", "")
        if DEFN_CUE_RE.search(text):
            boost += 0.25 if define_intent else 0.15
        if d.get("_type") == "equation":
            boost += 0.05
        if sel_tokens:
            for tok in sel_tokens:
                if re.search(rf"(?<![A-Za-z0-9_]){re.escape(tok)}(?![A-Za-z0-9_])", text):
                    boost += 0.08
                    break
        # Distance from top can be smoothed here if needed.
        d = dict(d)
        d["score"] = float(d.get("score", 0.0)) + boost
        reranked.append(d)

    reranked.sort(key=lambda x: x.get("score", 0.0), reverse=True)
    return reranked


# ---------------------------------------------------------------------------
# Context pack assembly
# ---------------------------------------------------------------------------


async def build_context_pack(
    db: AsyncIOMotorDatabase,
    orgId: str,
    doc_id: str,
    anchor_id: str,
    question: str,
    embed_fn,
) -> dict:
    """
    Build the complete retrieval context:

      1) Load anchor (org/doc-checked) and resolve primary block.
      2) Embed the combined query (question + selected text).
      3) Run vector + lexical, fuse + rerank.
      4) Assemble [LocalCluster (pinned)] + [Top fused] with dedup.

    Parameters
    ----------
    embed_fn : callable
        Async function: await embed_fn([text]) -> [[float]]
    """
    # 1) Anchor
    anchor = await db["anchors"].find_one({"_id": anchor_id, "orgId": orgId, "docId": doc_id})
    if not anchor:
        raise ValueError("Anchor not found or not accessible")

    selection_text: str = anchor.get("text", "") or ""
    # Resolve the primary block for local clustering
    blk = await block_for_anchor(db, anchor)  # may be None if page empty
    order_idx = int(blk.get("orderIdx", 0)) if blk else 0
    page = int(anchor["page"])

    # 2) Embed combined query
    query_text = question.strip()
    if selection_text:
        query_text = f"{question.strip()}\n\n{selection_text.strip()}"
    [query_vec] = await embed_fn([query_text])

    # 3) Global searches
    filters = {"type": {"$in": ["paragraph", "equation", "caption"]}}
    vec = await vector_search(db, doc_id, query_vec, filters=filters, k=48)
    lex = await lexical_search(db, doc_id, text=question, selected=selection_text, k=48)
    fused = rrf_fuse(vec, lex, k=60, top=16)
    fused = await rerank(db, fused, question=question, selection=selection_text)

    # 4) Local cluster (pinned)
    local = await local_cluster(db, doc_id=doc_id, page=page, order_idx=order_idx, k=3)

    # Deduplicate while preserving order: local first, then fused tail
    seen: set[str] = set()
    blocks: List[dict] = []
    for src in (local, fused):
        for d in src:
            bid = d["blockId"]
            if bid in seen:
                continue
            seen.add(bid)
            # Strip internal hints
            d.pop("_orderIdx", None)
            d.pop("_type", None)
            blocks.append(d)

    # Limit final context size (local pinned + top fused within token budget handled later)
    return {
        "anchor": {
            "anchorId": anchor_id,
            "docId": doc_id,
            "page": page,
            "bboxNorm": tuple(anchor.get("bboxNorm", (0.0, 0.0, 0.0, 0.0))),
            "text": selection_text,
            "blockId": str(blk["_id"]) if blk else None,
        },
        "selectionText": selection_text,
        "blocks": blocks,
    }


__all__ = [
    "block_for_anchor",
    "local_cluster",
    "vector_search",
    "lexical_search",
    "rrf_fuse",
    "rerank",
    "build_context_pack",
]
