# backend/app/routers.py
"""
All FastAPI routes in one file.

Routes
------
GET  /health                                -> { ok: true }

Auth:
POST /auth/google/exchange                  -> verify Google ID token, upsert user, issue backend JWT

Uploads:
POST /upload/init                           -> { docId, uploadUrl? | path? }
POST /upload/complete                       -> { docId }  (kicks off ingestion in background)

Documents:
GET  /doc/{docId}/status                    -> ingestion status & doc metadata

Anchors:
POST /anchor                                -> create anchor, returns { anchorId }
GET  /citation/{anchorId}                   -> full anchor payload for evidence panel

Query:
POST /query                                 -> allocate queryId, return { queryId, streamUrl }
GET  /query/{id}/stream                     -> SSE stream of LLM answer; persists message & retrieval log

Threads:
POST /threads                               -> create/upsert thread for (user,org,doc)
GET  /threads/{id}/messages                 -> latest first, limit=50

Notes
-----
- Multi-tenancy enforced via backend JWT (see auth_ctx()).
- SSE streaming uses `EventSource` protocol with events: 'token', 'trace', 'final'.
- For S3 uploads, we pre-sign a PUT URL; on complete we download to local path
  and run ingestion uniformly from disk.
"""
from __future__ import annotations

import asyncio
import hashlib
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple, Union

from bson import ObjectId
from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    HTTPException,
    Request,
    Response,
    status,
)
from fastapi.responses import JSONResponse, StreamingResponse
from motor.motor_asyncio import AsyncIOMotorDatabase
from pymongo import ReturnDocument

from .auth import (
    AuthContext,
    auth_ctx,
    ensure_user,
    issue_backend_jwt,
    verify_google_id_token,
)
from .config import Settings, get_settings
from .db import get_db
from .ingestion import ingest_document
from .llm import answer_stream_json, embed_texts
from .retrieval import build_context_pack
from .schemas import (
    AnchorDoc,
    AuthExchangeIn,
    AuthExchangeOut,
    CitationOut,
    CreateAnchorIn,
    CreateAnchorOut,
    DocStatusOut,
    ObjectIdStr,
    QueryIn,
    QueryOut,
    UserDoc,
)
from .sse import keepalive_comment, sse_event

router = APIRouter()

# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------

def _new_id() -> str:
    return str(ObjectId())


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _count_tokens_fast(text: str) -> int:
    """
    Small token counter: prefer tiktoken if available; fallback to ~4 chars/token.
    """
    try:
        import tiktoken  # type: ignore
        try:
            enc = tiktoken.get_encoding("o200k_base")
        except Exception:
            enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except Exception:
        return max(1, (len(text) + 3) // 4)


# Ephemeral store for pending queries (id -> payload). Simple & per-process.
_PENDING_QUERIES: Dict[str, Dict[str, Any]] = {}
_PENDING_LOCK = asyncio.Lock()


async def _put_pending(qid: str, payload: Dict[str, Any]) -> None:
    async with _PENDING_LOCK:
        _PENDING_QUERIES[qid] = payload


async def _pop_pending(qid: str) -> Optional[Dict[str, Any]]:
    async with _PENDING_LOCK:
        return _PENDING_QUERIES.pop(qid, None)


# -----------------------------------------------------------------------------
# Routes
# -----------------------------------------------------------------------------

@router.get("/health")
async def health() -> Dict[str, bool]:
    return {"ok": True}


# ----------------------------- AUTH -----------------------------------------


@router.post("/auth/google/exchange", response_model=AuthExchangeOut)
async def auth_google_exchange(body: AuthExchangeIn, request: Request) -> AuthExchangeOut:
    """
    Exchange a Google ID token for a backend JWT. Upserts the user document.
    """
    db: AsyncIOMotorDatabase = get_db()
    claims = await verify_google_id_token(body.idToken)
    user: UserDoc = await ensure_user(db, claims)

    token = issue_backend_jwt(sub=user._id, orgId=user.orgId, email=user.email)
    return AuthExchangeOut(accessToken=token, user={"_id": user._id, "email": user.email, "name": user.name, "picture": user.picture})


# ----------------------------- UPLOADS --------------------------------------


@router.post("/upload/init")
async def upload_init(request: Request, ctx: AuthContext = Depends(auth_ctx)) -> Dict[str, Any]:
    """
    Allocate a document record and return where to upload the raw PDF.
    S3 -> pre-signed PUT URL; local -> filesystem path (for dev tooling / server-side upload).
    """
    db = get_db()
    settings = get_settings()

    # Optional filename hint for nicer S3 keys (clients may pass ?filename=paper.pdf)
    filename = request.query_params.get("filename") or "upload.pdf"

    doc_id = _new_id()
    key = f"docs/{doc_id}/raw/{filename}"
    now = _now()

    doc = {
        "_id": doc_id,
        "ownerId": ctx.userId,
        "orgId": ctx.orgId,
        "sha256": "",
        "title": filename,
        "pages": 0,
        "mime": "application/pdf",
        "isOCR": False,
        "uploadAt": now,
        "ingest": {"status": "uploaded", "errors": [], "stats": {"blocks": 0, "ocrLines": 0}},
        "s3": None,
    }

    out: Dict[str, Any] = {"docId": doc_id}

    if settings.use_s3():
        s3 = settings.s3_client()
        bucket = settings.S3_BUCKET or ""
        # Pre-sign a PUT to upload the raw PDF
        upload_url = s3.generate_presigned_url(
            ClientMethod="put_object",
            Params={"Bucket": bucket, "Key": key, "ContentType": "application/pdf"},
            ExpiresIn=900,
        )
        doc["s3"] = {"bucket": bucket, "key": key}
        out["uploadUrl"] = upload_url
    else:
        # Local path within UPLOAD_DIR; client-side cannot write this path directly,
        # but dev scripts / server-side proxy can.
        local_path = get_settings().local_path_for_key(key)
        out["path"] = local_path

    await db["documents"].insert_one(doc)
    return out


@router.post("/upload/complete")
async def upload_complete(
    body: Dict[str, ObjectIdStr],
    background: BackgroundTasks,
    ctx: AuthContext = Depends(auth_ctx),
) -> Dict[str, Any]:
    """
    Mark upload complete and trigger ingestion as a background task.
    If STORAGE_BACKEND='s3', download to local path first.
    """
    db = get_db()
    settings = get_settings()
    doc_id = body.get("docId")
    if not doc_id:
        raise HTTPException(status_code=400, detail="docId is required")

    # Verify access to doc (org-level)
    doc = await db["documents"].find_one({"_id": doc_id, "orgId": ctx.orgId})
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    # Determine local path for ingestion
    if doc.get("s3") and settings.use_s3():
        bucket = doc["s3"]["bucket"]
        key = doc["s3"]["key"]
        local_path = settings.local_path_for_key(key)
        # Download from S3 to local path
        s3 = settings.s3_client()

        async def _dl():
            await asyncio.to_thread(s3.download_file, bucket, key, local_path)

        await _dl()
        fpath = Path(local_path)
    else:
        # Local path already computed from key (in init)
        key = f"docs/{doc_id}/raw/{doc.get('title') or 'upload.pdf'}"
        local_path = settings.local_path_for_key(key)
        fpath = Path(local_path)

    # Kick off ingestion
    background.add_task(ingest_document, db, doc_id, fpath, settings)
    return {"ok": True, "docId": doc_id}


# ----------------------------- DOCUMENTS ------------------------------------


@router.get("/doc/{docId}/status", response_model=DocStatusOut)
async def doc_status(docId: ObjectIdStr, ctx: AuthContext = Depends(auth_ctx)) -> DocStatusOut:
    db = get_db()
    doc = await db["documents"].find_one({"_id": docId, "orgId": ctx.orgId})
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    ingest = doc.get("ingest") or {}
    stats = ingest.get("stats") or {}
    return DocStatusOut(
        docId=docId,
        status=str(ingest.get("status", "uploaded")),
        errors=list(ingest.get("errors", [])),
        stats={"blocks": int(stats.get("blocks", 0)), "ocrLines": int(stats.get("ocrLines", 0))} if stats else None,
        pages=int(doc.get("pages", 0)),
        isOCR=bool(doc.get("isOCR", False)),
        title=str(doc.get("title", "")),
    )


# ----------------------------- ANCHORS --------------------------------------


@router.post("/anchor", response_model=CreateAnchorOut)
async def create_anchor(
    body: CreateAnchorIn,
    ctx: AuthContext = Depends(auth_ctx),
) -> CreateAnchorOut:
    """
    Create a user highlight anchor; returns the new anchorId.
    """
    db = get_db()
    # Ensure document exists for tenant
    doc = await db["documents"].find_one({"_id": body.docId, "orgId": ctx.orgId})
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    anchor_id = _new_id()
    doc_anchor = {
        "_id": anchor_id,
        "userId": ctx.userId,
        "orgId": ctx.orgId,
        "docId": body.docId,
        "page": int(body.page),
        "blockId": body.blockId,
        "bboxNorm": tuple(body.bboxNorm),
        "charStart": int(body.charStart),
        "charEnd": int(body.charEnd),
        "text": body.text,
        "ts": _now(),
    }
    await db["anchors"].insert_one(doc_anchor)
    return CreateAnchorOut(anchorId=anchor_id)


@router.get("/citation/{anchorId}", response_model=CitationOut)
async def get_citation(anchorId: ObjectIdStr, ctx: AuthContext = Depends(auth_ctx)) -> CitationOut:
    db = get_db()
    a = await db["anchors"].find_one({"_id": anchorId, "orgId": ctx.orgId})
    if not a:
        raise HTTPException(status_code=404, detail="Anchor not found")

    return CitationOut(
        anchorId=a["_id"],
        docId=a["docId"],
        page=int(a["page"]),
        bboxNorm=tuple(a.get("bboxNorm", (0.0, 0.0, 1.0, 1.0))),
        text=str(a.get("text", "")),
        blockId=str(a.get("blockId")) if a.get("blockId") else None,
        charStart=int(a.get("charStart", 0)),
        charEnd=int(a.get("charEnd", 0)),
    )


# ----------------------------- QUERY ----------------------------------------


@router.post("/query", response_model=QueryOut)
async def create_query(body: QueryIn, request: Request, ctx: AuthContext = Depends(auth_ctx)) -> QueryOut:
    """
    Allocate a queryId and stash minimal execution state in-process for the upcoming SSE call.
    If selection is inline, materialize an anchor now.
    """
    db = get_db()
    # Verify document access
    doc = await db["documents"].find_one({"_id": body.docId, "orgId": ctx.orgId})
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    # Ensure we have an anchorId
    anchor_id: Optional[str] = None
    if isinstance(body.selection, dict) and "anchorId" in body.selection:
        anchor_id = body.selection["anchorId"]
    else:
        # Inline selection -> create an anchor
        inline = body.selection  # type: ignore[assignment]
        anchor_id = _new_id()
        doc_anchor = {
            "_id": anchor_id,
            "userId": ctx.userId,
            "orgId": ctx.orgId,
            "docId": body.docId,
            "page": int(inline["page"]),
            "blockId": inline.get("blockId"),
            "bboxNorm": tuple(inline["bboxNorm"]),
            "charStart": int(inline["charStart"]),
            "charEnd": int(inline["charEnd"]),
            "text": str(inline["text"] or ""),
            "ts": _now(),
        }
        await db["anchors"].insert_one(doc_anchor)

    # Thread handling: if threadId not provided, upsert one per (user,doc)
    thread_id = body.threadId
    if not thread_id:
        tdoc = await db["threads"].find_one_and_update(
            {"userId": ctx.userId, "orgId": ctx.orgId, "docId": body.docId},
            {"$setOnInsert": {"_id": _new_id(), "title": None, "lastMessageAt": _now()}},
            upsert=True,
            return_document=ReturnDocument.AFTER,
        )
        thread_id = str(tdoc["_id"])

    # Allocate queryId and stash payload
    qid = _new_id()
    await _put_pending(
        qid,
        {
            "userId": ctx.userId,
            "orgId": ctx.orgId,
            "docId": body.docId,
            "question": body.question,
            "anchorId": anchor_id,
            "threadId": thread_id,
            "ts": time.time(),
        },
    )

    # Build absolute stream URL (App Router will call it with EventSource)
    stream_url = str(request.url_for("query_stream", id=qid))
    return QueryOut(queryId=qid, streamUrl=stream_url)


@router.get("/query/{id}/stream", name="query_stream")
async def query_stream(id: str, request: Request, ctx: AuthContext = Depends(auth_ctx)) -> StreamingResponse:
    """
    SSE stream producing:
      - event: trace -> telemetry (stage, meta)
      - event: token -> delta string
      - event: final -> JSON { answer, citations[] }
    """
    db = get_db()
    settings = get_settings()

    # Retrieve pending query
    pending = await _pop_pending(id)
    if not pending:
        raise HTTPException(status_code=404, detail="Query not found or expired")
    if pending["userId"] != ctx.userId or pending["orgId"] != ctx.orgId:
        raise HTTPException(status_code=403, detail="Forbidden")

    doc_id: str = pending["docId"]
    question: str = pending["question"]
    anchor_id: str = pending["anchorId"]
    thread_id: str = pending["threadId"]

    async def _gen() -> AsyncIterator[bytes]:
        # Emit retrieval traces around build_context_pack
        t0 = time.perf_counter()
        try:
            # Build retrieval context (includes local + global + fusion + rerank)
            context_pack = await build_context_pack(
                db=db,
                orgId=ctx.orgId,
                doc_id=doc_id,
                anchor_id=anchor_id,
                question=question,
                embed_fn=lambda xs: embed_texts(xs, model=settings.EMBED_MODEL, api_key=settings.OPENAI_API_KEY),
            )
            t1 = time.perf_counter()
            yield sse_event("trace", {"stage": "local", "meta": {"ms": int((t1 - t0) * 1000)}})

            # Derive per-block ephemeral anchors (first sentence / whole block bbox as fallback)
            derived_map: Dict[str, str] = {}
            for b in context_pack["blocks"]:
                bid = b["blockId"]
                # Create simple anchor for the block if none set
                anchor_id_b = _new_id()
                text = (b.get("text") or "")[:160]
                doc_anchor = {
                    "_id": anchor_id_b,
                    "userId": ctx.userId,
                    "orgId": ctx.orgId,
                    "docId": doc_id,
                    "page": int(b.get("page", 0)),
                    "blockId": bid,
                    "bboxNorm": tuple(b.get("bboxNorm", (0.0, 0.0, 1.0, 1.0))),
                    "charStart": 0,
                    "charEnd": min(len(text), 160),
                    "text": text,
                    "ts": _now(),
                }
                await db["anchors"].insert_one(doc_anchor)
                b["anchorId"] = anchor_id_b
                derived_map[bid] = anchor_id_b

            # Build compact blocks list (with derived anchorIds) for LLM
            ctx_blocks = context_pack["blocks"]

            # Stream LLM
            answer_started = False
            tokens_out = 0
            # Rough tokens-in estimation
            tokens_in = _count_tokens_fast(question + "\n" + (context_pack.get("selectionText") or ""))
            for blk in ctx_blocks:
                tokens_in += _count_tokens_fast(blk.get("text", ""))

            async for evt in answer_stream_json(
                question=question,
                selection=context_pack.get("selectionText") or None,
                context_blocks=ctx_blocks,
                model=settings.CHAT_MODEL,
                api_key=settings.OPENAI_API_KEY,
            ):
                typ = evt.get("type")
                payload = evt.get("payload")
                if typ == "trace":
                    yield sse_event("trace", payload or {})
                elif typ == "token":
                    answer_started = True
                    delta = (payload or {}).get("delta", "")
                    tokens_out += _count_tokens_fast(delta)
                    yield sse_event("token", {"delta": delta})
                elif typ == "final":
                    # Normalize citations: ensure anchorIds filled from derived_map if missing
                    final_obj = dict(payload or {})
                    cites = []
                    for c in final_obj.get("citations", []):
                        bid = str(c.get("blockId"))
                        page = int(c.get("page", 0))
                        aid = c.get("anchorId") or derived_map.get(bid, "")
                        cites.append({"blockId": bid, "page": page, "anchorId": aid})
                    final_obj["citations"] = cites

                    # Persist message & retrieval log (fire-and-forget)
                    asyncio.create_task(
                        _persist_message_and_log(
                            db=db,
                            ctx=ctx,
                            thread_id=thread_id,
                            question=question,
                            answer=str(final_obj.get("answer") or ""),
                            citations=final_obj.get("citations") or [],
                            anchor_id=anchor_id,
                            tokens_in=tokens_in,
                            tokens_out=tokens_out,
                            doc_id=doc_id,
                        )
                    )

                    yield sse_event("final", final_obj)
                else:
                    # Unknown event type; ignore
                    continue

            # Keep connection alive a bit (clients may process the final event)
            yield keepalive_comment("done")
        except Exception as e:
            yield sse_event("final", {"answer": f"Error: {e}", "citations": []})

    return StreamingResponse(_gen(), media_type="text/event-stream")


async def _persist_message_and_log(
    *,
    db,
    ctx: AuthContext,
    thread_id: str,
    question: str,
    answer: str,
    citations: List[Dict[str, Any]],
    anchor_id: str,
    tokens_in: int,
    tokens_out: int,
    doc_id: str,
) -> None:
    """
    Store the assistant message and retrieval log records.
    """
    try:
        now = _now()
        # Update thread lastMessageAt
        await db["threads"].update_one({"_id": thread_id}, {"$set": {"lastMessageAt": now}})

        # Persist message
        msg = {
            "_id": _new_id(),
            "threadId": thread_id,
            "userId": ctx.userId,
            "role": "assistant",
            "question": question,
            "answer": answer,
            "citations": citations,
            "selection": {"anchorId": anchor_id},
            "latencyMs": None,
            "ts": now,
        }
        await db["messages"].insert_one(msg)

        # Persist retrieval log (approx timings; detailed stage metrics can be added)
        stages = {
            "local": {"k": 3, "ms": 0},
            "vector": {"k": 48, "efSearch": None, "ms": 0},
            "lexical": {"k": 48, "ms": 0},
            "fusion": {"method": "rrf", "ms": 0},
            "rerank": {"model": "heuristic", "ms": 0},
        }
        log = {
            "_id": _new_id(),
            "queryId": _new_id(),  # not the transport id; generate for log record
            "userId": ctx.userId,
            "orgId": ctx.orgId,
            "docId": doc_id,
            "selectionAnchorId": anchor_id,
            "stages": stages,
            "tokensIn": int(tokens_in),
            "tokensOut": int(tokens_out),
            "ts": now,
        }
        await db["retrieval_logs"].insert_one(log)
    except Exception:
        # Best-effort persistence
        pass


# ----------------------------- THREADS --------------------------------------


@router.post("/threads")
async def upsert_thread(body: Dict[str, Any], ctx: AuthContext = Depends(auth_ctx)) -> Dict[str, Any]:
    """
    Create or fetch a thread for (user, org, doc). Optional 'title'.
    """
    db = get_db()
    doc_id: str = body.get("docId")
    title: Optional[str] = body.get("title")
    if not doc_id:
        raise HTTPException(status_code=400, detail="docId is required")

    # Verify doc access
    doc = await db["documents"].find_one({"_id": doc_id, "orgId": ctx.orgId})
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    now = _now()
    t = await db["threads"].find_one_and_update(
        {"userId": ctx.userId, "orgId": ctx.orgId, "docId": doc_id},
        {"$setOnInsert": {"_id": _new_id(), "title": title, "lastMessageAt": now}, "$set": {"title": title} if title else {}},
        upsert=True,
        return_document=ReturnDocument.AFTER,
    )
    return {"threadId": str(t["_id"]), "title": t.get("title")}


@router.get("/threads/{id}/messages")
async def list_messages(id: ObjectIdStr, ctx: AuthContext = Depends(auth_ctx)) -> Dict[str, Any]:
    """
    Return latest messages for the given thread (desc ts), limit=50.
    """
    db = get_db()
    # Basic access: ensure thread belongs to user/org
    thr = await db["threads"].find_one({"_id": id, "orgId": ctx.orgId, "userId": ctx.userId})
    if not thr:
        raise HTTPException(status_code=404, detail="Thread not found")

    cur = db["messages"].find({"threadId": id}, sort=[("ts", -1)], limit=50)
    rows = await cur.to_list(length=50)
    # Normalize ObjectIds
    for r in rows:
        r["_id"] = str(r["_id"])
    return {"messages": rows}


__all__ = ["router"]
