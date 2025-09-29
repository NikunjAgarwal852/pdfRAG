# backend/app/ingestion.py
"""
End-to-end PDF ingestion: parse -> (optional) OCR -> blocks -> embeddings -> ready.

High-level flow
---------------
1) Mark document.ingest.status = 'parsing'
2) Parse PDF pages with PyMuPDF (fitz):
   - Extract text blocks (page.get_text('dict')), fold lines into paragraphs
   - Compute bboxNorm in [0,1]
   - Heuristically detect standalone equations
   - Collect variable hints from trailing 'where|let|denote|means' phrases
3) If a page is scanned/near-empty, optionally run Tesseract OCR to create text blocks
   - Store average OCR confidence on each block as `ocrConf` (0..1)
4) Mark status = 'embedding', embed eligible blocks (paragraph/equation/caption), L2-normalize vectors
5) Insert blocks (batch insert_many) and upsert variables
6) Mark status = 'ready' (or 'error' on failure), write stats

Notes
-----
- IDs are created client-side (24-hex strings) to keep the DB consistently string-typed.
- Equation detection is heuristic—good enough to lift obvious display equations.
- Variables pass: extract tokens from "where/let/denote/means" phrases; store into `variables` with sentence spans.
"""
from __future__ import annotations

import math
import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import fitz  # PyMuPDF
import numpy as np
from bson import ObjectId
from motor.motor_asyncio import AsyncIOMotorDatabase
from pymongo import UpdateOne

from .config import Settings
from .schemas import Bbox

# Optional OCR imports are deferred inside function scope.

# ---------------------------
# Small datatypes
# ---------------------------


@dataclass
class ParsedBlock:
    """Intermediate representation before persistence/embedding."""
    _id: str
    docId: str
    page: int
    type: str  # 'paragraph'|'equation'|'figure'|'table'|'caption'|'footnote'
    orderIdx: int
    sectionPath: List[str]
    bboxNorm: Tuple[float, float, float, float]
    text: str
    charRanges: List[Dict[str, int]]
    eq: Optional[Dict[str, Any]]
    vars: Optional[List[Dict[str, Any]]]
    ocrConf: Optional[float] = None  # 0..1
    # embedding to be filled later
    embedding: Optional[Dict[str, Any]] = None


# ---------------------------
# Parsing helpers
# ---------------------------


MATH_CHARS = set("=±+−–*/×·÷^_()[]{}<>≤≥≠≈≃≡∑∏∫√∞→←↔°%|‖·,:;λμνπσφΩαβγδθΔΓΣΦΨω·′″")  # inkl. greek
EQ_NUM_RE = re.compile(r"[\(\[]\s*(\d{1,4})\s*[\)\]]\s*$")
DEFN_CUE_RE = re.compile(r"\b(where|let|denote|means)\b", flags=re.IGNORECASE)
TOKEN_RE = re.compile(r"(?<![A-Za-z0-9_])([A-Za-z](?:_[A-Za-z0-9]+)?)")


def l2_normalize(vec: Sequence[float]) -> List[float]:
    arr = np.asarray(vec, dtype=np.float32)
    norm = float(np.linalg.norm(arr))
    if norm == 0.0 or not math.isfinite(norm):
        return [0.0] * len(arr)
    return list((arr / norm).tolist())


def bbox_norm(rect: Tuple[float, float, float, float], w: float, h: float) -> Bbox:
    x0, y0, x1, y1 = rect
    # Clamp to page and normalize
    x0 = max(0.0, min(x0, w))
    y0 = max(0.0, min(y0, h))
    x1 = max(0.0, min(x1, w))
    y1 = max(0.0, min(y1, h))
    if x0 > x1:
        x0, x1 = x1, x0
    if y0 > y1:
        y0, y1 = y1, y0
    return (x0 / w, y0 / h, x1 / w, y1 / h)


def ratio_math(text: str) -> float:
    if not text:
        return 0.0
    m = sum(1 for ch in text if ch in MATH_CHARS)
    return m / max(1, len(text))


def is_equation_like(text: str, rect: Tuple[float, float, float, float], w: float) -> Tuple[bool, Optional[str]]:
    """Simple display-equation heuristic and optional number extraction."""
    t = text.strip()
    if not t:
        return False, None
    # Standalone short/medium line with many math operators OR contains '=' and minimal words
    words = [w for w in re.split(r"\s+", t) if w]
    has_equals = "=" in t or "≡" in t or "≈" in t
    mathy = ratio_math(t) >= 0.20 or (has_equals and len(words) <= 12)
    centered = abs(((rect[0] + rect[2]) * 0.5) - (w * 0.5)) <= w * 0.15
    eqnum = None
    m = EQ_NUM_RE.search(t)
    if m:
        eqnum = m.group(1)
    # Treat as equation if mathy AND either short-ish or centered block
    cond = mathy and (len(t) <= 260) and (centered or has_equals)
    return cond, eqnum


def join_block_text(block: dict) -> Tuple[str, Tuple[float, float, float, float], float]:
    """
    Join all spans into a single string. Return (text, bbox, avg_font_size).
    Non-text blocks are skipped upstream.
    """
    x0, y0, x1, y1 = block.get("bbox", (0, 0, 0, 0))
    lines = block.get("lines", [])
    parts: List[str] = []
    sizes: List[float] = []
    for ln in lines:
        spans = ln.get("spans", [])
        row = []
        for sp in spans:
            s = sp.get("text") or ""
            if not s:
                continue
            row.append(s)
            size = sp.get("size")
            if size:
                sizes.append(float(size))
        if row:
            parts.append("".join(row))
    text = "\n".join(parts).strip()
    avg_sz = sum(sizes) / len(sizes) if sizes else 0.0
    return text, (x0, y0, x1, y1), avg_sz


def page_is_scanned(pg: fitz.Page) -> bool:
    """Heuristic: if there are almost no text characters, assume scanned."""
    raw = (pg.get_text("text") or "").strip()
    return len(raw) < 10


def extract_variables_from_text(text: str) -> Tuple[List[Dict[str, Any]], List[Tuple[str, Dict[str, Any]]]]:
    """
    Find variable tokens and return:
      - vars list for the block (token, offset)
      - variable defs to upsert: [(token, {sentenceSpan, weight, sectionPath})]
    """
    vars_list: List[Dict[str, Any]] = []
    defs: List[Tuple[str, Dict[str, Any]]] = []

    # Collect raw token positions for vars list
    for m in TOKEN_RE.finditer(text):
        token = m.group(1)
        offset = m.start(1)
        vars_list.append({"token": token, "offset": offset})

    # Definitional cues: capture the sentence following the cue
    for m in DEFN_CUE_RE.finditer(text):
        cue_start = m.start()
        # Sentence ends at first '.', ';', or newline after cue
        tail = text[cue_start:]
        end_rel = re.search(r"[.;\n]", tail)
        span_end = cue_start + (end_rel.start() if end_rel else len(tail))
        sent_span = {"start": cue_start, "end": span_end}
        # Tokens appearing soon after the cue are candidates
        window = text[cue_start:span_end]
        for tok in set(t.group(1) for t in TOKEN_RE.finditer(window)):
            defs.append((tok, {"sentenceSpan": sent_span, "weight": 1.0, "sectionPath": []}))

    return vars_list, defs


def ocr_page(pg: fitz.Page, dpi: int = 200) -> List[Tuple[str, Tuple[float, float, float, float], float]]:
    """
    Run OCR with pytesseract (if available) and return a list of
    (line_text, bbox_page_coords, avg_conf[0..1]) tuples.

    Coordinates are normalized to page coordinate space.
    """
    try:
        import pytesseract
        from PIL import Image
    except Exception:
        return []

    mat = fitz.Matrix(dpi / 72.0, dpi / 72.0)
    pix = pg.get_pixmap(matrix=mat, alpha=False)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

    # image_to_data returns TSV; easier grouping
    data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
    n = len(data.get("level", []))
    if n == 0:
        return []

    results = []
    # Group by (block_num, par_num, line_num)
    groups: Dict[Tuple[int, int, int], List[int]] = defaultdict(list)
    for i in range(n):
        if data["text"][i].strip():
            groups[(data["block_num"][i], data["par_num"][i], data["line_num"][i])].append(i)

    for key, idxs in sorted(groups.items()):
        words = [data["text"][i] for i in idxs if data["text"][i].strip()]
        if not words:
            continue
        lefts = [data["left"][i] for i in idxs]
        tops = [data["top"][i] for i in idxs]
        rights = [data["left"][i] + data["width"][i] for i in idxs]
        bottoms = [data["top"][i] + data["height"][i] for i in idxs]
        confs = [float(data["conf"][i]) for i in idxs if data["conf"][i] not in ("-1", "")]
        # Normalize into page coordinates: pix dimensions map to page.rect * mat
        # Inverse map back to page coordinates by dividing by scale.
        sx, sy = (dpi / 72.0), (dpi / 72.0)
        rect = (min(lefts) / sx, min(tops) / sy, max(rights) / sx, max(bottoms) / sy)
        line_text = " ".join(words)
        avg_conf = (sum(confs) / len(confs) / 100.0) if confs else 0.0
        results.append((line_text, rect, avg_conf))

    return results


# ---------------------------
# Main ingestion
# ---------------------------


def _new_id() -> str:
    return str(ObjectId())


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _chunked(seq: Sequence[Any], size: int) -> Iterable[Sequence[Any]]:
    for i in range(0, len(seq), size):
        yield seq[i : i + size]


async def ingest_document(db: AsyncIOMotorDatabase, doc_id: str, file_path: Path, settings: Settings) -> None:
    """
    Parse a PDF and populate: blocks, variables; compute embeddings; mark ready.

    Parameters
    ----------
    db : AsyncIOMotorDatabase
    doc_id : str
        Target document ID (24-hex string).
    file_path : Path
        Location of the uploaded PDF on disk.
    settings : Settings
        Application settings (models, API keys, etc.)
    """
    # Defensive: persist status transitions
    await db["documents"].update_one(
        {"_id": doc_id},
        {"$set": {"ingest.status": "parsing", "ingest.errors": [], "ingest.stats": {"blocks": 0, "ocrLines": 0}}},
    )

    total_blocks = 0
    total_ocr_lines = 0
    all_blocks: List[ParsedBlock] = []
    var_defs_agg: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    try:
        with fitz.open(str(file_path)) as pdf:
            # Persist page count if not already present
            await db["documents"].update_one({"_id": doc_id}, {"$set": {"pages": pdf.page_count}})

            for pno in range(pdf.page_count):
                page = pdf.load_page(pno)
                w, h = page.rect.width, page.rect.height
                order = 0

                page_dict = page.get_text("dict") or {}
                text_blocks = [b for b in page_dict.get("blocks", []) if "lines" in b]

                if not text_blocks and page_is_scanned(page):
                    # OCR fallback
                    ocr_lines = ocr_page(page)
                    total_ocr_lines += len(ocr_lines)
                    for line_text, rect, conf in ocr_lines:
                        text = line_text.strip()
                        if not text:
                            continue
                        norm = bbox_norm(rect, w, h)
                        vars_list, var_defs = extract_variables_from_text(text)
                        block = ParsedBlock(
                            _id=_new_id(),
                            docId=doc_id,
                            page=pno,
                            type="paragraph",
                            orderIdx=order,
                            sectionPath=[],
                            bboxNorm=norm,
                            text=text,
                            charRanges=[{"start": 0, "end": len(text)}],
                            eq=None,
                            vars=vars_list or None,
                            ocrConf=float(conf),
                        )
                        all_blocks.append(block)
                        for tok, meta in var_defs:
                            var_defs_agg[tok].append(
                                {"blockId": block._id, "sentenceSpan": meta["sentenceSpan"], "weight": meta["weight"], "sectionPath": []}
                            )
                        order += 1
                    continue  # next page

                # Normal text parsing
                for b in text_blocks:
                    text, rect, _avg_sz = join_block_text(b)
                    if not text:
                        continue
                    norm = bbox_norm(rect, w, h)
                    is_eq, eqnum = is_equation_like(text, rect, w)
                    btype = "equation" if is_eq else "paragraph"
                    eq_info = {"latex": None, "mathml": None, "number": eqnum} if is_eq else None

                    vars_list, var_defs = extract_variables_from_text(text)

                    block = ParsedBlock(
                        _id=_new_id(),
                        docId=doc_id,
                        page=pno,
                        type=btype,
                        orderIdx=order,
                        sectionPath=[],
                        bboxNorm=norm,
                        text=text,
                        charRanges=[{"start": 0, "end": len(text)}],
                        eq=eq_info,
                        vars=vars_list or None,
                        ocrConf=None,
                    )
                    all_blocks.append(block)
                    total_blocks += 1

                    # Variable defs attached to this block
                    for tok, meta in var_defs:
                        # Boost weight if the defining sentence follows an equation on same page and close-by
                        weight = 1.2 if is_eq else 1.0
                        var_defs_agg[tok].append(
                            {"blockId": block._id, "sentenceSpan": meta["sentenceSpan"], "weight": weight, "sectionPath": []}
                        )

                    order += 1

        # ---- Embedding stage ----
        await db["documents"].update_one({"_id": doc_id}, {"$set": {"ingest.status": "embedding"}})

        # Eligible types for embeddings
        emb_eligible = {"paragraph", "equation", "caption"}
        to_embed = [b for b in all_blocks if b.type in emb_eligible and b.text]
        # Import lazily to avoid cyclic deps at import time
        from .llm import embed_texts  # type: ignore

        texts = [b.text for b in to_embed]
        # Batch to avoid request limits
        BATCH = 128
        embed_model = settings.EMBED_MODEL
        dims = settings.EMBEDDING_DIMS

        idx = 0
        for chunk in _chunked(texts, BATCH):
            vecs = await embed_texts(chunk, model=embed_model, api_key=settings.OPENAI_API_KEY)
            for off, vec in enumerate(vecs):
                b = to_embed[idx + off]
                b.embedding = {
                    "model": embed_model,
                    "dims": dims if dims else len(vec),
                    "vector": l2_normalize(vec),
                }
            idx += len(vecs)

        # Prepare DB docs and insert in chunks (avoid huge payload)
        def as_doc(pb: ParsedBlock) -> Dict[str, Any]:
            now = _now()
            doc = {
                "_id": pb._id,
                "docId": pb.docId,
                "page": pb.page,
                "type": pb.type,
                "orderIdx": pb.orderIdx,
                "sectionPath": pb.sectionPath,
                "bboxNorm": pb.bboxNorm,
                "text": pb.text,
                "charRanges": pb.charRanges,
                "eq": pb.eq,
                "vars": pb.vars,
                "embedding": pb.embedding,
                "ts": {"createdAt": now, "embeddedAt": now if pb.embedding else None},
            }
            if pb.ocrConf is not None:
                doc["ocrConf"] = float(max(0.0, min(1.0, pb.ocrConf)))
            return doc

        block_docs = [as_doc(b) for b in all_blocks]
        for chunk in _chunked(block_docs, 500):
            if chunk:
                await db["blocks"].insert_many(chunk, ordered=False)

        # Upsert variables (bulk)
        bulk_ops: List[UpdateOne] = []
        for token, defs in var_defs_agg.items():
            # Keep lastSeenPage as max page of supporting defs
            last_page = 0
            for d in defs:
                # We only have blockId (string), need page for lastSeenPage; derive from all_blocks
                # Build a small lookup to avoid extra DB hits
                pass
            # Build a lookup map for blockId -> page
        block_page_map = {b._id: b.page for b in all_blocks}

        for token, defs in var_defs_agg.items():
            last_seen = max((block_page_map.get(d["blockId"], 0) for d in defs), default=0)
            # Normalize defs to schema shape
            norm_defs = [
                {
                    "blockId": d["blockId"],
                    "sentenceSpan": d["sentenceSpan"],
                    "weight": float(d.get("weight", 1.0)),
                    "sectionPath": d.get("sectionPath", []),
                }
                for d in defs
            ]
            bulk_ops.append(
                UpdateOne(
                    {"docId": doc_id, "token": token},
                    {
                        "$setOnInsert": {"_id": _new_id(), "docId": doc_id, "token": token},
                        "$set": {"lastSeenPage": int(last_seen)},
                        "$push": {"defs": {"$each": norm_defs}},
                    },
                    upsert=True,
                )
            )

        if bulk_ops:
            await db["variables"].bulk_write(bulk_ops, ordered=False)

        # Finalize stats and status
        stats = {"blocks": len(block_docs), "ocrLines": total_ocr_lines}
        await db["documents"].update_one(
            {"_id": doc_id},
            {"$set": {"ingest.status": "ready", "ingest.stats": stats}},
        )

    except Exception as e:
        # Capture the error and mark status
        await db["documents"].update_one(
            {"_id": doc_id},
            {"$set": {"ingest.status": "error"}, "$push": {"ingest.errors": str(e)}},
        )
        # Optionally re-raise if upstream wants to log/trace
        # raise


__all__ = ["ingest_document"]
