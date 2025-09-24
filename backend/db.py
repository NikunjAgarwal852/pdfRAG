# backend/app/db.py
"""
Mongo client singleton, DB handle, and CRUD index initialization.

This module centralizes:
- Async Motor client creation (singleton)
- Database handle accessor
- Setup for standard CRUD indexes required by the app
- Convenience functions exposing Atlas Search / Vector index names used elsewhere

Atlas Search / Vector indexes are NOT created here (they are managed by JSON
specs in `backend/app/index_specs/*`). We only export their names so the rest
of the code can reference them consistently.
"""
from __future__ import annotations

import asyncio
from typing import Dict, List

from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
from pymongo import ASCENDING, DESCENDING, IndexModel

from .config import get_settings

# ---------------------------
# Collection name constants
# ---------------------------

COLL_USERS = "users"
COLL_DOCUMENTS = "documents"
COLL_SECTIONS = "sections"
COLL_BLOCKS = "blocks"
COLL_VARIABLES = "variables"
COLL_ANCHORS = "anchors"
COLL_THREADS = "threads"
COLL_MESSAGES = "messages"
COLL_RETRIEVAL_LOGS = "retrieval_logs"

# ---------------------------
# Atlas index names (referenced in code)
# Managed via JSON specs under app/index_specs
# ---------------------------

_BLOCKS_SEARCH_INDEX_NAME = "blocks_text"
_BLOCKS_VECTOR_INDEX_NAME = "blocks_vec"


def blocks_search_index_name() -> str:
    """Return the Atlas Search (BM25) index name used for `blocks.text`."""
    return _BLOCKS_SEARCH_INDEX_NAME


def blocks_vector_index_name() -> str:
    """Return the Atlas Vector Search (HNSW) index name used for `blocks.embedding.vector`."""
    return _BLOCKS_VECTOR_INDEX_NAME


# ---------------------------
# Client + DB accessors
# ---------------------------

_client: AsyncIOMotorClient | None = None


def get_client() -> AsyncIOMotorClient:
    """
    Get (or create) the global AsyncIOMotorClient.

    The client is safe to share across tasks. The URI is read from settings.
    """
    global _client
    if _client is None:
        settings = get_settings()
        # Let Motor/PyMongo manage pooling; keep defaults modest.
        _client = AsyncIOMotorClient(
            settings.MONGODB_URI,
            serverSelectionTimeoutMS=5000,
            uuidRepresentation="standard",
        )
    return _client


def get_db() -> AsyncIOMotorDatabase:
    """
    Get the application's database handle using `Settings.MONGODB_DB`.
    """
    settings = get_settings()
    return get_client()[settings.MONGODB_DB]


# ---------------------------
# Index initialization
# ---------------------------


async def _ensure_blocks_indexes(db: AsyncIOMotorDatabase) -> List[str]:
    """
    Create CRUD indexes for the `blocks` collection.

    From MASTER:
      - {docId:1, page:1, orderIdx:1}
      - {docId:1}
      - {docId:1, page:1}
    """
    col = db[COLL_BLOCKS]
    models = [
        IndexModel([("docId", ASCENDING), ("page", ASCENDING), ("orderIdx", ASCENDING)], name="doc_page_order"),
        IndexModel([("docId", ASCENDING)], name="doc"),
        IndexModel([("docId", ASCENDING), ("page", ASCENDING)], name="doc_page"),
    ]
    return await col.create_indexes(models)


async def _ensure_anchors_indexes(db: AsyncIOMotorDatabase) -> List[str]:
    """
    Create CRUD indexes for the `anchors` collection.

    From MASTER:
      - {userId:1, docId:1, page:1}
      - {_id:1} (implicit in MongoDB; no need to create)
    """
    col = db[COLL_ANCHORS]
    models = [
        IndexModel(
            [("userId", ASCENDING), ("docId", ASCENDING), ("page", ASCENDING)],
            name="user_doc_page",
        ),
    ]
    return await col.create_indexes(models)


async def _ensure_messages_indexes(db: AsyncIOMotorDatabase) -> List[str]:
    """
    Create CRUD indexes for the `messages` collection.

    From MASTER:
      - {threadId:1, ts:-1}
    """
    col = db[COLL_MESSAGES]
    models = [
        IndexModel([("threadId", ASCENDING), ("ts", DESCENDING)], name="thread_ts_desc"),
    ]
    return await col.create_indexes(models)


async def _ensure_documents_indexes(db: AsyncIOMotorDatabase) -> List[str]:
    """
    Create CRUD indexes for the `documents` collection.

    From MASTER:
      - {orgId:1, _id:1}
    """
    col = db[COLL_DOCUMENTS]
    models = [
        IndexModel([("orgId", ASCENDING), ("_id", ASCENDING)], name="org_doc"),
    ]
    return await col.create_indexes(models)


async def ensure_indexes(db: AsyncIOMotorDatabase) -> Dict[str, List[str]]:
    """
    Ensure all required CRUD indexes exist. Idempotent and safe to run on startup.

    Returns a mapping {collection_name: [index_names_created_or_existing]}.
    """
    results: Dict[str, List[str]] = {}

    created = await asyncio.gather(
        _ensure_blocks_indexes(db),
        _ensure_anchors_indexes(db),
        _ensure_messages_indexes(db),
        _ensure_documents_indexes(db),
        return_exceptions=False,
    )

    results[COLL_BLOCKS] = created[0]
    results[COLL_ANCHORS] = created[1]
    results[COLL_MESSAGES] = created[2]
    results[COLL_DOCUMENTS] = created[3]

    return results


__all__ = [
    "get_client",
    "get_db",
    "ensure_indexes",
    "blocks_search_index_name",
    "blocks_vector_index_name",
    # Collections (optional re-exports)
    "COLL_USERS",
    "COLL_DOCUMENTS",
    "COLL_SECTIONS",
    "COLL_BLOCKS",
    "COLL_VARIABLES",
    "COLL_ANCHORS",
    "COLL_THREADS",
    "COLL_MESSAGES",
    "COLL_RETRIEVAL_LOGS",
]
