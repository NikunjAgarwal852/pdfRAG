# backend/app/main.py
"""
App factory, CORS, startup hooks, and uvicorn runner.

- Creates a FastAPI app with ORJSON responses.
- Configures CORS for http://localhost:3000 and any origins derived from Settings.
- Startup hook: initializes Mongo client and ensures CRUD indexes.
- Serves uploaded files under /files/{path} when STORAGE_BACKEND='local'.
- Exposes `run()` used by the console script to start uvicorn at 0.0.0.0:8000.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Path as PathParam
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, ORJSONResponse, PlainTextResponse
from starlette.requests import Request

from .config import get_settings
from .db import ensure_indexes, get_client, get_db
from .routers import router as api_router

logger = logging.getLogger("pdf-rag-backend")
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")


def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title="PDF RAG Backend",
        version="0.1.0",
        default_response_class=ORJSONResponse,
        docs_url="/docs",
        redoc_url=None,
        openapi_url="/openapi.json",
    )

    # --- CORS ---
    origins = settings.CORS_ORIGINS or []
    if "http://localhost:3000" not in origins:
        origins.append("http://localhost:3000")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
        allow_headers=["*"],
        expose_headers=["*"],
        max_age=600,
    )
    logger.info("CORS origins: %s", origins)

    # --- Routers ---
    app.include_router(api_router)

    # --- Local file server (dev/local storage only) ---
    if not settings.use_s3():
        upload_root = Path(settings.UPLOAD_DIR).resolve()

        @app.get("/files/{path:path}")
        async def serve_file(path: str = PathParam(..., description="Key relative to UPLOAD_DIR")):
            """
            Serve uploaded assets from local disk (dev only).
            Prevent directory traversal by ensuring the resolved path stays under UPLOAD_DIR.
            """
            target = (upload_root / path).resolve()
            if not str(target).startswith(str(upload_root)):
                raise HTTPException(status_code=403, detail="Forbidden")
            if not target.exists() or not target.is_file():
                raise HTTPException(status_code=404, detail="Not found")
            # Let Starlette guess content-type; PDFs will be application/pdf
            return FileResponse(target)

    # --- Root path: minimal ping for quick checks ---
    @app.get("/", response_class=PlainTextResponse, include_in_schema=False)
    async def root(_: Request):
        return "pdf-rag-backend: OK"

    # --- Lifespan events ---
    @app.on_event("startup")
    async def on_startup():
        # Initialize Mongo client and ping
        client = get_client()
        try:
            await client.admin.command("ping")
            logger.info("MongoDB ping OK")
        except Exception as e:
            logger.error("MongoDB ping failed: %s", e)
            # Do not crash here; allow subsequent requests to retry

        # Ensure CRUD indexes exist
        try:
            db = get_db()
            created = await ensure_indexes(db)
            logger.info("Indexes ensured: %s", {k: len(v) for k, v in created.items()})
        except Exception as e:
            logger.error("Failed ensuring indexes: %s", e)

    return app


app = create_app()


def run() -> None:
    """
    Start uvicorn (no CLI args), bound to 0.0.0.0:8000.
    """
    import uvicorn

    # Use module path so uvicorn can reload in dev if the user adds --reload externally.
    uvicorn.run(
        "backend.app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        workers=1,
        log_level="info",
    )


__all__ = ["app", "run", "create_app"]
