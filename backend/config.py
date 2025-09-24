# backend/app/config.py
from __future__ import annotations

import os
import pathlib
from functools import lru_cache
from typing import Iterable, List, Literal, Optional
from urllib.parse import urlparse

from pydantic import BaseModel, Field, computed_field, model_validator


class Settings(BaseModel):
    """
    Centralized runtime settings for the backend. Values are sourced from environment variables.

    MASTER (expected env vars):
      - MONGODB_URI (str)                                [required]
      - MONGODB_DB (str, default 'pdf_rag')
      - OPENAI_API_KEY (str)                             [required]
      - EMBED_MODEL ('text-embedding-3-small'|'text-embedding-3-large')
      - CHAT_MODEL (str, default 'gpt-4o-mini')
      - JWT_ISSUER (str, e.g. 'http://localhost:8000')   [required]
      - JWT_AUDIENCE (str, default 'pdf-rag')
      - JWT_SECRET (str)                                 [required]
      - GOOGLE_JWKS_URI (str, default Google JWKS URL)
      - STORAGE_BACKEND ('local'|'s3', default 'local')
      - UPLOAD_DIR (str, default './uploads')            [used when STORAGE_BACKEND='local']
      - S3_ENDPOINT (str, optional)                      [MinIO / custom endpoint]
      - S3_BUCKET (str, required if STORAGE_BACKEND='s3')
      - AWS_ACCESS_KEY_ID (str, optional)
      - AWS_SECRET_ACCESS_KEY (str, optional)
      - NEXTAUTH_URL (str, optional)                     [used for CORS]
      - FRONTEND_ORIGIN (str|CSV, optional)              [used for CORS]
    """

    # --- Core ---
    MONGODB_URI: str = Field(..., description="MongoDB connection string")
    MONGODB_DB: str = "pdf_rag"

    # --- OpenAI / Models ---
    OPENAI_API_KEY: str
    EMBED_MODEL: str = "text-embedding-3-small"
    CHAT_MODEL: str = "gpt-4o-mini"

    # --- Auth / JWT ---
    JWT_ISSUER: str = "http://localhost:8000"
    JWT_AUDIENCE: str = "pdf-rag"
    JWT_SECRET: str
    GOOGLE_JWKS_URI: str = "https://www.googleapis.com/oauth2/v3/certs"

    # --- Storage ---
    STORAGE_BACKEND: Literal["local", "s3"] = "local"
    UPLOAD_DIR: str = "./uploads"

    # S3 / MinIO
    S3_ENDPOINT: Optional[str] = None
    S3_BUCKET: Optional[str] = None
    AWS_ACCESS_KEY_ID: Optional[str] = None
    AWS_SECRET_ACCESS_KEY: Optional[str] = None

    # --- Frontend / CORS hints ---
    NEXTAUTH_URL: Optional[str] = None
    FRONTEND_ORIGIN: Optional[str] = None  # can be CSV list

    # ---------------------------
    # Computed / derived fields
    # ---------------------------

    @computed_field  # type: ignore[misc]
    @property
    def DEV_MODE(self) -> bool:
        """
        Development mode when JWT_SECRET == 'dev' or any known URL is localhost.
        """
        if self.JWT_SECRET == "dev":
            return True
        candidates = [self.NEXTAUTH_URL, self.FRONTEND_ORIGIN, self.JWT_ISSUER]
        for c in candidates:
            if not c:
                continue
            # Handle CSV lists
            for piece in str(c).split(","):
                host = urlparse(piece.strip()).hostname or piece
                if "localhost" in host or "127.0.0.1" in host:
                    return True
        return False

    @computed_field  # type: ignore[misc]
    @property
    def EMBEDDING_DIMS(self) -> int:
        """
        Resolve embedding dimensions from EMBED_MODEL.
        Defaults to 1536 if unknown (keeps service bootable).
        """
        mapping = {
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
        }
        return mapping.get(self.EMBED_MODEL, 1536)

    @computed_field  # type: ignore[misc]
    @property
    def CORS_ORIGINS(self) -> List[str]:
        """
        Build CORS origins list using NEXTAUTH_URL and/or FRONTEND_ORIGIN (CSV allowed).
        Adds localhost dev origins when DEV_MODE is true.
        """
        origins: set[str] = set()

        def add_from(value: Optional[str]) -> None:
            if not value:
                return
            for item in value.split(","):
                item = item.strip()
                if not item:
                    continue
                origins.add(origin_from_url(item))

        add_from(self.FRONTEND_ORIGIN)
        add_from(self.NEXTAUTH_URL)

        if self.DEV_MODE:
            origins.update(
                {
                    "http://localhost:3000",
                    "http://127.0.0.1:3000",
                }
            )

        # It's safe to include the API origin as well for tools like Swagger UI.
        origins.add(self.backend_origin())

        return sorted(origins)

    # ---------------------------
    # Validators
    # ---------------------------

    @model_validator(mode="after")
    def _validate_storage(self) -> "Settings":
        if self.STORAGE_BACKEND == "s3" and not self.S3_BUCKET:
            raise ValueError("S3_BUCKET is required when STORAGE_BACKEND='s3'")
        return self

    # ---------------------------
    # Helpers: storage + URLs
    # ---------------------------

    def use_s3(self) -> bool:
        return self.STORAGE_BACKEND == "s3"

    def object_key(self, doc_id: str, *, filename: str) -> str:
        """
        Generate a stable object key for a document asset.
        Example: docs/<docId>/raw/<filename>
        """
        safe_doc = doc_id.strip()
        safe_name = filename.strip().replace("\\", "/").split("/")[-1]
        return f"docs/{safe_doc}/raw/{safe_name}"

    def thumb_key(self, doc_id: str, *, page: int) -> str:
        """
        Object key for a page thumbnail (PNG).
        Example: thumbs/<docId>/page-3.png
        """
        safe_doc = doc_id.strip()
        return f"thumbs/{safe_doc}/page-{page}.png"

    # --- Local filesystem paths (when STORAGE_BACKEND='local') ---

    def local_path_for_key(self, key: str) -> str:
        """
        Resolve a local filesystem path for an object key.
        """
        # Normalize separators to avoid path traversal
        norm = "/".join(p for p in key.split("/") if p not in ("", ".", ".."))
        path = pathlib.Path(self.UPLOAD_DIR).joinpath(norm)
        path.parent.mkdir(parents=True, exist_ok=True)
        return str(path)

    # --- Public URLs for objects (download/view) ---

    def public_url(self, key: str) -> str:
        """
        Build a URL that the frontend can GET to retrieve the object.
        For S3/MinIO we use virtual-hosting or path-style based on S3_ENDPOINT.
        For local storage we assume the backend exposes /files/<key>.
        """
        if self.use_s3():
            base = self._s3_base_url()
            return f"{base}/{key}"
        # Local file server route (served by FastAPI: GET /files/{path:path})
        return f"{self.backend_origin()}/files/{key}"

    def backend_origin(self) -> str:
        """
        Extract the scheme://host[:port] origin for the backend from JWT_ISSUER.
        """
        return origin_from_url(self.JWT_ISSUER)

    def _s3_base_url(self) -> str:
        """
        Base HTTP URL for S3/MinIO objects.
        - If S3_ENDPOINT is provided, use path-style: <endpoint>/<bucket>
        - Otherwise, fall back to AWS virtual-host style: https://<bucket>.s3.amazonaws.com
        """
        bucket = self.S3_BUCKET or ""
        if self.S3_ENDPOINT:
            return f"{self.S3_ENDPOINT.rstrip('/')}/{bucket}"
        return f"https://{bucket}.s3.amazonaws.com"

    # --- Optional: boto3 client factory (lazy) ---

    def s3_client(self):
        """
        Create a boto3 S3 client configured for AWS or custom endpoints (e.g., MinIO).
        Only call this when STORAGE_BACKEND='s3'.
        """
        if not self.use_s3():
            raise RuntimeError("S3 client requested but STORAGE_BACKEND!='s3'")
        import boto3  # local import to avoid overhead when unused

        kwargs = {}
        if self.S3_ENDPOINT:
            kwargs["endpoint_url"] = self.S3_ENDPOINT
        if self.AWS_ACCESS_KEY_ID and self.AWS_SECRET_ACCESS_KEY:
            kwargs["aws_access_key_id"] = self.AWS_ACCESS_KEY_ID
            kwargs["aws_secret_access_key"] = self.AWS_SECRET_ACCESS_KEY
        return boto3.client("s3", **kwargs)

    # ---------------------------
    # Factory
    # ---------------------------

    @classmethod
    def from_env(cls) -> "Settings":
        """
        Build a Settings instance from environment variables.
        Uses Pydantic for type coercion/validation.
        """
        # Collect only known fields; empty strings are ignored.
        def get(name: str) -> Optional[str]:
            val = os.getenv(name)
            return val if (val is not None and val != "") else None

        data = {
            "MONGODB_URI": get("MONGODB_URI"),
            "MONGODB_DB": get("MONGODB_DB"),
            "OPENAI_API_KEY": get("OPENAI_API_KEY"),
            "EMBED_MODEL": get("EMBED_MODEL"),
            "CHAT_MODEL": get("CHAT_MODEL"),
            "JWT_ISSUER": get("JWT_ISSUER"),
            "JWT_AUDIENCE": get("JWT_AUDIENCE"),
            "JWT_SECRET": get("JWT_SECRET"),
            "GOOGLE_JWKS_URI": get("GOOGLE_JWKS_URI"),
            "STORAGE_BACKEND": get("STORAGE_BACKEND"),
            "UPLOAD_DIR": get("UPLOAD_DIR"),
            "S3_ENDPOINT": get("S3_ENDPOINT"),
            "S3_BUCKET": get("S3_BUCKET"),
            "AWS_ACCESS_KEY_ID": get("AWS_ACCESS_KEY_ID"),
            "AWS_SECRET_ACCESS_KEY": get("AWS_SECRET_ACCESS_KEY"),
            "NEXTAUTH_URL": get("NEXTAUTH_URL"),
            "FRONTEND_ORIGIN": get("FRONTEND_ORIGIN"),
        }
        # Let Pydantic handle defaults + type conversion.
        return cls(**{k: v for k, v in data.items() if v is not None})


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Cached singleton accessor. Import and call anywhere:
        from backend.app.config import get_settings
        settings = get_settings()
    """
    return Settings.from_env()


# ---------------------------
# Small utility functions
# ---------------------------


def origin_from_url(url: str) -> str:
    """
    Reduce a URL to its origin (scheme://host[:port]).
    Accepts bare origins and comma-separated inputs are handled upstream.
    """
    parsed = urlparse(url)
    if parsed.scheme and parsed.netloc:
        return f"{parsed.scheme}://{parsed.netloc}"
    # If someone passed a bare host or already an origin, return as-is.
    return url.strip()


def split_csv(value: Optional[str]) -> List[str]:
    if not value:
        return []
    return [v.strip() for v in value.split(",") if v.strip()]


__all__ = ["Settings", "get_settings", "origin_from_url", "split_csv"]
