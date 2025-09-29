# backend/app/auth.py
"""
Auth utilities:
- Verify Google ID tokens against Google JWKS.
- Issue backend JWTs used by the web app to call APIs.
- FastAPI dependency to enforce tenancy using the backend JWT.
- Upsert (ensure) user documents from Google claims.

Design notes
------------
- Audience ('aud') check: if an expected GOOGLE_CLIENT_ID is configured on the
  backend (optional), we enforce it. Otherwise we skip 'aud' verification to
  support environments where only NextAuth knows the client ID.
- Issuer is always validated against Google's issuers.
- Backend JWT: HS256 with {iss, aud, exp, orgId, sub, email}.
"""
from __future__ import annotations

import base64
import hashlib
import json
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional, Tuple

import httpx
from fastapi import Depends, Header, HTTPException, status
from jose import jwt
from motor.motor_asyncio import AsyncIOMotorDatabase
from pymongo import ReturnDocument

from .config import get_settings
from .schemas import ObjectIdStr, UserDoc

# ---------------------------------------------------------------------------
# Google JWKS fetching & ID token verification
# ---------------------------------------------------------------------------

_GOOGLE_ISSUERS = {"https://accounts.google.com", "accounts.google.com"}
_JWKS_CACHE: dict[str, Any] = {"fetched_at": 0.0, "jwks": None}
_JWKS_TTL_SECONDS = 60 * 10  # 10 minutes


async def fetch_google_jwks() -> dict:
    """
    Fetch Google's JWKS and cache for a short TTL.

    Returns
    -------
    dict
        The JWKS JSON object: {"keys": [ ... ]}
    """
    now = time.time()
    if _JWKS_CACHE["jwks"] and (now - _JWKS_CACHE["fetched_at"] < _JWKS_TTL_SECONDS):
        return _JWKS_CACHE["jwks"]

    settings = get_settings()
    uri = settings.GOOGLE_JWKS_URI
    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.get(uri)
        resp.raise_for_status()
        jwks = resp.json()

    _JWKS_CACHE["jwks"] = jwks
    _JWKS_CACHE["fetched_at"] = now
    return jwks


def _find_jwk_for_kid(jwks: dict, kid: str) -> Optional[dict]:
    for k in jwks.get("keys", []):
        if k.get("kid") == kid:
            return k
    return None


async def verify_google_id_token(id_token: str) -> dict:
    """
    Verify a Google ID token using JWKS. Returns decoded claims if valid.

    Validation:
      - Signature (RS256) using Google's JWKS
      - 'iss' in allowed issuers
      - 'aud' matches GOOGLE_CLIENT_ID if provided in settings (optional)
      - 'exp' is in the future (enforced by jose)

    Parameters
    ----------
    id_token : str
        Google ID token obtained server-side by NextAuth.

    Returns
    -------
    dict
        Verified claims.

    Raises
    ------
    HTTPException
        401/400 on verification errors.
    """
    # Unverified header to select the proper JWK by 'kid'
    try:
        header = jwt.get_unverified_header(id_token)
        kid = header.get("kid")
        alg = header.get("alg")
        if not kid or alg != "RS256":
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token header")
    except Exception:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Malformed ID token")

    jwks = await fetch_google_jwks()
    jwk_key = _find_jwk_for_kid(jwks, kid)
    if not jwk_key:
        # Stale cache? Force refresh once.
        _JWKS_CACHE["jwks"] = None
        jwks = await fetch_google_jwks()
        jwk_key = _find_jwk_for_kid(jwks, kid)
        if not jwk_key:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Unknown key id")

    settings = get_settings()
    expected_aud = getattr(settings, "GOOGLE_CLIENT_ID", None)  # optional
    verify_aud = bool(expected_aud)

    try:
        # jose accepts RSA JWK dicts directly as the key parameter.
        claims = jwt.decode(
            id_token,
            jwk_key,
            algorithms=["RS256"],
            audience=expected_aud if verify_aud else None,
            issuer=None,  # we'll enforce 'iss' manually to allow both forms
            options={"verify_aud": verify_aud, "verify_iss": False},
        )
    except Exception:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="ID token verification failed")

    iss = claims.get("iss")
    if iss not in _GOOGLE_ISSUERS:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid issuer")

    return claims


# ---------------------------------------------------------------------------
# Backend JWT issuance & validation
# ---------------------------------------------------------------------------


def issue_backend_jwt(*, sub: str, orgId: str, email: str, ttl_hours: int = 12) -> str:
    """
    Create a backend session JWT used by the web app to call APIs.

    Claims:
      - sub: userId (Mongo ObjectId string)
      - orgId: tenant identifier
      - email: user email (for convenience & logs)
      - iss, aud, iat, exp
    """
    settings = get_settings()
    now = datetime.now(timezone.utc)
    payload = {
        "sub": sub,
        "orgId": orgId,
        "email": email,
        "iss": settings.JWT_ISSUER,
        "aud": settings.JWT_AUDIENCE,
        "iat": int(now.timestamp()),
        "exp": int((now + timedelta(hours=ttl_hours)).timestamp()),
    }
    token = jwt.encode(payload, settings.JWT_SECRET, algorithm="HS256")
    return token


@dataclass
class AuthContext:
    orgId: ObjectIdStr
    userId: ObjectIdStr
    email: str


def _extract_bearer(authorization: Optional[str]) -> str:
    if not authorization:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing Authorization")
    parts = authorization.split()
    if len(parts) != 2 or parts[0].lower() != "bearer":
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Use Bearer token")
    return parts[1]


def auth_ctx(authorization: Optional[str] = Header(None)) -> AuthContext:
    """
    FastAPI dependency: validates backend JWT and returns tenancy context.
    """
    token = _extract_bearer(authorization)
    settings = get_settings()
    try:
        claims = jwt.decode(
            token,
            settings.JWT_SECRET,
            algorithms=["HS256"],
            audience=settings.JWT_AUDIENCE,
            issuer=settings.JWT_ISSUER,
            options={"verify_aud": True, "verify_iss": True},
        )
    except Exception:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or expired token")

    user_id = claims.get("sub")
    org_id = claims.get("orgId")
    email = claims.get("email")
    if not (user_id and org_id and email):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Token missing required claims")

    return AuthContext(orgId=str(org_id), userId=str(user_id), email=str(email))


# ---------------------------------------------------------------------------
# User upsert (ensure) from Google claims
# ---------------------------------------------------------------------------


def _derive_org_id(email: str, hd: Optional[str] = None) -> str:
    """
    Deterministically derive a 24-hex orgId.

    Strategy:
      - Prefer Google's 'hd' (hosted domain) when available.
      - Fallback to email domain.
      - Hash -> 24-hex (truncate SHA1).
    """
    domain = (hd or email.split("@")[-1]).lower().strip()
    digest = hashlib.sha1(domain.encode("utf-8")).hexdigest()[:24]
    return digest


async def ensure_user(db: AsyncIOMotorDatabase, google_claims: dict) -> UserDoc:
    """
    Upsert a user document from verified Google claims.

    Parameters
    ----------
    db : AsyncIOMotorDatabase
    google_claims : dict
        Claims returned by verify_google_id_token().

    Returns
    -------
    UserDoc
        The upserted or existing user document.
    """
    email = google_claims.get("email")
    if not email:
        # Google sometimes omits email if scope not granted; we require it.
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Google profile missing email")

    name = google_claims.get("name") or f"{google_claims.get('given_name','')} {google_claims.get('family_name','')}".strip() or email
    picture = google_claims.get("picture")
    hd = google_claims.get("hd")

    org_id = _derive_org_id(email, hd=hd)

    now = datetime.now(timezone.utc)

    doc = await db["users"].find_one_and_update(
        {"email": email},
        {
            # Keep orgId stable once set; only set on insert
            "$setOnInsert": {
                "orgId": org_id,
                "roles": [],
                "createdAt": now,
            },
            "$set": {
                "email": email,
                "name": name,
                "picture": picture,
            },
        },
        upsert=True,
        return_document=ReturnDocument.AFTER,
    )

    # Coerce ObjectId -> str fields expected by Pydantic model
    # (Motor returns raw ObjectId for _id)
    doc["_id"] = str(doc["_id"])
    doc["orgId"] = str(doc["orgId"])

    return UserDoc.model_validate(doc)


__all__ = [
    "fetch_google_jwks",
    "verify_google_id_token",
    "issue_backend_jwt",
    "AuthContext",
    "auth_ctx",
    "ensure_user",
]
