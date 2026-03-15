"""
Supabase JWT authentication.

Verifies JWTs issued by Supabase Auth using the project's JWKS endpoint.
Supports both legacy HS256 and modern ECC (P-256) signing keys.

Required env vars:
  SUPABASE_URL   Supabase project URL (e.g. https://xxx.supabase.co)
"""
import os
import logging
import time
import threading

import jwt
import httpx
from jwt import PyJWK
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

logger = logging.getLogger(__name__)

_security = HTTPBearer(auto_error=False)

# JWKS cache
_jwks_cache: dict = {}
_jwks_lock = threading.Lock()
_JWKS_TTL = 300  # refresh every 5 minutes


def _supabase_url() -> str:
    url = os.environ.get("SUPABASE_URL", "").strip().rstrip("/")
    if not url:
        raise RuntimeError("SUPABASE_URL not configured")
    return url


def _fetch_jwks() -> dict:
    """Fetch JWKS from Supabase GoTrue endpoint."""
    url = f"{_supabase_url()}/auth/v1/.well-known/jwks.json"
    resp = httpx.get(url, timeout=10.0)
    resp.raise_for_status()
    return resp.json()


def _get_jwks() -> dict:
    """Get cached JWKS, refreshing if stale."""
    global _jwks_cache
    with _jwks_lock:
        now = time.time()
        if _jwks_cache.get("keys") and now - _jwks_cache.get("fetched_at", 0) < _JWKS_TTL:
            return _jwks_cache

        try:
            data = _fetch_jwks()
            _jwks_cache = {"keys": data.get("keys", []), "fetched_at": now}
            logger.info("Refreshed JWKS from Supabase (%d keys)", len(_jwks_cache["keys"]))
        except Exception as e:
            logger.warning("Failed to fetch JWKS: %s (using cached)", e)
            if not _jwks_cache.get("keys"):
                raise RuntimeError(f"Cannot fetch JWKS and no cache available: {e}")

    return _jwks_cache


def _verify_token(token: str) -> dict:
    """
    Verify a Supabase JWT using JWKS.
    Tries JWKS first, falls back to SUPABASE_JWT_SECRET for HS256 if available.
    """
    # Decode header to get kid and algorithm
    try:
        unverified_header = jwt.get_unverified_header(token)
    except jwt.DecodeError:
        raise jwt.InvalidTokenError("Cannot decode token header")

    alg = unverified_header.get("alg", "")
    kid = unverified_header.get("kid")

    # Try JWKS-based verification for asymmetric algorithms
    if alg in ("ES256", "RS256", "EdDSA"):
        jwks_data = _get_jwks()
        keys = jwks_data.get("keys", [])

        # Find matching key by kid
        matching_key = None
        for key_data in keys:
            if kid and key_data.get("kid") == kid:
                matching_key = key_data
                break

        if not matching_key and keys:
            # If no kid match, use the first key with matching algorithm
            for key_data in keys:
                if key_data.get("alg") == alg or key_data.get("kty") == ("EC" if alg == "ES256" else "RSA"):
                    matching_key = key_data
                    break

        if matching_key:
            public_key = PyJWK(matching_key).key
            return jwt.decode(
                token,
                public_key,
                algorithms=[alg],
                audience="authenticated",
            )

        raise jwt.InvalidTokenError(f"No matching JWKS key found for kid={kid}")

    # HS256 fallback — use JWT secret if available
    if alg == "HS256":
        secret = os.environ.get("SUPABASE_JWT_SECRET", "").strip()
        if secret:
            return jwt.decode(
                token,
                secret,
                algorithms=["HS256"],
                audience="authenticated",
            )
        raise jwt.InvalidTokenError("HS256 token but no SUPABASE_JWT_SECRET configured")

    raise jwt.InvalidTokenError(f"Unsupported algorithm: {alg}")


def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(_security),
) -> dict:
    """
    Verify a Supabase-issued JWT and return the payload.
    Raises HTTP 401 if the token is missing or invalid.
    """
    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required. Please sign in.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    token = credentials.credentials

    try:
        payload = _verify_token(token)
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Session expired. Please sign in again.",
        )
    except jwt.InvalidTokenError as e:
        logger.warning("JWT validation failed: %s", e)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token. Please sign in again.",
        )


def get_user_id(payload: dict = Depends(get_current_user)) -> str:
    """Extract the user ID (sub) from the Supabase JWT payload."""
    uid = payload.get("sub")
    if not uid:
        raise HTTPException(status_code=401, detail="User ID not found in token")
    return uid
