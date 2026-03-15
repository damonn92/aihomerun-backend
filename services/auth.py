"""
JWT authentication via Supabase.

Supabase now uses ECC (P-256 / ES256) as the primary signing algorithm.
We fetch public keys from the JWKS endpoint and verify accordingly.
Legacy HS256 (shared secret) is kept as a fallback for old tokens.

Required env vars:
  SUPABASE_URL          e.g. https://ypfpreoycrhujqucjiid.supabase.co
  SUPABASE_JWT_SECRET   Legacy HS256 secret (fallback)
"""
import os
import time
import threading
from typing import Optional

import httpx
import jwt
from jwt.algorithms import ECAlgorithm, RSAAlgorithm
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

_security = HTTPBearer(auto_error=False)

# ── JWKS cache ──────────────────────────────────────────────────────────────
_jwks_cache: dict = {}
_jwks_lock = threading.Lock()
_jwks_fetched_at: float = 0
_JWKS_TTL = 3600  # re-fetch once per hour


def _get_supabase_url() -> str:
    url = os.environ.get("SUPABASE_URL", "").rstrip("/")
    if not url:
        # Derive from JWT secret project ref as fallback — not ideal
        return ""
    return url


def _get_jwks() -> dict:
    """Return cached JWKS, refreshing if stale."""
    global _jwks_cache, _jwks_fetched_at
    with _jwks_lock:
        if time.time() - _jwks_fetched_at < _JWKS_TTL and _jwks_cache:
            return _jwks_cache
        supabase_url = _get_supabase_url()
        if not supabase_url:
            return {}
        url = f"{supabase_url}/auth/v1/.well-known/jwks.json"
        try:
            resp = httpx.get(url, timeout=5)
            resp.raise_for_status()
            _jwks_cache = resp.json()
            _jwks_fetched_at = time.time()
        except Exception as e:
            print(f"⚠️  JWKS fetch failed: {e}")
            if _jwks_cache:
                return _jwks_cache  # use stale cache on error
        return _jwks_cache


def _get_public_key(kid: str):
    """Look up a JWK by key ID and return a verification key object."""
    jwks = _get_jwks()
    for key_data in jwks.get("keys", []):
        if key_data.get("kid") == kid:
            kty = key_data.get("kty", "")
            if kty == "EC":
                return ECAlgorithm.from_jwk(key_data)
            elif kty == "RSA":
                return RSAAlgorithm.from_jwk(key_data)
    return None


def _get_jwt_secret() -> Optional[str]:
    return os.environ.get("SUPABASE_JWT_SECRET") or None


def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(_security),
) -> dict:
    """
    Verify the Supabase JWT and return the payload.
    Supports ES256 (ECC via JWKS) and HS256 (legacy shared secret).
    Raises HTTP 401 if the token is missing or invalid.
    """
    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required. Please sign in.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    token = credentials.credentials

    # ── Try ES256 / RS256 via JWKS first ─────────────────────────────────────
    try:
        header = jwt.get_unverified_header(token)
        kid = header.get("kid")

        if kid:
            pub_key = _get_public_key(kid)
            if pub_key:
                # Determine algorithm from JWK kty, NOT from untrusted token header
                jwks = _get_jwks()
                alg = "ES256"  # default
                for key_data in jwks.get("keys", []):
                    if key_data.get("kid") == kid:
                        kty = key_data.get("kty", "")
                        alg = "RS256" if kty == "RSA" else "ES256"
                        break
                payload = jwt.decode(
                    token,
                    pub_key,
                    algorithms=[alg],
                    audience="authenticated",
                )
                return payload

    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Session expired. Please sign in again.",
        )
    except jwt.InvalidTokenError:
        pass  # fall through to HS256

    # ── Fallback: HS256 legacy secret ─────────────────────────────────────────
    secret = _get_jwt_secret()
    if secret:
        try:
            payload = jwt.decode(
                token,
                secret,
                algorithms=["HS256"],
                audience="authenticated",
            )
            return payload
        except jwt.ExpiredSignatureError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Session expired. Please sign in again.",
            )
        except jwt.InvalidTokenError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token. Please sign in again.",
            )

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Token verification failed. Check server configuration.",
    )


def get_user_id(payload: dict = Depends(get_current_user)) -> str:
    """Extract the user ID (sub) from the JWT payload."""
    uid = payload.get("sub")
    if not uid:
        raise HTTPException(status_code=401, detail="User ID not found in token")
    return uid
