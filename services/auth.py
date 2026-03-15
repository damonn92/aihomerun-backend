"""
Self-hosted JWT authentication.

Replaces Supabase Auth — the backend issues its own JWTs after verifying
Apple / Google identity tokens or email+password credentials.

Required env vars:
  JWT_SECRET          Secret key for signing/verifying HS256 JWTs

Optional (for social login):
  GOOGLE_CLIENT_ID    Google OAuth client ID (for ID token audience check)
"""
import os
import time
import uuid
import logging
import threading
from datetime import datetime, timezone
from typing import Optional

import httpx
import jwt
from jwt.algorithms import ECAlgorithm, RSAAlgorithm
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

logger = logging.getLogger(__name__)

_security = HTTPBearer(auto_error=False)

# ── JWT config ─────────────────────────────────────────────────────────────
JWT_ALGORITHM = "HS256"
JWT_EXPIRY_SECONDS = 60 * 60 * 24 * 30  # 30 days


def _jwt_secret() -> str:
    secret = os.environ.get("JWT_SECRET", "").strip()
    if not secret:
        raise RuntimeError("JWT_SECRET not configured")
    return secret


def create_access_token(user_id: str, email: Optional[str] = None) -> str:
    """Issue a new JWT for the given user."""
    now = time.time()
    payload = {
        "sub": user_id,
        "iat": int(now),
        "exp": int(now + JWT_EXPIRY_SECONDS),
    }
    if email:
        payload["email"] = email
    return jwt.encode(payload, _jwt_secret(), algorithm=JWT_ALGORITHM)


# ── Apple JWKS ─────────────────────────────────────────────────────────────
_apple_jwks: dict = {}
_apple_jwks_lock = threading.Lock()
_apple_jwks_fetched_at: float = 0
_APPLE_JWKS_TTL = 3600


def _fetch_apple_jwks() -> dict:
    global _apple_jwks, _apple_jwks_fetched_at
    with _apple_jwks_lock:
        if time.time() - _apple_jwks_fetched_at < _APPLE_JWKS_TTL and _apple_jwks:
            return _apple_jwks
        try:
            resp = httpx.get("https://appleid.apple.com/auth/keys", timeout=5)
            resp.raise_for_status()
            _apple_jwks = resp.json()
            _apple_jwks_fetched_at = time.time()
        except Exception as e:
            logger.warning("Apple JWKS fetch failed: %s", e)
        return _apple_jwks


def verify_apple_id_token(id_token: str) -> dict:
    """
    Verify an Apple Sign-In identity token.
    Returns the decoded payload with 'sub' (Apple user ID) and 'email'.
    """
    try:
        header = jwt.get_unverified_header(id_token)
    except jwt.DecodeError:
        raise HTTPException(400, "Invalid Apple ID token format")

    kid = header.get("kid")
    if not kid:
        raise HTTPException(400, "Apple ID token missing kid")

    jwks = _fetch_apple_jwks()
    pub_key = None
    for key_data in jwks.get("keys", []):
        if key_data.get("kid") == kid:
            kty = key_data.get("kty", "")
            if kty == "RSA":
                pub_key = RSAAlgorithm.from_jwk(key_data)
            elif kty == "EC":
                pub_key = ECAlgorithm.from_jwk(key_data)
            break

    if not pub_key:
        raise HTTPException(400, "Apple public key not found for kid")

    try:
        payload = jwt.decode(
            id_token,
            pub_key,
            algorithms=["RS256", "ES256"],
            audience=os.environ.get("APPLE_BUNDLE_ID", "com.aihomerun.app"),
            issuer="https://appleid.apple.com",
        )
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(401, "Apple ID token expired")
    except jwt.InvalidTokenError as e:
        raise HTTPException(401, f"Apple ID token invalid: {e}")


# ── Google token verification ─────────────────────────────────────────────

_google_jwks: dict = {}
_google_jwks_lock = threading.Lock()
_google_jwks_fetched_at: float = 0
_GOOGLE_JWKS_TTL = 3600


def _fetch_google_jwks() -> dict:
    global _google_jwks, _google_jwks_fetched_at
    with _google_jwks_lock:
        if time.time() - _google_jwks_fetched_at < _GOOGLE_JWKS_TTL and _google_jwks:
            return _google_jwks
        try:
            resp = httpx.get("https://www.googleapis.com/oauth2/v3/certs", timeout=5)
            resp.raise_for_status()
            _google_jwks = resp.json()
            _google_jwks_fetched_at = time.time()
        except Exception as e:
            logger.warning("Google JWKS fetch failed: %s", e)
        return _google_jwks


def verify_google_id_token(id_token: str) -> dict:
    """
    Verify a Google Sign-In identity token.
    Returns the decoded payload with 'sub' (Google user ID) and 'email'.
    """
    try:
        header = jwt.get_unverified_header(id_token)
    except jwt.DecodeError:
        raise HTTPException(400, "Invalid Google ID token format")

    kid = header.get("kid")
    if not kid:
        raise HTTPException(400, "Google ID token missing kid")

    jwks = _fetch_google_jwks()
    pub_key = None
    for key_data in jwks.get("keys", []):
        if key_data.get("kid") == kid:
            pub_key = RSAAlgorithm.from_jwk(key_data)
            break

    if not pub_key:
        raise HTTPException(400, "Google public key not found for kid")

    google_client_id = os.environ.get("GOOGLE_CLIENT_ID", "")

    try:
        payload = jwt.decode(
            id_token,
            pub_key,
            algorithms=["RS256"],
            audience=google_client_id if google_client_id else None,
            issuer=["accounts.google.com", "https://accounts.google.com"],
            options={"verify_aud": bool(google_client_id)},
        )
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(401, "Google ID token expired")
    except jwt.InvalidTokenError as e:
        raise HTTPException(401, f"Google ID token invalid: {e}")


# ── Password hashing ──────────────────────────────────────────────────────

def hash_password(password: str) -> str:
    import bcrypt
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()


def verify_password(password: str, hashed: str) -> bool:
    import bcrypt
    return bcrypt.checkpw(password.encode(), hashed.encode())


# ── User lookup / creation via D1 ─────────────────────────────────────────

def find_or_create_social_user(provider: str, provider_id: str, email: Optional[str] = None) -> dict:
    """
    Look up a user by provider+provider_id, or create one.
    Returns {"id": ..., "email": ...}.
    """
    from services.d1_client import execute

    rows = execute(
        "SELECT id, email FROM users WHERE provider = ?1 AND provider_id = ?2 LIMIT 1",
        [provider, provider_id],
    )
    if rows:
        return rows[0]

    # Create new user
    user_id = uuid.uuid4().hex
    execute(
        "INSERT INTO users (id, email, provider, provider_id) VALUES (?1, ?2, ?3, ?4)",
        [user_id, email, provider, provider_id],
    )
    # Also create a blank profile
    execute(
        "INSERT INTO profiles (id, display_name) VALUES (?1, ?2)",
        [user_id, ""],
    )
    return {"id": user_id, "email": email}


def find_email_user(email: str) -> Optional[dict]:
    from services.d1_client import execute
    rows = execute(
        "SELECT id, email, password_hash FROM users WHERE email = ?1 AND provider = 'email' LIMIT 1",
        [email],
    )
    return rows[0] if rows else None


def create_email_user(email: str, password: str) -> dict:
    from services.d1_client import execute
    user_id = uuid.uuid4().hex
    pw_hash = hash_password(password)
    execute(
        "INSERT INTO users (id, email, password_hash, provider) VALUES (?1, ?2, ?3, 'email')",
        [user_id, email, pw_hash],
    )
    execute(
        "INSERT INTO profiles (id, display_name) VALUES (?1, ?2)",
        [user_id, ""],
    )
    return {"id": user_id, "email": email}


# ── FastAPI dependencies (same interface as before) ───────────────────────

def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(_security),
) -> dict:
    """
    Verify our self-issued JWT and return the payload.
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
        payload = jwt.decode(token, _jwt_secret(), algorithms=[JWT_ALGORITHM])
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


def get_user_id(payload: dict = Depends(get_current_user)) -> str:
    """Extract the user ID (sub) from the JWT payload."""
    uid = payload.get("sub")
    if not uid:
        raise HTTPException(status_code=401, detail="User ID not found in token")
    return uid
