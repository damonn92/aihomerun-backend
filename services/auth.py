"""
JWT authentication via Supabase.

Supabase signs all JWTs with the project's JWT Secret (HS256).
Set SUPABASE_JWT_SECRET in Railway environment variables:
  Dashboard → aihomerun-backend → Variables → Add SUPABASE_JWT_SECRET
Find it at: Supabase Dashboard → Settings → API → JWT Settings → JWT Secret
"""
import os
import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

_security = HTTPBearer(auto_error=False)


def _get_jwt_secret() -> str:
    secret = os.environ.get("SUPABASE_JWT_SECRET", "")
    if not secret:
        raise RuntimeError("SUPABASE_JWT_SECRET env var is not set")
    return secret


def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(_security),
) -> dict:
    """
    Verify the Supabase JWT and return the payload.
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
        payload = jwt.decode(
            token,
            _get_jwt_secret(),
            algorithms=["HS256"],
            options={"verify_aud": False},  # Supabase tokens have audience "authenticated"
        )
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Session expired. Please sign in again.",
        )
    except jwt.InvalidTokenError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid token: {e}",
        )


def get_user_id(payload: dict = Depends(get_current_user)) -> str:
    """Extract the user ID (sub) from the JWT payload."""
    uid = payload.get("sub")
    if not uid:
        raise HTTPException(status_code=401, detail="User ID not found in token")
    return uid
