"""
Supabase client wrapper.

Provides a singleton Supabase client using the service role key
for backend operations that bypass RLS.

Required env vars:
  SUPABASE_URL              Supabase project URL
  SUPABASE_SERVICE_ROLE_KEY Service role key (bypasses RLS)
"""
import os
import logging

from supabase import create_client, Client

logger = logging.getLogger(__name__)

_client: Client | None = None


def get_client() -> Client:
    """Return a singleton Supabase client."""
    global _client
    if _client is None:
        url = os.environ.get("SUPABASE_URL", "").strip()
        key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "").strip()
        if not url or not key:
            raise RuntimeError(
                "SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY must be set"
            )
        _client = create_client(url, key)
    return _client


def is_configured() -> bool:
    """Check if Supabase credentials are available."""
    url = os.environ.get("SUPABASE_URL", "").strip()
    key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "").strip()
    return bool(url and key)
