"""
Cloudflare D1 REST API client.

Required env vars:
  CF_ACCOUNT_ID       Cloudflare account ID
  CF_D1_DATABASE_ID   D1 database ID
  CF_API_TOKEN         Cloudflare API token with D1:Edit permission
"""
import os
import logging
from typing import Optional, List, Any

import httpx

logger = logging.getLogger(__name__)

_CF_ACCOUNT_ID: str = ""
_CF_D1_DATABASE_ID: str = ""
_CF_API_TOKEN: str = ""


def _ensure_config():
    global _CF_ACCOUNT_ID, _CF_D1_DATABASE_ID, _CF_API_TOKEN
    if not _CF_ACCOUNT_ID:
        _CF_ACCOUNT_ID = os.environ.get("CF_ACCOUNT_ID", "").strip()
        _CF_D1_DATABASE_ID = os.environ.get("CF_D1_DATABASE_ID", "").strip()
        _CF_API_TOKEN = os.environ.get("CF_API_TOKEN", "").strip()


def _api_url() -> str:
    _ensure_config()
    return (
        f"https://api.cloudflare.com/client/v4/accounts/"
        f"{_CF_ACCOUNT_ID}/d1/database/{_CF_D1_DATABASE_ID}/query"
    )


def _headers() -> dict:
    _ensure_config()
    return {
        "Authorization": f"Bearer {_CF_API_TOKEN}",
        "Content-Type": "application/json",
    }


def is_configured() -> bool:
    _ensure_config()
    return bool(_CF_ACCOUNT_ID and _CF_D1_DATABASE_ID and _CF_API_TOKEN)


def execute(sql: str, params: Optional[List[Any]] = None) -> List[dict]:
    """
    Execute a SQL query against D1 and return result rows.
    For INSERT/UPDATE/DELETE, returns an empty list on success.
    """
    if not is_configured():
        logger.warning("D1 not configured — skipping query")
        return []

    body = {"sql": sql}
    if params:
        body["params"] = params

    try:
        resp = httpx.post(_api_url(), json=body, headers=_headers(), timeout=15)
        resp.raise_for_status()
        data = resp.json()

        if not data.get("success"):
            errors = data.get("errors", [])
            logger.error("D1 query error: %s", errors)
            return []

        results = data.get("result", [])
        if results and "results" in results[0]:
            return results[0]["results"]
        return []

    except Exception as exc:
        logger.error("D1 request failed: %s", exc)
        return []


def execute_returning_meta(sql: str, params: Optional[List[Any]] = None) -> dict:
    """
    Execute a SQL query and return the full result including meta.
    Useful for checking rows_written, etc.
    """
    if not is_configured():
        return {"success": False}

    body = {"sql": sql}
    if params:
        body["params"] = params

    try:
        resp = httpx.post(_api_url(), json=body, headers=_headers(), timeout=15)
        resp.raise_for_status()
        return resp.json()
    except Exception as exc:
        logger.error("D1 request failed: %s", exc)
        return {"success": False, "error": str(exc)}
