"""
Session storage — persists analysis results to Supabase for history and before/after comparison.

Gracefully degrades to no-ops if SUPABASE_URL / SUPABASE_SERVICE_ROLE_KEY are not set,
so the rest of the application continues to work without a database.

Required Supabase table (run once in your project's SQL editor):
─────────────────────────────────────────────────────────────────
  CREATE TABLE analyses (
    id                      UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id                 UUID        NOT NULL,
    created_at              TIMESTAMPTZ DEFAULT NOW(),
    action_type             TEXT        NOT NULL,
    overall_score           INT,
    technique_score         INT,
    power_score             INT,
    balance_score           INT,
    peak_wrist_speed        FLOAT,
    hip_shoulder_separation FLOAT,
    balance_metric          FLOAT,
    follow_through          BOOLEAN,
    plain_summary           TEXT,
    video_id                TEXT,
    video_url               TEXT
  );

  -- If upgrading an existing table, run:
  -- ALTER TABLE analyses ADD COLUMN video_url TEXT;

  CREATE INDEX analyses_user_action
    ON analyses (user_id, action_type, created_at DESC);
─────────────────────────────────────────────────────────────────
"""
from __future__ import annotations

import logging
import os
from typing import Optional, List

logger = logging.getLogger(__name__)

# Singleton Supabase client — avoids re-creating on every request
_supabase_client = None
_supabase_init_attempted = False


def _client():
    """Return a cached Supabase admin client, or None if credentials are missing."""
    global _supabase_client, _supabase_init_attempted
    if _supabase_client is not None:
        return _supabase_client
    if _supabase_init_attempted:
        return None
    _supabase_init_attempted = True
    url = os.environ.get("SUPABASE_URL", "").strip()
    key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "").strip()
    if not url or not key:
        return None
    try:
        from supabase import create_client
        _supabase_client = create_client(url, key)
        return _supabase_client
    except Exception as exc:
        logger.warning("Supabase client init error: %s", exc)
        return None


def save_session(user_id: str, result) -> Optional[str]:
    """
    Persist an AnalysisResult to the `analyses` table.
    Returns the new row UUID, or None if the write failed / DB not configured.
    """
    client = _client()
    if not client:
        return None
    try:
        fb = result.feedback
        mt = result.metrics
        row = {
            "user_id":                  user_id,
            "action_type":              result.action_type,
            "overall_score":            fb.overall_score,
            "technique_score":          fb.technique_score,
            "power_score":              fb.power_score,
            "balance_score":            fb.balance_score,
            "peak_wrist_speed":         mt.peak_wrist_speed,
            "hip_shoulder_separation":  mt.hip_shoulder_separation,
            "balance_metric":           mt.balance_score,
            "follow_through":           mt.follow_through,
            "plain_summary":            fb.plain_summary,
            "video_id":                 result.video_id,
            "video_url":                getattr(result, "video_url", None),
        }
        res = client.table("analyses").insert(row).execute()
        return res.data[0]["id"] if res.data else None
    except Exception as exc:
        logger.warning("save_session failed: %s", exc)
        return None


def get_previous_session(user_id: str, action_type: str) -> Optional[dict]:
    """
    Return the single most-recent session row for this user + action_type,
    or None if there is none / DB not configured.
    Used to populate the before/after comparison card.
    """
    client = _client()
    if not client:
        return None
    try:
        res = (
            client.table("analyses")
            .select("id, created_at, action_type, overall_score, technique_score, power_score, balance_score, video_id, video_url")
            .eq("user_id", user_id)
            .eq("action_type", action_type)
            .order("created_at", desc=True)
            .limit(1)
            .execute()
        )
        return res.data[0] if res.data else None
    except Exception as exc:
        logger.warning("get_previous_session failed: %s", exc)
        return None


def get_history(
    user_id: str,
    action_type: Optional[str] = None,
    limit: int = 8,
) -> List[dict]:
    """
    Return the last `limit` sessions for a user (oldest first, for charting).
    Optionally filter by action_type.
    Returns empty list if DB not configured or on error.
    """
    client = _client()
    if not client:
        return []
    try:
        query = (
            client.table("analyses")
            .select("created_at, overall_score, technique_score, power_score, balance_score, video_id, video_url")
            .eq("user_id", user_id)
        )
        if action_type:
            query = query.eq("action_type", action_type)
        res = (
            query
            .order("created_at", desc=True)
            .limit(limit)
            .execute()
        )
        rows = res.data or []
        return list(reversed(rows))   # oldest → newest for chart x-axis
    except Exception as exc:
        logger.warning("get_history failed: %s", exc)
        return []
