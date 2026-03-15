"""
Session storage — persists analysis results to Supabase PostgreSQL.

Gracefully degrades to no-ops if Supabase is not configured.
"""
from __future__ import annotations

import logging
from typing import Optional, List

from services.supabase_client import get_client, is_configured

logger = logging.getLogger(__name__)


def save_session(user_id: str, result) -> Optional[str]:
    """
    Persist an AnalysisResult to the `analyses` table.
    Returns the new row ID, or None if the write failed / DB not configured.
    """
    if not is_configured():
        return None
    try:
        fb = result.feedback
        mt = result.metrics

        row = {
            "user_id": user_id,
            "action_type": result.action_type,
            "overall_score": fb.overall_score,
            "technique_score": fb.technique_score,
            "power_score": fb.power_score,
            "balance_score": fb.balance_score,
            "peak_wrist_speed": mt.peak_wrist_speed,
            "hip_shoulder_separation": mt.hip_shoulder_separation,
            "balance_metric": mt.balance_score,
            "follow_through": mt.follow_through,
            "plain_summary": fb.plain_summary,
            "video_id": result.video_id,
            "video_url": getattr(result, "video_url", None),
        }

        resp = get_client().table("analyses").insert(row).execute()
        if resp.data:
            return resp.data[0].get("id")
        return None
    except Exception as exc:
        logger.warning("save_session failed: %s", exc)
        return None


def get_previous_session(user_id: str, action_type: str) -> Optional[dict]:
    """
    Return the single most-recent session row for this user + action_type,
    or None if there is none / DB not configured.
    """
    if not is_configured():
        return None
    try:
        resp = (
            get_client()
            .table("analyses")
            .select(
                "id, created_at, action_type, overall_score, "
                "technique_score, power_score, balance_score, "
                "video_id, video_url"
            )
            .eq("user_id", user_id)
            .eq("action_type", action_type)
            .order("created_at", desc=True)
            .limit(1)
            .execute()
        )
        return resp.data[0] if resp.data else None
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
    """
    if not is_configured():
        return []
    try:
        query = (
            get_client()
            .table("analyses")
            .select(
                "created_at, overall_score, technique_score, "
                "power_score, balance_score, video_id, video_url"
            )
            .eq("user_id", user_id)
        )
        if action_type:
            query = query.eq("action_type", action_type)

        resp = query.order("created_at", desc=True).limit(limit).execute()
        return list(reversed(resp.data)) if resp.data else []
    except Exception as exc:
        logger.warning("get_history failed: %s", exc)
        return []
