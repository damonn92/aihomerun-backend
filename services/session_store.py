"""
Session storage — persists analysis results to Cloudflare D1 for history and comparison.

Gracefully degrades to no-ops if D1 is not configured.
"""
from __future__ import annotations

import uuid
import logging
from typing import Optional, List

from services.d1_client import execute, is_configured

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
        row_id = uuid.uuid4().hex

        execute(
            """INSERT INTO analyses
               (id, user_id, action_type, overall_score, technique_score,
                power_score, balance_score, peak_wrist_speed,
                hip_shoulder_separation, balance_metric, follow_through,
                plain_summary, video_id, video_url)
               VALUES (?1,?2,?3,?4,?5,?6,?7,?8,?9,?10,?11,?12,?13,?14)""",
            [
                row_id,
                user_id,
                result.action_type,
                fb.overall_score,
                fb.technique_score,
                fb.power_score,
                fb.balance_score,
                mt.peak_wrist_speed,
                mt.hip_shoulder_separation,
                mt.balance_score,
                1 if mt.follow_through else 0,
                fb.plain_summary,
                result.video_id,
                getattr(result, "video_url", None),
            ],
        )
        return row_id
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
        rows = execute(
            """SELECT id, created_at, action_type, overall_score,
                      technique_score, power_score, balance_score,
                      video_id, video_url
               FROM analyses
               WHERE user_id = ?1 AND action_type = ?2
               ORDER BY created_at DESC
               LIMIT 1""",
            [user_id, action_type],
        )
        return rows[0] if rows else None
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
        if action_type:
            rows = execute(
                """SELECT created_at, overall_score, technique_score,
                          power_score, balance_score, video_id, video_url
                   FROM analyses
                   WHERE user_id = ?1 AND action_type = ?2
                   ORDER BY created_at DESC
                   LIMIT ?3""",
                [user_id, action_type, limit],
            )
        else:
            rows = execute(
                """SELECT created_at, overall_score, technique_score,
                          power_score, balance_score, video_id, video_url
                   FROM analyses
                   WHERE user_id = ?1
                   ORDER BY created_at DESC
                   LIMIT ?2""",
                [user_id, limit],
            )
        return list(reversed(rows))  # oldest → newest for chart x-axis
    except Exception as exc:
        logger.warning("get_history failed: %s", exc)
        return []
