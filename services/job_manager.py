"""
Analysis job lifecycle management.

Creates, tracks, and completes async video analysis jobs
using Supabase for persistence and Redis for queuing.
"""
from __future__ import annotations

import logging
from typing import Optional
from datetime import datetime, timezone

from services.supabase_client import get_client
from services.redis_client import enqueue_job

logger = logging.getLogger(__name__)


def create_job(
    user_id: str,
    action_type: str,
    age: int,
    video_storage_key: str,
) -> str:
    """
    Create a new analysis job in Supabase and enqueue it in Redis.
    Returns the job ID (UUID).
    """
    sb = get_client()

    row = {
        "user_id": user_id,
        "action_type": action_type,
        "age": age,
        "video_storage_key": video_storage_key,
        "status": "pending",
        "progress": 0,
    }

    resp = sb.table("analysis_jobs").insert(row).execute()
    job_id = resp.data[0]["id"]

    # Enqueue in Redis for worker pickup
    enqueue_job(job_id, {
        "user_id": user_id,
        "action_type": action_type,
        "age": age,
        "video_storage_key": video_storage_key,
    })

    logger.info("Created job %s for user %s", job_id, user_id[:8])
    return job_id


def get_job_status(job_id: str, user_id: str) -> Optional[dict]:
    """
    Get the current status of a job.
    Returns None if job not found or doesn't belong to user.
    """
    sb = get_client()
    resp = (
        sb.table("analysis_jobs")
        .select("id, status, progress, error_message, analysis_id, completed_at")
        .eq("id", job_id)
        .eq("user_id", user_id)
        .limit(1)
        .execute()
    )
    if not resp.data:
        return None
    return resp.data[0]


def get_analysis_result(analysis_id: str) -> Optional[dict]:
    """Fetch the full analysis result for a completed job."""
    sb = get_client()
    resp = (
        sb.table("analyses")
        .select("*")
        .eq("id", analysis_id)
        .limit(1)
        .execute()
    )
    if not resp.data:
        return None
    return resp.data[0]


def update_job(
    job_id: str,
    status: Optional[str] = None,
    progress: Optional[int] = None,
    analysis_id: Optional[str] = None,
    error_message: Optional[str] = None,
) -> None:
    """Update job fields in Supabase."""
    sb = get_client()
    updates = {"updated_at": datetime.now(timezone.utc).isoformat()}

    if status is not None:
        updates["status"] = status
    if progress is not None:
        updates["progress"] = progress
    if analysis_id is not None:
        updates["analysis_id"] = analysis_id
    if error_message is not None:
        updates["error_message"] = error_message
    if status == "completed":
        updates["completed_at"] = datetime.now(timezone.utc).isoformat()

    sb.table("analysis_jobs").update(updates).eq("id", job_id).execute()


def complete_job(job_id: str, analysis_id: str) -> None:
    """Mark a job as completed with the analysis result ID."""
    update_job(job_id, status="completed", progress=100, analysis_id=analysis_id)
    logger.info("Job %s completed with analysis %s", job_id, analysis_id)


def fail_job(job_id: str, error: str) -> None:
    """Mark a job as failed with an error message."""
    update_job(job_id, status="failed", error_message=error)
    logger.error("Job %s failed: %s", job_id, error)
