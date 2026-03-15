"""
BaseAI — Youth Baseball AI Coaching Backend
FastAPI entry point
"""
import asyncio
import logging
import time
import os
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

import sentry_sdk
_sentry_dsn = os.environ.get("SENTRY_DSN", "")
if _sentry_dsn:
    sentry_sdk.init(
        dsn=_sentry_dsn,
        traces_sample_rate=0.2,
        profiles_sample_rate=0.1,
        environment=os.environ.get("ENVIRONMENT", "development"),
    )

from typing import Optional, List
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Depends, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

logger = logging.getLogger(__name__)

# Limit concurrent video analyses to prevent OOM on small instances
_analysis_semaphore = asyncio.Semaphore(int(os.getenv("MAX_CONCURRENT_ANALYSES", 3)))

from services.video_processor import save_upload, extract_frames, get_video_info, cleanup_video_dir, upload_to_storage
from services.pose_analyzer import PoseAnalyzer
from services.baseball_metrics import analyze_swing, analyze_pitch
from services.ai_analyzer import analyze_with_claude
from services.quality_gate import check_quality
from services.session_store import save_session, get_previous_session, get_history
from services.supabase_auth import get_user_id
from services.supabase_client import get_client as get_supabase
from services.redis_client import (
    cache_get, cache_set, make_cache_key,
    check_rate_limit, is_configured as redis_is_configured,
)
from models.schemas import (
    AnalysisResult, AnalysisError,
    PreviousSession, HistorySummary,
)

Path(os.getenv("UPLOAD_DIR", "uploads")).mkdir(exist_ok=True)

# Reuse a single PoseAnalyzer instance — MediaPipe startup is slow
pose_analyzer: Optional[PoseAnalyzer] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global pose_analyzer
    pose_analyzer = PoseAnalyzer()
    print("✅ MediaPipe PoseAnalyzer initialized")
    yield
    pose_analyzer.close()
    print("🔴 PoseAnalyzer shut down")


APP_VERSION = "0.3.0"

app = FastAPI(
    title="BaseAI API",
    description="AI-powered youth baseball coaching — upload a swing or pitch video and get instant feedback.",
    version=APP_VERSION,
    lifespan=lifespan,
)

# ALLOWED_ORIGINS: comma-separated list in env, e.g. "https://aihomerun.app,https://www.aihomerun.app"
# Falls back to "*" for local development
_raw_origins = os.getenv("ALLOWED_ORIGINS", "*").strip().strip('"').strip("'")
ALLOWED_ORIGINS = [o.strip() for o in _raw_origins.split(",")] if _raw_origins != "*" else ["*"]
print(f"🌐 CORS allowed origins: {ALLOWED_ORIGINS}")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["POST", "GET", "PUT", "DELETE"],
    allow_headers=["*"],
)


# ──────────────────────────────────────────
# Health
# ──────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "app": "BaseAI", "version": APP_VERSION}


# ──────────────────────────────────────────
# Leaderboard & Trend
# ──────────────────────────────────────────

@app.get("/leaderboard")
async def get_leaderboard(
    age_group: str = Query("all"),
    user_id: str = Depends(get_user_id),
):
    """Top 20 users by best overall_score."""
    # Check cache first (user-independent part)
    cache_key = make_cache_key("leaderboard", age_group)
    cached_rows = cache_get(cache_key) if redis_is_configured() else None

    if cached_rows is not None:
        rows = cached_rows
    else:
        sb = get_supabase()
        resp = await asyncio.to_thread(
            lambda: sb.rpc("leaderboard_top20", {}).execute()
        )
        rows = resp.data or []
        if redis_is_configured():
            cache_set(cache_key, rows, ttl=60)

    result = []
    for i, r in enumerate(rows):
        uid = r.get("user_id", "")
        name = r.get("display_name", "") or "Player"
        initials = name[:2].upper() if name else "??"
        result.append({
            "entry_id": f"lb_{i}",
            "initials": initials,
            "display_name": name,
            "score": r.get("score", 0),
            "is_me": uid == user_id,
            "is_real_user": True,
        })
    return result


@app.get("/trend")
async def get_trend(
    limit: int = Query(8, ge=1, le=50),
    user_id: str = Depends(get_user_id),
):
    """Get score trend for the authenticated user."""
    cache_key = make_cache_key("trend", user_id)
    cached = cache_get(cache_key) if redis_is_configured() else None

    if cached is not None:
        rows = cached
    else:
        sb = get_supabase()
        resp = await asyncio.to_thread(
            lambda: (
                sb.table("analyses")
                .select("overall_score, created_at")
                .eq("user_id", user_id)
                .order("created_at", desc=True)
                .limit(limit)
                .execute()
            )
        )
        rows = list(reversed(resp.data or []))
        if redis_is_configured():
            cache_set(cache_key, rows, ttl=30)
    result = []
    for i, r in enumerate(rows):
        result.append({
            "session_number": i + 1,
            "overall_score": r.get("overall_score", 0),
            "created_at": r.get("created_at", ""),
        })
    return result


@app.get("/best-score")
async def get_best_score(user_id: str = Depends(get_user_id)):
    cache_key = make_cache_key("best", user_id)
    cached = cache_get(cache_key) if redis_is_configured() else None
    if cached is not None:
        return cached

    sb = get_supabase()
    resp = await asyncio.to_thread(
        lambda: (
            sb.table("analyses")
            .select("overall_score")
            .eq("user_id", user_id)
            .order("overall_score", desc=True)
            .limit(1)
            .execute()
        )
    )
    best = resp.data[0]["overall_score"] if resp.data else None
    result = {"best_score": best}
    if redis_is_configured():
        cache_set(cache_key, result, ttl=30)
    return result


# ──────────────────────────────────────────
# Account deletion
# ──────────────────────────────────────────

@app.delete("/account")
async def delete_account(user_id: str = Depends(get_user_id)):
    """Delete all user data (analyses). Profiles/children managed via Supabase directly."""
    sb = get_supabase()
    await asyncio.to_thread(
        lambda: sb.table("analyses").delete().eq("user_id", user_id).execute()
    )
    return {"ok": True}


# ──────────────────────────────────────────
# Video Analysis (Async)
# ──────────────────────────────────────────

from services.job_manager import create_job, get_job_status, get_analysis_result
from fastapi.responses import JSONResponse


@app.post(
    "/analyze",
    summary="Upload a baseball video for AI coaching analysis",
)
async def analyze(
    file: UploadFile = File(..., description="Baseball video file (mp4/mov/avi, max 100 MB)"),
    action_type: str = Form("swing", description="Action type: 'swing' or 'pitch'"),
    age: int = Form(10, description="Player age (6-18)", ge=6, le=18),
    user_id: str = Depends(get_user_id),
):
    if action_type not in ("swing", "pitch"):
        raise HTTPException(status_code=400, detail="action_type must be 'swing' or 'pitch'")

    # Rate limit: 10 analyses per hour per user
    if redis_is_configured() and not check_rate_limit(user_id, "analyze", limit=10, window_seconds=3600):
        raise HTTPException(status_code=429, detail="Rate limit exceeded. Please try again later.")

    # Choose async (Redis + worker) or synchronous (direct) processing
    use_async = redis_is_configured()

    video_id = None
    try:
        # 1. Save upload to temp
        video_id, video_path = await save_upload(file)

        # 2. Upload raw video to R2
        video_url = await asyncio.to_thread(upload_to_storage, video_id, video_path, user_id)

        if use_async:
            # ── Async path: create job + return 202 ──
            if not video_url:
                raise HTTPException(status_code=500, detail="Failed to upload video. Please try again.")

            ext = video_path.suffix
            storage_key = f"{user_id}/{video_id}{ext}"

            job_id = await asyncio.to_thread(
                create_job, user_id, action_type, age, storage_key
            )

            return JSONResponse(
                status_code=202,
                content={"job_id": job_id, "status": "pending"},
            )
        else:
            # ── Synchronous path: process video directly ──
            return await _analyze_sync(
                video_id=video_id,
                video_path=video_path,
                video_url=video_url,
                action_type=action_type,
                age=age,
                user_id=user_id,
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Failed to process analysis")
        raise HTTPException(status_code=500, detail="Failed to start analysis. Please try again.")
    finally:
        if video_id:
            cleanup_video_dir(video_id)


async def _analyze_sync(
    video_id: str,
    video_path,
    video_url: str | None,
    action_type: str,
    age: int,
    user_id: str,
) -> AnalysisResult:
    """
    Synchronous analysis fallback — runs the full pipeline in-process.
    Used when Redis/worker infrastructure is not available.
    """
    import time as _time

    start = _time.time()

    async with _analysis_semaphore:
        # 1. Video info
        info = await asyncio.to_thread(get_video_info, video_path)
        logger.info("Video: %dx%d %.1ffps %.1fs",
                     info["width"], info["height"], info["fps"], info["duration_sec"])

        # 2. Extract frames
        frames = await asyncio.to_thread(extract_frames, video_path)
        logger.info("Extracted %d frames", len(frames))

        # 3. MediaPipe pose estimation
        frames_data = await asyncio.to_thread(pose_analyzer.analyze_frames, frames)
        valid_count = sum(1 for f in frames_data if f is not None)
        logger.info("Valid pose frames: %d/%d", valid_count, len(frames))

        # 4. Quality gate
        quality = await asyncio.to_thread(check_quality, frames, frames_data, info)
        if not quality.passed:
            raise HTTPException(
                status_code=422,
                detail={
                    "error": "quality_gate_failed",
                    "issues": [i.model_dump() for i in quality.issues],
                    "visibility_rate": quality.visibility_rate,
                },
            )

        # 5. Previous session for comparison
        prev_row = await asyncio.to_thread(get_previous_session, user_id, action_type)

        # 6. Compute metrics
        if action_type == "swing":
            metrics = await asyncio.to_thread(analyze_swing, frames_data)
        else:
            metrics = await asyncio.to_thread(analyze_pitch, frames_data)

        # 7. Claude AI feedback
        feedback = await asyncio.to_thread(analyze_with_claude, metrics, age)

        # 8. Build comparison
        previous_session = None
        if prev_row:
            previous_session = PreviousSession(
                session_date=prev_row["created_at"],
                action_type=prev_row["action_type"],
                overall_score=prev_row["overall_score"],
                technique_score=prev_row["technique_score"],
                power_score=prev_row["power_score"],
                balance_score=prev_row["balance_score"],
                video_id=prev_row.get("video_id"),
                video_url=prev_row.get("video_url"),
            )

        # 9. Assemble result
        elapsed = _time.time() - start
        result = AnalysisResult(
            video_id=video_id,
            action_type=action_type,
            metrics=metrics,
            feedback=feedback,
            processing_time_seconds=round(elapsed, 2),
            quality=quality,
            previous_session=previous_session,
            video_url=video_url,
        )

        # 10. Save to database
        await asyncio.to_thread(save_session, user_id, result)

        # 11. Add history
        hist_rows = await asyncio.to_thread(get_history, user_id, action_type, 8)
        if hist_rows:
            result.history = [
                HistorySummary(
                    session_date=r["created_at"],
                    overall_score=r["overall_score"],
                    video_id=r.get("video_id"),
                    video_url=r.get("video_url"),
                )
                for r in hist_rows
            ]

        logger.info("Sync analysis complete — score %d (%.1fs)", feedback.overall_score, elapsed)
        return result


@app.get(
    "/analyze/status/{job_id}",
    summary="Check the status of an analysis job",
)
async def analyze_status(
    job_id: str,
    user_id: str = Depends(get_user_id),
):
    job = await asyncio.to_thread(get_job_status, job_id, user_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    response = {
        "job_id": job["id"],
        "status": job["status"],
        "progress": job.get("progress", 0),
    }

    if job["status"] == "completed" and job.get("analysis_id"):
        result = await asyncio.to_thread(get_analysis_result, job["analysis_id"])
        if result:
            # Fetch history for the completed result
            hist_rows = await asyncio.to_thread(
                get_history, user_id, result.get("action_type"), 8
            )
            response["result"] = result
            if hist_rows:
                response["result"]["history"] = hist_rows

    elif job["status"] == "failed":
        response["error"] = job.get("error_message", "Analysis failed")

    return response


@app.get(
    "/history",
    summary="Get analysis history for the authenticated user",
)
async def history_endpoint(
    action_type: Optional[str] = Query(None, description="Filter by 'swing' or 'pitch'"),
    limit: int = Query(20, ge=1, le=100, description="Number of sessions to return"),
    user_id: str = Depends(get_user_id),
):
    rows = get_history(user_id, action_type=action_type, limit=limit)
    return {"history": rows, "count": len(rows)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000)
