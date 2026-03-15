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

from typing import Optional, List
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Depends, Query
from fastapi.middleware.cors import CORSMiddleware

logger = logging.getLogger(__name__)

# Limit concurrent video analyses to prevent OOM on small instances
_analysis_semaphore = asyncio.Semaphore(int(os.getenv("MAX_CONCURRENT_ANALYSES", 3)))

from services.video_processor import save_upload, extract_frames, get_video_info, cleanup_video_dir, upload_to_storage
from services.pose_analyzer import PoseAnalyzer
from services.baseball_metrics import analyze_swing, analyze_pitch
from services.ai_analyzer import analyze_with_claude
from services.quality_gate import check_quality
from services.session_store import save_session, get_previous_session, get_history
from services.auth import get_user_id
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


APP_VERSION = "0.2.0"

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
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)


# ──────────────────────────────────────────
# Routes
# ──────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "app": "BaseAI", "version": APP_VERSION}


@app.post(
    "/analyze",
    response_model=AnalysisResult,
    responses={400: {"model": AnalysisError}, 422: {"model": AnalysisError}},
    summary="Upload a baseball video and get AI coaching feedback",
)
async def analyze(
    file: UploadFile = File(..., description="Baseball video file (mp4/mov/avi, max 100 MB)"),
    action_type: str = Form("swing", description="Action type: 'swing' or 'pitch'"),
    age: int = Form(10, description="Player age (6-18)", ge=6, le=18),
    user_id: str = Depends(get_user_id),
):
    if action_type not in ("swing", "pitch"):
        raise HTTPException(status_code=400, detail="action_type must be 'swing' or 'pitch'")

    start = time.time()
    video_id = None
    logger.info("Authenticated user: %s…", user_id[:8])

    async with _analysis_semaphore:
        try:
            # 1. Save upload
            video_id, video_path = await save_upload(file)

            # 2. Video info (for logging + quality gate)
            info = await asyncio.to_thread(get_video_info, video_path)
            logger.info("[%s] %dx%d %.1ffps %.1fs",
                        video_id[:8], info['width'], info['height'],
                        info['fps'], info['duration_sec'])

            # 3. Extract frames
            frames = await asyncio.to_thread(extract_frames, video_path)
            logger.info("[%s] %d frames extracted", video_id[:8], len(frames))

            # 4. MediaPipe pose estimation
            frames_data = await asyncio.to_thread(pose_analyzer.analyze_frames, frames)
            valid_count = sum(1 for f in frames_data if f is not None)
            logger.info("[%s] Valid pose frames: %d/%d", video_id[:8], valid_count, len(frames))

            # 5. Quality Gate — run before expensive analysis
            quality = await asyncio.to_thread(check_quality, frames, frames_data, info)
            logger.info("[%s] Quality: %s (visibility %.0f%%)",
                        video_id[:8], "passed" if quality.passed else "failed",
                        quality.visibility_rate * 100)

            if not quality.passed:
                raise HTTPException(
                    status_code=422,
                    detail={
                        "error": "video_quality_check_failed",
                        "issues": [i.model_dump() for i in quality.issues],
                        "visibility_rate": quality.visibility_rate,
                    },
                )

            # 6. Fetch previous session BEFORE saving (for before/after comparison)
            prev_row = await asyncio.to_thread(get_previous_session, user_id, action_type)

            # 7. Compute baseball-specific metrics
            if action_type == "swing":
                metrics = await asyncio.to_thread(analyze_swing, frames_data)
            else:
                metrics = await asyncio.to_thread(analyze_pitch, frames_data)

            # 8. Claude AI feedback
            feedback = await asyncio.to_thread(analyze_with_claude, metrics, age)

            elapsed = round(time.time() - start, 2)
            logger.info("[%s] Done — score %d, %.2fs", video_id[:8], feedback.overall_score, elapsed)

            # 9. Upload video to cloud storage BEFORE cleanup
            video_url = await asyncio.to_thread(upload_to_storage, video_id, video_path, user_id)
            if video_url:
                logger.info("[%s] Video uploaded to storage: %s", video_id[:8], video_url[:80])

            # 10. Build before/after comparison object (if we had a previous session)
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

            # 11. Assemble result
            result = AnalysisResult(
                video_id=video_id,
                action_type=action_type,
                metrics=metrics,
                feedback=feedback,
                processing_time_seconds=elapsed,
                quality=quality,
                previous_session=previous_session,
                video_url=video_url,
            )

            # 12. Persist to Supabase (graceful failure — won't crash if DB not set up)
            await asyncio.to_thread(save_session, user_id, result)

            # 13. Fetch growth history AFTER saving so the current session is included
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

            return result

        except HTTPException:
            raise

        except ValueError as e:
            raise HTTPException(status_code=422, detail=str(e))

        except Exception as e:
            logger.exception("[%s] Analysis error", video_id[:8] if video_id else "?")
            raise HTTPException(status_code=500, detail="Analysis failed. Please try again.")

        finally:
            if video_id:
                cleanup_video_dir(video_id)


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
