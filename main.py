"""
BaseAI — Youth Baseball AI Coaching Backend
FastAPI entry point
"""
import time
import os
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

from typing import Optional, List
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Depends, Query
from fastapi.middleware.cors import CORSMiddleware

from services.video_processor import save_upload, extract_frames, get_video_info, cleanup_video_dir
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


app = FastAPI(
    title="BaseAI API",
    description="AI-powered youth baseball coaching — upload a swing or pitch video and get instant feedback.",
    version="0.2.0",
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
    return {"status": "ok", "app": "BaseAI", "version": "0.2.0", "model": "claude-haiku-4-5"}


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
    print(f"👤 Authenticated user: {user_id[:8]}…")

    try:
        # 1. Save upload
        video_id, video_path = await save_upload(file)

        # 2. Video info (for logging + quality gate)
        info = get_video_info(video_path)
        print(f"📹 [{video_id[:8]}] {info['width']}x{info['height']} "
              f"{info['fps']:.1f}fps {info['duration_sec']:.1f}s")

        # 3. Extract frames
        frames = extract_frames(video_path)
        print(f"🖼  [{video_id[:8]}] {len(frames)} frames extracted")

        # 4. MediaPipe pose estimation
        frames_data = pose_analyzer.analyze_frames(frames)
        valid_count = sum(1 for f in frames_data if f is not None)
        print(f"🦾 [{video_id[:8]}] Valid pose frames: {valid_count}/{len(frames)}")

        # 5. Quality Gate — run before expensive analysis
        quality = check_quality(frames, frames_data, info)
        warn_count = sum(1 for i in quality.issues if i.severity == "warning")
        err_count  = sum(1 for i in quality.issues if i.severity == "error")
        print(f"🔍 [{video_id[:8]}] Quality: {'✅ passed' if quality.passed else '❌ failed'} "
              f"(visibility {quality.visibility_rate:.0%}, {err_count} errors, {warn_count} warnings)")

        if not quality.passed:
            # Return structured error so the frontend can display per-check messages
            raise HTTPException(
                status_code=422,
                detail={
                    "error": "video_quality_check_failed",
                    "issues": [i.model_dump() for i in quality.issues],
                    "visibility_rate": quality.visibility_rate,
                },
            )

        # 6. Fetch previous session BEFORE saving (for before/after comparison)
        prev_row = get_previous_session(user_id, action_type)

        # 7. Compute baseball-specific metrics
        if action_type == "swing":
            metrics = analyze_swing(frames_data)
        else:
            metrics = analyze_pitch(frames_data)

        # 8. Claude AI feedback
        feedback = analyze_with_claude(metrics, age=age)

        elapsed = round(time.time() - start, 2)
        print(f"✅ [{video_id[:8]}] Done — score {feedback.overall_score}, {elapsed}s")

        # 9. Build before/after comparison object (if we had a previous session)
        previous_session = None
        if prev_row:
            previous_session = PreviousSession(
                session_date=prev_row["created_at"],
                action_type=prev_row["action_type"],
                overall_score=prev_row["overall_score"],
                technique_score=prev_row["technique_score"],
                power_score=prev_row["power_score"],
                balance_score=prev_row["balance_score"],
            )

        # 10. Assemble result
        result = AnalysisResult(
            video_id=video_id,
            action_type=action_type,
            metrics=metrics,
            feedback=feedback,
            processing_time_seconds=elapsed,
            quality=quality,
            previous_session=previous_session,
        )

        # 11. Persist to Supabase (graceful failure — won't crash if DB not set up)
        save_session(user_id, result)

        # 12. Fetch growth history AFTER saving so the current session is included
        hist_rows = get_history(user_id, action_type, limit=8)
        if hist_rows:
            result.history = [
                HistorySummary(
                    session_date=r["created_at"],
                    overall_score=r["overall_score"],
                )
                for r in hist_rows
            ]

        return result

    except HTTPException:
        raise

    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    except Exception as e:
        print(f"❌ [{video_id[:8] if video_id else '?'}] Error: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

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
