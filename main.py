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
from services.auth import (
    get_user_id,
    create_access_token,
    verify_apple_id_token,
    verify_google_id_token,
    find_or_create_social_user,
    find_email_user,
    create_email_user,
    verify_password,
)
from services.d1_client import execute as d1_execute, is_configured as d1_is_configured
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
# Auth request/response models
# ──────────────────────────────────────────

class AuthResponse(BaseModel):
    access_token: str
    user_id: str
    email: Optional[str] = None

class AppleSignInRequest(BaseModel):
    id_token: str
    nonce: Optional[str] = None

class GoogleSignInRequest(BaseModel):
    id_token: str

class EmailSignInRequest(BaseModel):
    email: str
    password: str

class EmailSignUpRequest(BaseModel):
    email: str
    password: str

class ProfileResponse(BaseModel):
    id: str
    display_name: Optional[str] = None
    avatar_url: Optional[str] = None
    age_group: Optional[str] = None

class ProfileUpdateRequest(BaseModel):
    display_name: Optional[str] = None
    avatar_url: Optional[str] = None
    age_group: Optional[str] = None

class ChildModel(BaseModel):
    id: Optional[str] = None
    name: Optional[str] = None
    age: Optional[int] = None

class LeaderboardRow(BaseModel):
    entry_id: str
    initials: str
    display_name: str
    score: int
    is_me: bool
    is_real_user: bool

class TrendRow(BaseModel):
    session_number: int
    overall_score: int
    created_at: str


# ──────────────────────────────────────────
# Health
# ──────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "app": "BaseAI", "version": APP_VERSION}


# ──────────────────────────────────────────
# Auth endpoints
# ──────────────────────────────────────────

@app.post("/auth/apple", response_model=AuthResponse)
async def auth_apple(req: AppleSignInRequest):
    """Authenticate with Apple Sign-In identity token."""
    apple_payload = await asyncio.to_thread(verify_apple_id_token, req.id_token)
    apple_user_id = apple_payload.get("sub")
    email = apple_payload.get("email")

    user = await asyncio.to_thread(
        find_or_create_social_user, "apple", apple_user_id, email
    )
    token = create_access_token(user["id"], user.get("email"))
    return AuthResponse(access_token=token, user_id=user["id"], email=user.get("email"))


@app.post("/auth/google", response_model=AuthResponse)
async def auth_google(req: GoogleSignInRequest):
    """Authenticate with Google Sign-In identity token."""
    google_payload = await asyncio.to_thread(verify_google_id_token, req.id_token)
    google_user_id = google_payload.get("sub")
    email = google_payload.get("email")

    user = await asyncio.to_thread(
        find_or_create_social_user, "google", google_user_id, email
    )
    token = create_access_token(user["id"], user.get("email"))
    return AuthResponse(access_token=token, user_id=user["id"], email=user.get("email"))


@app.post("/auth/email/signin", response_model=AuthResponse)
async def auth_email_signin(req: EmailSignInRequest):
    """Sign in with email and password."""
    user = await asyncio.to_thread(find_email_user, req.email)
    if not user:
        raise HTTPException(401, "Invalid email or password")
    if not verify_password(req.password, user.get("password_hash", "")):
        raise HTTPException(401, "Invalid email or password")

    token = create_access_token(user["id"], user.get("email"))
    return AuthResponse(access_token=token, user_id=user["id"], email=user.get("email"))


@app.post("/auth/email/signup", response_model=AuthResponse)
async def auth_email_signup(req: EmailSignUpRequest):
    """Create a new account with email and password."""
    existing = await asyncio.to_thread(find_email_user, req.email)
    if existing:
        raise HTTPException(409, "Email already registered")

    if len(req.password) < 6:
        raise HTTPException(400, "Password must be at least 6 characters")

    user = await asyncio.to_thread(create_email_user, req.email, req.password)
    token = create_access_token(user["id"], user.get("email"))
    return AuthResponse(access_token=token, user_id=user["id"], email=user.get("email"))


# ──────────────────────────────────────────
# Profile endpoints
# ──────────────────────────────────────────

@app.get("/profile", response_model=ProfileResponse)
async def get_profile(user_id: str = Depends(get_user_id)):
    rows = await asyncio.to_thread(
        d1_execute,
        "SELECT id, display_name, avatar_url, age_group FROM profiles WHERE id = ?1 LIMIT 1",
        [user_id],
    )
    if not rows:
        return ProfileResponse(id=user_id)
    return ProfileResponse(**rows[0])


@app.put("/profile", response_model=ProfileResponse)
async def update_profile(req: ProfileUpdateRequest, user_id: str = Depends(get_user_id)):
    # Upsert profile
    await asyncio.to_thread(
        d1_execute,
        """INSERT INTO profiles (id, display_name, avatar_url, age_group, updated_at)
           VALUES (?1, ?2, ?3, ?4, datetime('now'))
           ON CONFLICT(id) DO UPDATE SET
             display_name = ?2, avatar_url = ?3, age_group = ?4, updated_at = datetime('now')""",
        [user_id, req.display_name, req.avatar_url, req.age_group],
    )
    return ProfileResponse(
        id=user_id,
        display_name=req.display_name,
        avatar_url=req.avatar_url,
        age_group=req.age_group,
    )


# ──────────────────────────────────────────
# Children endpoints
# ──────────────────────────────────────────

@app.get("/children", response_model=List[ChildModel])
async def get_children(user_id: str = Depends(get_user_id)):
    rows = await asyncio.to_thread(
        d1_execute,
        "SELECT id, name, age FROM children WHERE parent_id = ?1 ORDER BY created_at",
        [user_id],
    )
    return [ChildModel(**r) for r in rows]


@app.post("/children", response_model=ChildModel)
async def create_child(child: ChildModel, user_id: str = Depends(get_user_id)):
    import uuid
    child_id = uuid.uuid4().hex
    await asyncio.to_thread(
        d1_execute,
        "INSERT INTO children (id, parent_id, name, age) VALUES (?1, ?2, ?3, ?4)",
        [child_id, user_id, child.name, child.age],
    )
    return ChildModel(id=child_id, name=child.name, age=child.age)


@app.put("/children/{child_id}", response_model=ChildModel)
async def update_child(child_id: str, child: ChildModel, user_id: str = Depends(get_user_id)):
    await asyncio.to_thread(
        d1_execute,
        "UPDATE children SET name = ?1, age = ?2 WHERE id = ?3 AND parent_id = ?4",
        [child.name, child.age, child_id, user_id],
    )
    return ChildModel(id=child_id, name=child.name, age=child.age)


@app.delete("/children/{child_id}")
async def delete_child(child_id: str, user_id: str = Depends(get_user_id)):
    await asyncio.to_thread(
        d1_execute,
        "DELETE FROM children WHERE id = ?1 AND parent_id = ?2",
        [child_id, user_id],
    )
    return {"ok": True}


# ──────────────────────────────────────────
# Leaderboard & Trend
# ──────────────────────────────────────────

@app.get("/leaderboard")
async def get_leaderboard(
    age_group: str = Query("all"),
    user_id: str = Depends(get_user_id),
):
    """
    Simple leaderboard: top 20 users by best overall_score.
    """
    rows = await asyncio.to_thread(
        d1_execute,
        """SELECT a.user_id, MAX(a.overall_score) as score,
                  COALESCE(p.display_name, '') as display_name
           FROM analyses a
           LEFT JOIN profiles p ON p.id = a.user_id
           GROUP BY a.user_id
           ORDER BY score DESC
           LIMIT 20""",
        [],
    )

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
    rows = await asyncio.to_thread(
        d1_execute,
        """SELECT overall_score, created_at
           FROM analyses
           WHERE user_id = ?1
           ORDER BY created_at DESC
           LIMIT ?2""",
        [user_id, limit],
    )
    rows = list(reversed(rows))  # oldest first
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
    rows = await asyncio.to_thread(
        d1_execute,
        """SELECT MAX(overall_score) as best_score
           FROM analyses WHERE user_id = ?1""",
        [user_id],
    )
    best = rows[0].get("best_score") if rows else None
    return {"best_score": best}


# ──────────────────────────────────────────
# Account deletion
# ──────────────────────────────────────────

@app.delete("/account")
async def delete_account(user_id: str = Depends(get_user_id)):
    """Delete all user data."""
    await asyncio.to_thread(d1_execute, "DELETE FROM analyses WHERE user_id = ?1", [user_id])
    await asyncio.to_thread(d1_execute, "DELETE FROM children WHERE parent_id = ?1", [user_id])
    await asyncio.to_thread(d1_execute, "DELETE FROM profiles WHERE id = ?1", [user_id])
    await asyncio.to_thread(d1_execute, "DELETE FROM users WHERE id = ?1", [user_id])
    return {"ok": True}


# ──────────────────────────────────────────
# Video Analysis
# ──────────────────────────────────────────

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
                logger.info("[%s] Video uploaded to R2: %s", video_id[:8], video_url[:80])

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

            # 12. Persist to D1 (graceful failure — won't crash if DB not set up)
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
