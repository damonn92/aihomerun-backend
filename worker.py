"""
Background worker for processing video analysis jobs.

Polls Redis queue, processes videos, and stores results in Supabase.
Run with: python worker.py
"""
import logging
import os
import sys
import time
import tempfile
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

# Sentry initialization
import sentry_sdk
_sentry_dsn = os.environ.get("SENTRY_DSN", "")
if _sentry_dsn:
    sentry_sdk.init(
        dsn=_sentry_dsn,
        traces_sample_rate=0.2,
        environment=os.environ.get("ENVIRONMENT", "development"),
    )

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("worker")

from services.redis_client import dequeue_job
from services.job_manager import update_job, complete_job, fail_job
from services.video_processor import extract_frames, get_video_info, upload_to_storage
from services.pose_analyzer import PoseAnalyzer
from services.baseball_metrics import analyze_swing, analyze_pitch
from services.ai_analyzer import analyze_with_claude
from services.quality_gate import check_quality
from services.session_store import save_session, get_previous_session, get_history
from models.schemas import AnalysisResult, PreviousSession, HistorySummary

# Download video from R2 for processing
import boto3


def _get_r2_client():
    return boto3.client(
        "s3",
        endpoint_url=os.environ.get("R2_ENDPOINT", ""),
        aws_access_key_id=os.environ.get("R2_ACCESS_KEY_ID", ""),
        aws_secret_access_key=os.environ.get("R2_SECRET_ACCESS_KEY", ""),
        region_name="auto",
    )


R2_BUCKET = "aihomerun-videos"

# Reuse a single PoseAnalyzer instance
pose_analyzer: PoseAnalyzer | None = None

MAX_RETRIES = 3
RETRY_DELAYS = [5, 30, 120]  # seconds


def download_from_r2(storage_key: str, local_path: Path) -> None:
    """Download a video file from R2 to a local path."""
    client = _get_r2_client()
    client.download_file(R2_BUCKET, storage_key, str(local_path))
    logger.info("Downloaded %s from R2 (%d bytes)", storage_key, local_path.stat().st_size)


def process_job(job: dict) -> None:
    """Process a single analysis job."""
    job_id = job["job_id"]
    user_id = job["user_id"]
    action_type = job["action_type"]
    age = job["age"]
    video_key = job["video_storage_key"]

    logger.info("[%s] Starting job for user %s", job_id[:8], user_id[:8])
    update_job(job_id, status="processing", progress=5)

    tmp_dir = None
    try:
        # 1. Download video from R2
        tmp_dir = tempfile.mkdtemp(prefix="analysis_")
        ext = Path(video_key).suffix or ".mp4"
        video_path = Path(tmp_dir) / f"video{ext}"
        download_from_r2(video_key, video_path)
        update_job(job_id, progress=15)

        # 2. Video info
        info = get_video_info(video_path)
        logger.info("[%s] %dx%d %.1ffps %.1fs",
                    job_id[:8], info["width"], info["height"],
                    info["fps"], info["duration_sec"])

        # 3. Extract frames
        frames = extract_frames(video_path)
        logger.info("[%s] %d frames extracted", job_id[:8], len(frames))
        update_job(job_id, progress=30)

        # 4. MediaPipe pose estimation
        frames_data = pose_analyzer.analyze_frames(frames)
        valid_count = sum(1 for f in frames_data if f is not None)
        logger.info("[%s] Valid pose frames: %d/%d", job_id[:8], valid_count, len(frames))
        update_job(job_id, progress=50)

        # 5. Quality gate
        quality = check_quality(frames, frames_data, info)
        if not quality.passed:
            issues_str = "; ".join(
                f"{i.check}: {i.message}" for i in quality.issues if i.severity == "error"
            )
            fail_job(job_id, f"Video quality check failed: {issues_str}")
            return

        update_job(job_id, progress=55)

        # 6. Fetch previous session
        prev_row = get_previous_session(user_id, action_type)

        # 7. Compute metrics
        if action_type == "swing":
            metrics = analyze_swing(frames_data)
        else:
            metrics = analyze_pitch(frames_data)
        update_job(job_id, progress=65)

        # 8. Claude AI feedback
        feedback = analyze_with_claude(metrics, age)
        update_job(job_id, progress=80)

        # 9. Build video URL from storage key
        r2_public = os.environ.get("R2_PUBLIC_URL", "")
        video_url = f"{r2_public}/{video_key}" if r2_public else None
        video_id = Path(video_key).stem.split("/")[-1]

        # 10. Build comparison
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
            processing_time_seconds=0,
            quality=quality,
            previous_session=previous_session,
            video_url=video_url,
        )

        # 12. Save to database
        analysis_id = save_session(user_id, result)
        update_job(job_id, progress=90)

        # 13. Add history
        hist_rows = get_history(user_id, action_type, 8)
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

        # 14. Complete
        if analysis_id:
            complete_job(job_id, analysis_id)
        else:
            fail_job(job_id, "Failed to save analysis result")

        logger.info("[%s] Job completed — score %d", job_id[:8], feedback.overall_score)

    except Exception as exc:
        logger.exception("[%s] Job processing error", job_id[:8])
        fail_job(job_id, str(exc)[:500])

    finally:
        # Cleanup temp files
        if tmp_dir:
            import shutil
            shutil.rmtree(tmp_dir, ignore_errors=True)


def main():
    """Main worker loop — polls Redis queue for jobs."""
    global pose_analyzer

    logger.info("Starting analysis worker...")
    pose_analyzer = PoseAnalyzer()
    logger.info("MediaPipe PoseAnalyzer initialized")

    consecutive_errors = 0

    while True:
        try:
            job = dequeue_job(timeout=5)
            if job is None:
                consecutive_errors = 0
                continue

            consecutive_errors = 0
            retry_count = job.get("_retry", 0)

            try:
                process_job(job)
            except Exception as exc:
                logger.exception("Unhandled error processing job %s", job.get("job_id", "?"))

                # Retry with exponential backoff
                if retry_count < MAX_RETRIES:
                    delay = RETRY_DELAYS[min(retry_count, len(RETRY_DELAYS) - 1)]
                    logger.info("Retrying job %s in %ds (attempt %d/%d)",
                               job.get("job_id", "?"), delay, retry_count + 1, MAX_RETRIES)
                    time.sleep(delay)
                    job["_retry"] = retry_count + 1
                    from services.redis_client import enqueue_job
                    enqueue_job(job["job_id"], job)
                else:
                    fail_job(job["job_id"], f"Max retries exceeded: {exc}")

        except KeyboardInterrupt:
            logger.info("Worker shutting down...")
            break
        except Exception as exc:
            consecutive_errors += 1
            logger.error("Queue polling error: %s", exc)
            if consecutive_errors > 10:
                logger.critical("Too many consecutive errors, sleeping 30s")
                time.sleep(30)
            else:
                time.sleep(1)

    if pose_analyzer:
        pose_analyzer.close()
    logger.info("Worker stopped")


if __name__ == "__main__":
    main()
