"""
Video processing module: handles upload validation and frame extraction.
Uses OpenCV (which bundles FFmpeg) — no separate FFmpeg installation required.
"""
import os
import uuid
import shutil
from pathlib import Path

import cv2
import aiofiles
from fastapi import UploadFile, HTTPException


ALLOWED_EXTENSIONS = {".mp4", ".mov", ".avi", ".m4v"}
MAX_VIDEO_SIZE_MB = int(os.getenv("MAX_VIDEO_SIZE_MB", 100))
UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", "uploads"))
FRAMES_PER_SECOND = int(os.getenv("FRAMES_PER_SECOND", 6))


async def save_upload(file: UploadFile) -> tuple:
    """Save the uploaded video file. Returns (video_id, file_path)."""
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported format '{suffix}'. Please upload one of: {', '.join(ALLOWED_EXTENSIONS)}"
        )

    video_id = uuid.uuid4().hex
    video_dir = UPLOAD_DIR / video_id
    video_dir.mkdir(parents=True, exist_ok=True)
    video_path = video_dir / f"input{suffix}"

    size = 0
    async with aiofiles.open(video_path, "wb") as out:
        while chunk := await file.read(1024 * 1024):  # 1 MB chunks
            size += len(chunk)
            if size > MAX_VIDEO_SIZE_MB * 1024 * 1024:
                shutil.rmtree(video_dir)
                raise HTTPException(
                    status_code=413,
                    detail=f"Video exceeds the {MAX_VIDEO_SIZE_MB} MB size limit."
                )
            await out.write(chunk)

    return video_id, video_path


MAX_FRAME_WIDTH = 640   # Downscale to max 640px wide — keeps MediaPipe accurate, saves ~85% memory


def extract_frames(video_path: Path, target_fps: int = FRAMES_PER_SECOND) -> list:
    """
    Uniformly sample frames from a video at the target frame rate.
    Returns a list of numpy arrays in BGR format (downscaled to MAX_FRAME_WIDTH).
    Only processes the first 15 seconds to keep latency and memory reasonable.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")

    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec = total_frames / video_fps

    max_duration = min(duration_sec, 8.0)   # Cap at 8s — swing/pitch is 1-3s, 8s is plenty
    frame_interval = max(1, int(video_fps / target_fps))

    frames = []
    current_frame = 0
    next_target = 0
    max_frame = int(max_duration * video_fps)

    while next_target < max_frame:
        ret, frame = cap.read()
        if not ret:
            break
        if current_frame == next_target:
            # Downscale wide frames to MAX_FRAME_WIDTH — reduces memory 85% for 1080p
            h, w = frame.shape[:2]
            if w > MAX_FRAME_WIDTH:
                scale = MAX_FRAME_WIDTH / w
                new_w = MAX_FRAME_WIDTH
                new_h = int(h * scale)
                frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
            frames.append(frame)
            next_target += frame_interval
        current_frame += 1

    cap.release()

    if len(frames) < 5:
        raise ValueError(
            f"Video too short — only {len(frames)} frames extracted. "
            "Please upload a clip of at least 1 second."
        )

    return frames


def get_video_info(video_path: Path) -> dict:
    """Return basic video metadata."""
    cap = cv2.VideoCapture(str(video_path))
    info = {
        "fps": cap.get(cv2.CAP_PROP_FPS),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "total_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "duration_sec": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) / (cap.get(cv2.CAP_PROP_FPS) or 30),
    }
    cap.release()
    return info


def upload_to_storage(video_id: str, video_path: Path, user_id: str) -> "str | None":
    """
    Upload video to Supabase Storage and return a public URL.
    Returns None if upload fails or Supabase is not configured.
    """
    url = os.environ.get("SUPABASE_URL", "").strip()
    key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "").strip()
    if not url or not key:
        return None

    try:
        from supabase import create_client
        client = create_client(url, key)
        bucket = "videos"
        ext = video_path.suffix  # e.g. ".mov"
        storage_path = f"{user_id}/{video_id}{ext}"

        with open(video_path, "rb") as f:
            data = f.read()

        # Determine content type
        content_types = {".mp4": "video/mp4", ".mov": "video/quicktime", ".avi": "video/x-msvideo", ".m4v": "video/x-m4v"}
        content_type = content_types.get(ext.lower(), "video/mp4")

        client.storage.from_(bucket).upload(
            path=storage_path,
            file=data,
            file_options={"content-type": content_type},
        )

        # Get public URL
        res = client.storage.from_(bucket).get_public_url(storage_path)
        return res
    except Exception as exc:
        import logging
        logging.getLogger(__name__).warning("upload_to_storage failed: %s", exc)
        return None


def cleanup_video_dir(video_id: str):
    """Remove the temporary upload directory for a given video_id."""
    video_dir = UPLOAD_DIR / video_id
    if video_dir.exists():
        shutil.rmtree(video_dir)
