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
FRAMES_PER_SECOND = int(os.getenv("FRAMES_PER_SECOND", 10))


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
                await out.close()
                shutil.rmtree(video_dir)
                raise HTTPException(
                    status_code=413,
                    detail=f"Video exceeds the {MAX_VIDEO_SIZE_MB} MB size limit."
                )
            await out.write(chunk)

    return video_id, video_path


def extract_frames(video_path: Path, target_fps: int = FRAMES_PER_SECOND) -> list:
    """
    Uniformly sample frames from a video at the target frame rate.
    Returns a list of numpy arrays in BGR format.
    Only processes the first 30 seconds to keep latency reasonable.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")

    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec = total_frames / video_fps

    max_duration = min(duration_sec, 30.0)
    frame_interval = max(1, int(video_fps / target_fps))

    frames = []
    frame_idx = 0
    max_frame = int(max_duration * video_fps)

    while frame_idx < max_frame:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
        frame_idx += frame_interval

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


def cleanup_video_dir(video_id: str):
    """Remove the temporary upload directory for a given video_id."""
    video_dir = UPLOAD_DIR / video_id
    if video_dir.exists():
        shutil.rmtree(video_dir)
