"""
Quality Gate — validates video frames and pose data before running the full analysis pipeline.

Checks performed (in order):
  1. Frame rate (fps)              — too low → bad temporal resolution
  2. Pose visibility rate          — too few detected frames → unreliable metrics
  3. Blur detection (Laplacian)    — too blurry → keypoint inaccuracy
  4. Brightness / lighting         — too dark or overexposed
  5. Frame count (completeness)    — clip too short to capture the full motion

Returns a QualityGateResult: passed=True if no *error*-severity issues found.
Warnings are included in the result but do NOT block analysis.
"""
import cv2
import numpy as np
from models.schemas import QualityGateResult, QualityIssue


def check_quality(
    frames: list,
    frames_data: list,
    video_info: dict,
) -> QualityGateResult:
    """
    Args:
        frames:       Raw BGR numpy frames from extract_frames()
        frames_data:  Pose-landmark dicts from PoseAnalyzer (None = detection failed)
        video_info:   Dict from get_video_info() — keys: fps, width, height, …

    Returns:
        QualityGateResult with passed=True/False and structured issue list
    """
    issues: list[QualityIssue] = []

    # ── 1. Frame rate ─────────────────────────────────────────────────────────
    fps = float(video_info.get("fps", 30))
    if fps < 20:
        issues.append(QualityIssue(
            check="low_fps",
            message=(
                f"Video frame rate is too low ({fps:.0f} fps). "
                "Please record at 24 fps or higher — most phones default to 30 fps in their camera app."
            ),
            severity="error",
        ))
    elif fps < 24:
        issues.append(QualityIssue(
            check="marginal_fps",
            message=(
                f"Frame rate is a little low ({fps:.0f} fps). "
                "Results may be slightly less accurate — 30 fps or higher is recommended."
            ),
            severity="warning",
        ))

    # ── 2. Pose visibility rate ───────────────────────────────────────────────
    total = len(frames_data)
    valid_count = sum(1 for f in frames_data if f is not None)
    visibility_rate = valid_count / total if total > 0 else 0.0

    if visibility_rate < 0.40:
        issues.append(QualityIssue(
            check="poor_visibility",
            message=(
                f"The AI could only detect the player in {int(visibility_rate * 100)}% of frames. "
                "Make sure the whole body — head to feet — is fully visible, "
                "and that there is enough light."
            ),
            severity="error",
        ))
    elif visibility_rate < 0.65:
        issues.append(QualityIssue(
            check="partial_visibility",
            message=(
                f"Player detection was inconsistent ({int(visibility_rate * 100)}% of frames). "
                "Try moving the camera farther back so the entire body fits comfortably in frame."
            ),
            severity="warning",
        ))

    # ── 3. Blur detection (Laplacian variance) ────────────────────────────────
    if frames:
        step = max(1, len(frames) // 6)
        sample = frames[::step][:6]
        blur_scores = []
        for fr in sample:
            gray = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
            blur_scores.append(cv2.Laplacian(gray, cv2.CV_64F).var())
        avg_blur = float(np.mean(blur_scores))

        if avg_blur < 25:
            issues.append(QualityIssue(
                check="blurry_video",
                message=(
                    "The video is too blurry for reliable pose tracking. "
                    "Keep the camera completely still — a tripod or resting it on a stable surface helps a lot."
                ),
                severity="error",
            ))
        elif avg_blur < 70:
            issues.append(QualityIssue(
                check="slightly_blurry",
                message=(
                    "The video is a little blurry. "
                    "Using a tripod or propping the phone on something stable will improve accuracy."
                ),
                severity="warning",
            ))

    # ── 4. Brightness / lighting ──────────────────────────────────────────────
    if frames:
        mid_frame = frames[len(frames) // 2]
        brightness = float(np.mean(cv2.cvtColor(mid_frame, cv2.COLOR_BGR2GRAY)))

        if brightness < 35:
            issues.append(QualityIssue(
                check="too_dark",
                message=(
                    "The video is too dark for pose detection. "
                    "Please record in a well-lit area — outdoors or near a bright indoor light."
                ),
                severity="error",
            ))
        elif brightness > 235:
            issues.append(QualityIssue(
                check="overexposed",
                message=(
                    "The video looks overexposed (too bright). "
                    "Avoid pointing the camera directly at the sun or bright lights behind the player."
                ),
                severity="warning",
            ))

    # ── 5. Frame count / action completeness ──────────────────────────────────
    if total < 6:
        issues.append(QualityIssue(
            check="too_short",
            message=(
                f"Only {total} frames were captured — the clip is too short. "
                "Please upload a video that shows the complete motion from start to finish."
            ),
            severity="error",
        ))

    # ── Result ────────────────────────────────────────────────────────────────
    has_errors = any(i.severity == "error" for i in issues)
    return QualityGateResult(
        passed=not has_errors,
        issues=issues,
        visibility_rate=round(visibility_rate, 2),
    )
