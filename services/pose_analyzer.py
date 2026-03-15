"""
姿态分析模块：使用 MediaPipe 提取人体关键点
MediaPipe Pose 提供 33 个关键点，覆盖全身
"""
import math
from typing import Optional
import numpy as np
import mediapipe as mp

# MediaPipe 关键点索引常量（对应 PoseLandmark）
LM = mp.solutions.pose.PoseLandmark


class PoseAnalyzer:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,      # 视频模式，利用时序信息
            model_complexity=0,           # 0=轻量(快), 1=标准, 2=高精度 — 0 够用且快很多
            smooth_landmarks=True,        # 跨帧平滑
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def analyze_frames(self, frames: list) -> list:
        """
        对每一帧做姿态分析
        返回: list of landmark dicts，检测失败的帧为 None
        """
        import cv2
        results = []
        for frame in frames:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            try:
                result = self.pose.process(rgb)
            except Exception:
                results.append(None)
                continue
            if result.pose_landmarks:
                results.append(self._landmarks_to_dict(result.pose_landmarks, frame.shape))
            else:
                results.append(None)
        return results

    def _landmarks_to_dict(self, landmarks, shape: tuple) -> dict:
        """将 MediaPipe landmarks 转为 {name: (x_px, y_px, visibility)} 字典"""
        h, w = shape[:2]
        points = {}
        for lm in LM:
            pt = landmarks.landmark[lm]
            points[lm.name] = (pt.x * w, pt.y * h, pt.visibility)
        return points

    def close(self):
        self.pose.close()


# ──────────────────────────────────────────
# 几何计算工具函数
# ──────────────────────────────────────────

def angle_between(a: tuple, b: tuple, c: tuple) -> float:
    """
    计算 a-b-c 三点构成的角度（b 为顶点），返回角度 [0, 180]
    每个点为 (x, y, ...) 格式
    """
    ax, ay = a[0] - b[0], a[1] - b[1]
    cx, cy = c[0] - b[0], c[1] - b[1]
    cos_val = (ax * cx + ay * cy) / (
        math.hypot(ax, ay) * math.hypot(cx, cy) + 1e-6
    )
    return math.degrees(math.acos(max(-1.0, min(1.0, cos_val))))


def point_distance(a: tuple, b: tuple) -> float:
    """两点像素距离"""
    return math.hypot(a[0] - b[0], a[1] - b[1])


def line_angle_horizontal(a: tuple, b: tuple) -> float:
    """a→b 连线与水平线的夹角（度）"""
    dx, dy = b[0] - a[0], b[1] - a[1]
    return math.degrees(math.atan2(dy, dx))


def visibility_ok(frame_dict: dict, *names: str, threshold: float = 0.5) -> bool:
    """检查指定关键点的可见度是否都达标"""
    return all(
        frame_dict.get(n, (0, 0, 0))[2] >= threshold
        for n in names
    )


def valid_frames(frames_data: list, required_points: list) -> list:
    """过滤掉检测失败或关键点不可见的帧"""
    return [
        f for f in frames_data
        if f is not None and visibility_ok(f, *required_points)
    ]
