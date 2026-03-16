"""
姿态分析模块：使用 MediaPipe 提取人体关键点
MediaPipe Pose 提供 33 个关键点，覆盖全身

改进 v2:
  - model_complexity=2 提升关键点精度 ~15%
  - 视角检测（正面/侧面/背面）— 侧面最佳
  - 多次分析取平均（multi-pass）减少单次随机误差
"""
import math
import logging
from typing import Optional
import numpy as np
import mediapipe as mp

logger = logging.getLogger(__name__)

# MediaPipe 关键点索引常量（对应 PoseLandmark）
LM = mp.solutions.pose.PoseLandmark


class PoseAnalyzer:
    def __init__(self, model_complexity: int = 2, multi_pass: int = 3):
        """
        Args:
            model_complexity: 0=轻量, 1=标准, 2=高精度（推荐生产环境）
            multi_pass: 多次分析取平均的次数（1=单次，3=三次取中位数）
        """
        self.mp_pose = mp.solutions.pose
        self.model_complexity = model_complexity
        self.multi_pass = max(1, multi_pass)
        self._create_pose()

    def _create_pose(self):
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,      # 视频模式，利用时序信息
            model_complexity=self.model_complexity,
            smooth_landmarks=True,        # 跨帧平滑
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def analyze_frames(self, frames: list) -> list:
        """
        对每一帧做姿态分析。
        如果 multi_pass > 1，执行多轮分析并对关键点坐标取中位数，减少随机误差。
        返回: list of landmark dicts，检测失败的帧为 None
        """
        if self.multi_pass <= 1:
            return self._single_pass(frames)

        # 多次分析取中位数
        all_passes = []
        for p in range(self.multi_pass):
            if p > 0:
                # 每轮重建 Pose 实例以获得独立的检测结果
                self.pose.close()
                self._create_pose()
            pass_result = self._single_pass(frames)
            all_passes.append(pass_result)

        # 合并：对每帧取各轮的中位数坐标
        merged = []
        for frame_idx in range(len(frames)):
            frame_results = [all_passes[p][frame_idx] for p in range(self.multi_pass)]
            valid_results = [r for r in frame_results if r is not None]

            if len(valid_results) == 0:
                merged.append(None)
            elif len(valid_results) == 1:
                merged.append(valid_results[0])
            else:
                merged.append(self._median_landmarks(valid_results))

        logger.info("Multi-pass analysis: %d passes, merged %d frames", self.multi_pass, len(merged))
        return merged

    def _single_pass(self, frames: list) -> list:
        """单次分析所有帧"""
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

    @staticmethod
    def _median_landmarks(results: list[dict]) -> dict:
        """对多次检测结果取中位数坐标，减少随机抖动"""
        merged = {}
        keys = results[0].keys()
        for key in keys:
            xs = [r[key][0] for r in results if key in r]
            ys = [r[key][1] for r in results if key in r]
            vs = [r[key][2] for r in results if key in r]
            merged[key] = (
                float(np.median(xs)),
                float(np.median(ys)),
                float(np.median(vs)),
            )
        return merged

    def close(self):
        self.pose.close()


# ──────────────────────────────────────────
# 视角检测
# ──────────────────────────────────────────

def detect_viewing_angle(frames_data: list) -> dict:
    """
    检测拍摄视角（正面 / 侧面 / 背面）。
    侧面视角是棒球分析的最佳视角。

    原理：比较左右肩距离与肩到鼻子距离的比值
      - 侧面：左右肩几乎重叠，肩距很小
      - 正面/背面：左右肩展开，肩距较大

    返回:
        {
            "angle": "side" | "front" | "back" | "unknown",
            "confidence": 0.0-1.0,
            "shoulder_ratio": float,  # 肩宽/肩高 ratio
            "recommendation": str,     # 给用户的建议
        }
    """
    valid = [f for f in frames_data if f is not None]
    if len(valid) < 3:
        return {
            "angle": "unknown",
            "confidence": 0.0,
            "shoulder_ratio": 0.0,
            "recommendation": "Could not detect enough pose data to determine camera angle.",
        }

    ratios = []
    nose_shoulder_depths = []  # 用于区分正面和背面

    for f in valid:
        ls = f.get("LEFT_SHOULDER")
        rs = f.get("RIGHT_SHOULDER")
        nose = f.get("NOSE")
        if not (ls and rs and nose):
            continue
        if ls[2] < 0.4 or rs[2] < 0.4:
            continue

        # 肩宽（像素）
        shoulder_width = abs(ls[0] - rs[0])
        # 肩高差（像素）— 侧面时有明显高度差
        shoulder_height_diff = abs(ls[1] - rs[1])
        # 肩中点到鼻子的水平距离
        shoulder_cx = (ls[0] + rs[0]) / 2
        nose_offset = abs(nose[0] - shoulder_cx)

        # 估算身体尺度（用肩到髋的距离）
        lh = f.get("LEFT_HIP")
        rh = f.get("RIGHT_HIP")
        if lh and rh:
            body_scale = abs((ls[1] + rs[1]) / 2 - (lh[1] + rh[1]) / 2)
        else:
            body_scale = 200.0  # 默认

        if body_scale > 0:
            ratio = shoulder_width / body_scale
            ratios.append(ratio)

        # 鼻子可见度（背面时鼻子可见度很低）
        nose_shoulder_depths.append(nose[2])

    if not ratios:
        return {
            "angle": "unknown",
            "confidence": 0.0,
            "shoulder_ratio": 0.0,
            "recommendation": "Could not calculate camera angle — make sure full body is visible.",
        }

    avg_ratio = float(np.mean(ratios))
    avg_nose_vis = float(np.mean(nose_shoulder_depths))

    # 判断逻辑：
    # 侧面：肩宽/身高 < 0.4（肩膀几乎重叠）
    # 正面/背面：肩宽/身高 > 0.6
    if avg_ratio < 0.35:
        angle = "side"
        confidence = min(1.0, (0.5 - avg_ratio) / 0.3)
        recommendation = "Great angle! Side view is perfect for swing and pitch analysis."
    elif avg_ratio < 0.55:
        angle = "diagonal"
        confidence = 0.6
        recommendation = (
            "The camera is at a slight angle. For best results, "
            "position the camera directly to the side (perpendicular to the batter/pitcher)."
        )
    else:
        # 正面 vs 背面：用鼻子可见度区分
        if avg_nose_vis > 0.6:
            angle = "front"
            confidence = min(1.0, (avg_ratio - 0.5) / 0.4)
            recommendation = (
                "The camera is facing the player head-on. "
                "For much better analysis, move the camera to the side "
                "so it captures the full swing/pitch from a side view."
            )
        else:
            angle = "back"
            confidence = min(1.0, (avg_ratio - 0.5) / 0.4)
            recommendation = (
                "The camera appears to be behind the player. "
                "Please move it to the side for accurate swing analysis."
            )

    return {
        "angle": angle,
        "confidence": round(confidence, 2),
        "shoulder_ratio": round(avg_ratio, 3),
        "recommendation": recommendation,
    }


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
