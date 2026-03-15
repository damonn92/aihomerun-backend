"""
棒球指标计算模块
从 MediaPipe 关键点数据中提取棒球专项动作指标

挥棒关键指标：
  - 髋肩分离角：髋部先于肩膀旋转的角度差（越大越好，理想 >30°）
  - 肘部角度：后肘弯曲角（理想 90°±20°）
  - 重心平衡：双脚踝与髋部质心的稳定性
  - 随挥：击球后手腕是否越过身体中线

投球关键指标：
  - 肩膀倾斜角：出球瞬间肩线与水平线夹角
  - 步幅：跨步脚与支撑脚的距离（相对身高归一化）
  - 手肘高度：投球手肘是否高于肩膀（防止"肘下投"伤害）
"""
from __future__ import annotations
import numpy as np
from services.pose_analyzer import (
    angle_between, point_distance, line_angle_horizontal,
    visibility_ok, valid_frames
)
from models.schemas import JointAngles, MotionMetrics

# ──────────────────── 关键点名称 ────────────────────
# 左侧（拍摄者视角）
LS = "LEFT_SHOULDER"
RS = "RIGHT_SHOULDER"
LE = "LEFT_ELBOW"
RE = "RIGHT_ELBOW"
LW = "LEFT_WRIST"
RW = "RIGHT_WRIST"
LH = "LEFT_HIP"
RH = "RIGHT_HIP"
LK = "LEFT_KNEE"
RK = "RIGHT_KNEE"
LA = "LEFT_ANKLE"
RA = "RIGHT_ANKLE"
NOSE = "NOSE"


# ──────────────────── 挥棒分析 ────────────────────

def analyze_swing(frames_data: list[dict | None]) -> MotionMetrics:
    """分析挥棒动作，返回 MotionMetrics"""
    swing_points = [LS, RS, LH, RH, LE, RE, LW, RW, LK, RK, LA, RA]
    good = valid_frames(frames_data, swing_points)

    if len(good) < 3:
        raise ValueError("Too few valid pose frames. Make sure your full body is visible and the lighting is good.")

    # 1. 髋肩分离角（取各帧均值）
    hip_shoulder_seps = []
    for f in good:
        hip_angle = line_angle_horizontal(f[LH], f[RH])
        shoulder_angle = line_angle_horizontal(f[LS], f[RS])
        hip_shoulder_seps.append(abs(hip_angle - shoulder_angle))
    hip_shoulder_separation = float(np.mean(hip_shoulder_seps))

    # 2. 肘部角度（取击球瞬间，即手腕速度最大帧）
    wrist_speeds = _wrist_speeds(good)
    peak_idx = int(np.argmax(wrist_speeds))
    peak_frame = good[peak_idx]

    # 判断惯用手（右打者：右肘；左打者：左肘）
    is_right_batter = _is_right_side_dominant(peak_frame)
    elbow_angle = angle_between(
        peak_frame[RS if is_right_batter else LS],
        peak_frame[RE if is_right_batter else LE],
        peak_frame[RW if is_right_batter else LW],
    )

    # 3. 肩膀倾斜角
    shoulder_tilt = abs(line_angle_horizontal(peak_frame[LS], peak_frame[RS]))

    # 4. 膝盖弯曲角（取均值）
    knee_bends = [
        angle_between(f[LH], f[LK], f[LA]) for f in good
        if visibility_ok(f, LH, LK, LA)
    ]
    knee_bend = float(np.mean(knee_bends)) if knee_bends else 150.0

    # 5. 脊柱倾斜（鼻子与髋部中点的垂直偏差）
    spine_tilts = []
    for f in good:
        if visibility_ok(f, NOSE, LH, RH):
            hip_center_x = (f[LH][0] + f[RH][0]) / 2
            spine_tilts.append(abs(f[NOSE][0] - hip_center_x))
    # 归一化为角度近似值
    spine_tilt = float(np.mean(spine_tilts) / 10) if spine_tilts else 0.0

    # 6. 重心平衡（双脚踝稳定性方差，越小越稳）
    ankle_vars = [
        abs(f[LA][1] - f[RA][1]) for f in good
        if visibility_ok(f, LA, RA)
    ]
    balance_score = max(0.0, 1.0 - float(np.std(ankle_vars)) / 50.0) if ankle_vars else 0.5

    # 7. 手腕峰值速度（像素/帧）
    peak_wrist_speed = float(wrist_speeds[peak_idx]) if len(wrist_speeds) > peak_idx else 0.0

    # 8. 随挥检测：后挥手腕是否超过身体中线
    follow_through = _check_follow_through(good, is_right_batter)

    # 9. 球棒平面效率 & 路径一致性（从手腕轨迹估算）
    plane_eff, bat_consistency = _calculate_plane_metrics(good, wrist_speeds, peak_idx, is_right_batter)

    return MotionMetrics(
        action_type="swing",
        frames_analyzed=len(good),
        peak_wrist_speed=round(peak_wrist_speed, 2),
        hip_shoulder_separation=round(hip_shoulder_separation, 2),
        balance_score=round(balance_score, 3),
        follow_through=follow_through,
        joint_angles=JointAngles(
            elbow_angle=round(elbow_angle, 1),
            shoulder_angle=round(shoulder_tilt, 1),
            hip_rotation=round(hip_shoulder_separation, 1),
            knee_bend=round(knee_bend, 1),
            spine_tilt=round(spine_tilt, 1),
        ),
        plane_efficiency=plane_eff,
        bat_path_consistency=bat_consistency,
    )


# ──────────────────── 投球分析 ────────────────────

def analyze_pitch(frames_data: list[dict | None]) -> MotionMetrics:
    """分析投球动作，返回 MotionMetrics"""
    pitch_points = [LS, RS, LH, RH, LE, RE, LW, RW, LK, RK, LA, RA]
    good = valid_frames(frames_data, pitch_points)

    if len(good) < 3:
        raise ValueError("Too few valid pose frames. Make sure your full body is visible and the lighting is good.")

    # 投球瞬间 = 手腕速度最大帧
    wrist_speeds = _wrist_speeds(good)
    peak_idx = int(np.argmax(wrist_speeds))
    peak_frame = good[peak_idx]

    is_right_pitcher = _is_right_side_dominant(peak_frame)
    throw_shoulder = RS if is_right_pitcher else LS
    throw_elbow = RE if is_right_pitcher else LE
    throw_wrist = RW if is_right_pitcher else LW
    stride_ankle = LA if is_right_pitcher else RA
    plant_ankle = RA if is_right_pitcher else LA

    # 1. 肩膀倾斜角
    shoulder_tilt = abs(line_angle_horizontal(peak_frame[LS], peak_frame[RS]))

    # 2. 投球肘角度
    elbow_angle = angle_between(
        peak_frame[throw_shoulder],
        peak_frame[throw_elbow],
        peak_frame[throw_wrist],
    )

    # 3. 肘是否高于肩（防伤关键）
    elbow_above_shoulder = peak_frame[throw_elbow][1] < peak_frame[throw_shoulder][1]

    # 4. 步幅（归一化到身高）
    body_height = _estimate_height(peak_frame)
    stride = point_distance(peak_frame[stride_ankle], peak_frame[plant_ankle])
    normalized_stride = stride / body_height if body_height > 0 else 0.5

    # 5. 髋肩分离
    hip_angle = line_angle_horizontal(peak_frame[LH], peak_frame[RH])
    shoulder_angle = line_angle_horizontal(peak_frame[LS], peak_frame[RS])
    hip_shoulder_separation = abs(hip_angle - shoulder_angle)

    # 6. 膝盖弯曲（支撑腿）
    plant_knee = RK if is_right_pitcher else LK
    plant_hip = RH if is_right_pitcher else LH
    plant_ankle_pt = RA if is_right_pitcher else LA
    knee_bend = angle_between(
        peak_frame[plant_hip], peak_frame[plant_knee], peak_frame[plant_ankle_pt]
    )

    # 7. 重心平衡
    ankle_vars = [
        abs(f[LA][1] - f[RA][1]) for f in good
        if visibility_ok(f, LA, RA)
    ]
    balance_score = max(0.0, 1.0 - float(np.std(ankle_vars)) / 50.0) if ankle_vars else 0.5

    peak_wrist_speed = float(wrist_speeds[peak_idx]) if len(wrist_speeds) > peak_idx else 0.0

    # 随挥：投球后手腕是否过身体中线
    follow_through = _check_follow_through(good[peak_idx:], is_right_pitcher)

    return MotionMetrics(
        action_type="pitch",
        frames_analyzed=len(good),
        peak_wrist_speed=round(peak_wrist_speed, 2),
        hip_shoulder_separation=round(hip_shoulder_separation, 2),
        balance_score=round(balance_score, 3),
        follow_through=follow_through,
        joint_angles=JointAngles(
            elbow_angle=round(elbow_angle, 1),
            shoulder_angle=round(shoulder_tilt, 1),
            hip_rotation=round(hip_shoulder_separation, 1),
            knee_bend=round(knee_bend, 1),
            spine_tilt=round(normalized_stride * 10, 1),   # 用步幅替代脊柱倾斜
        ),
    )


# ──────────────────── 辅助函数 ────────────────────

def _wrist_speeds(frames: list[dict]) -> list[float]:
    """计算逐帧手腕速度（左右手腕取较大值）"""
    speeds = [0.0]
    for i in range(1, len(frames)):
        prev, curr = frames[i - 1], frames[i]
        lw_speed = point_distance(curr[LW], prev[LW])
        rw_speed = point_distance(curr[RW], prev[RW])
        speeds.append(max(lw_speed, rw_speed))
    return speeds


def _is_right_side_dominant(frame: dict) -> bool:
    """简单判断右侧是否为主力侧（右打/右投）"""
    # 右肩比左肩更靠近画面中心（侧面拍摄时更明显）
    # 这里用简单启发式：右手腕 x 坐标比左手腕更靠画面右侧
    return frame[RW][0] > frame[LW][0]


def _estimate_height(frame: dict) -> float:
    """用鼻子到踝关节中点估算身高（像素）"""
    if not (visibility_ok(frame, NOSE) and visibility_ok(frame, LA, RA)):
        return 200.0  # 默认值
    ankle_y = (frame[LA][1] + frame[RA][1]) / 2
    return abs(ankle_y - frame[NOSE][1])


def _calculate_plane_metrics(
    frames: list[dict],
    wrist_speeds: list[float],
    peak_idx: int,
    is_right_batter: bool,
) -> tuple[float | None, float | None]:
    """
    Estimate bat plane efficiency and bat path consistency from 2D wrist trajectory.

    Plane efficiency: What percentage of the wrist path during the swing zone
    stays on the ideal swing plane.  We approximate the "ideal plane" as the
    best-fit line through the wrist positions during the active swing phase,
    then measure how tightly the actual path hugs that line (R² value → 0-100%).

    Bat path consistency: How smooth / jitter-free the wrist velocity profile
    is during the swing.  Computed as 1 − normalised jerk (rate of speed change).

    Returns (plane_efficiency, bat_path_consistency) each 0-100 or None.
    """
    wrist_key = RW if is_right_batter else LW

    # Define the "swing zone" — frames around peak wrist speed (±40% of total frames, at least 3)
    half_window = max(2, len(frames) // 5)
    start = max(0, peak_idx - half_window)
    end = min(len(frames), peak_idx + half_window + 1)
    swing_frames = frames[start:end]

    if len(swing_frames) < 3:
        return None, None

    # Extract wrist (x, y) during swing zone
    xs = np.array([f[wrist_key][0] for f in swing_frames])
    ys = np.array([f[wrist_key][1] for f in swing_frames])

    # ── Plane efficiency (line-fit R²) ──
    # Fit a line to the wrist path: y = mx + b
    # R² tells us how much of the variance is explained by a straight path
    try:
        coeffs = np.polyfit(xs, ys, 1)
        y_pred = np.polyval(coeffs, xs)
        ss_res = np.sum((ys - y_pred) ** 2)
        ss_tot = np.sum((ys - np.mean(ys)) ** 2)
        if ss_tot > 0:
            r_squared = 1.0 - ss_res / ss_tot
        else:
            r_squared = 1.0  # Perfectly flat — all same y, perfect plane
        # R² can be negative for very bad fits; clamp to [0, 1]
        plane_efficiency = round(max(0.0, min(1.0, r_squared)) * 100.0, 1)
    except (np.linalg.LinAlgError, ValueError):
        plane_efficiency = None

    # ── Bat path consistency (velocity smoothness) ──
    # Calculate frame-to-frame speed changes (jerk proxy)
    swing_speeds = wrist_speeds[start:end]
    if len(swing_speeds) >= 3:
        speed_arr = np.array(swing_speeds, dtype=float)
        # Jerk = second derivative of position ≈ first derivative of speed
        speed_diffs = np.diff(speed_arr)
        # Normalise jerk by mean speed to get a dimensionless roughness metric
        mean_speed = np.mean(speed_arr) if np.mean(speed_arr) > 0 else 1.0
        normalised_jerk = np.std(speed_diffs) / mean_speed
        # Map to 0-100: jerk=0 → 100% consistent, jerk≥1.5 → ~0%
        bat_consistency = round(max(0.0, min(1.0, 1.0 - normalised_jerk / 1.5)) * 100.0, 1)
    else:
        bat_consistency = None

    return plane_efficiency, bat_consistency


def _check_follow_through(frames: list[dict], is_right: bool) -> bool:
    """检测随挥：击球后手腕是否越过身体髋部中线"""
    wrist_key = RW if is_right else LW
    hip_keys = (LH, RH)
    for f in frames[-3:]:  # 只看最后几帧
        if visibility_ok(f, wrist_key, *hip_keys):
            hip_cx = (f[LH][0] + f[RH][0]) / 2
            wrist_x = f[wrist_key][0]
            # 右打者：随挥后手腕应在髋部中线左侧
            if is_right and wrist_x < hip_cx:
                return True
            if not is_right and wrist_x > hip_cx:
                return True
    return False
