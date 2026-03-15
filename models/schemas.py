from __future__ import annotations
from pydantic import BaseModel, Field
from typing import Optional, List, Literal


class JointAngles(BaseModel):
    elbow_angle: float        # 肘部角度
    shoulder_angle: float     # 肩膀倾斜角
    hip_rotation: float       # 髋部旋转角
    knee_bend: float          # 膝盖弯曲角
    spine_tilt: float         # 脊柱倾斜角


class MotionMetrics(BaseModel):
    action_type: Literal["swing", "pitch"]  # "swing" 或 "pitch"
    frames_analyzed: int                    # 分析帧数
    peak_wrist_speed: float                 # 手腕峰值速度（像素/帧）
    hip_shoulder_separation: float          # 髋肩分离角度
    balance_score: float                    # 重心平衡分（0-1）
    joint_angles: JointAngles
    follow_through: bool                    # 是否有充分随挥
    plane_efficiency: Optional[float] = None      # 球棒在挥棒平面上的时间占比 (0-100%)
    bat_path_consistency: Optional[float] = None  # 球棒路径一致性 (0-100%)


class DrillInfo(BaseModel):
    name: str               # Short drill name, e.g. "Tee Drill"
    description: str        # 2-3 sentence description
    reps: Optional[str] = None  # e.g. "20 swings", "3 × 12 reps"


class AIFeedback(BaseModel):
    overall_score: int = Field(ge=0, le=100)       # 总评分 0-100
    technique_score: int = Field(ge=0, le=100)     # 技术分
    power_score: int = Field(ge=0, le=100)         # 力量分
    balance_score: int = Field(ge=0, le=100)       # 平衡分
    strengths: List[str]                    # 优点列表（3条）
    improvements: List[str]                 # 改进建议（2条）
    drill: DrillInfo                        # 1个针对核心问题的练习（结构化）
    encouragement: str                      # 鼓励语（针对儿童）
    plain_summary: str = ""                 # 白话文总结（一句话，无术语）
    parent_tip: str = ""                    # 家长版：今日练习建议（≤10分钟）


# ── Quality Gate ──────────────────────────────────────────────────────────────

class QualityIssue(BaseModel):
    check: str               # Machine-readable check name, e.g. "low_fps"
    message: str             # Human-readable explanation
    severity: Literal["warning", "error"]


class QualityGateResult(BaseModel):
    passed: bool
    issues: List[QualityIssue]
    visibility_rate: float   # Fraction of frames where pose was detected


# ── History / Comparison ──────────────────────────────────────────────────────

class PreviousSession(BaseModel):
    """Scores from the most recent prior session — used for before/after card."""
    session_date: str
    action_type: str
    overall_score: int
    technique_score: int
    power_score: int
    balance_score: int
    video_id: Optional[str] = None
    video_url: Optional[str] = None


class HistorySummary(BaseModel):
    """Lightweight row for the growth sparkline chart."""
    session_date: str
    overall_score: int
    video_id: Optional[str] = None
    video_url: Optional[str] = None


# ── API Response ──────────────────────────────────────────────────────────────

class AnalysisResult(BaseModel):
    video_id: str
    action_type: str
    metrics: MotionMetrics
    feedback: AIFeedback
    processing_time_seconds: float
    quality: Optional[QualityGateResult] = None
    previous_session: Optional[PreviousSession] = None
    history: Optional[List[HistorySummary]] = None
    video_url: Optional[str] = None


class AnalysisError(BaseModel):
    error: str
    detail: Optional[str] = None
