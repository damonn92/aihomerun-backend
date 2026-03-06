from __future__ import annotations
from pydantic import BaseModel
from typing import Optional, List


class JointAngles(BaseModel):
    elbow_angle: float        # 肘部角度
    shoulder_angle: float     # 肩膀倾斜角
    hip_rotation: float       # 髋部旋转角
    knee_bend: float          # 膝盖弯曲角
    spine_tilt: float         # 脊柱倾斜角


class MotionMetrics(BaseModel):
    action_type: str                        # "swing" 或 "pitch"
    frames_analyzed: int                    # 分析帧数
    peak_wrist_speed: float                 # 手腕峰值速度（像素/帧）
    hip_shoulder_separation: float          # 髋肩分离角度
    balance_score: float                    # 重心平衡分（0-1）
    joint_angles: JointAngles
    follow_through: bool                    # 是否有充分随挥


class AIFeedback(BaseModel):
    overall_score: int                      # 总评分 0-100
    technique_score: int                    # 技术分
    power_score: int                        # 力量分
    balance_score: int                      # 平衡分
    strengths: List[str]                    # 优点列表
    improvements: List[str]                 # 改进建议
    drill: str                              # 一个具体练习建议
    encouragement: str                      # 鼓励语（针对儿童）


class AnalysisResult(BaseModel):
    video_id: str
    action_type: str
    metrics: MotionMetrics
    feedback: AIFeedback
    processing_time_seconds: float


class AnalysisError(BaseModel):
    error: str
    detail: Optional[str] = None
