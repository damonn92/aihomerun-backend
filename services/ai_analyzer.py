"""
AI analysis module: sends motion metrics to Claude and generates scores + coaching feedback
"""
import json
import os
import anthropic
from models.schemas import MotionMetrics, AIFeedback


def _get_client():
    key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not key:
        raise RuntimeError("ANTHROPIC_API_KEY environment variable is not set")
    return anthropic.Anthropic(api_key=key)


SYSTEM_PROMPT = """You are an expert youth baseball coach with 10+ years of experience training kids aged 6-18.
Your job is to analyze motion capture data from a player's video and provide professional, encouraging feedback.

Scoring rubric:
- technique_score (0-100): joint angles, body mechanics, form correctness
- power_score (0-100): wrist speed, hip-shoulder separation, follow-through
- balance_score (0-100): center-of-gravity stability, weight transfer, foot stability

Feedback rules:
1. Use simple, age-appropriate language — avoid overly technical jargon
2. Strengths must be specific: tell the player exactly what they did well
3. Improvements must be actionable: name the body part and describe the motion clearly
4. The drill must be fun and easy to practice at home (2-3 sentences)
5. The encouragement must feel genuine and match the overall performance level
6. ALL output must be in English

Output must be strict JSON only — no extra text outside the JSON."""


def build_metrics_description(metrics: MotionMetrics, age: int) -> str:
    """Convert metrics into a natural-language description for Claude"""
    action = "batting swing" if metrics.action_type == "swing" else "pitching"
    angles = metrics.joint_angles

    lines = [
        f"Action type: {action}",
        f"Player age: {age} years old",
        f"Frames analyzed: {metrics.frames_analyzed}",
        "",
        "[Motion Data]",
        f"- Peak wrist speed: {metrics.peak_wrist_speed:.1f} px/frame (reference: >15 is good for youth players)",
        f"- Hip-shoulder separation: {metrics.hip_shoulder_separation:.1f}° (ideal >25°, indicates full trunk rotation)",
        f"- Balance score: {metrics.balance_score:.2f} (0-1 scale, closer to 1 = more stable)",
        f"- Follow-through completed: {'Yes' if metrics.follow_through else 'No'}",
        "",
        "[Joint Angles]",
        f"- Hitting/throwing elbow angle: {angles.elbow_angle:.1f}° (ideal for swing: 90-110°, pitch: 85-105°)",
        f"- Shoulder tilt: {angles.shoulder_angle:.1f}° (closer to 0° = more level, ideal <15°)",
        f"- Hip rotation: {angles.hip_rotation:.1f}°",
        f"- Front knee bend: {angles.knee_bend:.1f}° (ideal 130-160°, too straight or too bent both hurt power)",
        f"- Spine/stride index: {angles.spine_tilt:.1f}",
    ]
    return "\n".join(lines)


USER_PROMPT_TEMPLATE = """Here is the motion analysis data for a {age}-year-old player's {action}:

{metrics_desc}

Based on this data, generate a professional coaching report. Output must follow this exact JSON format:

{{
  "overall_score": <integer 0-100>,
  "technique_score": <integer 0-100>,
  "power_score": <integer 0-100>,
  "balance_score": <integer 0-100>,
  "strengths": ["strength 1", "strength 2", "strength 3"],
  "improvements": ["improvement tip 1", "improvement tip 2"],
  "drill": "A specific, fun drill to practice at home (2-3 sentences)",
  "encouragement": "One genuine encouraging sentence for a {age}-year-old player"
}}"""


def analyze_with_claude(metrics: MotionMetrics, age: int = 10) -> AIFeedback:
    """
    Call Claude API to analyze motion metrics and generate coaching feedback.

    Args:
        metrics: MotionMetrics computed from baseball_metrics module
        age: player age in years (default 10)

    Returns:
        AIFeedback object with scores, strengths, improvements, drill, and encouragement
    """
    action = "batting swing" if metrics.action_type == "swing" else "pitching"
    metrics_desc = build_metrics_description(metrics, age)

    user_prompt = USER_PROMPT_TEMPLATE.format(
        age=age,
        action=action,
        metrics_desc=metrics_desc,
    )

    response = _get_client().messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_prompt}],
    )

    raw = response.content[0].text.strip()

    # Defensive JSON extraction in case model wraps in code fences
    if "```" in raw:
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]

    data = json.loads(raw)

    return AIFeedback(
        overall_score=int(data["overall_score"]),
        technique_score=int(data["technique_score"]),
        power_score=int(data["power_score"]),
        balance_score=int(data["balance_score"]),
        strengths=data["strengths"],
        improvements=data["improvements"],
        drill=data["drill"],
        encouragement=data["encouragement"],
    )
