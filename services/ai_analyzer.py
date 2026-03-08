"""
AI analysis module: sends motion metrics to Claude and generates coaching feedback.
"""
import json
import os
import anthropic
from models.schemas import MotionMetrics, AIFeedback, DrillInfo


def _get_client():
    key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not key:
        raise RuntimeError("ANTHROPIC_API_KEY environment variable is not set")
    return anthropic.Anthropic(api_key=key)


SYSTEM_PROMPT = """You are an expert youth baseball coach with 10+ years of experience training players aged 6-18.
Your job is to analyze motion capture data and produce friendly, actionable coaching feedback.

Scoring rubric:
- technique_score (0-100): joint angles, body mechanics, form correctness
- power_score (0-100): wrist speed, hip-shoulder separation, follow-through
- balance_score (0-100): center-of-gravity stability, weight transfer, foot stability

Output rules — follow every rule exactly:
1. Use simple, age-appropriate language. No technical jargon.
2. Focus on the TOP 1-2 issues only. Do not overwhelm the player with a long list.
3. "strengths": give EXACTLY 3 specific things the player did well.
4. "improvements": give EXACTLY 2 short, actionable tips. Name the body part and the motion.
5. "drill": a JSON object with three fields:
     "name"  — a 2-4 word title for the drill (e.g. "Tee Drill", "Hip Rotation"),
     "description" — 2-3 sentences, fun and doable at home. No lists, flowing prose.
     "reps"  — a short repetition target (e.g. "20 swings", "3 × 12 reps", "10 sec hold") or null.
6. "encouragement": one genuine motivating sentence sized for the player's age and score.
7. "plain_summary": write ONE casual, conversational sentence (like texting a friend) that
   explains what you noticed about the overall mechanics. Use plain everyday English.
   Good example: "Your hips weren't opening up before your shoulders turned,
   so a lot of the power you generated in your legs didn't make it to the bat."
   Bad example: "Hip-shoulder separation was suboptimal, reducing kinetic chain efficiency."
8. "parent_tip": write ONE practical sentence for the parent describing what the player
   should practice today — keep the time under 10 minutes and make it sound easy and fun.

ALL output must be in English.
Output must be strict JSON only — no markdown, no extra text outside the JSON object."""


def build_metrics_description(metrics: MotionMetrics, age: int) -> str:
    """Convert metrics into a natural-language description for Claude."""
    action = "batting swing" if metrics.action_type == "swing" else "pitching"
    a = metrics.joint_angles

    lines = [
        f"Action type: {action}",
        f"Player age: {age} years old",
        f"Frames analyzed: {metrics.frames_analyzed}",
        "",
        "[Motion Data]",
        f"- Peak wrist speed: {metrics.peak_wrist_speed:.1f} px/frame  (reference: >15 is good for youth players)",
        f"- Hip-shoulder separation: {metrics.hip_shoulder_separation:.1f}°  (ideal >25°, indicates full trunk rotation)",
        f"- Balance score: {metrics.balance_score:.2f}  (0–1 scale, closer to 1 = more stable)",
        f"- Follow-through completed: {'Yes' if metrics.follow_through else 'No'}",
        "",
        "[Joint Angles]",
        f"- Hitting/throwing elbow angle: {a.elbow_angle:.1f}°  (ideal for swing: 90–110°, pitch: 85–105°)",
        f"- Shoulder tilt: {a.shoulder_angle:.1f}°  (closer to 0° = level shoulders; ideal <15°)",
        f"- Hip rotation: {a.hip_rotation:.1f}°",
        f"- Front knee bend: {a.knee_bend:.1f}°  (ideal 130–160°; too straight or too bent both reduce power)",
        f"- Spine/stride index: {a.spine_tilt:.1f}",
    ]
    return "\n".join(lines)


USER_PROMPT_TEMPLATE = """\
Here is the motion analysis data for a {age}-year-old player's {action}:

{metrics_desc}

Generate a professional coaching report. Output must be ONLY this JSON object — no other text:

{{
  "overall_score": <integer 0-100>,
  "technique_score": <integer 0-100>,
  "power_score": <integer 0-100>,
  "balance_score": <integer 0-100>,
  "strengths": ["<specific strength 1>", "<specific strength 2>", "<specific strength 3>"],
  "improvements": ["<actionable tip 1>", "<actionable tip 2>"],
  "drill": {"name": "<2-4 word drill name>", "description": "<2-3 sentences, fun, easy at home>", "reps": "<e.g. '20 swings' or null>"},
  "encouragement": "<one genuine encouraging sentence for a {age}-year-old>",
  "plain_summary": "<one casual everyday-English sentence about what you saw>",
  "parent_tip": "<one practical sentence for the parent: what to do today in under 10 minutes>"
}}\
"""


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
        model="claude-haiku-4-5",
        max_tokens=800,
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

    # Build DrillInfo — handle both object (new) and plain string (legacy)
    raw_drill = data["drill"]
    if isinstance(raw_drill, dict):
        drill_obj = DrillInfo(
            name=raw_drill.get("name", "Practice Drill"),
            description=raw_drill.get("description", ""),
            reps=raw_drill.get("reps") or None,
        )
    else:
        # Legacy: Claude returned a plain string — wrap it
        drill_obj = DrillInfo(name="Practice Drill", description=str(raw_drill), reps=None)

    return AIFeedback(
        overall_score=int(data["overall_score"]),
        technique_score=int(data["technique_score"]),
        power_score=int(data["power_score"]),
        balance_score=int(data["balance_score"]),
        strengths=data["strengths"],
        improvements=data["improvements"],
        drill=drill_obj,
        encouragement=data["encouragement"],
        plain_summary=data.get("plain_summary", ""),
        parent_tip=data.get("parent_tip", ""),
    )
