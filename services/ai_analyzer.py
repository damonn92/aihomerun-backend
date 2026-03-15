"""
AI analysis module: sends motion metrics to Claude and generates coaching feedback.

Includes caching (via Redis) and graceful degradation when Claude is unavailable.
"""
from __future__ import annotations

import json
import logging
import os
import re
import anthropic
from models.schemas import MotionMetrics, AIFeedback, DrillInfo

logger = logging.getLogger(__name__)

# Singleton client — reuses connection pool and honours max_retries
_anthropic_client: anthropic.Anthropic | None = None


def _get_client() -> anthropic.Anthropic:
    global _anthropic_client
    if _anthropic_client is None:
        key = os.environ.get("ANTHROPIC_API_KEY", "")
        if not key:
            raise RuntimeError("ANTHROPIC_API_KEY environment variable is not set")
        _anthropic_client = anthropic.Anthropic(
            api_key=key,
            max_retries=3,
            timeout=30.0,
        )
    return _anthropic_client


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
        f"- Hip-shoulder separation: {metrics.hip_shoulder_separation:.1f}\u00b0  (ideal >25\u00b0, indicates full trunk rotation)",
        f"- Balance score: {metrics.balance_score:.2f}  (0\u20131 scale, closer to 1 = more stable)",
        f"- Follow-through completed: {'Yes' if metrics.follow_through else 'No'}",
        "",
        "[Joint Angles]",
        f"- Hitting/throwing elbow angle: {a.elbow_angle:.1f}\u00b0  (ideal for swing: 90\u2013110\u00b0, pitch: 85\u2013105\u00b0)",
        f"- Shoulder tilt: {a.shoulder_angle:.1f}\u00b0  (closer to 0\u00b0 = level shoulders; ideal <15\u00b0)",
        f"- Hip rotation: {a.hip_rotation:.1f}\u00b0",
        f"- Front knee bend: {a.knee_bend:.1f}\u00b0  (ideal 130\u2013160\u00b0; too straight or too bent both reduce power)",
        f"- Spine/stride index: {a.spine_tilt:.1f}",
    ]

    # Bat plane metrics (swing only, when available)
    if metrics.plane_efficiency is not None:
        lines.append(f"- Bat plane efficiency: {metrics.plane_efficiency:.1f}%  (how much of the swing path stays on the ideal swing plane; >80% is excellent)")
    if metrics.bat_path_consistency is not None:
        lines.append(f"- Bat path consistency: {metrics.bat_path_consistency:.1f}%  (smoothness of the bat path; >75% is good, lower means jerky swing)")

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
  "drill": {{"name": "<2-4 word drill name>", "description": "<2-3 sentences, fun, easy at home>", "reps": "<e.g. '20 swings' or null>"}},
  "encouragement": "<one genuine encouraging sentence for a {age}-year-old>",
  "plain_summary": "<one casual everyday-English sentence about what you saw>",
  "parent_tip": "<one practical sentence for the parent: what to do today in under 10 minutes>"
}}\
"""


def _fallback_feedback(metrics: MotionMetrics) -> AIFeedback:
    """
    Generate a basic feedback response from metrics alone when Claude is unavailable.
    No AI commentary — just heuristic scores.
    """
    # Simple heuristic scoring based on metrics
    tech = min(100, max(0, int(50 + (metrics.joint_angles.elbow_angle - 90) * 0.5)))
    power = min(100, max(0, int(metrics.peak_wrist_speed * 3)))
    balance = min(100, max(0, int(metrics.balance_score * 100)))
    overall = (tech + power + balance) // 3

    return AIFeedback(
        overall_score=overall,
        technique_score=tech,
        power_score=power,
        balance_score=balance,
        strengths=[
            "Good effort on this swing!",
            "You completed the full motion.",
            "Keep practicing to improve!",
        ],
        improvements=[
            "Focus on your hip rotation for more power.",
            "Try to keep your balance through the follow-through.",
        ],
        drill=DrillInfo(
            name="Basic Tee Drill",
            description="Set up a tee at waist height and take 20 easy swings focusing on smooth form.",
            reps="20 swings",
        ),
        encouragement="Great job getting out there and practicing!",
        plain_summary="We captured your motion but our AI coach is taking a break — scores are estimated from your data.",
        parent_tip="Have your player take 20 easy tee swings focusing on balance and form.",
    )


def _try_cache_get(cache_key: str):
    """Try to get cached result, returns None if Redis not available."""
    try:
        from services.redis_client import cache_get, is_configured
        if is_configured():
            return cache_get(cache_key)
    except Exception:
        pass
    return None


def _try_cache_set(cache_key: str, value, ttl: int = 86400):
    """Try to cache result, silently fails if Redis not available."""
    try:
        from services.redis_client import cache_set, is_configured
        if is_configured():
            cache_set(cache_key, value, ttl)
    except Exception:
        pass


def analyze_with_claude(metrics: MotionMetrics, age: int = 10) -> AIFeedback:
    """
    Call Claude API to analyze motion metrics and generate coaching feedback.
    Falls back to heuristic scoring if Claude is unavailable.
    Results are cached in Redis for 24 hours.
    """
    action = "batting swing" if metrics.action_type == "swing" else "pitching"
    metrics_desc = build_metrics_description(metrics, age)

    # Check cache
    try:
        from services.redis_client import hash_for_cache, make_cache_key
        cache_key = make_cache_key("ai", hash_for_cache({
            "metrics": metrics_desc,
            "age": age,
        }))
    except Exception:
        cache_key = None

    if cache_key:
        cached = _try_cache_get(cache_key)
        if cached:
            logger.info("AI feedback cache hit")
            return AIFeedback(**cached)

    user_prompt = USER_PROMPT_TEMPLATE.format(
        age=age,
        action=action,
        metrics_desc=metrics_desc,
    )

    try:
        response = _get_client().messages.create(
            model="claude-haiku-4-5",
            max_tokens=800,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_prompt}],
        )
    except Exception as exc:
        logger.error("Claude API call failed: %s", exc)
        return _fallback_feedback(metrics)

    raw = response.content[0].text.strip()

    # Robust JSON extraction — try code fences first, then regex for outermost {}
    if "```" in raw:
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    else:
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            raw = match.group(0)

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        logger.error("Failed to parse Claude response: %s", exc)
        return _fallback_feedback(metrics)

    # Clamp all scores to 0-100 range
    for key in ("overall_score", "technique_score", "power_score", "balance_score"):
        data[key] = max(0, min(100, int(data.get(key, 50))))

    # Build DrillInfo — handle both object (new) and plain string (legacy)
    raw_drill = data.get("drill", {})
    if isinstance(raw_drill, dict):
        drill_obj = DrillInfo(
            name=raw_drill.get("name", "Practice Drill"),
            description=raw_drill.get("description", ""),
            reps=raw_drill.get("reps") or None,
        )
    else:
        # Legacy: Claude returned a plain string — wrap it
        drill_obj = DrillInfo(name="Practice Drill", description=str(raw_drill), reps=None)

    feedback = AIFeedback(
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

    # Cache result for 24 hours
    if cache_key:
        _try_cache_set(cache_key, feedback.model_dump(), ttl=86400)

    return feedback
