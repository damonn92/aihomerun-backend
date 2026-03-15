"""
Upstash Redis client for caching, job queuing, and rate limiting.

Required env vars:
  UPSTASH_REDIS_URL     Redis REST URL (e.g., https://xxx.upstash.io)
  UPSTASH_REDIS_TOKEN   Redis REST token
"""
import json
import hashlib
import logging
import os
import time
from typing import Any, Optional

from upstash_redis import Redis

logger = logging.getLogger(__name__)

_redis: Redis | None = None

QUEUE_KEY = "analysis_jobs"


def get_redis() -> Redis:
    """Return a singleton Upstash Redis client."""
    global _redis
    if _redis is None:
        url = os.environ.get("UPSTASH_REDIS_URL", "").strip()
        token = os.environ.get("UPSTASH_REDIS_TOKEN", "").strip()
        if not url or not token:
            raise RuntimeError("UPSTASH_REDIS_URL and UPSTASH_REDIS_TOKEN must be set")
        _redis = Redis(url=url, token=token)
    return _redis


def is_configured() -> bool:
    url = os.environ.get("UPSTASH_REDIS_URL", "").strip()
    token = os.environ.get("UPSTASH_REDIS_TOKEN", "").strip()
    return bool(url and token)


# ── Job Queue ────────────────────────────────────────────────────────────────

def enqueue_job(job_id: str, payload: dict) -> None:
    """Push a job onto the analysis queue."""
    data = json.dumps({"job_id": job_id, **payload})
    get_redis().lpush(QUEUE_KEY, data)
    logger.info("Enqueued job %s", job_id)


def dequeue_job(timeout: int = 5) -> Optional[dict]:
    """
    Block-pop a job from the queue.
    Returns the job dict or None if timeout.
    """
    result = get_redis().brpop(QUEUE_KEY, timeout)
    if result:
        _, data = result
        return json.loads(data)
    return None


# ── Cache ────────────────────────────────────────────────────────────────────

def cache_get(key: str) -> Optional[Any]:
    """Get a cached value (JSON-deserialized)."""
    try:
        val = get_redis().get(key)
        if val is None:
            return None
        return json.loads(val) if isinstance(val, str) else val
    except Exception as exc:
        logger.warning("cache_get(%s) failed: %s", key, exc)
        return None


def cache_set(key: str, value: Any, ttl: int = 60) -> None:
    """Set a cached value with TTL in seconds."""
    try:
        get_redis().setex(key, ttl, json.dumps(value))
    except Exception as exc:
        logger.warning("cache_set(%s) failed: %s", key, exc)


def cache_delete(key: str) -> None:
    """Delete a cached key."""
    try:
        get_redis().delete(key)
    except Exception as exc:
        logger.warning("cache_delete(%s) failed: %s", key, exc)


def make_cache_key(*parts: str) -> str:
    """Build a namespaced cache key."""
    return ":".join(parts)


def hash_for_cache(data: dict) -> str:
    """Create a stable hash of a dict for cache key usage."""
    serialized = json.dumps(data, sort_keys=True, default=str)
    return hashlib.sha256(serialized.encode()).hexdigest()[:16]


# ── Rate Limiting ────────────────────────────────────────────────────────────

def check_rate_limit(user_id: str, action: str, limit: int, window_seconds: int) -> bool:
    """
    Sliding-window rate limiter.
    Returns True if the request is allowed, False if rate limited.
    """
    try:
        r = get_redis()
        key = f"rl:{action}:{user_id}"
        now = time.time()
        window_start = now - window_seconds

        pipe = r.pipeline()
        # Remove old entries outside the window
        pipe.zremrangebyscore(key, 0, window_start)
        # Count current entries
        pipe.zcard(key)
        # Add current request
        pipe.zadd(key, {str(now): now})
        # Set expiry on the key
        pipe.expire(key, window_seconds)
        results = pipe.exec()

        current_count = results[1]
        return current_count < limit
    except Exception as exc:
        logger.warning("Rate limit check failed: %s", exc)
        return True  # fail open
