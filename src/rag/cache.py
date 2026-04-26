import hashlib
import json
from typing import Any

from src.core.config import get_settings

_settings = get_settings()


def _get_redis():  # type: ignore[return]
    try:
        import redis
        return redis.from_url(_settings.redis_url, decode_responses=True)
    except Exception:
        return None


_redis = _get_redis()


def _cache_key(meeting_id: str, query: str) -> str:
    h = hashlib.sha256(f"{meeting_id}:{query}".encode()).hexdigest()[:24]
    return f"msumm:{h}"


def get_cached(meeting_id: str, query: str) -> dict[str, Any] | None:
    if _redis is None:
        return None
    try:
        raw = _redis.get(_cache_key(meeting_id, query))
        return json.loads(raw) if raw else None
    except Exception:
        return None


def set_cached(meeting_id: str, query: str, value: dict[str, Any]) -> None:
    if _redis is None:
        return
    try:
        _redis.setex(
            _cache_key(meeting_id, query),
            _settings.cache_ttl_seconds,
            json.dumps(value),
        )
    except Exception:
        pass  # cache failures are non-fatal


def invalidate(meeting_id: str) -> None:
    """Clear all cached responses for a meeting."""
    if _redis is None:
        return
    try:
        pattern = "msumm:*"
        keys = _redis.keys(pattern)
        if keys:
            _redis.delete(*keys)
    except Exception:
        pass
