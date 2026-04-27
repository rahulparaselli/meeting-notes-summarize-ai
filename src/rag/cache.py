"""In-memory LRU cache with optional Redis backend.

Falls back to a simple dict-based cache when Redis is not available.
"""
import hashlib
import json
import logging
from collections import OrderedDict
from typing import Any

from src.core.config import get_settings

logger = logging.getLogger(__name__)
_settings = get_settings()


# ── In-memory LRU cache (always available) ───────────────────────────────────

class _LRUCache:
    """Simple in-memory LRU cache with a max size."""

    def __init__(self, max_size: int = 128):
        self._cache: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self._max_size = max_size

    def get(self, key: str) -> dict[str, Any] | None:
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        return None

    def set(self, key: str, value: dict[str, Any]) -> None:
        if key in self._cache:
            self._cache.move_to_end(key)
        self._cache[key] = value
        if len(self._cache) > self._max_size:
            self._cache.popitem(last=False)

    def invalidate(self, prefix: str = "") -> None:
        if not prefix:
            self._cache.clear()
        else:
            keys_to_remove = [k for k in self._cache if k.startswith(prefix)]
            for k in keys_to_remove:
                del self._cache[k]


_local_cache = _LRUCache(max_size=256)


# ── Redis (optional) ─────────────────────────────────────────────────────────

def _get_redis():
    try:
        import redis
        r = redis.from_url(_settings.redis_url, decode_responses=True, socket_timeout=2)
        r.ping()  # verify connection
        logger.info("Redis cache connected")
        return r
    except Exception:
        logger.info("Redis not available — using in-memory cache only")
        return None


_redis = _get_redis()


# ── Public API ────────────────────────────────────────────────────────────────

def _cache_key(meeting_id: str, query: str) -> str:
    h = hashlib.sha256(f"{meeting_id}:{query}".encode()).hexdigest()[:24]
    return f"msumm:{meeting_id[:8]}:{h}"


def get_cached(meeting_id: str, query: str) -> dict[str, Any] | None:
    key = _cache_key(meeting_id, query)

    # Try local first
    result = _local_cache.get(key)
    if result is not None:
        logger.debug(f"Cache HIT (local): {key}")
        return result

    # Try Redis
    if _redis is not None:
        try:
            raw = _redis.get(key)
            if raw:
                result = json.loads(raw)
                _local_cache.set(key, result)  # warm local cache
                logger.debug(f"Cache HIT (redis): {key}")
                return result
        except Exception:
            pass

    return None


def set_cached(meeting_id: str, query: str, value: dict[str, Any]) -> None:
    key = _cache_key(meeting_id, query)

    # Always set local
    _local_cache.set(key, value)

    # Set Redis if available
    if _redis is not None:
        try:
            _redis.setex(
                key,
                _settings.cache_ttl_seconds,
                json.dumps(value, default=str),
            )
        except Exception:
            pass  # non-fatal


def invalidate(meeting_id: str) -> None:
    """Clear all cached responses for a meeting."""
    prefix = f"msumm:{meeting_id[:8]}:"
    _local_cache.invalidate(prefix)

    if _redis is not None:
        try:
            keys = _redis.keys(f"{prefix}*")
            if keys:
                _redis.delete(*keys)
        except Exception:
            pass
