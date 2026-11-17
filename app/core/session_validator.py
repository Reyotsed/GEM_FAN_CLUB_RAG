"""Session-id ↔ user-id ownership validator.

Ensures the same ``session_id`` cannot be hijacked by a different ``user_id``.
Uses Redis as the system-of-record (key ``session:owner:{session_id}``); falls
back to an in-process dict when Redis is unavailable so the main flow still
works (security guarantee then degrades to per-instance only — acceptable for
a degraded-mode deployment).
"""
from __future__ import annotations

import re
from datetime import datetime
from typing import Optional, Protocol

from app.core.logging import get_logger

_logger = get_logger(__name__)


SESSION_ID_PATTERN = re.compile(r"^[A-Za-z0-9-]{1,64}$")
ANONYMOUS_USER = "anonymous"


class SessionIdError(ValueError):
    """Raised when the supplied session_id is malformed."""


class SessionUserMismatch(PermissionError):
    """Raised when (user_id, session_id) binding conflicts with the stored owner."""


class _RedisLike(Protocol):
    async def get(self, key: str) -> Optional[bytes]: ...
    async def set(self, key: str, value: str, ex: Optional[int] = None) -> object: ...


def normalise_session_id(session_id: Optional[str], user_id: str) -> str:
    """Validate or auto-generate a session id.

    - Empty/None → ``f"{user_id}-{YYYYMMDD}"`` (URL-safe, deterministic per
      day per user; avoids accidental cross-day session bloat).
    - Otherwise must match ``^[A-Za-z0-9-]{1,64}$``.
    """
    if not session_id:
        today = datetime.utcnow().strftime("%Y%m%d")
        # Keep only legal chars from user_id; fall back to "anon" if everything
        # was stripped.
        safe_uid = re.sub(r"[^A-Za-z0-9]", "", user_id) or "anon"
        return f"{safe_uid[:48]}-{today}"

    if not SESSION_ID_PATTERN.match(session_id):
        raise SessionIdError(
            "session_id must match ^[A-Za-z0-9-]{1,64}$"
        )
    return session_id


def normalise_user_id(user_id: Optional[str]) -> str:
    """Empty user_id collapses to the well-known anonymous id."""
    if not user_id or not user_id.strip():
        return ANONYMOUS_USER
    return user_id.strip()[:128]


class SessionOwnershipValidator:
    """Binds (session_id → user_id) on first sight; rejects on conflict.

    Backed by Redis when available; otherwise uses a process-local dict.
    """

    KEY_TPL = "session:owner:{sid}"
    BIND_TTL_SECONDS = 60 * 60 * 24 * 30  # 30 days, matches session TTL default

    def __init__(self, redis: Optional[_RedisLike] = None) -> None:
        self._redis = redis
        self._fallback: dict = {}

    async def ensure(self, user_id: str, session_id: str) -> None:
        """Either bind the session to the user, or raise on mismatch.

        Anonymous users are not bound (their session_id is ephemeral and
        deterministic per day, so collisions are by design and harmless).
        """
        if user_id == ANONYMOUS_USER:
            return

        key = self.KEY_TPL.format(sid=session_id)
        existing: Optional[str] = None

        if self._redis is not None:
            try:
                raw = await self._redis.get(key)
                if raw is not None:
                    existing = raw.decode() if isinstance(raw, (bytes, bytearray)) else str(raw)
            except Exception as e:  # noqa: BLE001
                _logger.warning("session_validator.redis_get_failed", error=str(e))
                existing = self._fallback.get(session_id)
        else:
            existing = self._fallback.get(session_id)

        if existing is None:
            # First sighting — claim the binding.
            if self._redis is not None:
                try:
                    await self._redis.set(key, user_id, ex=self.BIND_TTL_SECONDS)
                except Exception as e:  # noqa: BLE001
                    _logger.warning("session_validator.redis_set_failed", error=str(e))
                    self._fallback[session_id] = user_id
            else:
                self._fallback[session_id] = user_id
            return

        if existing != user_id:
            raise SessionUserMismatch(
                f"session_id '{session_id}' is already bound to a different user"
            )
