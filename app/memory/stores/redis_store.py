"""Redis-backed SessionStore.

Uses ``redis.asyncio`` (>=5.0). One Redis List per session for messages and
one String per session for the rolling summary; both share the same TTL so
old sessions garbage-collect themselves.

Schema
------
- ``session:{user_id}:{session_id}:messages`` — RPUSH JSON blobs, LRANGE on
  read. TTL refreshed on every write.
- ``session:{user_id}:{session_id}:summary`` — plain UTF-8 string. Same TTL.
"""
from __future__ import annotations

import json
import time
from typing import List, Optional

from app.core.logging import get_logger
from app.memory.stores.base import SessionStore, StoredMessage

_logger = get_logger(__name__)

_KEY_MSG = "session:{u}:{s}:messages"
_KEY_SUM = "session:{u}:{s}:summary"


class RedisSessionStore(SessionStore):
    """Async Redis implementation of SessionStore."""

    def __init__(self, redis, ttl_seconds: int) -> None:
        self._r = redis
        self._ttl = ttl_seconds

    # ------------------------------------------------------------------
    async def append_message(self, user_id, session_id, message: StoredMessage) -> None:
        if not message.timestamp:
            message.timestamp = time.time()
        payload = json.dumps({
            "role": message.role,
            "content": message.content,
            "timestamp": message.timestamp,
            "trace_id": message.trace_id,
            "tools_called": message.tools_called or [],
            "extra": message.extra or {},
        }, ensure_ascii=False)
        key = _KEY_MSG.format(u=user_id, s=session_id)
        async with self._r.pipeline(transaction=False) as pipe:
            pipe.rpush(key, payload)
            pipe.expire(key, self._ttl)
            await pipe.execute()

    async def load_messages(self, user_id, session_id, max_count=None) -> List[StoredMessage]:
        key = _KEY_MSG.format(u=user_id, s=session_id)
        if max_count is not None and max_count > 0:
            raw = await self._r.lrange(key, -max_count, -1)
        else:
            raw = await self._r.lrange(key, 0, -1)
        return [_decode(b) for b in raw if b]

    async def message_count(self, user_id, session_id) -> int:
        return int(await self._r.llen(_KEY_MSG.format(u=user_id, s=session_id)))

    async def get_summary(self, user_id, session_id) -> Optional[str]:
        v = await self._r.get(_KEY_SUM.format(u=user_id, s=session_id))
        if v is None:
            return None
        return v.decode() if isinstance(v, (bytes, bytearray)) else str(v)

    async def set_summary(self, user_id, session_id, summary: str) -> None:
        await self._r.set(
            _KEY_SUM.format(u=user_id, s=session_id),
            summary,
            ex=self._ttl,
        )

    async def forget_session(self, user_id, session_id=None) -> None:
        if session_id is None:
            # Best-effort scan-and-delete; OK for moderate-cardinality keyspaces.
            pattern = f"session:{user_id}:*"
            cursor = 0
            while True:
                cursor, keys = await self._r.scan(cursor=cursor, match=pattern, count=200)
                if keys:
                    await self._r.delete(*keys)
                if cursor == 0:
                    break
        else:
            await self._r.delete(
                _KEY_MSG.format(u=user_id, s=session_id),
                _KEY_SUM.format(u=user_id, s=session_id),
            )

    async def ping(self) -> bool:
        try:
            return bool(await self._r.ping())
        except Exception as e:  # noqa: BLE001
            _logger.warning("redis_store.ping_failed", error=str(e))
            return False

    async def close(self) -> None:
        try:
            await self._r.aclose()  # redis>=5.0
        except AttributeError:
            try:
                await self._r.close()
            except Exception:  # noqa: BLE001
                pass


def _decode(raw) -> StoredMessage:
    if isinstance(raw, (bytes, bytearray)):
        raw = raw.decode("utf-8")
    obj = json.loads(raw)
    return StoredMessage(
        role=obj.get("role", "user"),
        content=obj.get("content", ""),
        timestamp=float(obj.get("timestamp") or 0.0),
        trace_id=obj.get("trace_id"),
        tools_called=obj.get("tools_called") or [],
        extra=obj.get("extra") or {},
    )
