"""In-process fallback for SessionStore + LongTermStore.

Used when ``MEMORY_BACKEND=inmemory`` or when Redis/Postgres are unreachable
at startup. Provides correctness (not durability) — sessions live until the
process exits or until LRU eviction kicks in.

LongTermStore recall uses cosine similarity against the cached embeddings in
pure Python. Adequate for development and single-instance demos; not intended
for production load.
"""
from __future__ import annotations

import math
import time
from collections import OrderedDict
from datetime import datetime
from typing import Dict, List, Optional, Sequence

from app.memory.stores.base import (
    Episode,
    LongTermStore,
    RecalledEpisode,
    SessionStore,
    StoredMessage,
    UserProfile,
)


class _LRU(OrderedDict):
    def __init__(self, capacity: int = 1000) -> None:
        super().__init__()
        self._cap = capacity

    def touch(self, key) -> None:
        if key in self:
            self.move_to_end(key)

    def __setitem__(self, key, value) -> None:
        super().__setitem__(key, value)
        self.move_to_end(key)
        while len(self) > self._cap:
            self.popitem(last=False)


def _key(user_id: str, session_id: str) -> str:
    return f"{user_id}::{session_id}"


class InMemorySessionStore(SessionStore):
    def __init__(self, capacity: int = 1000) -> None:
        self._messages: _LRU = _LRU(capacity)
        self._summaries: _LRU = _LRU(capacity)

    async def append_message(self, user_id, session_id, message: StoredMessage) -> None:
        k = _key(user_id, session_id)
        bucket: List[StoredMessage] = self._messages.get(k) or []
        if not message.timestamp:
            message.timestamp = time.time()
        bucket.append(message)
        self._messages[k] = bucket

    async def load_messages(self, user_id, session_id, max_count=None) -> List[StoredMessage]:
        k = _key(user_id, session_id)
        bucket = self._messages.get(k) or []
        self._messages.touch(k)
        if max_count is not None and max_count >= 0:
            return list(bucket[-max_count:])
        return list(bucket)

    async def message_count(self, user_id, session_id) -> int:
        return len(self._messages.get(_key(user_id, session_id)) or [])

    async def get_summary(self, user_id, session_id) -> Optional[str]:
        return self._summaries.get(_key(user_id, session_id))

    async def set_summary(self, user_id, session_id, summary: str) -> None:
        self._summaries[_key(user_id, session_id)] = summary

    async def forget_session(self, user_id, session_id=None) -> None:
        if session_id is None:
            for k in list(self._messages.keys()):
                if k.startswith(f"{user_id}::"):
                    self._messages.pop(k, None)
                    self._summaries.pop(k, None)
        else:
            k = _key(user_id, session_id)
            self._messages.pop(k, None)
            self._summaries.pop(k, None)

    async def ping(self) -> bool:
        return True

    async def close(self) -> None:
        return None


class InMemoryLongTermStore(LongTermStore):
    def __init__(self) -> None:
        self._episodes: Dict[str, List[Episode]] = {}
        self._profiles: Dict[str, UserProfile] = {}

    async def insert_episode(self, episode: Episode) -> None:
        if episode.created_at is None:
            episode.created_at = datetime.utcnow()
        self._episodes.setdefault(episode.user_id, []).append(episode)

    async def recall_similar(
        self,
        user_id: str,
        query_embedding: Sequence[float],
        *,
        top_k: int = 3,
        exclude_session: Optional[str] = None,
    ) -> List[RecalledEpisode]:
        episodes = self._episodes.get(user_id) or []
        if not episodes or not query_embedding:
            return []
        scored: List[tuple] = []
        for ep in episodes:
            if exclude_session and ep.session_id == exclude_session:
                continue
            if not ep.embedding:
                continue
            sim = _cosine(ep.embedding, query_embedding)
            scored.append((sim, ep))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [
            RecalledEpisode(
                content=ep.content,
                role=ep.role,
                session_id=ep.session_id,
                score=float(score),
                created_at=ep.created_at,
            )
            for score, ep in scored[:top_k]
        ]

    async def upsert_profile(self, profile: UserProfile) -> None:
        if profile.updated_at is None:
            profile.updated_at = datetime.utcnow()
        self._profiles[profile.user_id] = profile

    async def get_profile(self, user_id: str) -> Optional[UserProfile]:
        return self._profiles.get(user_id)

    async def forget_user(self, user_id: str) -> None:
        self._episodes.pop(user_id, None)
        self._profiles.pop(user_id, None)

    async def ping(self) -> bool:
        return True

    async def close(self) -> None:
        return None


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    if not a or not b:
        return 0.0
    n = min(len(a), len(b))
    dot = sum(a[i] * b[i] for i in range(n))
    na = math.sqrt(sum(a[i] * a[i] for i in range(n)))
    nb = math.sqrt(sum(b[i] * b[i] for i in range(n)))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)
