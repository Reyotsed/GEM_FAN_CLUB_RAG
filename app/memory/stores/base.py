"""Abstract interfaces for the layered memory subsystem.

Two stores are defined:

- ``SessionStore`` — append-only chronological log per (user, session) plus
  an optional rolling summary. Hot-path read/write; backed by Redis in prod.
- ``LongTermStore`` — durable, queryable per-user store with vector recall
  for cross-session memory and a ``user_profile`` blob. Backed by Postgres
  with pgvector in prod.

Both interfaces are async and stateless from the consumer's point of view —
implementations own their connection pools and lifecycle.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence


# ----------------------------- Data shapes ---------------------------------

@dataclass
class StoredMessage:
    role: str
    content: str
    timestamp: float = 0.0
    trace_id: Optional[str] = None
    tools_called: List[str] = field(default_factory=list)
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Episode:
    """Long-term-memory record (one user/assistant message)."""
    user_id: str
    session_id: str
    role: str
    content: str
    embedding: Optional[List[float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: Optional[datetime] = None


@dataclass
class RecalledEpisode:
    content: str
    role: str
    session_id: str
    score: float
    created_at: Optional[datetime] = None


@dataclass
class UserProfile:
    user_id: str
    preferences: Dict[str, Any] = field(default_factory=dict)
    summary: str = ""
    updated_at: Optional[datetime] = None


# ----------------------------- Abstract stores ------------------------------

class SessionStore(ABC):
    @abstractmethod
    async def append_message(
        self, user_id: str, session_id: str, message: StoredMessage
    ) -> None: ...

    @abstractmethod
    async def load_messages(
        self,
        user_id: str,
        session_id: str,
        max_count: Optional[int] = None,
    ) -> List[StoredMessage]: ...

    @abstractmethod
    async def message_count(self, user_id: str, session_id: str) -> int: ...

    @abstractmethod
    async def get_summary(self, user_id: str, session_id: str) -> Optional[str]: ...

    @abstractmethod
    async def set_summary(self, user_id: str, session_id: str, summary: str) -> None: ...

    @abstractmethod
    async def forget_session(self, user_id: str, session_id: Optional[str] = None) -> None: ...

    @abstractmethod
    async def ping(self) -> bool: ...

    @abstractmethod
    async def close(self) -> None: ...


class LongTermStore(ABC):
    @abstractmethod
    async def insert_episode(self, episode: Episode) -> None: ...

    @abstractmethod
    async def recall_similar(
        self,
        user_id: str,
        query_embedding: Sequence[float],
        *,
        top_k: int = 3,
        exclude_session: Optional[str] = None,
    ) -> List[RecalledEpisode]: ...

    @abstractmethod
    async def upsert_profile(self, profile: UserProfile) -> None: ...

    @abstractmethod
    async def get_profile(self, user_id: str) -> Optional[UserProfile]: ...

    @abstractmethod
    async def forget_user(self, user_id: str) -> None: ...

    @abstractmethod
    async def ping(self) -> bool: ...

    @abstractmethod
    async def close(self) -> None: ...
