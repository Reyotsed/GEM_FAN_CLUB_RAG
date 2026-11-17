"""High-level memory orchestration.

``MemoryService`` is the only thing the rest of the application talks to.
It hides the details of:
  - which session backend is in use (Redis vs in-memory),
  - whether long-term memory is reachable at all,
  - the outbox handoff for asynchronous Postgres writes,
  - failure isolation: any internal error becomes an empty result + WARNING,
    so a memory outage never breaks the chat hot path.
"""
from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Dict, List, Optional, Sequence

from app.context.manager import HistoryTurn
from app.core.logging import get_logger
from app.core.session_validator import ANONYMOUS_USER
from app.memory.stores.base import (
    Episode,
    LongTermStore,
    RecalledEpisode,
    SessionStore,
    StoredMessage,
    UserProfile,
)

_logger = get_logger(__name__)

OUTBOX_KEY = "mem:outbox"

# Embedder type: callable that returns a list[float] for a single text; may
# be sync or async. We normalize via ``_aembed``.
EmbedderFn = Callable[[str], "Awaitable[List[float]] | List[float]"]


@dataclass
class SessionLoadResult:
    history: List[HistoryTurn]
    summary: Optional[str]
    raw_messages: List[StoredMessage]


class MemoryService:
    """Single entry point for the agent / API to read & write memory."""

    def __init__(
        self,
        *,
        session_store: SessionStore,
        long_term_store: Optional[LongTermStore],
        embedder: Optional[EmbedderFn] = None,
        outbox_redis=None,  # async redis used for the outbox queue (optional)
        max_session_messages: int = 200,
    ) -> None:
        self._sessions = session_store
        self._long_term = long_term_store
        self._embedder = embedder
        self._outbox = outbox_redis
        self._max_session_messages = max_session_messages

    # ==================================================================
    # Reading
    # ==================================================================
    async def load_session(self, user_id: str, session_id: str) -> SessionLoadResult:
        """Load most-recent messages + rolling summary for a session.

        Returns an empty result on any internal failure; never raises.
        """
        try:
            raw = await self._sessions.load_messages(
                user_id, session_id, max_count=self._max_session_messages
            )
            summary = await self._sessions.get_summary(user_id, session_id)
            return SessionLoadResult(
                history=_to_turns(raw),
                summary=summary,
                raw_messages=raw,
            )
        except Exception as e:  # noqa: BLE001
            _logger.warning("memory.load_session_failed", error=str(e))
            return SessionLoadResult(history=[], summary=None, raw_messages=[])

    async def recall_long_term(
        self,
        user_id: str,
        current_session_id: str,
        query: str,
        top_k: int = 3,
    ) -> List[RecalledEpisode]:
        """Cross-session vector recall. Empty for anonymous users."""
        if user_id == ANONYMOUS_USER:
            return []
        if self._long_term is None or self._embedder is None:
            return []
        try:
            embedding = await _aembed(self._embedder, query)
            if not embedding:
                return []
            return await self._long_term.recall_similar(
                user_id,
                embedding,
                top_k=top_k,
                exclude_session=current_session_id,
            )
        except Exception as e:  # noqa: BLE001
            _logger.warning("memory.recall_long_term_failed", error=str(e))
            return []

    async def get_profile(self, user_id: str) -> Optional[UserProfile]:
        if user_id == ANONYMOUS_USER or self._long_term is None:
            return None
        try:
            return await self._long_term.get_profile(user_id)
        except Exception as e:  # noqa: BLE001
            _logger.warning("memory.get_profile_failed", error=str(e))
            return None

    # ==================================================================
    # Writing
    # ==================================================================
    async def save_turn(
        self,
        *,
        user_id: str,
        session_id: str,
        user_message: str,
        assistant_message: str,
        trace_id: Optional[str] = None,
        tools_called: Optional[List[str]] = None,
    ) -> None:
        """Persist a finished round of chat.

        - Always synchronously writes to the SessionStore (fast path).
        - For non-anonymous users, enqueues both messages onto the Redis
          outbox so an OutboxWorker can persist them to Postgres later.
        - Never raises.
        """
        ts = time.time()
        try:
            if user_message:
                await self._sessions.append_message(
                    user_id,
                    session_id,
                    StoredMessage(
                        role="user",
                        content=user_message,
                        timestamp=ts,
                        trace_id=trace_id,
                    ),
                )
            if assistant_message:
                await self._sessions.append_message(
                    user_id,
                    session_id,
                    StoredMessage(
                        role="assistant",
                        content=assistant_message,
                        timestamp=ts + 0.001,  # preserve ordering
                        trace_id=trace_id,
                        tools_called=tools_called or [],
                    ),
                )
        except Exception as e:  # noqa: BLE001
            _logger.warning("memory.save_session_failed", error=str(e))

        if user_id == ANONYMOUS_USER or self._outbox is None:
            return

        for role, content in (("user", user_message), ("assistant", assistant_message)):
            if not content:
                continue
            try:
                payload = json.dumps({
                    "user_id": user_id,
                    "session_id": session_id,
                    "role": role,
                    "content": content,
                    "trace_id": trace_id,
                    "ts": ts,
                }, ensure_ascii=False)
                await self._outbox.rpush(OUTBOX_KEY, payload)
            except Exception as e:  # noqa: BLE001
                _logger.warning("memory.outbox_enqueue_failed", error=str(e))

    async def forget(self, user_id: str) -> None:
        """Erase a user's memory across all stores. Idempotent."""
        try:
            await self._sessions.forget_session(user_id, None)
        except Exception as e:  # noqa: BLE001
            _logger.warning("memory.forget_session_failed", error=str(e))
        if self._long_term is not None and user_id != ANONYMOUS_USER:
            try:
                await self._long_term.forget_user(user_id)
            except Exception as e:  # noqa: BLE001
                _logger.warning("memory.forget_long_term_failed", error=str(e))

    # ==================================================================
    # Misc helpers
    # ==================================================================
    async def message_count(self, user_id: str, session_id: str) -> int:
        try:
            return await self._sessions.message_count(user_id, session_id)
        except Exception:  # noqa: BLE001
            return 0

    async def set_summary(self, user_id: str, session_id: str, summary: str) -> None:
        try:
            await self._sessions.set_summary(user_id, session_id, summary)
        except Exception as e:  # noqa: BLE001
            _logger.warning("memory.set_summary_failed", error=str(e))

    @property
    def long_term_available(self) -> bool:
        return self._long_term is not None

    @property
    def session_store(self) -> SessionStore:
        return self._sessions

    @property
    def long_term_store(self) -> Optional[LongTermStore]:
        return self._long_term

    @property
    def embedder(self) -> Optional[EmbedderFn]:
        return self._embedder


# ===========================================================================
# Helpers
# ===========================================================================

def _to_turns(messages: Sequence[StoredMessage]) -> List[HistoryTurn]:
    """Pair consecutive (user, assistant) messages into turns.

    Stray messages (e.g. two user messages in a row) are flushed as separate
    turns with the missing role left empty so the ContextManager preserves
    them anyway.
    """
    turns: List[HistoryTurn] = []
    pending: Optional[HistoryTurn] = None
    for m in messages:
        if m.role == "user":
            if pending is not None:
                turns.append(pending)
            pending = HistoryTurn(user=m.content)
        elif m.role == "assistant":
            if pending is None:
                turns.append(HistoryTurn(assistant=m.content))
            else:
                pending.assistant = m.content
                turns.append(pending)
                pending = None
    if pending is not None:
        turns.append(pending)
    return turns


async def _aembed(embedder: EmbedderFn, text: str) -> List[float]:
    out = embedder(text)
    if asyncio.iscoroutine(out):
        out = await out
    return list(out or [])
