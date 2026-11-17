"""Postgres + pgvector implementation of LongTermStore.

Schema (created by ``db/init.sql``):

    CREATE EXTENSION IF NOT EXISTS vector;

    CREATE TABLE episodic_memory (
        id            BIGSERIAL PRIMARY KEY,
        user_id       TEXT NOT NULL,
        session_id    TEXT NOT NULL,
        role          TEXT NOT NULL,
        content       TEXT NOT NULL,
        embedding     vector(1024),
        metadata      JSONB DEFAULT '{}'::jsonb,
        created_at    TIMESTAMPTZ NOT NULL DEFAULT NOW()
    );
    CREATE INDEX idx_episodic_user_created  ON episodic_memory(user_id, created_at DESC);
    CREATE INDEX idx_episodic_embedding ON episodic_memory USING ivfflat (embedding vector_cosine_ops);

    CREATE TABLE user_profile (
        user_id      TEXT PRIMARY KEY,
        preferences  JSONB DEFAULT '{}'::jsonb,
        summary      TEXT DEFAULT '',
        updated_at   TIMESTAMPTZ NOT NULL DEFAULT NOW()
    );

This module assumes asyncpg's ``pool`` is injected from the container.
"""
from __future__ import annotations

import json
from datetime import datetime
from typing import List, Optional, Sequence

from app.core.logging import get_logger
from app.memory.stores.base import (
    Episode,
    LongTermStore,
    RecalledEpisode,
    UserProfile,
)

_logger = get_logger(__name__)


def _to_pgvector(values: Sequence[float]) -> str:
    """Format a Python sequence as the pgvector text literal ``[v1,v2,...]``."""
    return "[" + ",".join(f"{float(v):.6f}" for v in values) + "]"


class PgLongTermStore(LongTermStore):
    def __init__(self, pool, embedding_dim: int = 1024) -> None:
        self._pool = pool
        self._dim = embedding_dim

    # ------------------------------------------------------------------
    async def insert_episode(self, episode: Episode) -> None:
        meta = json.dumps(episode.metadata or {}, ensure_ascii=False)
        # Branch on embedding presence to avoid feeding NULL through ``::vector``
        # which some pgvector versions reject when the cast is applied as a
        # parameter type rather than a column-default.
        if episode.embedding:
            sql = """
            INSERT INTO episodic_memory
                (user_id, session_id, role, content, embedding, metadata, created_at)
            VALUES ($1, $2, $3, $4, $5::vector, $6::jsonb, COALESCE($7, NOW()))
            """
            args = (
                episode.user_id, episode.session_id, episode.role, episode.content,
                _to_pgvector(episode.embedding), meta, episode.created_at,
            )
        else:
            sql = """
            INSERT INTO episodic_memory
                (user_id, session_id, role, content, metadata, created_at)
            VALUES ($1, $2, $3, $4, $5::jsonb, COALESCE($6, NOW()))
            """
            args = (
                episode.user_id, episode.session_id, episode.role, episode.content,
                meta, episode.created_at,
            )
        async with self._pool.acquire() as conn:
            await conn.execute(sql, *args)

    async def recall_similar(
        self,
        user_id: str,
        query_embedding: Sequence[float],
        *,
        top_k: int = 3,
        exclude_session: Optional[str] = None,
    ) -> List[RecalledEpisode]:
        if not query_embedding:
            return []
        embedding_lit = _to_pgvector(query_embedding)

        params: list = [user_id, embedding_lit]
        where = "user_id = $1 AND embedding IS NOT NULL"
        if exclude_session:
            where += " AND session_id <> $3"
            params.append(exclude_session)
        params.append(top_k)
        limit_param = f"${len(params)}"

        sql = f"""
        SELECT content, role, session_id, created_at,
               1 - (embedding <=> $2::vector) AS score
        FROM episodic_memory
        WHERE {where}
        ORDER BY embedding <=> $2::vector
        LIMIT {limit_param}
        """
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(sql, *params)
        return [
            RecalledEpisode(
                content=r["content"],
                role=r["role"],
                session_id=r["session_id"],
                score=float(r["score"]),
                created_at=r["created_at"],
            )
            for r in rows
        ]

    async def upsert_profile(self, profile: UserProfile) -> None:
        prefs = json.dumps(profile.preferences or {}, ensure_ascii=False)
        sql = """
        INSERT INTO user_profile (user_id, preferences, summary, updated_at)
        VALUES ($1, $2::jsonb, $3, COALESCE($4, NOW()))
        ON CONFLICT (user_id) DO UPDATE
        SET preferences = EXCLUDED.preferences,
            summary     = EXCLUDED.summary,
            updated_at  = EXCLUDED.updated_at
        """
        async with self._pool.acquire() as conn:
            await conn.execute(sql, profile.user_id, prefs, profile.summary or "", profile.updated_at)

    async def get_profile(self, user_id: str) -> Optional[UserProfile]:
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT user_id, preferences, summary, updated_at FROM user_profile WHERE user_id = $1",
                user_id,
            )
        if not row:
            return None
        prefs = row["preferences"]
        if isinstance(prefs, str):
            try:
                prefs = json.loads(prefs)
            except Exception:  # noqa: BLE001
                prefs = {}
        return UserProfile(
            user_id=row["user_id"],
            preferences=prefs or {},
            summary=row["summary"] or "",
            updated_at=row["updated_at"],
        )

    async def forget_user(self, user_id: str) -> None:
        async with self._pool.acquire() as conn:
            async with conn.transaction():
                await conn.execute(
                    "DELETE FROM episodic_memory WHERE user_id = $1", user_id
                )
                await conn.execute(
                    "DELETE FROM user_profile WHERE user_id = $1", user_id
                )

    async def ping(self) -> bool:
        try:
            async with self._pool.acquire() as conn:
                await conn.execute("SELECT 1")
            return True
        except Exception as e:  # noqa: BLE001
            _logger.warning("pg_store.ping_failed", error=str(e))
            return False

    async def close(self) -> None:
        try:
            await self._pool.close()
        except Exception:  # noqa: BLE001
            pass
