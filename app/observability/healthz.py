"""Async health probes for downstream dependencies."""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Dict

from app.core.logging import get_logger

_logger = get_logger(__name__)

_TIMEOUT = 2.0


@dataclass
class CheckResult:
    name: str
    ok: bool
    detail: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "ok": self.ok, "detail": self.detail}


async def check_llm(llm) -> CheckResult:
    """Lightweight LLM connectivity probe.

    LangChain's ChatZhipuAI doesn't expose a cheap HEAD endpoint. We treat the
    object's mere presence as ``ok`` because firing a real LLM call here would
    be wasteful; the readiness probe should still surface a 503 if Redis/PG
    are down, which is the more common failure mode.
    """
    if llm is None:
        return CheckResult("llm", False, "not initialized")
    return CheckResult("llm", True, "constructed")


async def check_chroma(db) -> CheckResult:
    if db is None:
        return CheckResult("chroma", False, "not initialized")
    try:
        # Cheapest call we can make on a Chroma vectorstore.
        await asyncio.wait_for(asyncio.to_thread(lambda: db._collection.count()), timeout=_TIMEOUT)
        return CheckResult("chroma", True, "ok")
    except Exception as e:  # noqa: BLE001
        return CheckResult("chroma", False, str(e)[:120])


async def check_redis(client) -> CheckResult:
    if client is None:
        return CheckResult("redis", False, "not configured")
    try:
        ok = await asyncio.wait_for(client.ping(), timeout=_TIMEOUT)
        return CheckResult("redis", bool(ok), "pong" if ok else "no pong")
    except Exception as e:  # noqa: BLE001
        return CheckResult("redis", False, str(e)[:120])


async def check_postgres(pool) -> CheckResult:
    if pool is None:
        return CheckResult("postgres", False, "not configured")
    try:
        async def _q():
            async with pool.acquire() as conn:
                await conn.execute("SELECT 1")

        await asyncio.wait_for(_q(), timeout=_TIMEOUT)
        return CheckResult("postgres", True, "ok")
    except Exception as e:  # noqa: BLE001
        return CheckResult("postgres", False, str(e)[:120])


async def aggregate(container) -> Dict[str, Any]:
    """Run all probes in parallel; return a summary dict."""
    results = await asyncio.gather(
        check_llm(getattr(container, "llm", None)),
        check_chroma(getattr(container, "db", None)),
        check_redis(getattr(container, "_redis_client", None)),
        check_postgres(getattr(container, "_pg_pool", None)),
        return_exceptions=False,
    )
    overall = all(r.ok for r in results)
    return {
        "ok": overall,
        "checks": [r.to_dict() for r in results],
    }
