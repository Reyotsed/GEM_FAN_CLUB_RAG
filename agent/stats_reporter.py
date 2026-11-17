"""
Agent 调用指标推送：fire-and-forget 上报到 Java 后端 /stats/track
"""
from __future__ import annotations

import asyncio
import logging
import os
import threading
import time
from typing import Optional

import httpx

logger = logging.getLogger(__name__)

_BACKEND = os.getenv("JAVA_BACKEND_URL", "http://127.0.0.1:7071")
_API_PREFIX = os.getenv("JAVA_API_PREFIX", "")
_TRACK_URL = f"{_BACKEND}{_API_PREFIX}/stats/track"
_TIMEOUT = 2.0

# 异步 client（绑定到首个使用的 loop），同步 client 用于无 loop 兜底
_async_client: Optional[httpx.AsyncClient] = None
_async_client_loop: Optional[asyncio.AbstractEventLoop] = None
_sync_client: Optional[httpx.Client] = None
_lock = threading.Lock()


def _get_sync_client() -> httpx.Client:
    global _sync_client
    if _sync_client is None:
        with _lock:
            if _sync_client is None:
                _sync_client = httpx.Client(timeout=_TIMEOUT)
    return _sync_client


async def _async_post(payload: dict) -> None:
    global _async_client, _async_client_loop
    try:
        loop = asyncio.get_running_loop()
        # AsyncClient 必须绑定到同一个 loop，loop 切换时重建
        if _async_client is None or _async_client_loop is not loop:
            _async_client = httpx.AsyncClient(timeout=_TIMEOUT)
            _async_client_loop = loop
        await _async_client.post(_TRACK_URL, json=payload)
    except Exception as e:
        logger.debug("Stats push failed (ignored): %s", e)


def _sync_post_in_thread(payload: dict) -> None:
    """没有运行中的 loop 时（极少见），用线程发同步请求兜底。"""
    def _run():
        try:
            _get_sync_client().post(_TRACK_URL, json=payload)
        except Exception as e:
            logger.debug("Stats sync push failed (ignored): %s", e)
    threading.Thread(target=_run, daemon=True).start()


def report_agent_invoke(
    user_id: Optional[str],
    duration_ms: float,
    tokens_total: int = 0,
    cost_usd: float = 0.0,
) -> None:
    """fire-and-forget 上报 Agent 调用指标，绝不抛异常。"""
    payload = {
        "type": "event",
        "key": "agent:invoke",
        "userId": user_id or "",
        "extra": {
            "duration_ms": int(duration_ms),
            "tokens_total": int(tokens_total),
            "cost_usd": float(cost_usd),
            "ts": int(time.time()),
        },
    }
    try:
        loop = asyncio.get_running_loop()
        # 在 FastAPI handler 内调用 → 直接 schedule 到当前 loop
        loop.create_task(_async_post(payload))
    except RuntimeError:
        # 没有正在运行的 loop（同步上下文）→ 线程内同步发送
        _sync_post_in_thread(payload)
    except Exception as e:
        logger.debug("Stats push schedule failed: %s", e)
