"""Outbox-pattern background worker.

Drains the ``mem:outbox`` Redis list, embeds each payload's ``content``, and
inserts a row into Postgres ``episodic_memory``. Failures retry up to
``MAX_RETRIES`` times; permanent failures are dropped (with WARNING) so a
poison message can't stall the queue indefinitely.

Lifecycle is owned by ``AppContainer`` — call ``start()`` on FastAPI startup
and ``stop()`` on shutdown.
"""
from __future__ import annotations

import asyncio
import json
from typing import Optional

from app.core.logging import get_logger
from app.memory.service import OUTBOX_KEY, _aembed
from app.memory.stores.base import Episode, LongTermStore

_logger = get_logger(__name__)


class OutboxWorker:
    MAX_RETRIES = 3
    BLOCK_TIMEOUT_SEC = 5  # BLPOP block window; lets graceful shutdown wake up

    def __init__(
        self,
        *,
        redis,
        long_term_store: LongTermStore,
        embedder,
        on_lag: Optional[callable] = None,
    ) -> None:
        self._r = redis
        self._lt = long_term_store
        self._embedder = embedder
        self._on_lag = on_lag
        self._task: Optional[asyncio.Task] = None
        self._stopped = asyncio.Event()

    # ------------------------------------------------------------------
    def start(self) -> None:
        if self._task is not None:
            return
        self._stopped.clear()
        self._task = asyncio.create_task(self._run(), name="memory-outbox-worker")
        _logger.info("outbox.started")

    async def stop(self, timeout: float = 5.0) -> None:
        self._stopped.set()
        if self._task is None:
            return
        self._task.cancel()
        try:
            await asyncio.wait_for(self._task, timeout=timeout)
        except (asyncio.CancelledError, asyncio.TimeoutError):
            pass
        self._task = None
        _logger.info("outbox.stopped")

    # ------------------------------------------------------------------
    async def _run(self) -> None:
        backoff = 1.0
        while not self._stopped.is_set():
            try:
                # BLPOP returns (key, value) or None on timeout
                popped = await self._r.blpop(OUTBOX_KEY, timeout=self.BLOCK_TIMEOUT_SEC)
                if popped is None:
                    await self._report_lag()
                    backoff = 1.0
                    continue
                _, raw = popped
                await self._handle(raw)
                backoff = 1.0
            except asyncio.CancelledError:
                break
            except Exception as e:  # noqa: BLE001
                _logger.warning("outbox.loop_error", error=str(e), backoff=backoff)
                try:
                    await asyncio.wait_for(self._stopped.wait(), timeout=backoff)
                except asyncio.TimeoutError:
                    pass
                backoff = min(30.0, backoff * 2)

    async def _handle(self, raw) -> None:
        if isinstance(raw, (bytes, bytearray)):
            raw = raw.decode("utf-8")
        try:
            obj = json.loads(raw)
        except Exception as e:  # noqa: BLE001
            _logger.warning("outbox.decode_failed", error=str(e))
            return

        for attempt in range(1, self.MAX_RETRIES + 1):
            try:
                embedding = await _aembed(self._embedder, obj.get("content", ""))
                ep = Episode(
                    user_id=obj["user_id"],
                    session_id=obj["session_id"],
                    role=obj.get("role", "user"),
                    content=obj.get("content", ""),
                    embedding=embedding or None,
                    metadata={
                        "trace_id": obj.get("trace_id"),
                        "ts": obj.get("ts"),
                    },
                )
                await self._lt.insert_episode(ep)
                return
            except Exception as e:  # noqa: BLE001
                _logger.warning(
                    "outbox.insert_failed",
                    attempt=attempt,
                    max=self.MAX_RETRIES,
                    error=str(e),
                )
                # exponential pause between retries
                await asyncio.sleep(min(5.0, 0.5 * 2 ** attempt))

        _logger.error("outbox.dropping_message", payload_preview=str(obj)[:200])

    async def _report_lag(self) -> None:
        if self._on_lag is None:
            return
        try:
            n = int(await self._r.llen(OUTBOX_KEY))
            self._on_lag(n)
        except Exception:  # noqa: BLE001
            pass
