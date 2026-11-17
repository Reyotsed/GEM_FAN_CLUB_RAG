"""
Agent Trace — lightweight request-level tracing for the GemAgent pipeline.

Records timing and metadata for each stage of the agent loop, then prints
a structured summary to the logger.  No persistence — purely log output.

Usage::

    trace = AgentTrace()
    with trace.stage("rewrite"):
        rewritten = ...
    with trace.stage("planning"):
        planned = ...
    trace.print_summary()
"""

from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class StageRecord:
    """Timing record for a single pipeline stage."""

    name: str
    duration_ms: float = 0.0
    success: bool = True
    detail: str = ""


class AgentTrace:
    """
    Accumulates stage-level timing data for one agent request.

    Designed to be zero-overhead when not used — simply don't call any methods.
    """

    def __init__(self, question: str = "") -> None:
        self._start = time.perf_counter()
        self._stages: List[StageRecord] = []
        self._question = question
        self._extra: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Recording helpers
    # ------------------------------------------------------------------

    @contextmanager
    def stage(self, name: str, detail: str = ""):
        """
        Context manager that times a named stage.

        Usage::

            with trace.stage("rewrite"):
                result = do_something()
        """
        record = StageRecord(name=name, detail=detail)
        t0 = time.perf_counter()
        try:
            yield record
        except Exception:
            record.success = False
            raise
        finally:
            record.duration_ms = (time.perf_counter() - t0) * 1000
            self._stages.append(record)

    def record(
        self,
        name: str,
        duration_ms: float,
        success: bool = True,
        detail: str = "",
    ) -> None:
        """Manually record a stage (for cases where context manager is awkward)."""
        self._stages.append(
            StageRecord(
                name=name,
                duration_ms=duration_ms,
                success=success,
                detail=detail,
            )
        )

    def set_extra(self, key: str, value: Any) -> None:
        """Attach arbitrary metadata to the trace (e.g. tools_called, strategy)."""
        self._extra[key] = value

    # ------------------------------------------------------------------
    # Summary output
    # ------------------------------------------------------------------

    @property
    def total_ms(self) -> float:
        return (time.perf_counter() - self._start) * 1000

    def summary(self) -> Dict[str, Any]:
        """Return a structured dict suitable for JSON logging."""
        return {
            "question": self._question[:80] if self._question else "",
            "total_ms": round(self.total_ms, 1),
            "stages": [
                {
                    "name": s.name,
                    "ms": round(s.duration_ms, 1),
                    "ok": s.success,
                    **({"detail": s.detail} if s.detail else {}),
                }
                for s in self._stages
            ],
            **self._extra,
        }

    def print_summary(self) -> None:
        """
        Print a human-readable summary to the logger.

        Example output::

            ┌─ AgentTrace ─────────────────────────────────
            │ Question : 邓紫棋什么时候生日
            │ Total    : 3,241 ms
            ├─ rewrite    :   812 ms ✓
            ├─ strategy   :     0 ms ✓  [精确事实]
            ├─ planning   :   923 ms ✓
            ├─ execute    :   456 ms ✓  lookup_artist_profile
            ├─ generate   : 1,034 ms ✓
            └──────────────────────────────────────────────
        """
        # Best-effort: feed per-stage durations to Prometheus.
        try:
            from app.observability import metrics as _m

            for s in self._stages:
                _m.chat_request_duration_seconds.labels(stage=s.name).observe(
                    s.duration_ms / 1000.0
                )
        except Exception:  # noqa: BLE001
            pass

        total = round(self.total_ms, 1)
        lines = [
            f"┌─ AgentTrace ─────────────────────────────────",
            f"│ Question : {self._question[:60]}",
            f"│ Total    : {total:,.1f} ms",
        ]

        for s in self._stages:
            status = "✓" if s.success else "✗"
            detail = f"  {s.detail}" if s.detail else ""
            lines.append(
                f"├─ {s.name:<12}: {s.duration_ms:>8,.1f} ms {status}{detail}"
            )

        # Extra metadata
        if self._extra:
            for k, v in self._extra.items():
                lines.append(f"│ {k}: {v}")

        lines.append(f"└──────────────────────────────────────────────")

        logger.info("\n".join(lines))
