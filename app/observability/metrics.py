"""Prometheus metrics.

A minimal-but-useful set defined upfront. Initialization is wrapped in a
try/except so that a missing ``prometheus_client`` package never blocks
startup — in that case all metric helpers become no-ops.
"""
from __future__ import annotations

from typing import Optional

from app.core.logging import get_logger

_logger = get_logger(__name__)


class _Noop:
    def labels(self, *args, **kwargs):
        return self

    def inc(self, *args, **kwargs):
        pass

    def observe(self, *args, **kwargs):
        pass

    def set(self, *args, **kwargs):
        pass


_NOOP = _Noop()


# Public metric handles — always assigned; either real or _NOOP.
chat_requests_total = _NOOP
chat_request_duration_seconds = _NOOP
context_tokens_used = _NOOP
memory_recall_total = _NOOP
memory_outbox_lag = _NOOP

_initialized = False
_registry = None


def init_metrics() -> bool:
    """Idempotent metrics initialization. Returns True on success."""
    global chat_requests_total, chat_request_duration_seconds
    global context_tokens_used, memory_recall_total, memory_outbox_lag
    global _initialized, _registry

    if _initialized:
        return True
    try:
        from prometheus_client import (
            CollectorRegistry,
            Counter,
            Gauge,
            Histogram,
        )

        _registry = CollectorRegistry()

        chat_requests_total = Counter(
            "chat_requests_total",
            "Total chat requests handled, labelled by mode and status.",
            labelnames=("mode", "status"),
            registry=_registry,
        )
        chat_request_duration_seconds = Histogram(
            "chat_request_duration_seconds",
            "End-to-end / per-stage chat duration.",
            labelnames=("stage",),
            buckets=(0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30),
            registry=_registry,
        )
        context_tokens_used = Histogram(
            "context_tokens_used",
            "Tokens consumed by each context section.",
            labelnames=("section",),
            buckets=(50, 100, 250, 500, 1000, 2000, 4000, 8000),
            registry=_registry,
        )
        memory_recall_total = Counter(
            "memory_recall_total",
            "Memory recall operations, labelled by layer and hit-status.",
            labelnames=("layer", "hit"),
            registry=_registry,
        )
        memory_outbox_lag = Gauge(
            "memory_outbox_lag",
            "Number of messages currently sitting in the memory outbox.",
            registry=_registry,
        )
        _initialized = True
        return True
    except Exception as e:  # noqa: BLE001
        _logger.error("metrics.init_failed", error=str(e))
        return False


def render() -> bytes:
    """Render the registry as a Prometheus exposition format payload."""
    if not _initialized or _registry is None:
        return b""
    try:
        from prometheus_client import generate_latest

        return generate_latest(_registry)
    except Exception as e:  # noqa: BLE001
        _logger.warning("metrics.render_failed", error=str(e))
        return b""


def content_type() -> str:
    try:
        from prometheus_client import CONTENT_TYPE_LATEST  # type: ignore

        return CONTENT_TYPE_LATEST
    except Exception:  # noqa: BLE001
        return "text/plain; version=0.0.4; charset=utf-8"


# Convenience wrappers used from hot paths -----------------------------------

def observe_outbox_lag(n: int) -> None:
    if _initialized:
        try:
            memory_outbox_lag.set(n)
        except Exception:  # noqa: BLE001
            pass
