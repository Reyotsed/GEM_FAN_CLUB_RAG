"""Structured logging based on structlog with request-scoped context vars.

Every log line emitted via `get_logger()` automatically carries the current
`trace_id` / `session_id` / `user_id` if they have been bound by the
`TraceIdMiddleware` for the in-flight request.
"""
from __future__ import annotations

import logging
import sys
from contextvars import ContextVar
from typing import Any, Dict, Optional

try:  # structlog is the preferred path; degrade gracefully when missing.
    import structlog  # type: ignore

    _HAVE_STRUCTLOG = True
except Exception:  # noqa: BLE001
    structlog = None  # type: ignore
    _HAVE_STRUCTLOG = False

from app.core.settings import get_settings


# Request-scoped context variables. Default to None so that out-of-request
# log lines (startup / background tasks) still work cleanly.
_trace_id_var: ContextVar[Optional[str]] = ContextVar("trace_id", default=None)
_session_id_var: ContextVar[Optional[str]] = ContextVar("session_id", default=None)
_user_id_var: ContextVar[Optional[str]] = ContextVar("user_id", default=None)


def bind_request_context(
    trace_id: Optional[str] = None,
    session_id: Optional[str] = None,
    user_id: Optional[str] = None,
) -> None:
    """Bind request-scoped fields. Safe to call multiple times per request."""
    if trace_id is not None:
        _trace_id_var.set(trace_id)
    if session_id is not None:
        _session_id_var.set(session_id)
    if user_id is not None:
        _user_id_var.set(user_id)


def clear_request_context() -> None:
    _trace_id_var.set(None)
    _session_id_var.set(None)
    _user_id_var.set(None)


def get_trace_id() -> Optional[str]:
    return _trace_id_var.get()


def get_session_id() -> Optional[str]:
    return _session_id_var.get()


def get_user_id() -> Optional[str]:
    return _user_id_var.get()


def _inject_context(_, __, event_dict: Dict[str, Any]) -> Dict[str, Any]:
    """structlog processor: enrich every event with request context vars."""
    tid = _trace_id_var.get()
    sid = _session_id_var.get()
    uid = _user_id_var.get()
    if tid is not None:
        event_dict.setdefault("trace_id", tid)
    if sid is not None:
        event_dict.setdefault("session_id", sid)
    if uid is not None:
        event_dict.setdefault("user_id", uid)
    return event_dict


_configured = False


def configure_logging() -> None:
    """Idempotent global logging configuration. Call once at startup."""
    global _configured
    if _configured:
        return

    obs = get_settings().observability

    # --- stdlib logging baseline (so 3rd-party libs route through us) -------
    root = logging.getLogger()
    # remove existing handlers to avoid double emission when reloaded
    for h in list(root.handlers):
        root.removeHandler(h)
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(obs.log_level_value)
    # 给 stdlib 日志统一加上时间戳（含毫秒），方便按 query 时间排查问题。
    # 注意：structlog 走自己的管线（已带 iso 时间戳），这里只影响通过
    # logging.getLogger() 打的日志（如 agent/skills 等业务模块）。
    _formatter = logging.Formatter(
        fmt="%(asctime)s.%(msecs)03d [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(_formatter)
    root.addHandler(handler)
    root.setLevel(obs.log_level_value)

    if not _HAVE_STRUCTLOG:
        # Plain stdlib logging fallback. Good enough for unit tests / minimal
        # environments where structlog is not installed.
        _configured = True
        return

    # --- structlog pipeline -------------------------------------------------
    timestamper = structlog.processors.TimeStamper(fmt="iso", utc=False)
    shared_processors = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        timestamper,
        _inject_context,
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]

    if obs.log_json:
        renderer = structlog.processors.JSONRenderer(ensure_ascii=False)
    else:
        renderer = structlog.dev.ConsoleRenderer(colors=False)

    structlog.configure(
        processors=shared_processors + [renderer],
        wrapper_class=structlog.make_filtering_bound_logger(obs.log_level_value),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(file=sys.stdout),
        cache_logger_on_first_use=True,
    )

    _configured = True


class _StdlibAdapter:
    """Minimal structlog-compatible interface backed by stdlib logging."""

    def __init__(self, name: Optional[str]) -> None:
        self._log = logging.getLogger(name)

    def _emit(self, level: int, event: str, **kwargs: Any) -> None:
        if not kwargs:
            self._log.log(level, event)
            return
        # Render kwargs as ``key=value`` so debug output is still readable.
        ctx = " ".join(f"{k}={v}" for k, v in kwargs.items())
        self._log.log(level, "%s %s", event, ctx)

    def info(self, event: str, **kwargs: Any) -> None:
        self._emit(logging.INFO, event, **kwargs)

    def debug(self, event: str, **kwargs: Any) -> None:
        self._emit(logging.DEBUG, event, **kwargs)

    def warning(self, event: str, **kwargs: Any) -> None:
        self._emit(logging.WARNING, event, **kwargs)

    def error(self, event: str, **kwargs: Any) -> None:
        self._emit(logging.ERROR, event, **kwargs)

    def critical(self, event: str, **kwargs: Any) -> None:
        self._emit(logging.CRITICAL, event, **kwargs)

    def bind(self, **_: Any) -> "_StdlibAdapter":
        return self


def get_logger(name: Optional[str] = None):
    """Get a structlog-bound logger. ``configure_logging()`` is called lazily."""
    if not _configured:
        configure_logging()
    if _HAVE_STRUCTLOG:
        return structlog.get_logger(name)
    return _StdlibAdapter(name)
