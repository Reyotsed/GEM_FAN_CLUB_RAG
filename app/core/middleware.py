"""HTTP middlewares: trace-id propagation."""
from __future__ import annotations

import uuid
from typing import Callable

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from app.core.logging import (
    bind_request_context,
    clear_request_context,
)


REQUEST_ID_HEADER = "X-Request-Id"


class TraceIdMiddleware(BaseHTTPMiddleware):
    """Assign or propagate a per-request trace id, and bind it to logging context.

    - Reads ``X-Request-Id`` from incoming headers; if absent, generates a
      UUID v4.
    - Stores the value in ``request.state.trace_id`` for downstream handlers.
    - Echoes the same value back in the response header.
    - Binds it (along with any pre-existing session/user) into the structlog
      contextvars so every log line emitted while serving the request carries
      the trace id automatically.
    """

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        incoming = request.headers.get(REQUEST_ID_HEADER)
        trace_id = incoming.strip() if incoming else uuid.uuid4().hex
        # Cap length to a sane bound to avoid log abuse.
        trace_id = trace_id[:64] or uuid.uuid4().hex

        request.state.trace_id = trace_id
        bind_request_context(trace_id=trace_id)
        try:
            response = await call_next(request)
        finally:
            # Don't unbind here; FastAPI may still log after middleware returns.
            # Cleanup happens at the end of the request via a separate hook.
            pass

        response.headers[REQUEST_ID_HEADER] = trace_id
        # Best-effort cleanup so background tasks don't inherit stale ids.
        clear_request_context()
        return response
