"""G.E.M. AI Chat API server.

Industrial-grade entrypoint:
- Configuration via ``app.core.settings`` (pydantic-settings).
- Structured logging via ``app.core.logging`` (structlog + contextvars).
- Per-request tracing via ``TraceIdMiddleware``.
- Long-lived dependencies owned by ``AppContainer`` (DI).
- Layered memory + token-budget context manager.
- Backward-compatible ``/chat_gem`` and ``/chat_gem/stream`` routes.
"""
from __future__ import annotations

import argparse
import asyncio
import os
import re
import time as _time
from contextlib import asynccontextmanager
from typing import Dict, List, Optional

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from langchain.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field, field_validator

# Importing ``config`` triggers the backward-compat shim which propagates the
# new Settings into legacy module-level constants used by ``agent/`` & co.
import config  # noqa: F401
from app.core.container import AppContainer
from app.core.logging import bind_request_context, get_logger
from app.core.middleware import TraceIdMiddleware
from app.core.session_validator import (
    SessionIdError,
    SessionUserMismatch,
    normalise_session_id,
    normalise_user_id,
)
from app.core.settings import get_settings
from app.core.text_utils import (
    get_current_time,
    is_time_related_query,
    sort_docs_by_date,
)
from app.observability import healthz, metrics
from rag_modules.content_safety import check_content_safety
from rag_modules.prompts import (
    GEM_SYSTEM_PROMPT,
    GEM_USER_PROMPT,
    QUESTION_REWRITE_SYSTEM_PROMPT,
    QUESTION_REWRITE_USER_PROMPT,
)

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Settings + logging are configured at import time by ``app.core.*``.
# ---------------------------------------------------------------------------
_settings = get_settings()

# Quiet noisy 3rd-party loggers (structlog still wins for our own loggers).
import logging as _stdlogging  # noqa: E402

for _lib in (
    "httpcore",
    "httpx",
    "urllib3",
    "requests",
    "langchain",
    "langchain_community",
    "langchain_chroma",
):
    _stdlogging.getLogger(_lib).setLevel(_stdlogging.WARNING)


# ---------------------------------------------------------------------------
# Helper utilities removed: now live in app.core.text_utils.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# CLI args
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="G.E.M. AI Chat API 服务器")
    parser.add_argument("--update-db", action="store_true",
                        help="更新向量数据库（默认：不存在则创建，存在则跳过）")
    parser.add_argument("--mode", choices=["agent", "rag"], default=None,
                        help="运行模式：agent（默认）或 rag（传统RAG链）")
    args, _ = parser.parse_known_args()
    return args


# ---------------------------------------------------------------------------
# Container & lifespan
# ---------------------------------------------------------------------------

_container: Optional[AppContainer] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _container
    args = parse_args()
    if args.mode is not None:
        os.environ["RUNTIME_MODE"] = args.mode

    container = AppContainer()
    try:
        await container.startup(update_db=args.update_db)
    except Exception as e:
        logger.error("startup_failed", error=str(e))
        raise

    # Best-effort: initialize prometheus metrics. Failure is non-fatal.
    metrics.init_metrics()

    _container = container
    app.state.container = container
    logger.info("application.startup_complete")
    try:
        yield
    finally:
        timeout = container.settings.server.graceful_shutdown_timeout
        try:
            await asyncio.wait_for(container.shutdown(), timeout=timeout)
        except asyncio.TimeoutError:
            logger.warning("application.shutdown_timeout", timeout=timeout)
        _container = None
        logger.info("application.shutdown_complete")


def get_container() -> AppContainer:
    if _container is None:
        raise HTTPException(status_code=503, detail="Service is not ready")
    return _container


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title=_settings.server.api_title,
    description=_settings.server.api_description,
    lifespan=lifespan,
)

# trace-id BEFORE CORS so all log lines (including CORS pre-flight rejections)
# carry a request id.
app.add_middleware(TraceIdMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=_settings.server.cors_origins,
    allow_credentials=_settings.server.cors_allow_credentials,
    allow_methods=_settings.server.cors_allow_methods,
    allow_headers=_settings.server.cors_allow_headers,
)


# ---------------------------------------------------------------------------
# API models
# ---------------------------------------------------------------------------

_SESSION_ID_RE = re.compile(r"^[A-Za-z0-9-]{1,64}$")


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=500,
                          description="User question, max 500 characters")
    history: List[Dict[str, str]] = Field(
        default_factory=list, max_length=20,
        description="Conversation history, max 20 turns",
    )
    user_id: str = Field(default="", description="调用方用户ID")
    session_id: Optional[str] = Field(
        default=None,
        max_length=64,
        description="会话ID（可选）。为空时按 (user_id, 当日) 自动生成。"
                    "格式：^[A-Za-z0-9-]{1,64}$",
    )

    @field_validator("session_id")
    @classmethod
    def _validate_sid(cls, v: Optional[str]) -> Optional[str]:
        if v is None or v == "":
            return None
        if not _SESSION_ID_RE.match(v):
            raise ValueError("session_id must match ^[A-Za-z0-9-]{1,64}$")
        return v


# ---------------------------------------------------------------------------
# Question rewrite (legacy RAG fallback path)
# ---------------------------------------------------------------------------

_question_rewrite_prompt = ChatPromptTemplate.from_messages([
    ("system", QUESTION_REWRITE_SYSTEM_PROMPT),
    ("user", QUESTION_REWRITE_USER_PROMPT),
])


def format_conversation_history(history: List[Dict[str, str]]) -> str:
    if not history:
        return "无对话历史"
    out: List[str] = []
    for i, turn in enumerate(history, 1):
        role = turn.get("role", "unknown")
        content = turn.get("content", "")
        if role == "user":
            out.append(f"用户{i}: {content}")
        elif role == "assistant":
            out.append(f"我: {content}")
        else:
            out.append(f"{role}: {content}")
    return "\n".join(out)


def enhanced_rag_chain(container: AppContainer, input_data: Dict[str, Any]) -> str:
    """Legacy RAG chain. Now uses the ContextManager for message assembly."""
    llm = container.llm
    retriever = container.retriever
    db = container.db
    time_retriever = container.time_retriever
    if llm is None or retriever is None:
        raise RuntimeError("组件未初始化")

    original_question = input_data["question"]
    history = input_data.get("history", [])

    try:
        rewritten = (_question_rewrite_prompt | llm).invoke({
            "current_question": original_question,
            "conversation_history": format_conversation_history(history),
        }).content
    except Exception as e:  # noqa: BLE001
        logger.warning("rag.rewrite_failed", error=str(e))
        rewritten = original_question

    try:
        if is_time_related_query(original_question):
            docs = time_retriever.retrieve(
                query=original_question,
                vector_query=rewritten,
                bm25_query=original_question,
            )
            docs = sort_docs_by_date(docs, reverse=True)
            docs = docs[: container.settings.retriever.hybrid_retriever_num_results]
        else:
            docs = retriever.retrieve(
                query=original_question,
                vector_query=rewritten,
                bm25_query=original_question,
            )
    except Exception as e:  # noqa: BLE001
        logger.warning("rag.retrieve_failed", error=str(e))
        try:
            docs = db.as_retriever(
                search_kwargs={"k": container.settings.retriever.hybrid_retriever_num_results}
            ).invoke(rewritten)
        except Exception as fb_e:  # noqa: BLE001
            logger.error("rag.fallback_retrieve_failed", error=str(fb_e))
            raise RuntimeError("检索服务不可用") from fb_e

    # Assemble messages via ContextManager (token-budgeted).
    turns = _to_history_turns(history)
    docs_text = "\n\n".join(d.page_content for d in docs)
    user_text = GEM_USER_PROMPT.format(
        current_time=get_current_time(),
        context=docs_text,
        question=original_question,
    )
    build = container.context_manager.build(
        system_prompt=GEM_SYSTEM_PROMPT,
        question=user_text,
        history=turns,
        retrieved_docs="",  # already embedded into the user message above
    )
    msgs = [(m["role"], m["content"]) for m in build.messages]
    response = llm.invoke(ChatPromptTemplate.from_messages(msgs).invoke({}))
    return response.content


def _to_history_turns(history: List[Dict[str, str]]):
    from app.context.manager import HistoryTurn

    turns: List[HistoryTurn] = []
    pending: Optional[HistoryTurn] = None
    for t in history:
        role = t.get("role", "")
        content = t.get("content", "") or ""
        if role == "user":
            if pending is not None:
                turns.append(pending)
            pending = HistoryTurn(user=content)
        elif role == "assistant":
            if pending is None:
                turns.append(HistoryTurn(assistant=content))
            else:
                pending.assistant = content
                turns.append(pending)
                pending = None
    if pending is not None:
        turns.append(pending)
    return turns


# ---------------------------------------------------------------------------
# Request prep helper (shared between /chat_gem and /chat_gem/stream)
# ---------------------------------------------------------------------------

class _PreparedRequest:
    __slots__ = (
        "user_id", "session_id", "trace_id", "safe_question", "history",
        "memory_context",
    )

    def __init__(
        self, *, user_id, session_id, trace_id, safe_question, history,
        memory_context,
    ):
        self.user_id = user_id
        self.session_id = session_id
        self.trace_id = trace_id
        self.safe_question = safe_question
        self.history = history
        self.memory_context = memory_context


# 过滤历史消息中旧版代码遗留的 [foto] 行（带 URL 的照片引用），
# 防止 LLM 看到 URL 后原文输出到前端。
_FOTO_LINE_RE = re.compile(r"\n?\[foto\].*$", re.MULTILINE)

def _strip_foto_line(content: str) -> str:
    return _FOTO_LINE_RE.sub("", content)


# 每个会话最近一次发过的照片 URL 映射（模块级，进程内），
# key = "user_id::session_id", value = {0: url0, 1: url1, …}
# 仅用于可能的 URL 解析，不暴露给 LLM。
_photo_refs: Dict[str, Dict[int, str]] = {}


async def _prepare(req: Request, body: QueryRequest, container: AppContainer) -> _PreparedRequest:
    trace_id = getattr(req.state, "trace_id", None) or ""
    user_id = normalise_user_id(body.user_id)
    try:
        session_id = normalise_session_id(body.session_id, user_id)
    except SessionIdError as e:
        raise HTTPException(status_code=422, detail=str(e))

    bind_request_context(trace_id=trace_id, session_id=session_id, user_id=user_id)

    if container.session_validator is not None:
        try:
            await container.session_validator.ensure(user_id, session_id)
        except SessionUserMismatch:
            raise HTTPException(
                status_code=403,
                detail={"code": "SESSION_USER_MISMATCH",
                        "message": "session_id is bound to a different user"},
            )

    # Content safety
    safety = check_content_safety(body.question)
    if safety.is_violation:
        logger.warning(
            "safety.intercepted",
            violation_type=safety.violation_type,
            matched=safety.matched_keyword,
        )
        safe_question = safety.safe_replacement
    else:
        safe_question = body.question

    # Prefer server-side history when available (req 5.4).
    history: List[Dict[str, str]] = list(body.history or [])
    memory_context: Dict[str, str] = {"summary": "", "long_term": "", "profile": ""}
    if container.memory_service is not None:
        _ml_t0 = _time.perf_counter()
        try:
            loaded = await container.memory_service.load_session(user_id, session_id)
            if loaded.raw_messages:
                history = []
                for m in loaded.raw_messages:
                    # 过滤掉历史中遗留的 [foto] 行（旧版代码曾把 URL 追加到消息末尾）
                    clean = _strip_foto_line(m.content)
                    history.append({"role": m.role, "content": clean})
            memory_context["summary"] = loaded.summary or ""
        except Exception as e:  # noqa: BLE001
            logger.warning("memory.preload_failed", error=str(e))

        # Long-term recall (skipped for anonymous; service handles that)
        try:
            recalled = await container.memory_service.recall_long_term(
                user_id, session_id, safe_question,
                top_k=container.settings.memory.long_term_top_k,
            )
            if recalled:
                memory_context["long_term"] = "\n".join(
                    f"- ({r.session_id[:8]}) {r.content}" for r in recalled
                )
            try:
                metrics.memory_recall_total.labels(
                    layer="long_term", hit="true" if recalled else "false",
                ).inc()
            except Exception:  # noqa: BLE001
                pass
        except Exception as e:  # noqa: BLE001
            logger.warning("memory.recall_failed", error=str(e))

        # Profile
        try:
            profile = await container.memory_service.get_profile(user_id)
            if profile and (profile.preferences or profile.summary):
                bits: List[str] = []
                if profile.summary:
                    bits.append(profile.summary)
                if profile.preferences:
                    import json as _j
                    bits.append("preferences=" + _j.dumps(
                        profile.preferences, ensure_ascii=False,
                    ))
                memory_context["profile"] = "\n".join(bits)
        except Exception as e:  # noqa: BLE001
            logger.warning("memory.profile_failed", error=str(e))

        try:
            metrics.chat_request_duration_seconds.labels(stage="memory_load").observe(
                _time.perf_counter() - _ml_t0
            )
        except Exception:  # noqa: BLE001
            pass

    return _PreparedRequest(
        user_id=user_id,
        session_id=session_id,
        trace_id=trace_id,
        safe_question=safe_question,
        history=history,
        memory_context=memory_context,
    )

# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.post("/chat_gem", summary="与邓紫棋AI聊天")
async def chat_with_gem(
    body: QueryRequest,
    request: Request,
    container: AppContainer = Depends(get_container),
):
    from agent.stats_reporter import report_agent_invoke

    t0 = _time.perf_counter()
    prep = await _prepare(request, body, container)
    runtime_mode = os.getenv("RUNTIME_MODE", container.settings.agent.runtime_mode)

    logger.info("chat.received", question=prep.safe_question, mode=runtime_mode)

    status = "ok"
    try:
        if runtime_mode == "agent" and container.agent is not None:
            response_text = await asyncio.to_thread(
                container.agent.run, prep.safe_question, prep.history, prep.memory_context
            )
        else:
            response_text = await asyncio.to_thread(
                enhanced_rag_chain,
                container,
                {"question": prep.safe_question, "history": prep.history},
            )

        # Persist to memory asynchronously (fire-and-forget).
        if container.memory_service is not None:
            # 照片 URL 存入内部映射，不暴露给 LLM
            if container.agent is not None:
                photo_urls = getattr(container.agent, "last_photo_urls", None) or []
                if photo_urls:
                    ref_key = f"{prep.user_id}::{prep.session_id}"
                    _photo_refs[ref_key] = {
                        i: url for i, url in enumerate(photo_urls)
                    }
            asyncio.create_task(container.memory_service.save_turn(
                user_id=prep.user_id,
                session_id=prep.session_id,
                user_message=prep.safe_question,
                assistant_message=response_text,
                trace_id=prep.trace_id,
            ))

        report_agent_invoke(
            user_id=prep.user_id,
            duration_ms=(_time.perf_counter() - t0) * 1000,
        )

        return {
            "answer": response_text,
            "session_id": prep.session_id,
            "trace_id": prep.trace_id,
        }
    except HTTPException:
        status = "client_error"
        raise
    except Exception as e:  # noqa: BLE001
        status = "error"
        logger.error("chat.failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"AI服务处理请求时发生内部错误: {e}")
    finally:
        try:
            metrics.chat_requests_total.labels(mode=runtime_mode, status=status).inc()
            metrics.chat_request_duration_seconds.labels(stage="end_to_end").observe(
                _time.perf_counter() - t0
            )
        except Exception:  # noqa: BLE001
            pass


@app.post("/chat_gem/stream", summary="与邓紫棋AI聊天（流式响应）")
async def chat_with_gem_stream(
    body: QueryRequest,
    request: Request,
    container: AppContainer = Depends(get_container),
):
    import json as _json

    from agent.stats_reporter import report_agent_invoke

    t0 = _time.perf_counter()
    prep = await _prepare(request, body, container)
    runtime_mode = os.getenv("RUNTIME_MODE", container.settings.agent.runtime_mode)

    logger.info("chat.stream_received", question=prep.safe_question, mode=runtime_mode)

    async def event_generator():
        accumulated: List[str] = []
        # 缓存本轮照片 url，仅用于内部存储，不暴露给 LLM
        _photo_urls: List[str] = []
        try:
            if runtime_mode == "agent" and container.agent is not None:
                queue: asyncio.Queue = asyncio.Queue()
                _SENTINEL = object()
                loop = asyncio.get_event_loop()

                def _producer():
                    try:
                        for evt in container.agent.run_stream(
                            prep.safe_question, prep.history, prep.memory_context,
                        ):
                            loop.call_soon_threadsafe(queue.put_nowait, evt)
                    except Exception as exc:  # noqa: BLE001
                        try:
                            loop.call_soon_threadsafe(
                                queue.put_nowait,
                                {"event": "error", "data": {"message": str(exc)}},
                            )
                        except RuntimeError:
                            pass
                    finally:
                        try:
                            loop.call_soon_threadsafe(queue.put_nowait, _SENTINEL)
                        except RuntimeError:
                            pass

                loop.run_in_executor(None, _producer)

                while True:
                    evt = await queue.get()
                    if evt is _SENTINEL:
                        break
                    event_type = evt.get("event", "token")
                    if event_type == "token":
                        accumulated.append(evt.get("data", {}).get("content", ""))
                    elif event_type == "photos":
                        items = (evt.get("data") or {}).get("items") or []
                        _photo_urls.extend(
                            item["url"] for item in items if isinstance(item, dict) and item.get("url")
                        )
                    if event_type == "done":
                        evt.setdefault("data", {})
                        evt["data"]["session_id"] = prep.session_id
                        evt["data"]["trace_id"] = prep.trace_id
                    data = _json.dumps(evt.get("data", {}), ensure_ascii=False)
                    yield f"event: {event_type}\ndata: {data}\n\n"
            else:
                yield (
                    "event: status\n"
                    f"data: {_json.dumps({'stage': 'generating', 'message': '正在生成回答...'}, ensure_ascii=False)}\n\n"
                )
                response_text = await asyncio.to_thread(
                    enhanced_rag_chain,
                    container,
                    {"question": prep.safe_question, "history": prep.history},
                )
                accumulated.append(response_text)
                yield f"event: token\ndata: {_json.dumps({'content': response_text}, ensure_ascii=False)}\n\n"
                yield (
                    "event: done\n"
                    f"data: {_json.dumps({'session_id': prep.session_id, 'trace_id': prep.trace_id}, ensure_ascii=False)}\n\n"
                )
        except Exception as exc:  # noqa: BLE001
            logger.error("chat.stream_failed", error=str(exc))
            err = _json.dumps({"message": str(exc)}, ensure_ascii=False)
            yield f"event: error\ndata: {err}\n\n"
        finally:
            full_answer = "".join(accumulated).strip()
            if full_answer and container.memory_service is not None:
                # 照片 URL 存入内部映射，不暴露给 LLM（防 LLM 直接输出 URL）
                if _photo_urls:
                    ref_key = f"{prep.user_id}::{prep.session_id}"
                    _photo_refs[ref_key] = {
                        i: url for i, url in enumerate(_photo_urls)
                    }
                asyncio.create_task(container.memory_service.save_turn(
                    user_id=prep.user_id,
                    session_id=prep.session_id,
                    user_message=prep.safe_question,
                    assistant_message=full_answer,
                    trace_id=prep.trace_id,
                ))
            report_agent_invoke(
                user_id=prep.user_id,
                duration_ms=(_time.perf_counter() - t0) * 1000,
            )

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/", summary="服务健康检查")
def read_root():
    return {"message": "欢迎使用 G.E.M. AI Chat API，服务运行正常！"}


# ---------------------------------------------------------------------------
# Health / metrics / memory-admin endpoints
# ---------------------------------------------------------------------------

@app.get("/health/live", summary="Liveness probe")
def health_live():
    return {"status": "alive"}


@app.get("/health/ready", summary="Readiness probe")
async def health_ready(container: AppContainer = Depends(get_container)):
    from fastapi.responses import JSONResponse

    summary = await healthz.aggregate(container)
    return JSONResponse(
        status_code=200 if summary["ok"] else 503,
        content=summary,
    )


@app.get("/metrics", summary="Prometheus metrics")
def metrics_endpoint():
    from fastapi.responses import Response

    body = metrics.render()
    return Response(content=body, media_type=metrics.content_type())


@app.delete("/memory/forget", summary="忘掉某个用户的全部记忆")
async def memory_forget(
    user_id: str,
    container: AppContainer = Depends(get_container),
):
    if not user_id or user_id == "anonymous":
        raise HTTPException(status_code=400, detail="user_id is required")
    if container.memory_service is None:
        raise HTTPException(status_code=503, detail="memory service unavailable")
    await container.memory_service.forget(user_id)
    logger.info("memory.forget_done", user_id=user_id)
    return {"ok": True, "user_id": user_id}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    logger.info("server.start", host=_settings.server.host, port=_settings.server.port)
    uvicorn.run(app, host=_settings.server.host, port=_settings.server.port)