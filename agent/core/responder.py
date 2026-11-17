"""回答生成模块 — 上下文构建 + LLM 调用 + 流式输出 + 兜底。

从 ``agent/gem_agent.py`` 拆分出来。
"""

from __future__ import annotations

import logging
import time
from datetime import datetime
from typing import Any, Dict, Generator, List, Optional, TYPE_CHECKING

from langchain.prompts import ChatPromptTemplate

from rag_modules.prompts import GEM_SYSTEM_PROMPT, GEM_USER_PROMPT

if TYPE_CHECKING:
    from agent.skills import FunctionResult
    from agent.prompt_strategy import PromptStrategyResult

logger = logging.getLogger(__name__)


# ─── 纯工具函数（原 GemAgent 静态方法）───────────────


def format_history(history: List[Dict[str, str]]) -> str:
    """格式化对话历史，用于提示词注入。"""
    if not history:
        return "无对话历史"
    parts = []
    for turn in history:
        role = turn.get("role", "unknown")
        content = turn.get("content", "")
        if role == "user":
            parts.append(f"粉丝: {content}")
        elif role == "assistant":
            parts.append(f"邓紫棋: {content}")
    return "\n".join(parts)


def compress_photo_items(items: List[Dict[str, Any]]) -> List[str]:
    """压缩图片结果：去除 url / score / match_score 等技术字段，
    仅保留 LLM ＂看图＂所需的人类可读文本描述。
    每张图从 ~500-800 字符压缩到 ~100-150 字符。
    """
    if not items:
        return []
    lines = []
    for i, item in enumerate(items, 1):
        parts = []
        caption = (item.get("captionZh") or "").strip()
        tags = item.get("tags") or []
        note = (item.get("note") or "").strip()
        dims = item.get("dims") or {}

        if caption:
            parts.append(caption)
        if tags:
            tag_str = "、".join(str(t)[:20] for t in tags[:5])
            parts.append(f"[{tag_str}]")
        if note:
            note_trunc = note[:60] + "…" if len(note) > 60 else note
            parts.append(note_trunc)

        # 压缩构图维度（仅保留人眼可感知的字段）
        dim_texts = []
        if isinstance(dims, dict):
            for k, v in dims.items():
                if isinstance(v, str) and v.strip():
                    dim_texts.append(f"{k}={v[:12]}")
                elif isinstance(v, (int, float)):
                    dim_texts.append(f"{k}={v}")
        if dim_texts:
            parts.append("(" + " ".join(dim_texts) + ")")

        lines.append(f"{i}. {'｜'.join(parts)}" if parts else f"{i}. <照片{i}>")
    return lines


def extract_current_time(function_results: List["FunctionResult"]) -> str:
    """从函数结果获取时间，如无则生成默认值。"""
    for fr in function_results:
        if fr.function_name == "get_current_datetime" and fr.success:
            return fr.data
    return datetime.now().strftime("%Y年%m月%d日 %H:%M")


def build_context_text(
    function_results: List["FunctionResult"],
) -> tuple[str, str]:
    """从函数结果构建上下文文本，对图片结果做压缩后注入。

    返回 (context_text, photo_text)：
      - context_text — 非图片类结果，嵌入 GEM_USER_PROMPT 的 {context}
      - photo_text   — 压缩后的图片描述，独立通过 ContextManager 的 photo_context 传入
    """

    def _escape_braces(s: str) -> str:
        return s.replace("{", "{{").replace("}", "}}")

    if not function_results:
        return ("（无需检索资料，直接与粉丝聊天即可）", "")

    context_parts: List[str] = []
    photo_parts: List[str] = []

    for fr in function_results:
        if not fr.success:
            context_parts.append(
                f"【{fr.function_name} 调用失败】{_escape_braces(fr.error or '')}"
            )
            continue
        if not fr.data:
            continue

        # 图片结果：压缩后走独立通道
        if fr.skill_name == "photo":
            try:
                import json as _json

                obj = _json.loads(fr.data)
                items = (obj or {}).get("items") or []
                compressed = compress_photo_items(items)
                if compressed:
                    photo_parts.append(
                        f"【search_gem_photos 返回 {len(compressed)} 张照片】\n"
                        + "\n".join(compressed)
                    )
                else:
                    photo_parts.append("【search_gem_photos 返回结果】未找到匹配照片。")
            except Exception:
                photo_parts.append(
                    f"【{fr.function_name} 返回结果】\n{_escape_braces(fr.data)}"
                )
        else:
            context_parts.append(
                f"【{fr.function_name} 返回结果】\n{_escape_braces(fr.data)}"
            )

    ctx = "\n\n---\n\n".join(context_parts) if context_parts else "（无需检索资料，直接与粉丝聊天即可）"
    photo = "\n\n---\n\n".join(photo_parts) if photo_parts else ""
    return (ctx, photo)


def apply_memory_context(
    memory_fields: Dict[str, str],
    memory_context: Optional[Dict[str, str]],
) -> None:
    """Apply caller-provided memory snippets to the given dict (mutates in-place)."""
    if not memory_context:
        memory_fields["summary"] = ""
        memory_fields["long_term"] = ""
        memory_fields["profile"] = ""
        return
    memory_fields["summary"] = (memory_context.get("summary") or "").strip()
    memory_fields["long_term"] = (memory_context.get("long_term") or "").strip()
    memory_fields["profile"] = (memory_context.get("profile") or "").strip()


# ─── 消息构建 ──────────────────────────────────────


def build_messages(
    *,
    question: str,
    history: List[Dict[str, str]],
    context_text: str,
    photo_context: str = "",
    current_time: str,
    strategy_result: Optional["PromptStrategyResult"] = None,
    context_manager=None,
    memory_fields: Optional[Dict[str, str]] = None,
    history_max_turns: int = 10,
    active_trace=None,
) -> List[tuple]:
    """Assemble the (role, content) tuple list passed to ChatPromptTemplate.

    支持两种路径：
      • Token-budgeted（有 context_manager 注入时）
      • Legacy simple slicing（无 context_manager 时，向前兼容测试）
    """
    # Strategy-augmented system prompt
    system_prompt = GEM_SYSTEM_PROMPT
    if strategy_result and strategy_result.injection_text:
        system_prompt = (
            f"{GEM_SYSTEM_PROMPT}\n\n"
            f"---\n\n"
            f"## 🎯 本次回答策略增强（由策略引擎自动匹配）\n\n"
            f"{strategy_result.injection_text}"
        )

    # 如果存在独立图片上下文，user message 里用占位符代替完整 context
    if photo_context:
        context_placeholder = "（见【照片描述】与【参考资料】）"
    else:
        context_placeholder = context_text

    user_message = GEM_USER_PROMPT.format(
        current_time=current_time,
        context=context_placeholder,
        question=question,
    )

    if context_manager is None:
        # Legacy path: simple [-N:] slicing.
        messages: List[tuple] = [("system", system_prompt)]
        for turn in history[-history_max_turns:]:
            role = turn.get("role", "")
            content = turn.get("content", "")
            if role in ("user", "assistant"):
                messages.append((role, content))
        messages.append(("user", user_message))
        return messages

    # Token-budgeted path
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

    mf = memory_fields or {}
    build = context_manager.build(
        system_prompt=system_prompt,
        question=user_message,
        history=turns,
        retrieved_docs=context_text,
        photo_context=photo_context,
        session_summary=mf.get("summary", "") or "",
        long_term_memory=mf.get("long_term", "") or "",
        user_profile=mf.get("profile", "") or "",
    )
    # Stash stats on the trace if available.
    if active_trace is not None:
        try:
            active_trace.set_extra("context_tokens_used", build.stats.get("tokens_used"))
            active_trace.set_extra("context_budget", build.stats.get("budget"))
            active_trace.set_extra("context_dropped", build.stats.get("dropped_sections"))
        except Exception:  # noqa: BLE001
            pass
    return [(m["role"], m["content"]) for m in build.messages]


# ─── 答案生成 ──────────────────────────────────────


def generate_answer(
    *,
    llm,
    question: str,
    history: List[Dict[str, str]],
    function_results: List["FunctionResult"],
    strategy_result: Optional["PromptStrategyResult"] = None,
    guard_config,
    context_manager=None,
    memory_fields: Optional[Dict[str, str]] = None,
    history_max_turns: int = 10,
    active_trace=None,
) -> str:
    """生成最终的邓紫棋人设回答（含重试 + 输出守护）。"""
    from agent.llm_guard import retry_llm_call, validate_answer, get_fallback_reply

    context_text, photo_context = build_context_text(function_results)
    current_time = extract_current_time(function_results)

    messages = build_messages(
        question=question,
        history=history,
        context_text=context_text,
        photo_context=photo_context,
        current_time=current_time,
        strategy_result=strategy_result,
        context_manager=context_manager,
        memory_fields=memory_fields,
        history_max_turns=history_max_turns,
        active_trace=active_trace,
    )

    try:
        prompt = ChatPromptTemplate.from_messages(messages)

        def _do_generate() -> str:
            response = llm.invoke(prompt.invoke({}))
            return response.content or ""

        raw_answer = retry_llm_call(
            _do_generate,
            max_retries=guard_config.max_retries,
            base_delay=guard_config.retry_base_delay,
            jitter=guard_config.retry_jitter,
            label="generate",
        )

        answer = validate_answer(raw_answer, guard_config)
        logger.info("模型回复:\n%s", answer)
        return answer

    except Exception as exc:
        logger.error("回答生成重试后仍失败: %s", exc, exc_info=True)
        return get_fallback_reply()


def generate_answer_stream(
    *,
    llm,
    question: str,
    history: List[Dict[str, str]],
    function_results: List["FunctionResult"],
    strategy_result: Optional["PromptStrategyResult"] = None,
    guard_config,
    context_manager=None,
    memory_fields: Optional[Dict[str, str]] = None,
    history_max_turns: int = 10,
    active_trace=None,
) -> Generator[str, None, None]:
    """generate_answer 的流式版本（含输出守护）。

    鲁棒性：
        • 瞬态LLM错误自动重试
        • 监控累积缓冲区中的禁用词
        • 流输出为空时回退到人设回复
    """
    from agent.llm_guard import validate_stream_buffer, get_fallback_reply

    context_text, photo_context = build_context_text(function_results)
    current_time = extract_current_time(function_results)

    messages = build_messages(
        question=question,
        history=history,
        context_text=context_text,
        photo_context=photo_context,
        current_time=current_time,
        strategy_result=strategy_result,
        context_manager=context_manager,
        memory_fields=memory_fields,
        history_max_turns=history_max_turns,
        active_trace=active_trace,
    )

    prompt = ChatPromptTemplate.from_messages(messages)
    compiled_prompt = prompt.invoke({})

    # 流式重试包装器：失败时重试整个流
    last_exc: Optional[Exception] = None
    for attempt in range(1 + guard_config.max_retries):
        try:
            buffer = ""
            token_count = 0
            aborted = False

            for chunk in llm.stream(compiled_prompt):
                token_text = getattr(chunk, "content", None) or ""
                if token_text:
                    buffer += token_text
                    token_count += 1

                    # 每10个token检查一次禁用内容（避免逐token开销）
                    if token_count % 10 == 0:
                        if not validate_stream_buffer(buffer, guard_config):
                            logger.warning(
                                "Stream aborted: forbidden content detected "
                                "after %d tokens",
                                token_count,
                            )
                            aborted = True
                            break

                    yield token_text

            if aborted:
                logger.warning(
                    "Stream was aborted due to content guard. "
                    "Partial output (%d chars) already sent to client.",
                    len(buffer),
                )
                logger.info("模型回复(流式-中止):\n%s", buffer)
                return

            if buffer.strip():
                logger.info("模型回复(流式):\n%s", buffer)

            if not buffer.strip():
                logger.warning(
                    "Stream produced empty output (attempt %d/%d)",
                    attempt + 1,
                    1 + guard_config.max_retries,
                )
                if attempt < guard_config.max_retries:
                    time.sleep(guard_config.retry_base_delay)
                    continue
                yield get_fallback_reply()

            return  # success

        except Exception as exc:
            last_exc = exc
            if attempt < guard_config.max_retries:
                logger.warning(
                    "Stream generation attempt %d/%d failed: %s, retrying…",
                    attempt + 1,
                    1 + guard_config.max_retries,
                    exc,
                )
                time.sleep(guard_config.retry_base_delay * (2**attempt))
                continue

    logger.error("流式回答生成重试后仍失败: %s", last_exc, exc_info=True)
    yield get_fallback_reply()
