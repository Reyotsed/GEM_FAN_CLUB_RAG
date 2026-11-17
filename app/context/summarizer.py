"""Rolling-summary helper.

Compresses the oldest slice of a session's raw messages into a <=500 char
summary using a fast LLM (e.g. ``glm-4-flash``). Designed to be called from
a background worker — never on the request hot path.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Sequence

from app.core.logging import get_logger
from app.core.settings import get_settings

_logger = get_logger(__name__)


class SummarizeError(RuntimeError):
    """Raised when the summarizer fails to produce a usable summary."""


@dataclass
class SessionMessage:
    role: str
    content: str


_SYSTEM = (
    "你是对话摘要助手，请把下面这一段对话压缩为不超过 500 字的中文摘要，"
    "保留关键事实、用户偏好与未解决的问题，去掉寒暄和重复表述。"
    "只输出摘要本身，不要包含任何解释或前缀。"
)


class RollingSummarizer:
    """Wrap a callable that invokes a fast-tier LLM.

    The LLM client is injected as ``invoke`` so that this module remains
    decoupled from any specific SDK; the caller supplies a closure that takes
    a list of ``{role, content}`` dicts and returns a string.
    """

    def __init__(self, invoke: Callable[[List[dict]], str], max_chars: int = 500) -> None:
        self._invoke = invoke
        self._max_chars = max_chars

    def summarize(
        self,
        messages: Sequence[SessionMessage],
        previous_summary: Optional[str] = None,
    ) -> str:
        if not messages:
            raise SummarizeError("nothing to summarize")

        rendered = "\n".join(f"{m.role}: {m.content}" for m in messages if m.content)
        prompt_msgs: List[dict] = [{"role": "system", "content": _SYSTEM}]
        if previous_summary:
            prompt_msgs.append({
                "role": "system",
                "content": f"已有摘要（请在此基础上更新）：\n{previous_summary}",
            })
        prompt_msgs.append({"role": "user", "content": rendered})

        try:
            text = self._invoke(prompt_msgs)
        except Exception as e:  # noqa: BLE001
            _logger.warning("summarizer.invoke_failed", error=str(e))
            raise SummarizeError(str(e)) from e

        if not text or not text.strip():
            raise SummarizeError("empty summary returned")
        text = text.strip()
        if len(text) > self._max_chars:
            text = text[: self._max_chars].rstrip() + "…"
        return text


def make_zhipu_invoker() -> Callable[[List[dict]], str]:
    """Default invoker using the configured fast LLM (provider-agnostic).

    Imports inside the function so that test environments without the SDK can
    still import this module.
    """
    s = get_settings()

    # DeepSeek / OpenAI 走 ChatOpenAI（LangChain 统一接口）
    if s.llm.provider in ("deepseek", "openai"):
        from langchain_community.chat_models import ChatOpenAI

        model = ChatOpenAI(
            model=s.llm.chat_fast_model,
            temperature=0.2,
            openai_api_key=s.llm.chat_api_key,
            openai_api_base=s.llm.chat_base_url,
            request_timeout=s.llm.timeout,
        )
    else:
        from zhipuai import ZhipuAI  # type: ignore

        _z_client = ZhipuAI(api_key=s.llm.api_key)

        def _zhipu_invoke(msgs: List[dict]) -> str:
            resp = _z_client.chat.completions.create(
                model=s.llm.fast_model,
                messages=msgs,
                temperature=0.2,
                timeout=s.llm.timeout,
            )
            return (resp.choices[0].message.content or "").strip()

        return _zhipu_invoke

    def _invoke(messages: List[dict]) -> str:
        resp = model.invoke(messages)
        return (resp.content or "").strip()

    return _invoke
