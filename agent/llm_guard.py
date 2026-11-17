"""
LLM 输出守护 — 不可靠模型输出的鲁棒性层。

提供重试逻辑、输出校验和兜底策略，覆盖 Agent 流水线中的每个 LLM 调用。
确保模型产出异常输出时系统优雅降级而非崩溃。

守护的 LLM 调用：
    1. 问题重写 → 校验非空，去除多余前缀
    2. 函数规划 → 校验JSON，验证工具名，自动重试
    3. 回答生成 → 检测空/脱角色输出，提供兜底回复

设计原则：
    • 每个守护函数无状态、无副作用（除日志外）。
    • 重试逻辑使用指数退避 + 抖动。
    • 兜底回复始终保持邓紫棋人设。
    • 所有守护均可选 — Agent 无守护也能运行，但有守护会更稳健。
"""

from __future__ import annotations

import json
import logging
import random
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


# ─────────────────────────────────────────────────────────────────────
# 配置
# ─────────────────────────────────────────────────────────────────────

@dataclass
class GuardConfig:
    """输出守护层的可调参数。"""

    # 重试设置
    max_retries: int = 2                # 总尝试次数 = 1 + max_retries
    retry_base_delay: float = 0.5       # 秒，每次重试翻倍
    retry_jitter: float = 0.3           # 延迟随机抖动 ±jitter

    # 重写守护
    rewrite_max_length: int = 300       # 重写问题不应超过此长度
    rewrite_min_length: int = 2         # 至少需要此字符数

    # 规划守护
    planning_max_retries: int = 2       # 关键规划步骤允许更多重试
    planning_known_tools: Set[str] = field(default_factory=set)  # 初始化时填充

    # 回答守护
    answer_min_length: int = 4          # 短于此长度触发兜底
    answer_forbidden_phrases: List[str] = field(default_factory=lambda: [
        "作为AI", "作为一个AI", "作为人工智能", "我是AI", "我是一个AI",
        "我是人工智能", "语言模型", "大模型", "我没有感情",
        "我是机器人", "我是程序", "作为一个语言模型",
        "as an AI", "I'm an AI", "language model",
    ])


# 默认单例配置（GemAgent初始化时会覆盖）
_DEFAULT_CONFIG = GuardConfig()


# ─────────────────────────────────────────────────────────────────────
# 邓紫棋人设兜底回复
# ─────────────────────────────────────────────────────────────────────

_FALLBACK_REPLIES: List[str] = [
    "哈哈哈哈对不起，我刚脑袋突然短路了一下下 😅 你再问我一次嘛，我保证这次认真回答你！",
    "诶怎么说呢，刚才想到一半突然卡壳了～其实你的问题超有趣的，再说一遍让我好好想想嘛！😊",
    "说实话我刚才在想别的事情想入迷了哈哈哈哈！你刚说什么来着？再问一次好不好～ 💕",
    "天哪不好意思！其实我刚在组织语言，但是越想越乱了 😂 你再给我一次机会嘛，这次一定好好说！",
    "嘿！让你等了不好意思呢～我刚本来想到一个超好的回答，然后就……忘了 🤦‍♀️ 再来一次嘛好不好？🎵",
    "其实我刚想回答你的，然后脑子突然放空了一下下，哈哈哈你们不要笑我啦！再问一次嘛～ ✨",
]


def get_fallback_reply() -> str:
    """随机返回一条邓紫棋人设兜底回复。"""
    return random.choice(_FALLBACK_REPLIES)


# ─────────────────────────────────────────────────────────────────────
# 重试辅助
# ─────────────────────────────────────────────────────────────────────

def retry_llm_call(
    fn: Callable[..., T],
    *args: Any,
    max_retries: int = 2,
    base_delay: float = 0.5,
    jitter: float = 0.3,
    label: str = "LLM call",
    **kwargs: Any,
) -> T:
    """调用 fn 并支持重试 + 指数退避。全部尝试失败则抛出最后的异常。"""
    last_exc: Optional[Exception] = None
    for attempt in range(1 + max_retries):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:
            last_exc = exc
            if attempt < max_retries:
                delay = base_delay * (2 ** attempt) + random.uniform(-jitter, jitter)
                delay = max(0.1, delay)
                logger.warning(
                    "[%s] attempt %d/%d failed (%s), retrying in %.1fs …",
                    label, attempt + 1, 1 + max_retries, exc, delay,
                )
                time.sleep(delay)
            else:
                logger.error(
                    "[%s] all %d attempts failed. Last error: %s",
                    label, 1 + max_retries, exc,
                )
    raise last_exc  # type: ignore[misc]


# ─────────────────────────────────────────────────────────────────────
# 重写守护
# ─────────────────────────────────────────────────────────────────────

# LLM有时会擅自添加的前缀
_REWRITE_NOISE_PREFIXES = [
    "重写后的问题：", "重写后：", "重写：",
    "重写后的问题:", "重写后:", "重写:",
    "Rewritten:", "Rewritten question:",
    "查询：", "查询:",
]


def validate_rewrite(
    rewritten: str,
    original: str,
    config: GuardConfig = _DEFAULT_CONFIG,
) -> str:
    """校验并清理重写后的问题。返回清理后的重写结果，不可用时回退到原始问题。"""
    if not rewritten or not rewritten.strip():
        logger.warning("Rewrite returned empty output, falling back to original")
        return original

    cleaned = rewritten.strip()

    # 去除模型可能添加的前缀（循环直到无前缀可去）
    changed = True
    while changed:
        changed = False
        for prefix in _REWRITE_NOISE_PREFIXES:
            if cleaned.startswith(prefix):
                cleaned = cleaned[len(prefix):].strip()
                logger.debug("去除重写前缀 '%s'", prefix)
                changed = True
                break  # 去除后重新扫描

    # 去除首尾引号
    if (cleaned.startswith('"') and cleaned.endswith('"')) or \
       (cleaned.startswith("'") and cleaned.endswith("'")):
        cleaned = cleaned[1:-1].strip()

    # 去除 markdown 粗体标记
    if cleaned.startswith("**") and cleaned.endswith("**"):
        cleaned = cleaned[2:-2].strip()

    # 过短 → 可能是乱码
    if len(cleaned) < config.rewrite_min_length:
        logger.warning(
            "Rewrite too short (%d chars), falling back to original: '%s'",
            len(cleaned), cleaned,
        )
        return original

    # 过长 → 模型可能附带了解释文本
    if len(cleaned) > config.rewrite_max_length:
        # 尝试只取第一行
        first_line = cleaned.split("\n")[0].strip()
        if config.rewrite_min_length <= len(first_line) <= config.rewrite_max_length:
            logger.warning(
                "Rewrite too long (%d chars), using first line (%d chars)",
                len(cleaned), len(first_line),
            )
            cleaned = first_line
        else:
            logger.warning(
                "Rewrite too long (%d chars) and first line unusable, "
                "falling back to original",
                len(cleaned),
            )
            return original

    return cleaned


# ─────────────────────────────────────────────────────────────────────
# 规划守护
# ─────────────────────────────────────────────────────────────────────

def validate_planning_output(
    raw: str,
    known_tools: Set[str],
    parse_fn: Callable[[str], List[Dict[str, Any]]],
) -> List[Dict[str, Any]]:
    """
    校验LLM规划输出：
      1. 用多策略解析器解析JSON
      2. 过滤幻觉工具名
      3. 校验参数类型
    返回清理后的函数调用列表（可能为空）。
    """
    calls = parse_fn(raw)

    if not calls:
        return calls

    validated: List[Dict[str, Any]] = []
    for fc in calls:
        tool_name = fc.get("tool", "")

        # Check for hallucinated tool names
        if tool_name and tool_name not in known_tools:
            # Try fuzzy matching: maybe LLM used a slightly different name
            best_match = _fuzzy_match_tool(tool_name, known_tools)
            if best_match:
                logger.warning(
                    "Planning: corrected hallucinated tool '%s' → '%s'",
                    tool_name, best_match,
                )
                fc["tool"] = best_match
                tool_name = best_match
            else:
                logger.warning(
                    "Planning: removing hallucinated tool '%s' (not in %s)",
                    tool_name, known_tools,
                )
                continue

        # Ensure arguments is a dict
        args = fc.get("arguments", {})
        if not isinstance(args, dict):
            logger.warning(
                "Planning: tool '%s' has non-dict arguments (%s), replacing with {}",
                tool_name, type(args).__name__,
            )
            fc["arguments"] = {}

        validated.append(fc)

    if len(validated) < len(calls):
        logger.info(
            "Planning: filtered %d → %d valid function calls",
            len(calls), len(validated),
        )

    return validated


def _fuzzy_match_tool(name: str, known_tools: Set[str]) -> Optional[str]:
    """
    Try to match a hallucinated tool name to a known tool.

    Handles common LLM mistakes:
      • Extra/missing underscores: "search_knowledgebase" → "search_knowledge_base"
      • CamelCase: "SearchKnowledgeBase" → "search_knowledge_base"
      • Partial match: "knowledge_base" → "search_knowledge_base"
    """
    name_lower = name.lower().replace("-", "_")

    # Exact match (case-insensitive)
    for tool in known_tools:
        if tool.lower() == name_lower:
            return tool

    # Underscore-collapsed match
    name_collapsed = name_lower.replace("_", "")
    for tool in known_tools:
        if tool.lower().replace("_", "") == name_collapsed:
            return tool

    # CamelCase → snake_case conversion
    snake_name = re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower()
    for tool in known_tools:
        if tool.lower() == snake_name:
            return tool

    # Suffix/substring match (e.g. "knowledge_base" → "search_knowledge_base")
    for tool in known_tools:
        if tool.endswith(name_lower) or name_lower.endswith(tool):
            return tool

    return None


# ─────────────────────────────────────────────────────────────────────
# Answer guard
# ─────────────────────────────────────────────────────────────────────

def validate_answer(
    answer: str,
    config: GuardConfig = _DEFAULT_CONFIG,
) -> str:
    """
    Validate the final answer.

    Returns the answer if valid, or a persona-consistent fallback reply.
    """
    if not answer or not answer.strip():
        logger.warning("Answer generation returned empty content, using fallback")
        return get_fallback_reply()

    cleaned = answer.strip()

    if len(cleaned) < config.answer_min_length:
        logger.warning(
            "Answer too short (%d chars): '%s', using fallback",
            len(cleaned), cleaned,
        )
        return get_fallback_reply()

    # Check for AI identity leaks — scan all forbidden phrases
    for phrase in config.answer_forbidden_phrases:
        if phrase.lower() in cleaned.lower():
            logger.warning(
                "Answer contains forbidden phrase '%s', patching out",
                phrase,
            )
            # Instead of replacing the whole answer, try to salvage by removing
            # the offending sentence
            cleaned = _remove_sentence_containing(cleaned, phrase)
            if not cleaned or len(cleaned.strip()) < config.answer_min_length:
                return get_fallback_reply()

    return cleaned


def validate_stream_buffer(
    buffer: str,
    config: GuardConfig = _DEFAULT_CONFIG,
) -> bool:
    """
    Check if the accumulated stream buffer contains forbidden content.

    Returns True if the buffer is OK, False if it should be aborted.
    (Used as an early-exit check during streaming.)
    """
    if not buffer:
        return True
    buffer_lower = buffer.lower()
    for phrase in config.answer_forbidden_phrases:
        if phrase.lower() in buffer_lower:
            logger.warning("Stream buffer contains forbidden phrase '%s'", phrase)
            return False
    return True


def _remove_sentence_containing(text: str, phrase: str) -> str:
    """Remove sentences that contain the given phrase.

    Returns the cleaned text.  If *all* sentences contain the phrase,
    returns an empty string so the caller can trigger fallback.
    """
    # Split on Chinese and Western sentence terminators
    sentences = re.split(r'(?<=[。！？.!?\n])', text)
    cleaned_sentences = []
    phrase_lower = phrase.lower()
    for sent in sentences:
        if phrase_lower not in sent.lower():
            cleaned_sentences.append(sent)
    return "".join(cleaned_sentences).strip()
