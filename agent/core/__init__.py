"""Agent 核心模块：将 GemAgent 的重逻辑按阶段拆分为独立子模块。

模块职责：
    planner   — 工具规划（LLM 决策 + JSON 解析 + 校验）
    rewriter  — 问题重写（提升检索命中率）
    executor  — 函数调用执行（通过 SkillRegistry 分派）
    responder — 回答生成（上下文构建 + LLM 调用 + 流式输出）
"""

from agent.core.planner import (
    TOOL_PLANNING_SYSTEM_PROMPT,
    TOOL_PLANNING_USER_PROMPT,
    plan_functions,
    parse_function_calls,
    validate_function_calls,
    extract_photo_count,
)
from agent.core.rewriter import rewrite_question
from agent.core.executor import execute_functions, extract_photo_payload
from agent.core.responder import (
    build_messages,
    generate_answer,
    generate_answer_stream,
    apply_memory_context,
    build_context_text,
    compress_photo_items,
    extract_current_time,
    format_history,
)

__all__ = [
    "TOOL_PLANNING_SYSTEM_PROMPT",
    "TOOL_PLANNING_USER_PROMPT",
    "plan_functions",
    "parse_function_calls",
    "validate_function_calls",
    "extract_photo_count",
    "rewrite_question",
    "execute_functions",
    "extract_photo_payload",
    "build_messages",
    "generate_answer",
    "generate_answer_stream",
    "apply_memory_context",
    "build_context_text",
    "compress_photo_items",
    "extract_current_time",
    "format_history",
]
