"""
邓紫棋粉丝俱乐部 Agent 模块
基于技能的 RAG Agent，处理粉丝互动。

v2.5: 新增 PromptStrategySelector（规则策略池）和 AgentTrace（请求追踪）。
v2.5.1: 新增 LLM 输出守护（重试、校验、兜底），提升流水线鲁棒性。
"""

from agent.gem_agent import GemAgent
from agent.prompt_strategy import PromptStrategySelector, PromptStrategyResult
from agent.trace import AgentTrace
from agent.llm_guard import GuardConfig
from agent.skills import (
    SkillRegistry,
    BaseSkill,
    FunctionResult,
    KnowledgeSkill,
    RecommendSkill,
    ProfileSkill,
    UtilitySkill,
    WebSearchSkill,
    PhotoSkill,
)

__all__ = [
    "GemAgent",
    "PromptStrategySelector",
    "PromptStrategyResult",
    "AgentTrace",
    "GuardConfig",
    "SkillRegistry",
    "BaseSkill",
    "FunctionResult",
    "KnowledgeSkill",
    "RecommendSkill",
    "ProfileSkill",
    "UtilitySkill",
    "WebSearchSkill",
    "PhotoSkill",
]
