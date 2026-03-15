"""
内容安全检查模块 — Content Safety / Moderation for GEM Fan Club RAG Service.

在用户提问进入 RAG 或 Agent 流程之前，先检测是否包含违规内容（涉政、涉黄、
暴力、违法等）。若检测到违规，则将问题替换为一个随机的、安全的邓紫棋相关话题，
从而避免生成不当回复。

检测策略：
    1. **关键词匹配**（快速）：维护一份违规关键词列表，命中即判定违规。
    2. **正则模式匹配**（补充）：覆盖组合式敏感表述。

设计原则：
    - 误判时宁可放行（false-negative 友好），因为 LLM 本身也有内置安全过滤。
    - 关键词列表聚焦于高风险、无歧义的词汇，避免将歌词/日常用语误判。
    - 安全话题全部与邓紫棋相关，保持角色沉浸感。
"""

from __future__ import annotations

import logging
import random
import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────
# 违规关键词库（按类别分组，便于维护）
# ──────────────────────────────────────────────────────────────

# 涉黄关键词
_PORNOGRAPHIC_KEYWORDS: List[str] = [
    "色情", "裸体", "做爱", "性交", "一夜情", "约炮",
    "卖淫", "嫖娼", "三级片", "黄片", "AV女优", "成人影片",
    "口交", "肛交", "自慰", "性爱视频", "援交", "情色",
    "淫秽", "淫荡", "荡妇", "色狼", "强奸", "性侵",
    "猥亵", "恋童", "儿童色情", "未成年色情",
]

# 涉政敏感词
_POLITICAL_KEYWORDS: List[str] = [
    "颠覆政权", "分裂国家", "反华势力", "政治迫害",
    "六四事件", "天安门事件", "反共",
    "藏独", "疆独", "台独", "港独",
    "法轮功", "邪教",
]

# 暴力/违法关键词
_VIOLENCE_ILLEGAL_KEYWORDS: List[str] = [
    "制造炸弹", "制作炸药", "制造武器", "枪支制造",
    "如何杀人", "杀人方法", "投毒", "下毒方法",
    "自杀方法", "自残方法", "如何自杀",
    "贩毒", "制毒", "吸毒教程", "冰毒制作",
    "洗钱方法", "诈骗教程", "黑客攻击教程",
]

# 歧视/仇恨言论
_HATE_SPEECH_KEYWORDS: List[str] = [
    "种族灭绝", "纳粹万岁", "白人至上",
    "屠杀穆斯林", "杀光黑人",
]

# 合并所有违规关键词
_ALL_VIOLATION_KEYWORDS: List[str] = (
    _PORNOGRAPHIC_KEYWORDS
    + _POLITICAL_KEYWORDS
    + _VIOLENCE_ILLEGAL_KEYWORDS
    + _HATE_SPEECH_KEYWORDS
)

# ──────────────────────────────────────────────────────────────
# 正则模式补充（捕获变体/组合敏感表述）
# ──────────────────────────────────────────────────────────────

_VIOLATION_PATTERNS: List[re.Pattern] = [
    # 涉黄变体
    re.compile(r"[做发打]一?[炮飞]", re.IGNORECASE),
    re.compile(r"(怎么|如何|教我).{0,6}(制毒|制炸|杀人|自杀|投毒|下毒|贩毒|制造武器)", re.IGNORECASE),
    re.compile(r"(怎么|如何|教我).{0,6}(入侵|攻击|破解).{0,4}(系统|网站|服务器|密码)", re.IGNORECASE),
]

# ──────────────────────────────────────────────────────────────
# 安全替代话题（全部与邓紫棋相关，保持角色沉浸感）
# ──────────────────────────────────────────────────────────────

_SAFE_TOPICS: List[str] = [
    "邓紫棋的《泡沫》这首歌背后有什么创作故事？",
    "邓紫棋最近有什么新歌推荐吗？",
    "邓紫棋的演唱会有哪些经典曲目？",
    "邓紫棋是怎么开始学音乐的？",
    "邓紫棋有哪些获奖经历？",
    "邓紫棋的《光年之外》是怎么创作出来的？",
    "邓紫棋在《我是歌手》节目上有什么精彩表现？",
    "邓紫棋的音乐风格是怎样的？",
    "邓紫棋最喜欢的食物是什么？",
    "邓紫棋有哪些有趣的演唱会幕后故事？",
    "邓紫棋的《句号》这首歌你了解吗？",
    "邓紫棋有哪些经典的现场演出？",
    "邓紫棋的创作灵感通常来自哪里？",
    "邓紫棋有什么有趣的童年故事？",
    "邓紫棋的粉丝名'棋士'是怎么来的？",
    "邓紫棋最新的巡演计划是什么？",
    "邓紫棋有哪些跨界合作的歌曲？",
    "给我推荐几首邓紫棋的歌吧！",
    "邓紫棋最感人的歌曲是哪首？",
    "邓紫棋在社交媒体上有什么有趣的日常分享？",
]


# ──────────────────────────────────────────────────────────────
# 检测结果数据类
# ──────────────────────────────────────────────────────────────

@dataclass
class SafetyCheckResult:
    """内容安全检测结果。"""
    is_violation: bool
    """是否违规"""
    violation_type: Optional[str] = None
    """违规类型描述（用于日志，不暴露给用户）"""
    matched_keyword: Optional[str] = None
    """命中的关键词或模式（用于日志）"""
    safe_replacement: Optional[str] = None
    """违规时的安全替代问题"""


# ──────────────────────────────────────────────────────────────
# 核心检测函数
# ──────────────────────────────────────────────────────────────

def check_content_safety(text: str) -> SafetyCheckResult:
    """
    检查用户输入文本是否包含违规内容。

    Args:
        text: 用户原始输入文本。

    Returns:
        SafetyCheckResult — 包含是否违规、违规类型、安全替代问题等信息。
    """
    if not text or not text.strip():
        return SafetyCheckResult(is_violation=False)

    normalized = text.strip().lower()

    # ---- 阶段 1: 关键词匹配 ----
    for keyword in _ALL_VIOLATION_KEYWORDS:
        if keyword.lower() in normalized:
            # 判断违规类型
            vtype = _classify_violation(keyword)
            replacement = _pick_safe_topic()
            logger.warning(
                "内容安全检测命中关键词 [%s]（类型: %s），原始输入: %s",
                keyword, vtype, text[:80],
            )
            return SafetyCheckResult(
                is_violation=True,
                violation_type=vtype,
                matched_keyword=keyword,
                safe_replacement=replacement,
            )

    # ---- 阶段 2: 正则模式匹配 ----
    for pattern in _VIOLATION_PATTERNS:
        match = pattern.search(normalized)
        if match:
            replacement = _pick_safe_topic()
            logger.warning(
                "内容安全检测命中正则模式 [%s]，原始输入: %s",
                match.group(), text[:80],
            )
            return SafetyCheckResult(
                is_violation=True,
                violation_type="组合敏感表述",
                matched_keyword=match.group(),
                safe_replacement=replacement,
            )

    return SafetyCheckResult(is_violation=False)


def get_random_safe_topic() -> str:
    """获取一个随机的安全话题（外部也可调用）。"""
    return _pick_safe_topic()


# ──────────────────────────────────────────────────────────────
# 内部辅助函数
# ──────────────────────────────────────────────────────────────

def _classify_violation(keyword: str) -> str:
    """根据命中的关键词判断违规类别。"""
    kw_lower = keyword.lower()
    if kw_lower in [k.lower() for k in _PORNOGRAPHIC_KEYWORDS]:
        return "涉黄内容"
    if kw_lower in [k.lower() for k in _POLITICAL_KEYWORDS]:
        return "涉政内容"
    if kw_lower in [k.lower() for k in _VIOLENCE_ILLEGAL_KEYWORDS]:
        return "暴力/违法内容"
    if kw_lower in [k.lower() for k in _HATE_SPEECH_KEYWORDS]:
        return "仇恨/歧视内容"
    return "其他违规内容"


def _pick_safe_topic() -> str:
    """随机挑选一个安全替代话题。"""
    return random.choice(_SAFE_TOPICS)
