"""
Recommendation tools for the GEM Agent.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

from langchain.schema import Document

if TYPE_CHECKING:
    from rag_modules.hybrid_retriever import HybridRetriever

logger = logging.getLogger(__name__)

# Module-level reference (injected at startup)
_retriever: Optional["HybridRetriever"] = None


def configure(retriever: "HybridRetriever") -> None:
    """Inject runtime dependencies."""
    global _retriever
    _retriever = retriever
    logger.info("Recommend tools configured")


# =========================================================================
# Tool: get_hot_songs_recommendation
# =========================================================================

HOT_SONGS_SCHEMA = {
    "type": "object",
    "properties": {
        "preference": {
            "type": "string",
            "description": "Optional: fan's preference or mood, e.g. '快歌', '慢歌', '励志', '伤感'",
        }
    },
    "required": [],
}

HOT_SONGS_DESCRIPTION = (
    "Get GEM's popular / hot songs list for recommendation. "
    "Use this when the fan asks for song recommendations, "
    "wants to know her most popular tracks, or asks 'what should I listen to?'. "
    "You can optionally pass a preference keyword to tailor the recommendation."
)


def get_hot_songs_recommendation(preference: str = "") -> str:
    """Search for hot songs, optionally filtered by a preference keyword."""
    if _retriever is None:
        return "错误：检索器尚未初始化。"

    base_query = "邓紫棋热门歌曲排行榜推荐"
    if preference:
        base_query = f"邓紫棋 {preference} 风格的热门歌曲推荐"

    docs = _retriever.retrieve(
        query=base_query,
        vector_query=base_query,
        bm25_query=f"热门 歌曲 {preference}".strip(),
    )
    if not docs:
        return "暂无热门歌曲数据。"

    parts = []
    for i, doc in enumerate(docs[:8], 1):
        parts.append(f"[{i}] {doc.page_content}")
    logger.info("get_hot_songs_recommendation: returned %d results", len(parts))
    return "\n\n".join(parts)
