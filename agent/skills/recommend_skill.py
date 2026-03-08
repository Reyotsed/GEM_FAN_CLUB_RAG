"""
RecommendSkill — song recommendation capabilities.

Functions:
    • get_hot_songs_recommendation — recommend popular songs, optionally filtered by preference

Dependencies (injected via ``configure``):
    • retriever — ``HybridRetriever``
"""

from __future__ import annotations

import logging
from typing import Any, Optional, TYPE_CHECKING

from agent.skills.base_skill import BaseSkill

if TYPE_CHECKING:
    from rag_modules.hybrid_retriever import HybridRetriever

logger = logging.getLogger(__name__)


class RecommendSkill(BaseSkill):
    """Skill for recommending GEM's songs based on popularity or preference."""

    def __init__(self) -> None:
        super().__init__()
        self._retriever: Optional[HybridRetriever] = None

    # ---- BaseSkill interface ----------------------------------------

    @property
    def name(self) -> str:
        return "recommend"

    @property
    def description(self) -> str:
        return (
            "Song recommendation skill — suggests GEM's popular tracks, "
            "optionally filtered by mood or style preference."
        )

    def configure(self, **kwargs: Any) -> None:
        """
        Required kwargs:
            retriever — HybridRetriever
        """
        self._retriever = kwargs["retriever"]
        self._finalize()

    def _register_functions(self) -> None:
        self._add_function(
            name="get_hot_songs_recommendation",
            description=(
                "Get GEM's popular / hot songs list for recommendation. "
                "Use this when the fan asks for song recommendations, "
                "wants to know her most popular tracks, or asks "
                "'what should I listen to?'. "
                "You can optionally pass a preference keyword to tailor "
                "the recommendation."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "preference": {
                        "type": "string",
                        "description": "Optional: fan's preference or mood, "
                                       "e.g. '快歌', '慢歌', '励志', '伤感'",
                    }
                },
                "required": [],
            },
            func=self._get_hot_songs_recommendation,
        )

    # ---- Function implementation ------------------------------------

    def _get_hot_songs_recommendation(self, preference: str = "") -> str:
        if self._retriever is None:
            return "错误：检索器尚未初始化。"

        base_query = "邓紫棋热门歌曲排行榜推荐"
        if preference:
            base_query = f"邓紫棋 {preference} 风格的热门歌曲推荐"

        docs = self._retriever.retrieve(
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
