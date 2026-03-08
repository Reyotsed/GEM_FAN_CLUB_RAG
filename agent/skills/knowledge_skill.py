"""
KnowledgeSkill — encapsulates all knowledge-retrieval functions.

Functions:
    • search_knowledge_base   — general knowledge search (career, awards, …)
    • search_concert_schedule — concert / tour schedule search (date-sorted)
    • search_song_info        — song-specific search (lyrics, background, …)

Dependencies (injected via ``configure``):
    • retriever       — ``HybridRetriever`` for general queries
    • time_retriever  — ``HybridRetriever`` with expanded K for time queries
    • sort_docs_fn    — callable to sort documents by date
"""

from __future__ import annotations

import logging
from typing import Any, Callable, List, Optional, TYPE_CHECKING

from langchain.schema import Document

from agent.skills.base_skill import BaseSkill

if TYPE_CHECKING:
    from rag_modules.hybrid_retriever import HybridRetriever

logger = logging.getLogger(__name__)


class KnowledgeSkill(BaseSkill):
    """Skill that wraps the HybridRetriever for knowledge-base searches."""

    def __init__(self) -> None:
        super().__init__()
        self._retriever: Optional[HybridRetriever] = None
        self._time_retriever: Optional[HybridRetriever] = None
        self._sort_docs_by_date: Optional[Callable] = None
        self._num_results: int = 5

    # ---- BaseSkill interface ----------------------------------------

    @property
    def name(self) -> str:
        return "knowledge"

    @property
    def description(self) -> str:
        return (
            "Knowledge retrieval skill — searches GEM's knowledge base "
            "for career facts, concert schedules, and song information "
            "using hybrid (vector + BM25) retrieval with RRF re-ranking."
        )

    def configure(self, **kwargs: Any) -> None:
        """
        Required kwargs:
            retriever       — HybridRetriever (general)
            time_retriever  — HybridRetriever (expanded K for time queries)
            sort_docs_fn    — callable(docs, reverse) → sorted docs

        Optional kwargs:
            num_results     — int (default 5)
        """
        self._retriever = kwargs["retriever"]
        self._time_retriever = kwargs["time_retriever"]
        self._sort_docs_by_date = kwargs["sort_docs_fn"]
        self._num_results = kwargs.get("num_results", 5)
        self._finalize()

    def _register_functions(self) -> None:
        self._add_function(
            name="search_knowledge_base",
            description=(
                "Search GEM's general knowledge base for information about her career, "
                "personal life, awards, and general facts. "
                "Use this for broad questions that are NOT specifically about "
                "concert schedules or song details."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query for GEM's knowledge base "
                                       "(career, personal life, general info)",
                    }
                },
                "required": ["query"],
            },
            func=self._search_knowledge_base,
        )

        self._add_function(
            name="search_concert_schedule",
            description=(
                "Search for GEM's concert and tour schedule information. "
                "Use this when the fan asks about upcoming or past concerts, "
                "tour dates, venues, cities, or ticket info. "
                "Results are sorted by date (newest first)."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query about GEM's concerts, tours, "
                                       "dates, venues, or tickets",
                    }
                },
                "required": ["query"],
            },
            func=self._search_concert_schedule,
        )

        self._add_function(
            name="search_song_info",
            description=(
                "Search for specific song information including lyrics, "
                "creation background, album details, and related facts. "
                "Use this when the fan mentions a particular song by name."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "song_name": {
                        "type": "string",
                        "description": "The name of the song to search for "
                                       "(Chinese or English)",
                    }
                },
                "required": ["song_name"],
            },
            func=self._search_song_info,
        )

    # ---- Function implementations -----------------------------------

    def _search_knowledge_base(self, query: str) -> str:
        if self._retriever is None:
            return "错误：检索器尚未初始化。"
        docs = self._retriever.retrieve(
            query=query, vector_query=query, bm25_query=query
        )
        logger.info("search_knowledge_base: retrieved %d docs for '%s'", len(docs), query)
        return self._docs_to_text(docs)

    def _search_concert_schedule(self, query: str) -> str:
        if self._time_retriever is None:
            return "错误：时间检索器尚未初始化。"
        docs = self._time_retriever.retrieve(
            query=query, vector_query=query, bm25_query=query
        )
        if self._sort_docs_by_date is not None:
            docs = self._sort_docs_by_date(docs, reverse=True)

        # Prioritise docs whose city matches the query to prevent relevant
        # results from being truncated by the date-sorted cutoff.
        city_matched = [d for d in docs if d.metadata.get("city") and d.metadata["city"] in query]
        city_unmatched = [d for d in docs if d not in city_matched]
        # Deduplicate: city-matched first, then fill remaining slots
        docs = (city_matched + city_unmatched)[: self._num_results]

        logger.info("search_concert_schedule: retrieved %d docs for '%s'", len(docs), query)
        return self._docs_to_text(docs)

    def _search_song_info(self, song_name: str) -> str:
        if self._retriever is None:
            return "错误：检索器尚未初始化。"
        enriched_query = f"邓紫棋 {song_name} 歌曲 歌词 创作背景"
        docs = self._retriever.retrieve(
            query=enriched_query,
            vector_query=enriched_query,
            bm25_query=song_name,
        )
        logger.info("search_song_info: retrieved %d docs for '%s'", len(docs), song_name)
        return self._docs_to_text(docs)

    # ---- Helpers ----------------------------------------------------

    @staticmethod
    def _docs_to_text(docs: List[Document], max_docs: int = 10) -> str:
        """Convert retrieved Documents to a numbered text block."""
        if not docs:
            return "未找到相关信息。"
        parts = []
        for i, doc in enumerate(docs[:max_docs], 1):
            category = doc.metadata.get("category", "未知")
            parts.append(f"[{i}] ({category}) {doc.page_content}")
        return "\n\n".join(parts)
