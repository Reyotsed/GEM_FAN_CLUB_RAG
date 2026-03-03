"""
Knowledge-related tools for the GEM Agent.

These tools wrap the existing HybridRetriever to provide
structured search capabilities that the agent can invoke autonomously.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, List, Optional

from langchain.schema import Document

if TYPE_CHECKING:
    from rag_modules.hybrid_retriever import HybridRetriever

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level references (injected at startup by GemAgent.initialize)
# ---------------------------------------------------------------------------
_retriever: Optional["HybridRetriever"] = None
_time_retriever: Optional["HybridRetriever"] = None
_sort_docs_by_date = None  # Will be set to the sort function from rag_server
_hybrid_num_results: int = 5


def configure(
    retriever: "HybridRetriever",
    time_retriever: "HybridRetriever",
    sort_docs_fn,
    hybrid_num_results: int = 5,
) -> None:
    """Inject runtime dependencies — called once during agent initialisation."""
    global _retriever, _time_retriever, _sort_docs_by_date, _hybrid_num_results
    _retriever = retriever
    _time_retriever = time_retriever
    _sort_docs_by_date = sort_docs_fn
    _hybrid_num_results = hybrid_num_results
    logger.info("Knowledge tools configured")


def _docs_to_text(docs: List[Document], max_docs: int = 10) -> str:
    """Convert a list of Documents to a readable string."""
    if not docs:
        return "未找到相关信息。"
    parts = []
    for i, doc in enumerate(docs[:max_docs], 1):
        category = doc.metadata.get("category", "未知")
        parts.append(f"[{i}] ({category}) {doc.page_content}")
    return "\n\n".join(parts)


# =========================================================================
# Tool: search_knowledge_base
# =========================================================================

SEARCH_KB_SCHEMA = {
    "type": "object",
    "properties": {
        "query": {
            "type": "string",
            "description": "Search query for GEM's knowledge base (career, personal life, general info)",
        }
    },
    "required": ["query"],
}

SEARCH_KB_DESCRIPTION = (
    "Search GEM's general knowledge base for information about her career, "
    "personal life, awards, and general facts. "
    "Use this for broad questions that are NOT specifically about concert schedules or song details."
)


def search_knowledge_base(query: str) -> str:
    """Execute a general knowledge-base search."""
    if _retriever is None:
        return "错误：检索器尚未初始化。"
    docs = _retriever.retrieve(query=query, vector_query=query, bm25_query=query)
    logger.info("search_knowledge_base: retrieved %d docs for '%s'", len(docs), query)
    return _docs_to_text(docs)


# =========================================================================
# Tool: search_concert_schedule
# =========================================================================

SEARCH_CONCERT_SCHEMA = {
    "type": "object",
    "properties": {
        "query": {
            "type": "string",
            "description": "Search query about GEM's concerts, tours, dates, venues, or tickets",
        }
    },
    "required": ["query"],
}

SEARCH_CONCERT_DESCRIPTION = (
    "Search for GEM's concert and tour schedule information. "
    "Use this when the fan asks about upcoming or past concerts, "
    "tour dates, venues, cities, or ticket info. "
    "Results are sorted by date (newest first)."
)


def search_concert_schedule(query: str) -> str:
    """Execute a concert-schedule search with time-aware sorting."""
    if _time_retriever is None:
        return "错误：时间检索器尚未初始化。"
    docs = _time_retriever.retrieve(query=query, vector_query=query, bm25_query=query)
    if _sort_docs_by_date is not None:
        docs = _sort_docs_by_date(docs, reverse=True)
    docs = docs[:_hybrid_num_results]
    logger.info("search_concert_schedule: retrieved %d docs for '%s'", len(docs), query)
    return _docs_to_text(docs)


# =========================================================================
# Tool: search_song_info
# =========================================================================

SEARCH_SONG_SCHEMA = {
    "type": "object",
    "properties": {
        "song_name": {
            "type": "string",
            "description": "The name of the song to search for (Chinese or English)",
        }
    },
    "required": ["song_name"],
}

SEARCH_SONG_DESCRIPTION = (
    "Search for specific song information including lyrics, "
    "creation background, album details, and related facts. "
    "Use this when the fan mentions a particular song by name."
)


def search_song_info(song_name: str) -> str:
    """Execute a song-specific search."""
    if _retriever is None:
        return "错误：检索器尚未初始化。"
    enriched_query = f"邓紫棋 {song_name} 歌曲 歌词 创作背景"
    docs = _retriever.retrieve(
        query=enriched_query,
        vector_query=enriched_query,
        bm25_query=song_name,
    )
    logger.info("search_song_info: retrieved %d docs for '%s'", len(docs), song_name)
    return _docs_to_text(docs)
