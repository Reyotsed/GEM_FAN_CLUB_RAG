"""
Tool definitions for GEM Agent.
Each tool encapsulates a specific RAG capability that the agent can invoke.
"""

from agent.tools.tool_registry import ToolRegistry, ToolResult
from agent.tools.knowledge_tools import (
    search_knowledge_base,
    search_concert_schedule,
    search_song_info,
)
from agent.tools.time_tools import get_current_datetime
from agent.tools.recommend_tools import get_hot_songs_recommendation
from agent.tools.structured_data_tools import lookup_artist_profile, lookup_milestones

__all__ = [
    "ToolRegistry",
    "ToolResult",
    "search_knowledge_base",
    "search_concert_schedule",
    "search_song_info",
    "get_current_datetime",
    "get_hot_songs_recommendation",
    "lookup_artist_profile",
    "lookup_milestones",
]
