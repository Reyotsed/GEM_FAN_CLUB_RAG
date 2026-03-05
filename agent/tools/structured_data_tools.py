"""
Structured data lookup tool for the GEM Agent.

Provides direct, zero-hallucination access to structured JSON data
(artist profile, milestones) without going through vector search.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level cache (loaded once at configure time)
# ---------------------------------------------------------------------------
_profile: Optional[Dict[str, Any]] = None
_milestones: Optional[List[Dict[str, Any]]] = None


def configure(structured_data_dir: str) -> None:
    """Load structured JSON files into memory. Called once at startup."""
    global _profile, _milestones
    base = Path(structured_data_dir)

    profile_path = base / "artist_profile.json"
    milestones_path = base / "milestones.json"

    if profile_path.exists():
        with open(profile_path, "r", encoding="utf-8") as f:
            _profile = json.load(f)
        logger.info("Loaded artist_profile.json (%d top-level keys)", len(_profile))
    else:
        logger.warning("artist_profile.json not found at %s", profile_path)

    if milestones_path.exists():
        with open(milestones_path, "r", encoding="utf-8") as f:
            _milestones = json.load(f)
        logger.info("Loaded milestones.json (%d events)", len(_milestones))
    else:
        logger.warning("milestones.json not found at %s", milestones_path)


# =========================================================================
# Tool: lookup_artist_profile
# =========================================================================

LOOKUP_PROFILE_SCHEMA = {
    "type": "object",
    "properties": {
        "field": {
            "type": "string",
            "description": (
                "The profile field to look up. Available fields: "
                "name, english_name, full_name, birthday, birthplace, nationality, "
                "zodiac, profession, debut_date, debut_age, family, education, "
                "labels, fan_name, fan_nickname_for_gem, charity, social_media, "
                "records, billion_mvs, discography, tours, major_awards, "
                "filmography, books, physical. "
                "Use 'all' to return the complete profile."
            ),
        }
    },
    "required": ["field"],
}

LOOKUP_PROFILE_DESCRIPTION = (
    "Look up specific factual information about GEM (邓紫棋) from her structured profile. "
    "Use this for precise questions about her birthday, zodiac, family, albums, "
    "tours, awards, records, YouTube stats, discography, etc. "
    "This tool returns accurate structured data — prefer it over search_knowledge_base "
    "for factual lookups."
)


def lookup_artist_profile(field: str) -> str:
    """Look up a specific field from the artist profile."""
    if _profile is None:
        return "错误：艺人资料尚未加载。"

    if field == "all":
        return json.dumps(_profile, ensure_ascii=False, indent=2)

    # Support nested dot notation like "family.father"
    keys = field.split(".")
    data: Any = _profile
    for key in keys:
        if isinstance(data, dict) and key in data:
            data = data[key]
        else:
            # Try fuzzy match on top-level keys
            available = list(_profile.keys()) if isinstance(_profile, dict) else []
            return f"未找到字段 '{field}'。可用字段: {', '.join(available)}"

    if isinstance(data, (dict, list)):
        return json.dumps(data, ensure_ascii=False, indent=2)
    return str(data)


# =========================================================================
# Tool: lookup_milestones
# =========================================================================

LOOKUP_MILESTONES_SCHEMA = {
    "type": "object",
    "properties": {
        "year": {
            "type": "string",
            "description": (
                "Filter milestones by year (e.g. '2014') or year range (e.g. '2019-2020'). "
                "Use 'all' to return all milestones."
            ),
        },
        "category": {
            "type": "string",
            "description": (
                "Optional category filter: 个人, 出道, 专辑发行, 单曲发布, "
                "演唱会, 综艺, 奖项, 荣誉, 社交媒体, 里程碑, 事业转折, 慈善, 电影"
            ),
        },
    },
    "required": ["year"],
}

LOOKUP_MILESTONES_DESCRIPTION = (
    "Look up key milestones and events in GEM's career by year or category. "
    "Use this when the fan asks 'what happened in [year]', 'when did [event] happen', "
    "or for timeline-based questions. Returns structured event data."
)


def lookup_milestones(year: str, category: str = "") -> str:
    """Look up milestones filtered by year and/or category."""
    if _milestones is None:
        return "错误：里程碑数据尚未加载。"

    results = _milestones

    # Filter by year
    if year and year != "all":
        if "-" in year and len(year) > 4:
            # Year range like "2019-2020"
            parts = year.split("-")
            try:
                start_year, end_year = int(parts[0]), int(parts[1])
                results = [
                    m for m in results
                    if any(
                        start_year <= int(y) <= end_year
                        for y in _extract_years(m.get("date", ""))
                    )
                ]
            except ValueError:
                pass
        else:
            # Single year
            results = [m for m in results if year in m.get("date", "")]

    # Filter by category
    if category:
        results = [m for m in results if m.get("category", "") == category]

    if not results:
        return f"未找到 {year} 年" + (f" {category} 类别" if category else "") + "的里程碑事件。"

    # Format output
    lines = []
    for m in results:
        date = m.get("date", "")
        event = m.get("event", "")
        cat = m.get("category", "")
        lines.append(f"[{date}] ({cat}) {event}")

    return "\n".join(lines)


def _extract_years(date_str: str) -> List[str]:
    """Extract 4-digit year strings from a date string."""
    import re
    return re.findall(r"\d{4}", date_str)
