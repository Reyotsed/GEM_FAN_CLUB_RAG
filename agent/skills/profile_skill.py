"""
ProfileSkill — structured data lookups (zero-hallucination).

Functions:
    • lookup_artist_profile — look up specific fields from GEM's structured profile
    • lookup_milestones     — query career milestones by year / category

Dependencies (injected via ``configure``):
    • structured_data_dir — path containing artist_profile.json & milestones.json
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from agent.skills.base_skill import BaseSkill

logger = logging.getLogger(__name__)


class ProfileSkill(BaseSkill):
    """Skill that provides direct access to structured JSON data."""

    def __init__(self) -> None:
        super().__init__()
        self._profile: Optional[Dict[str, Any]] = None
        self._milestones: Optional[List[Dict[str, Any]]] = None

    # ---- BaseSkill interface ----------------------------------------

    @property
    def name(self) -> str:
        return "profile"

    @property
    def description(self) -> str:
        return (
            "Structured profile skill — provides direct, zero-hallucination "
            "access to GEM's personal profile and career milestones from "
            "curated JSON data."
        )

    def configure(self, **kwargs: Any) -> None:
        """
        Required kwargs:
            structured_data_dir — str, path to the directory containing
                                  artist_profile.json and milestones.json
        """
        structured_dir = Path(kwargs["structured_data_dir"])
        self._load_data(structured_dir)
        self._finalize()

    def _register_functions(self) -> None:
        self._add_function(
            name="lookup_artist_profile",
            description=(
                "Look up specific factual information about GEM (邓紫棋) from "
                "her structured profile. Use this for precise questions about "
                "her birthday, zodiac, family, albums, tours, awards, records, "
                "YouTube stats, discography, etc. "
                "This tool returns accurate structured data — prefer it over "
                "search_knowledge_base for factual lookups."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "field": {
                        "type": "string",
                        "description": (
                            "The profile field to look up. Available fields: "
                            "name, english_name, full_name, birthday, birthplace, "
                            "nationality, zodiac, profession, debut_date, debut_age, "
                            "family, education, labels, fan_name, "
                            "fan_nickname_for_gem, charity, social_media, records, "
                            "billion_mvs, discography, tours, major_awards, "
                            "filmography, books, physical. "
                            "Use 'all' to return the complete profile."
                        ),
                    }
                },
                "required": ["field"],
            },
            func=self._lookup_artist_profile,
        )

        self._add_function(
            name="lookup_milestones",
            description=(
                "Look up key milestones and events in GEM's career by year or "
                "category. Use this when the fan asks 'what happened in [year]', "
                "'when did [event] happen', or for timeline-based questions. "
                "Returns structured event data."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "year": {
                        "type": "string",
                        "description": (
                            "Filter milestones by year (e.g. '2014') or year range "
                            "(e.g. '2019-2020'). Use 'all' to return all milestones."
                        ),
                    },
                    "category": {
                        "type": "string",
                        "description": (
                            "Optional category filter: 个人, 出道, 专辑发行, "
                            "单曲发布, 演唱会, 综艺, 奖项, 荣誉, 社交媒体, "
                            "里程碑, 事业转折, 慈善, 电影"
                        ),
                    },
                },
                "required": ["year"],
            },
            func=self._lookup_milestones,
        )

    # ---- Data loading -----------------------------------------------

    def _load_data(self, structured_dir: Path) -> None:
        """
        Load structured JSON files into memory.

        Both files are optional — a warning is logged if either is missing,
        but the skill will still be usable for the file that *was* found.
        """
        profile_path = structured_dir / "artist_profile.json"
        milestones_path = structured_dir / "milestones.json"

        if profile_path.exists():
            with open(profile_path, "r", encoding="utf-8") as f:
                self._profile = json.load(f)
            logger.info(
                "Loaded artist_profile.json (%d top-level keys)", len(self._profile)
            )
        else:
            logger.warning("artist_profile.json not found at %s", profile_path)

        if milestones_path.exists():
            with open(milestones_path, "r", encoding="utf-8") as f:
                self._milestones = json.load(f)
            logger.info("Loaded milestones.json (%d events)", len(self._milestones))
        else:
            logger.warning("milestones.json not found at %s", milestones_path)

    # ---- Function implementations -----------------------------------

    def _lookup_artist_profile(self, field: str) -> str:
        if self._profile is None:
            return "错误：艺人资料尚未加载。"

        if field == "all":
            return json.dumps(self._profile, ensure_ascii=False, indent=2)

        # Support nested dot notation like "family.father"
        keys = field.split(".")
        data: Any = self._profile
        for key in keys:
            if isinstance(data, dict) and key in data:
                data = data[key]
            else:
                available = (
                    list(self._profile.keys()) if isinstance(self._profile, dict) else []
                )
                return f"未找到字段 '{field}'。可用字段: {', '.join(available)}"

        if isinstance(data, (dict, list)):
            return json.dumps(data, ensure_ascii=False, indent=2)
        return str(data)

    def _lookup_milestones(self, year: str, category: str = "") -> str:
        if self._milestones is None:
            return "错误：里程碑数据尚未加载。"

        results = self._milestones

        # Filter by year
        if year and year != "all":
            if "-" in year and len(year) > 4:
                # Year range like "2019-2020"
                parts = year.split("-")
                try:
                    start_year, end_year = int(parts[0]), int(parts[1])
                    results = [
                        m
                        for m in results
                        if any(
                            start_year <= int(y) <= end_year
                            for y in self._extract_years(m.get("date", ""))
                        )
                    ]
                except (ValueError, IndexError) as exc:
                    logger.warning(
                        "Failed to parse year range '%s': %s", year, exc
                    )
                    # Fall back to substring match
                    results = [m for m in results if year in m.get("date", "")]
            else:
                # Single year
                results = [m for m in results if year in m.get("date", "")]

        # Filter by category
        if category:
            results = [m for m in results if m.get("category", "") == category]

        if not results:
            return (
                f"未找到 {year} 年"
                + (f" {category} 类别" if category else "")
                + "的里程碑事件。"
            )

        lines = []
        for m in results:
            date = m.get("date", "")
            event = m.get("event", "")
            cat = m.get("category", "")
            lines.append(f"[{date}] ({cat}) {event}")

        return "\n".join(lines)

    # ---- Helpers ----------------------------------------------------

    @staticmethod
    def _extract_years(date_str: str) -> List[str]:
        """Extract 4-digit year strings from a date string."""
        return re.findall(r"\d{4}", date_str)
