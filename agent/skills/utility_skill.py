"""
UtilitySkill — general-purpose utility functions.

Functions:
    • get_current_datetime — return current date/time in Chinese format

Dependencies: none
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

from agent.skills.base_skill import BaseSkill

logger = logging.getLogger(__name__)

# Pre-computed constant — avoids re-creating the tuple on every call
_WEEKDAY_NAMES = ("星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日")


class UtilitySkill(BaseSkill):
    """Skill providing miscellaneous utility functions."""

    # ---- BaseSkill interface ----------------------------------------

    @property
    def name(self) -> str:
        return "utility"

    @property
    def description(self) -> str:
        return (
            "Utility skill — provides helper functions such as getting "
            "the current date and time."
        )

    def configure(self, **kwargs: Any) -> None:
        """No external dependencies required."""
        self._finalize()

    def _register_functions(self) -> None:
        self._add_function(
            name="get_current_datetime",
            description=(
                "Get the current date and time. "
                "Use this when you need to determine whether a concert or event "
                "is in the past or in the future, or when the fan asks about "
                "'today', 'now', or time-relative questions."
            ),
            parameters={
                "type": "object",
                "properties": {},
                "required": [],
            },
            func=self._get_current_datetime,
        )

    # ---- Function implementation ------------------------------------

    @staticmethod
    def _get_current_datetime() -> str:
        """Return the current date/time in a human-friendly Chinese format."""
        now = datetime.now()
        weekday = _WEEKDAY_NAMES[now.weekday()]
        result = f"当前时间：{now.strftime('%Y年%m月%d日 %H:%M')} {weekday}"
        logger.info("get_current_datetime: %s", result)
        return result
